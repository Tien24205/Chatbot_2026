"""
BƯỚC 2-3 — Kiểm thử tầng loader + chunking.

Plan §Bước 10 yêu cầu test từng tầng chứ không đợi xong hết mới test.
Đây là Test 1 (Document Processing) và Test 2 (Chunking).

Không cần API key, không cần Docker. Chạy:  python 01_test_chunking.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from rag import config
from rag.checks import check, report
from rag.chunker import (
    MIN_CHUNK_CHARS,
    OVERLAP_CHARS,
    TARGET_CHARS,
    TARGET_TOKENS,
    chunk_document,
)
from rag.loader import Document, load_directory
from rag.text import BULLET

KB_DIR = config.KB_DIR


def main() -> None:
    print("=" * 66)
    print("TEST 1 — DOCUMENT PROCESSING (đọc + làm sạch)")
    print("=" * 66)

    docs = load_directory(KB_DIR)
    print(f"\n  Đọc được {len(docs)} tài liệu từ {KB_DIR.name}/\n")
    for d in docs:
        print(f"    {d.source_name:<32} {d.char_count:>6} ký tự")

    print()
    check(all(d.text for d in docs), "Không tài liệu nào rỗng")
    check(
        all("\r" not in d.text for d in docs),
        "Đã thống nhất xuống dòng về LF (không còn \\r)",
    )
    check(
        all("\n\n\n" not in d.text for d in docs),
        "Không còn >1 dòng trống liên tiếp",
    )
    check(
        all(not any(ln != ln.rstrip() for ln in d.text.split("\n")) for d in docs),
        "Không còn khoảng trắng cuối dòng",
    )
    check(
        all(d.text == unicodedata.normalize("NFC", d.text) for d in docs),
        "Tiếng Việt đã chuẩn hoá Unicode NFC",
    )

    print("\n" + "=" * 66)
    print("TEST 2 — CHUNKING")
    print("=" * 66)
    print(
        f"\n  Cấu hình: target={TARGET_CHARS} ký tự (~{TARGET_TOKENS} token), "
        f"overlap={OVERLAP_CHARS} ký tự\n"
    )

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        sizes = [c.char_count for c in chunks]
        sections = len({c.section for c in chunks})
        print(
            f"    {doc.source_name:<32} {len(chunks):>2} chunk / {sections:>2} mục"
            f"   ký tự: min {min(sizes)} · trung bình {sum(sizes) // len(sizes)} · max {max(sizes)}"
        )

    print(f"\n  Tổng: {len(all_chunks)} chunk\n")

    # --- Kiểm tra kích thước ---
    oversized = [c for c in all_chunks if c.char_count > TARGET_CHARS * 1.15]
    check(
        not oversized,
        f"Không chunk nào vượt quá {int(TARGET_CHARS * 1.15)} ký tự",
        f"vượt: {[(c.source_name, c.chunk_index, c.char_count) for c in oversized]}",
    )

    tiny = [c for c in all_chunks if c.char_count < MIN_CHUNK_CHARS]
    check(
        not tiny,
        f"Không chunk nào ngắn hơn {MIN_CHUNK_CHARS} ký tự (chunk rác)",
        f"quá ngắn: {[(c.source_name, c.chunk_index, c.char_count) for c in tiny]}",
    )

    # --- Kiểm tra không mất nội dung ---
    # QUAN TRỌNG: phép kiểm tra này KHÔNG dùng _split_units. Bản trước dùng chính
    # hàm đang được kiểm thử để tính kỳ vọng, nên khi hàm đó có bug thì test vẫn
    # PASS (so output sai với kỳ vọng sai). Ở đây dùng oracle độc lập: đọc thẳng
    # từng dòng của file nguồn.
    lost: list[tuple[str, str]] = []
    for doc in docs:
        blob = " ".join(c.content for c in chunk_document(doc))
        blob = " ".join(blob.split())  # chuẩn hoá khoảng trắng để so khớp
        for line in doc.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " ".join(line.split()) not in blob:
                lost.append((doc.source_name, line[:60]))
    check(
        not lost,
        "Không mất nội dung — mọi dòng của file nguồn đều nằm trong ít nhất 1 chunk",
        f"thiếu {len(lost)} dòng, ví dụ: {lost[:3]}",
    )

    # --- Regression: ranh giới gạch đầu dòng phải khớp chính xác với nguồn ---
    # Oracle độc lập: đọc thẳng file nguồn, dựng danh sách mục mong đợi (một mục =
    # dòng gạch đầu dòng + các dòng THỤT LỀ nối tiếp ngay sau nó), rồi đòi hỏi mỗi
    # mục xuất hiện nguyên vẹn thành MỘT DÒNG trong chunk.
    # Bug đã sửa: "- Nham (Geo)" từng bị dính với câu dẫn của đoạn sau.

    def expected_bullets(text: str) -> list[str]:
        lines, out, i = text.split("\n"), [], 0
        while i < len(lines):
            if BULLET.match(lines[i]):
                unit, i = lines[i].strip(), i + 1
                while (
                    i < len(lines)
                    and re.match(r"^\s+\S", lines[i])
                    and not BULLET.match(lines[i])
                ):
                    unit += " " + lines[i].strip()
                    i += 1
                out.append(unit)
            else:
                i += 1
        return out

    mismatched: list[tuple[str, str]] = []
    for doc in docs:
        chunk_lines = {ln for c in chunk_document(doc) for ln in c.content.split("\n")}
        for unit in expected_bullets(doc.text):
            if unit not in chunk_lines:
                mismatched.append((doc.source_name, unit[:70]))
    check(
        not mismatched,
        "Ranh giới gạch đầu dòng khớp chính xác nguồn (không dính câu đoạn kế tiếp)",
        f"lệch {len(mismatched)} mục, ví dụ: {mismatched[:2]}",
    )

    # --- Kiểm tra không cắt giữa câu ---
    # Mỗi chunk phải kết thúc bằng dấu câu, dấu đóng ngoặc, hoặc là một mục danh sách.
    bad_endings = [
        c
        for c in all_chunks
        if not c.content.rstrip().endswith((".", "!", "?", ":", ")", "”", '"'))
    ]
    check(
        not bad_endings,
        "Không chunk nào bị cắt giữa câu",
        f"kết thúc bất thường: "
        f"{[(c.source_name, c.chunk_index, c.content[-40:]) for c in bad_endings[:3]]}",
    )

    # --- Kiểm tra metadata cho trích nguồn ---
    check(
        all(c.source_name and c.section for c in all_chunks),
        "Mọi chunk đều có source_name + section (phục vụ trích nguồn ở UI)",
    )

    # --- TEST 3: nhánh overlap ---
    # Với KB hiện tại, mỗi mục `##` đều nhỏ hơn ngưỡng 1800 ký tự nên mỗi mục ra
    # đúng 1 chunk -> nhánh overlap KHÔNG bao giờ chạy. Không hạ ngưỡng để ép test
    # pass (đó là chỉnh cấu hình cho vừa bài test); thay vào đó kiểm thử nhánh này
    # bằng dữ liệu tổng hợp đủ dài.
    print("\n" + "=" * 66)
    print("TEST 3 — NHÁNH OVERLAP (dữ liệu tổng hợp)")
    print("=" * 66)

    long_body = "\n\n".join(
        f"Câu số {i} nói về một cơ chế trong game và dài vừa đủ để lấp đầy chunk. "
        f"Đây là phần bổ sung của câu {i} để tăng độ dài đoạn văn bản này lên."
        for i in range(1, 41)
    )
    synthetic = Document(
        source_name="synthetic.txt",
        path=Path("synthetic.txt"),
        text=f"# Tài liệu thử\n\n## Mục dài\n\n{long_body}",
    )
    syn_chunks = chunk_document(synthetic)
    print(f"\n  Tài liệu tổng hợp: {len(synthetic.text)} ký tự -> {len(syn_chunks)} chunk")

    check(len(syn_chunks) > 1, "Tài liệu dài bị cắt thành nhiều chunk")

    shared = 0
    for a, b in zip(syn_chunks, syn_chunks[1:], strict=False):
        tail_lines = [ln for ln in a.content.split("\n") if ln.strip()][1:]  # bỏ dòng ngữ cảnh
        if tail_lines and tail_lines[-1] in b.content:
            shared += 1
    check(
        shared == len(syn_chunks) - 1,
        f"Mọi cặp chunk liền kề đều có overlap ({shared}/{len(syn_chunks) - 1} cặp)",
    )
    check(
        all(c.char_count <= TARGET_CHARS * 1.15 for c in syn_chunks),
        "Chunk tổng hợp vẫn nằm trong ngưỡng kích thước",
    )

    # --- In thử một chunk để mắt người kiểm tra ---
    print("\n" + "=" * 66)
    print("CHUNK MẪU (chunk #1 của tài liệu đầu tiên)")
    print("=" * 66)
    sample = chunk_document(docs[0])[1]
    print(f"\n  nguồn : {sample.source_name}")
    print(f"  mục   : {sample.section}")
    print(f"  cỡ    : {sample.char_count} ký tự (~{sample.approx_tokens} token)")
    print("  ---")
    for line in sample.content.split("\n"):
        print(f"  {line}")

    # --- Kết luận ---
    report(
        success_note=(
            f"\n{len(all_chunks)} chunk đã sẵn sàng để embedding.\n"
            "Bước tiếp theo (Bước 4-5) cần API key + Docker:\n"
            "  1. Chạy 00_smoke_test.py để lấy EMBED_DIMENSION\n"
            "  2. Bật Docker Desktop rồi dựng pgvector\n"
        )
    )


if __name__ == "__main__":
    main()

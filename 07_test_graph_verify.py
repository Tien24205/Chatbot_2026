"""
Kiểm thử tầng đồ thị (GraphRAG) và các lớp chống hallucination tất định.

Plan §Bước 10 yêu cầu test từng tầng. Đây là Test 8 (Graph) và Test 9 (Verification).

KHÔNG cần API key. Cần Docker cho phần đồ thị (đọc bảng chunks); phần kiểm chứng
chạy hoàn toàn offline. Chạy:

  .venv\\Scripts\\python.exe 07_test_graph_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rag import config, graph, store, verify
from rag.loader import load_directory
from rag.store import SearchHit

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        if detail:
            print(f"         {detail}")
        failures.append(label)


def hit(content: str, similarity: float = 0.7, via: str = "vector", cid: int = 1) -> SearchHit:
    return SearchHit(
        content=content, source_name="test.txt", section="Mục thử", similarity=similarity,
        id=cid, via=via,
    )


# ===========================================================================
# TEST 8 — TRÍCH THỰC THỂ VÀ ĐỒ THỊ
# ===========================================================================


def test_extraction() -> None:
    print("=" * 66)
    print("TEST 8 — TRÍCH THỰC THỂ (offline, không cần database)")
    print("=" * 66 + "\n")

    docs = load_directory(config.KB_DIR)
    entities, edges = graph.extract(docs)
    by_name = {e.name: e for e in entities}

    check(len(entities) > 50, f"Trích được nhiều thực thể ({len(entities)})")

    check("Baron Nashor" in by_name, "Bắt được thực thể từ dòng gạch đầu dòng")
    check("Đường trên" in by_name, "Bắt được thực thể từ dòng đánh số")

    # Bí danh trong ngoặc — thứ khiến "Rift Herald" tìm được dù KB viết tiếng Việt.
    sgkn = by_name.get("Sứ Giả Khe Nứt")
    check(
        sgkn is not None and "Rift Herald" in sgkn.aliases,
        "Tách được bí danh trong ngoặc",
        f"aliases={sgkn.aliases if sgkn else None}",
    )
    # Tra theo tên là KHÔNG đủ: "Xạ thủ" tồn tại ở cả hai game, bản Liên Quân
    # không có bí danh. Phải tra đúng bản của Liên Minh.
    adc = next(
        (e for e in entities
         if e.name == "Xạ thủ" and e.source_name == "lien_minh_huyen_thoai.txt"),
        None,
    )
    check(
        adc is not None and {"ADC", "Bot"} <= set(adc.aliases),
        "Tách được nhiều bí danh ngăn bởi dấu /",
        f"aliases={adc.aliases if adc else None}",
    )

    # Văn xuôi bám đuôi tên: "Thánh Di Vật (Artifact), gồm năm vị trí: ..."
    check(
        "Thánh Di Vật" in by_name,
        "Cắt được phần văn xuôi bám sau tên",
        f"tên gần đúng: {[n for n in by_name if 'Thánh' in n]}",
    )
    # ...nhưng KHÔNG được cắt nhầm tên vốn có dấu phẩy.
    check(
        any("Bo3" in n and "Bo5" in n for n in by_name),
        "Không cắt nhầm tên có dấu phẩy đi kèm chữ hoa (Bo3, Bo5)",
        f"tên gần đúng: {[n for n in by_name if 'Bo3' in n]}",
    )

    # Vai trò suy ra từ tiêu đề mục — đây là thứ tạo ra cạnh `tương_tự`.
    check(
        by_name.get("Baron Nashor") is not None
        and by_name["Baron Nashor"].kind == "mục tiêu trung lập",
        "Suy được vai trò từ tiêu đề mục",
        f"kind={by_name['Baron Nashor'].kind if 'Baron Nashor' in by_name else None}",
    )

    # Cùng một tên ở hai game phải là HAI thực thể, không được gộp.
    duong_giua = [e for e in entities if e.name.lower() == "đường giữa"]
    check(
        len({e.source_name for e in duong_giua}) == 2,
        "Giữ riêng 'Đường Giữa' của hai game thay vì gộp theo tên",
        f"tìm thấy {len(duong_giua)} bản: {[e.source_name for e in duong_giua]}",
    )

    # Cạnh tương_tự PHẢI nối chéo tài liệu, không bao giờ nối trong cùng một file.
    by_key = {e.key: e for e in entities}
    similar = [e for e in edges if e.relation == "tương_tự"]
    orphan = [e for e in edges if e.src not in by_key or e.dst not in by_key]
    check(not orphan, f"Mọi cạnh đều trỏ tới thực thể có thật ({len(orphan)} cạnh mồ côi)")

    same_file = [
        e for e in similar if by_key[e.src].source_name == by_key[e.dst].source_name
    ]
    check(len(similar) > 0, f"Có cạnh tương_tự giữa các game ({len(similar)} cạnh)")
    check(not same_file, "Cạnh tương_tự không bao giờ nối trong cùng một tài liệu")

    pairs = {(by_key[e.src].name, by_key[e.dst].name) for e in similar}
    pairs |= {(b, a) for a, b in pairs}
    check(
        ("Baron Nashor", "Thần Rừng") in pairs,
        "Nối được Baron Nashor <-> Thần Rừng (thứ vector KHÔNG nối được)",
    )


def test_lexicon() -> None:
    print("\n" + "=" * 66)
    print("TEST 8b — DÒ TÊN TRONG VĂN BẢN")
    print("=" * 66 + "\n")

    docs = load_directory(config.KB_DIR)
    entities, _ = graph.extract(docs)
    lex = graph.Lexicon(entities)

    found = lex.find("Baron Nashor cho buff toàn đội")
    check("Baron Nashor" in found, "Tìm được tên đầy đủ")

    # Khớp dài trước: "Rồng Ngàn Tuổi" KHÔNG được tính thành "Rồng".
    # (Nếu câu có thêm chỗ nhắc "Rồng" đứng riêng thì trả về cả hai là ĐÚNG —
    #  ở đây kiểm tra riêng trường hợp chỉ có một lần xuất hiện.)
    found = lex.find("Rồng Ngàn Tuổi xuất hiện muộn trong trận")
    check(
        "Rồng Ngàn Tuổi" in found and "Rồng" not in found,
        "Khớp cụm dài trước, không cắt thành thực thể ngắn hơn",
        f"tìm thấy: {found}",
    )

    check("Sứ Giả Khe Nứt" in lex.find("Rift Herald dùng để phá trụ"), "Tìm được qua bí danh")
    check(not lex.find("Hôm nay trời đẹp và tôi muốn nấu phở"), "Không bắt nhầm văn bản vô quan")


def test_graph_db() -> None:
    print("\n" + "=" * 66)
    print("TEST 8c — ĐỒ THỊ TRONG DATABASE")
    print("=" * 66 + "\n")

    try:
        conn = store.connect()
    except Exception as e:
        print(f"  [BỎ QUA] Không kết nối được database: {e}")
        print("           Chạy `docker compose up -d` rồi thử lại.\n")
        return

    with conn:
        if not graph.is_built(conn):
            print("  [BỎ QUA] Đồ thị chưa dựng. Chạy 06_build_graph.py trước.\n")
            return

        lex = graph.load_lexicon(conn)
        check(bool(lex.find("Baron Nashor")), "Nạp được bộ dò tên từ database")

        nbrs = graph.neighbors(conn, ["Baron Nashor"], ("tương_tự",))
        names = [n["name"] for n in nbrs]
        check("Thần Rừng" in names, "Đi được một bước sang thực thể cùng vai trò", f"{names}")
        check("Baron Nashor" not in names, "Không trả về chính nó làm hàng xóm")
        check(
            all(n["from_name"] == "Baron Nashor" for n in nbrs),
            "Ghi lại được đi TỪ thực thể nào, để giải thích được vì sao đoạn được kéo vào",
        )
        check(
            all(n["kind"] == "mục tiêu trung lập" for n in nbrs),
            "Hàng xóm qua cạnh tương_tự luôn cùng vai trò",
            f"{[(n['name'], n['kind']) for n in nbrs]}",
        )

        chunks = graph.chunks_mentioning(conn, names, limit=3)
        sources = {c["source_name"] for c in chunks}
        check(
            "lien_quan_mobile.txt" in sources,
            "Từ thực thể hàng xóm lấy được chunk của game khác",
            f"{sources}",
        )

        comms = graph.communities(conn)
        check(len(comms) == 4, f"Tóm tắt cộng đồng theo 4 tài liệu ({len(comms)})")
        check(
            all(c["entities"] > 0 and c["kinds"] for c in comms),
            "Mỗi cộng đồng đều có thực thể và vai trò",
        )


# ===========================================================================
# TEST 9 — CÁC LỚP CHỐNG HALLUCINATION TẤT ĐỊNH
# ===========================================================================


def test_citations() -> None:
    print("\n" + "=" * 66)
    print("TEST 9 — LỚP 3: TRÍCH DẪN CÓ CẤU TRÚC")
    print("=" * 66 + "\n")

    text, invalid = verify.strip_invalid_citations("Baron cho buff [1] và tiếp sức lính [7].", 4)
    check(invalid == [7], "Phát hiện chỉ số trích dẫn không tồn tại", f"invalid={invalid}")
    check("[7]" not in text, "Đã bỏ chỉ số không tồn tại khỏi câu trả lời", text)
    check("[1]" in text, "Giữ nguyên chỉ số hợp lệ", text)
    check(text.endswith("."), "Không để lại khoảng trắng trước dấu câu", repr(text))

    _, invalid = verify.strip_invalid_citations("Không trích gì cả.", 4)
    check(invalid == [], "Câu không trích dẫn thì không bị báo lỗi")

    r = verify.check("Ăn đủ bốn rồng nhận Linh Hồn Rồng [1]. Đây là lợi thế lớn.",
                     [hit("Ăn đủ bốn rồng sẽ nhận Linh Hồn Rồng, một lợi thế rất lớn.")])
    check(0 < r.citation_coverage < 1, f"Đo được tỉ lệ câu có trích dẫn ({r.citation_coverage:.0%})")


def test_grounding() -> None:
    print("\n" + "=" * 66)
    print("TEST 9b — LỚP 4: KIỂM CHỨNG GROUNDING TẤT ĐỊNH")
    print("=" * 66 + "\n")

    context = hit("Mỗi lượt cầu nguyện tốn 160 Nguyên Thạch. Bảo hiểm 90 lượt.")

    r = verify.check("Mỗi lượt tốn 160 Nguyên Thạch, bảo hiểm ở lượt 90.", [context])
    check(r.ok, "Câu trả lời đúng số liệu thì sạch cờ", str(r.flags))

    # Đây là lỗi mà system prompt CHỈ có thể khuyên, không thể chặn.
    r = verify.check("Mỗi lượt tốn 180 Nguyên Thạch.", [context])
    check(r.ungrounded_numbers == ["180"], "Bắt được số bịa", str(r.ungrounded_numbers))
    check(verify.escalate_needed(r), "Số bịa thì leo thang sang lớp 5")

    # Số do người dùng nêu trong câu hỏi không phải là bịa.
    r = verify.check("Bạn hỏi về 5 lượt thì tốn 800.", [context], question="Tôi quay 5 lượt?")
    check("5" not in r.ungrounded_numbers, "Số lấy từ câu hỏi không bị tính là bịa")
    check("800" in r.ungrounded_numbers, "Nhưng số tự tính ra thì vẫn bị bắt")

    # Số nhỏ trong lối diễn đạt thông thường không tính.
    r = verify.check("Có hai đội, mỗi đội 1 người dẫn.", [context])
    check(r.ok, "Số 0/1/2 trong lối nói không gây báo động giả", str(r.flags))

    # Tên riêng không có trong ngữ cảnh — model lôi từ kiến thức nền ra.
    r = verify.check("Cơ chế này giống Honkai Star Rail của HoYoverse.", [context])
    check(bool(r.ungrounded_names), "Bắt được tên riêng lạ", str(r.ungrounded_names))

    # Chữ đầu câu viết hoa KHÔNG phải tên riêng. Ca thật gặp khi chạy 08_eval_answers:
    # "Trong Liên Minh Huyền Thoại..." từng bị báo là bịa tên "Trong Liên".
    lmht = hit("Liên Minh Huyền Thoại là game MOBA do Riot Games phát triển.")
    r = verify.check("Trong Liên Minh Huyền Thoại, Riot Games là nhà phát triển.", [lmht])
    check(
        not r.ungrounded_names,
        "Chữ đầu câu viết hoa không bị tính là tên riêng bịa",
        str(r.ungrounded_names),
    )
    r = verify.check("Game này do Riot Games và Tencent Holdings phát triển.", [lmht])
    check(
        any("Tencent" in n for n in r.ungrounded_names),
        "Nhưng tên riêng thật sự lạ giữa câu vẫn bị bắt",
        str(r.ungrounded_names),
    )

    docs = load_directory(config.KB_DIR)
    entities, _ = graph.extract(docs)
    lex = graph.Lexicon(entities)

    # Thực thể CÓ THẬT trong knowledge base nhưng KHÔNG có trong đoạn được truy hồi:
    # thông tin nghe đúng nhưng không đến từ nguồn đã trích — rất khó thấy bằng mắt.
    r = verify.check(
        "Nguyên Thạch dùng cho gacha, tương tự Baron Nashor.", [context], lexicon=lex
    )
    check(
        "Baron Nashor" in r.ungrounded_entities,
        "Bắt được thực thể có thật nhưng nằm ngoài ngữ cảnh đã truy hồi",
        str(r.ungrounded_entities),
    )


def main() -> None:
    test_extraction()
    test_lexicon()
    test_graph_db()
    test_citations()
    test_grounding()

    print("\n" + "=" * 66)
    if failures:
        print(f"THẤT BẠI — {len(failures)} kiểm thử không đạt")
        for f in failures:
            print(f"  - {f}")
        print("=" * 66)
        sys.exit(1)
    print("TẤT CẢ ĐỀU ĐẠT")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()

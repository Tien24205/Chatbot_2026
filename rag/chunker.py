"""
BƯỚC 3 — Chunking.

Chiến lược: cắt theo CẤU TRÚC trước, theo KÍCH THƯỚC sau.

Lý do: các file trong knowledge_base/ có tiêu đề `##` phân tách chủ đề rõ ràng
("## Hệ thống Nguyên Tố", "## Mục tiêu trung lập"...). Cắt theo tiêu đề giữ trọn
đơn vị ngữ nghĩa; cắt mù theo số ký tự sẽ xé đôi một khái niệm thành 2 chunk và
không chunk nào trả lời được câu hỏi.

Mỗi chunk được gắn thêm dòng ngữ cảnh "<tên tài liệu> — <tiêu đề mục>" ở đầu, để
khi đứng một mình chunk vẫn biết nó đang nói về game nào.

Không phụ thuộc API key — chạy được offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Document

# --- Ước lượng token ------------------------------------------------------
# Chưa gọi được API nên chưa đếm token chính xác. Với tiếng Việt, Gemini
# tokenize khoảng 3 ký tự / token. Sau khi có API key, kiểm chứng lại bằng
# client.models.count_tokens() rồi chỉnh hằng số này nếu lệch.
CHARS_PER_TOKEN = 3.0

TARGET_TOKENS = 600  # plan §Bước 3 đề xuất 500-1000
OVERLAP_TOKENS = 100  # plan §Bước 3 đề xuất 50-150

TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)  # ~1800
OVERLAP_CHARS = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)  # ~300
MIN_CHUNK_CHARS = 80  # dưới ngưỡng này thì gộp vào chunk trước, tránh chunk rác

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# Ranh giới câu tiếng Việt: dấu kết câu + khoảng trắng + chữ hoa/số.
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+(?=[A-ZĐÀ-Ỹ0-9])")


@dataclass(frozen=True)
class Chunk:
    """Một đoạn văn bản sẵn sàng để embedding."""

    content: str  # text sẽ được embedding (đã kèm dòng ngữ cảnh)
    source_name: str  # ví dụ "genshin_impact.txt" -> dùng để trích nguồn
    section: str  # tiêu đề mục, ví dụ "Hệ thống Nguyên Tố"
    chunk_index: int  # thứ tự trong tài liệu, bắt đầu từ 0

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def approx_tokens(self) -> int:
        return round(self.char_count / CHARS_PER_TOKEN)


def _split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Tách tài liệu thành (tiêu đề tài liệu, [(tiêu đề mục, nội dung mục), ...]).

    Dòng `# ...` là tiêu đề tài liệu; `## ...` trở xuống mở một mục mới.
    """
    doc_title = ""
    sections: list[tuple[str, list[str]]] = []
    current_title = "Mở đầu"
    current_lines: list[str] = []

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if not m:
            current_lines.append(line)
            continue

        level, title = len(m.group(1)), m.group(2).strip()
        if level == 1 and not doc_title:
            doc_title = title
            continue

        if current_lines and any(l.strip() for l in current_lines):
            sections.append((current_title, current_lines))
        current_title, current_lines = title, []

    if current_lines and any(l.strip() for l in current_lines):
        sections.append((current_title, current_lines))

    return doc_title, [(t, "\n".join(ls).strip()) for t, ls in sections]


def _split_units(body: str) -> list[str]:
    """
    Cắt nội dung mục thành các đơn vị nhỏ nhất KHÔNG được phép xé đôi.

    - Đoạn văn bản thường -> cắt theo câu.
    - Đoạn dạng danh sách (`- ...`, `1. ...`) -> mỗi mục là một đơn vị,
      không cắt giữa một gạch đầu dòng.
    """
    units: list[str] = []

    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue

        lines = para.split("\n")
        is_list = sum(bool(re.match(r"^\s*(?:[-*•]|\d+\.)\s+", l)) for l in lines) >= max(
            1, len(lines) // 2
        )

        if is_list:
            # Gom dòng nối tiếp vào mục danh sách phía trên — nhưng CHỈ khi dòng đó
            # thực sự thụt lề. Dòng không thụt lề là câu dẫn của đoạn, phải đứng riêng.
            # (Dùng para_units cục bộ, không dùng `units` toàn cục: nếu không, câu dẫn
            #  của đoạn này sẽ bị dính vào mục cuối của đoạn TRƯỚC.)
            para_units: list[str] = []
            for line in lines:
                is_bullet = bool(re.match(r"^\s*(?:[-*•]|\d+\.)\s+", line))
                is_continuation = bool(re.match(r"^\s+\S", line)) and not is_bullet
                if is_continuation and para_units:
                    para_units[-1] += " " + line.strip()
                else:
                    para_units.append(line.strip())
            units.extend(u for u in para_units if u)
        else:
            flat = " ".join(l.strip() for l in lines)
            units.extend(s.strip() for s in _SENTENCE_END.split(flat) if s.strip())

    return units


def _pack(units: list[str], target: int, overlap: int) -> list[str]:
    """Gom các đơn vị thành chunk <= target ký tự, có overlap giữa 2 chunk liền kề."""
    if not units:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    for unit in units:
        # +1 cho ký tự nối. Nếu buf đã có nội dung mà thêm vào thì vượt -> chốt chunk.
        if buf and size + len(unit) + 1 > target:
            chunks.append("\n".join(buf))

            # Overlap: giữ lại các đơn vị cuối cùng cho đủ ~overlap ký tự.
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(buf):
                if tail_size + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            buf, size = tail, tail_size

        buf.append(unit)
        size += len(unit) + 1

    if buf:
        chunks.append("\n".join(buf))

    # Chunk cuối quá ngắn thì gộp ngược vào chunk trước, tránh sinh chunk rác.
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] = chunks[-2] + "\n" + chunks[-1]
        chunks.pop()

    return chunks


def chunk_document(
    doc: Document,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Cắt một tài liệu thành danh sách Chunk."""
    doc_title, sections = _split_sections(doc.text)
    doc_title = doc_title or doc.source_name

    chunks: list[Chunk] = []
    index = 0

    for section_title, body in sections:
        # Dòng ngữ cảnh chiếm chỗ trong ngân sách ký tự -> trừ ra trước khi pack.
        header = f"{doc_title} — {section_title}"
        budget = max(target_chars - len(header) - 1, MIN_CHUNK_CHARS)

        for piece in _pack(_split_units(body), budget, overlap_chars):
            chunks.append(
                Chunk(
                    content=f"{header}\n{piece}",
                    source_name=doc.source_name,
                    section=section_title,
                    chunk_index=index,
                )
            )
            index += 1

    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, **kwargs)]

"""
CHỐNG HALLUCINATION — kiểm chứng câu trả lời SAU khi LLM sinh ra.

Toàn bộ hệ thống có 5 lớp, mỗi lớp chặn một loại lỗi khác nhau. File này chứa
lớp 3 và lớp 4; lớp 1-2-5 nằm ở chỗ khác nhưng ghi lại đây để thấy toàn cảnh:

  Lớp 1 — Ràng buộc phạm vi bằng system prompt        (rag/prompt.py)
          Chặn: LLM dùng kiến thức nền thay vì ngữ cảnh.
          Chi phí: 0.

  Lớp 2 — Ngưỡng similarity τ, hiệu chuẩn bằng đo đạc  (rag/pipeline.py)
          Chặn: không có chunk nào đủ liên quan -> từ chối TRƯỚC khi gọi LLM.
          Chi phí: 0 (còn tiết kiệm được một lệnh gọi API).

  Lớp 3 — Trích dẫn có cấu trúc, kiểm tra chỉ số       (file này)
          Chặn: LLM bịa nguồn — trích [7] trong khi chỉ được đưa 4 đoạn.
          Chi phí: 0.

  Lớp 4 — Kiểm chứng grounding tất định                (file này)
          Chặn: bịa số liệu và bịa tên riêng.
          Chi phí: 0.

  Lớp 5 — Vòng entailment bằng LLM                     (rag/pipeline.py)
          Chặn: câu SUY DIỄN sai từ ngữ cảnh đúng — thứ mà regex không thấy.
          Chi phí: 1 lệnh gọi API, nên mặc định chỉ chạy khi lớp 3-4 đã cảnh báo.

VÌ SAO CẦN LỚP 3-4 khi đã có lớp 1 (system prompt)?

Vì system prompt là YÊU CẦU, không phải RÀNG BUỘC. Quy tắc số 7 trong prompt ghi
"không bịa số liệu" — nhưng không có gì bảo đảm model tuân theo, và khi nó không
tuân thì hệ thống hoàn toàn không biết. Lớp 3-4 biến yêu cầu đó thành phép kiểm
tra chạy được: mọi con số trong câu trả lời phải tìm thấy được trong ngữ cảnh, nếu
không thì hệ thống tự phát hiện chứ không chờ người dùng phát hiện hộ.

Nguyên tắc: kiểm chứng ở đây CHỈ báo cờ, không tự ý sửa câu trả lời. Quyết định
làm gì với cờ là việc của pipeline (xem escalate_needed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .store import SearchHit
from .text import UPPER as _UPPER

# Chỉ số trích dẫn: [1], [2]...
_CITATION = re.compile(r"\[(\d+)\]")
# Số: 90, 1.5, 1,5, 50%, 12 — bỏ qua phần đơn vị.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_SENTENCE_END = re.compile(rf"(?<=[.!?:])\s+(?=[{_UPPER}0-9])")
# Cụm danh từ riêng: từ hai chữ viết hoa liên tiếp trở lên ("Rồng Bạo Chúa").
# Yêu cầu hai chữ để tránh bắt nhầm chữ đầu câu.
_PROPER = re.compile(rf"(?:[{_UPPER}]\w*(?:\s+|$)){{2,}}")

# Số nhỏ hay xuất hiện trong lối diễn đạt ("một trong những", "hai đội") chứ không
# phải số liệu trích từ tài liệu. Kiểm tra chúng chỉ tạo báo động giả.
_NOT_A_FACT = {"0", "1", "2"}


@dataclass
class Report:
    """Kết quả kiểm chứng một câu trả lời. Chỉ ghi nhận, không phán xử."""

    invalid_citations: list[int] = field(default_factory=list)
    citation_coverage: float = 0.0  # tỉ lệ câu có ít nhất một [n]
    ungrounded_numbers: list[str] = field(default_factory=list)
    ungrounded_entities: list[str] = field(default_factory=list)
    ungrounded_names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.flags

    @property
    def flags(self) -> list[str]:
        out = []
        if self.invalid_citations:
            out.append(f"trích dẫn không tồn tại: {self.invalid_citations}")
        if self.ungrounded_numbers:
            out.append(f"số không có trong ngữ cảnh: {self.ungrounded_numbers}")
        if self.ungrounded_entities:
            out.append(f"thực thể không có trong ngữ cảnh: {self.ungrounded_entities}")
        if self.ungrounded_names:
            out.append(f"tên riêng không có trong ngữ cảnh: {self.ungrounded_names}")
        return out


def _normalize_number(token: str) -> str:
    """"1,5" và "1.5" là cùng một số; so sánh dạng chuẩn hoá."""
    return token.replace(",", ".").rstrip("0").rstrip(".") if "," in token or "." in token else token


def strip_invalid_citations(text: str, n_sources: int) -> tuple[str, list[int]]:
    """
    LỚP 3 — bỏ các chỉ số trích dẫn trỏ ra ngoài số đoạn thực sự được cung cấp.

    Hiển thị "[7]" cho người dùng trong khi chỉ có 4 nguồn thì tệ hơn là không
    trích gì: người đọc tưởng có một nguồn thứ 7 nào đó đã được đối chiếu.
    """
    invalid: list[int] = []

    def replace(m: re.Match) -> str:
        idx = int(m.group(1))
        if 1 <= idx <= n_sources:
            return m.group(0)
        invalid.append(idx)
        return ""

    cleaned = _CITATION.sub(replace, text)
    # Dọn khoảng trắng thừa để lại sau khi xoá.
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip(), invalid


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def check(
    answer: str,
    hits: list[SearchHit],
    question: str = "",
    lexicon=None,
) -> Report:
    """
    LỚP 3 + 4 — soi câu trả lời trên chính ngữ cảnh đã đưa cho LLM.

    `lexicon` là rag.graph.Lexicon (tuỳ chọn). Có thì bắt được cả trường hợp LLM
    kéo một thực thể có thật trong knowledge base vào câu trả lời dù thực thể đó
    KHÔNG nằm trong các đoạn được truy hồi — dạng lỗi rất khó thấy bằng mắt vì
    thông tin nghe đúng, chỉ là không đến từ nguồn được trích.
    """
    report = Report()
    context = "\n".join(h.content for h in hits)
    haystack = f"{context}\n{question}"  # số do người dùng nêu trong câu hỏi không tính là bịa

    # --- Lớp 3: trích dẫn -------------------------------------------------
    cited = [int(m.group(1)) for m in _CITATION.finditer(answer)]
    report.invalid_citations = sorted({c for c in cited if not 1 <= c <= len(hits)})

    sents = _sentences(answer)
    if sents:
        report.citation_coverage = sum(bool(_CITATION.search(s)) for s in sents) / len(sents)

    # --- Lớp 4a: số liệu --------------------------------------------------
    context_numbers = {_normalize_number(m.group(0)) for m in _NUMBER.finditer(haystack)}
    seen_num: set[str] = set()
    for m in _NUMBER.finditer(_CITATION.sub("", answer)):  # bỏ [1] khỏi phép đếm số
        raw = m.group(0)
        norm = _normalize_number(raw)
        if norm in _NOT_A_FACT or norm in seen_num:
            continue
        seen_num.add(norm)
        if norm not in context_numbers:
            report.ungrounded_numbers.append(raw)

    # --- Lớp 4b: thực thể có trong knowledge base --------------------------
    if lexicon is not None:
        in_context = set(lexicon.find(context))
        for name in lexicon.find(answer):
            if name not in in_context:
                report.ungrounded_entities.append(name)

    # --- Lớp 4c: cụm danh từ riêng lạ --------------------------------------
    # Bắt cả những tên KHÔNG có trong knowledge base — tức là model lôi từ kiến
    # thức nền của nó ra, đúng thứ mà quy tắc số 1 cấm.
    context_lower = haystack.lower()
    for m in _PROPER.finditer(answer):
        phrase = m.group(0).strip()
        if len(phrase) < 6 or phrase.lower() in context_lower:
            continue

        words = phrase.split()

        # Chữ đầu câu viết hoa vì nó đứng đầu câu, không phải vì là tên riêng.
        # Không loại nó ra thì "Trong Liên Minh Huyền Thoại..." sinh ra cặp
        # "Trong Liên" — không có trong ngữ cảnh, và bị báo là tên bịa. Đo được
        # 2/5 câu trả lời đúng bị gắn cờ oan đúng vì lý do này.
        before = answer[: m.start()].rstrip(" \t")
        if not before.strip() or before[-1:] in {"\n", ".", "!", "?", ":", "-", "•"}:
            words = words[1:]

        # Cụm dài có thể là nhiều tên đứng cạnh nhau; soi từng cặp từ để biết
        # chính xác chỗ nào không khớp, thay vì loại cả cụm.
        # strict=False: words[1:] cố tình ngắn hơn words một phần tử.
        for a, b in zip(words, words[1:], strict=False):
            pair = f"{a} {b}"
            if pair.lower() not in context_lower and pair not in report.ungrounded_names:
                report.ungrounded_names.append(pair)

    return report


def escalate_needed(report: Report) -> bool:
    """
    Có đáng tiêu một lệnh gọi API cho lớp 5 (entailment) không?

    Hạn mức free tier đo được chỉ 20 request, nên không thể chạy entailment cho
    mọi câu trả lời. Cách dùng: chỉ leo thang khi lớp 3-4 đã thấy dấu hiệu khả
    nghi — thứ mà regex bắt được thì rẻ, thứ nó không bắt được mới cần LLM.
    """
    return bool(report.invalid_citations or report.ungrounded_numbers or report.ungrounded_entities)

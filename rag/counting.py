"""
Đếm TẤT ĐỊNH cho câu "có bao nhiêu <loại>?" — hướng B của kế hoạch suy luận.

VÌ SAO KHÔNG NHỜ LLM ĐẾM: đo được ở nhóm eval suy_luan rằng model nhỏ hoặc từ
chối đếm, hoặc đếm rồi bị nghi ngờ. Trong khi đó đồ thị tri thức ĐÃ có sẵn mỗi
trang wiki là một thực thể mang `kind` lấy từ infobox — đếm là một câu SQL,
kết quả đúng theo đúng nghĩa đen, 0 lệnh gọi API, 0 khả năng bịa.

PHẠM VI TRẢ LỜI: con số là SỐ TRANG TRONG KHO TÀI LIỆU (wiki đã cào có chọn
lọc), không nhất thiết bằng tổng số trong game — câu trả lời phải nói rõ điều
đó, và luôn nói.

CHỈ khớp CHÍNH XÁC tên loại hoặc bí danh đã kiểm chứng bằng dữ liệu. Tuyệt đối
không khớp chuỗi con: "phản ứng nguyên tố" chứa "nguyên tố", mà kind "nguyên tố"
là một rổ tạp (Aura, Ấn Nguyên Tố...) — khớp lỏng là trả lời sai tự tin. Câu
không khớp thì trả None để rơi về pipeline thường (suy luận có đánh dấu, lớp 8).
"""

from __future__ import annotations

import re

import psycopg

from . import graph

# "Có tổng cộng bao nhiêu Kiếm Đơn trong game?" -> "kiếm đơn"
_COUNT_Q = re.compile(
    r"bao nhiêu\s+(?:loại\s+|trang\s+)?(.+?)\s*"
    r"(?:trong\s+(?:kho|tài liệu|game|trò chơi|genshin)\b[^?]*)?[?.!…\s]*$",
    re.IGNORECASE,
)

# WHITELIST các kind đếm được — không phải mọi kind trong đồ thị.
#
# Điều kiện vào danh sách: MỘT TRANG = MỘT CÁ THỂ phải đúng thật. Đúng với loại
# danh mục (mỗi vũ khí, mỗi nhân vật một trang). SAI với loại khái niệm: kind
# "Phản Ứng Nguyên Tố" chỉ có 2 trang (Khuếch Tán, Kết Tinh — hai phản ứng đủ
# phức tạp để có trang riêng), đếm trang mà trả lời "có 2 phản ứng" là sai tự
# tin — đo được đúng ca này khi thử matcher. Câu hỏi về loại khái niệm phải rơi
# về pipeline thường, nơi quy tắc 8 xử lý thận trọng.
#
# Từng dòng đối chiếu với phân bố kind thật (06_build_graph in ra) trước khi
# thêm. Thêm loại mới vào wiki thì phải thêm ở đây một cách có ý thức.
_WEAPON_KINDS = ["Kiếm Đơn", "Trọng Kiếm", "Cung", "Pháp Khí", "Vũ Khí Cán Dài"]
_NPC_KINDS = ["NPC", "NPC Nhiệm Vụ", "NPC Đại Thế Giới", "NPC Sự Kiện"]
_COUNTABLE = {k.lower(): [k] for k in _WEAPON_KINDS + _NPC_KINDS + ["Chơi Được"]}
_COUNTABLE |= {
    "vũ khí": _WEAPON_KINDS,
    "nhân vật chơi được": ["Chơi Được"],
    "npc": _NPC_KINDS,
}


def _kinds_for(phrase: str) -> list[str]:
    return _COUNTABLE.get(phrase.strip().lower(), [])


def try_count(conn: psycopg.Connection, question: str) -> str | None:
    """
    Câu trả lời đếm tất định, hoặc None nếu câu hỏi không thuộc dạng đếm-theo-loại.

    Chỉ đếm thực thể CẤP TRANG (definition = ''): mỗi trang wiki một thực thể,
    kind từ infobox. Thực thể con trong trang (mục `- Tên: định nghĩa`) không
    tính — kind của chúng suy từ tiêu đề mục, nhiễu hơn hẳn infobox.
    """
    if not graph.is_built(conn):
        return None
    m = _COUNT_Q.search(question)
    if not m:
        return None
    kinds = _kinds_for(m.group(1))
    if not kinds:
        return None

    rows = conn.execute(
        """
        SELECT kind, name FROM graph_entities
        WHERE definition = '' AND kind = ANY(%s)
        ORDER BY kind, name
        """,
        (kinds,),
    ).fetchall()
    if not rows:
        return None

    by_kind: dict[str, list[str]] = {}
    for kind, name in rows:
        by_kind.setdefault(kind, []).append(name)

    total = len(rows)
    parts = [f"Trong kho tài liệu hiện có **{total} trang** loại {m.group(1).strip()}."]
    if len(by_kind) > 1:
        parts.append(
            "Chia theo loại: " + " · ".join(f"{k} {len(v)}" for k, v in sorted(by_kind.items()))
        )
    sample = [n for names in by_kind.values() for n in names][:8]
    parts.append(f"Ví dụ: {', '.join(sample)}{'…' if total > len(sample) else '.'}")
    parts.append(
        "_Đếm tất định từ đồ thị tri thức (mỗi trang wiki một thực thể, loại lấy từ "
        "infobox) — không qua LLM. Con số phản ánh kho tài liệu đã cào, không nhất "
        "thiết bằng tổng số trong game._"
    )
    return "\n\n".join(parts)

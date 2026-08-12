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
_TAIL = r"\s*(?:trong\s+(?:kho|tài liệu|game|trò chơi|genshin)\b[^?]*)?[?.!…\s]*$"
_COUNT_Q = re.compile(
    r"bao nhiêu\s+(?:loại\s+|trang\s+)?(.+?)" + _TAIL,
    re.IGNORECASE,
)
# "Liệt kê các nhân vật chơi được" / "có những Trọng Kiếm nào?" -> liệt kê ĐỦ.
# Cùng gốc với đếm: danh sách đầy đủ chỉ tồn tại dưới dạng TẬP HỢP TRANG, không
# trang nào viết sẵn — đo được: "liệt kê các nhân vật chơi được" bị model từ
# chối thật thà dù kho có đủ 120 trang, vì ngữ cảnh không thể chứa danh sách.
#
# Cụm dấu hiệu cho phép LẶP: câu nối tiếp "liệt kê" được LLM diễn giải thành
# "liệt kê danh sách các nhân vật chơi được" — hai dấu hiệu chồng nhau, bắt một
# lần thì phần dư "danh sách các..." dính vào cụm loại và trượt whitelist.
# Mạo từ cũng lặp được ("tất cả các nhân vật...") — đo từ diễn giải thật của LLM.
_LIST_Q = re.compile(
    r"(?:(?:liệt kê|kể tên|danh sách)\s+)+(?:(?:các|những|tất cả)\s+)*(.+?)" + _TAIL
    + r"|có những\s+(.+?)\s+nào\b",
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
    Câu trả lời đếm HOẶC liệt kê tất định; None nếu câu hỏi không thuộc dạng
    đếm/liệt-kê-theo-loại (rơi về pipeline thường).

    Chỉ đếm thực thể CẤP TRANG (definition = ''): mỗi trang wiki một thực thể,
    kind từ infobox. Thực thể con trong trang (mục `- Tên: định nghĩa`) không
    tính — kind của chúng suy từ tiêu đề mục, nhiễu hơn hẳn infobox.
    """
    if not graph.is_built(conn):
        return None
    mode, phrase = "", ""
    if m := _COUNT_Q.search(question):
        mode, phrase = "count", m.group(1)
    elif m := _LIST_Q.search(question):
        mode, phrase = "list", m.group(1) or m.group(2)
    if not phrase:
        return None
    kinds = _kinds_for(phrase)
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
    phrase = phrase.strip()
    # Nói bằng đơn vị NGƯỜI HỎI dùng ("120 nhân vật chơi được"), không phải đơn
    # vị kỹ thuật ("120 trang loại nhân vật chơi được") — người dùng đã phản hồi
    # đúng chỗ này. Chú thích cuối câu vẫn nói rõ cách đếm là theo trang.
    parts = [f"Trong kho tài liệu hiện có **{total} {phrase}**."]
    if mode == "list":
        # Liệt kê ĐỦ, không cắt mẫu: đây chính là điều người hỏi cần, và danh
        # sách dài nhất (vũ khí, 227 tên) vẫn chỉ vài KB chữ.
        for kind, names in sorted(by_kind.items()):
            head = f"**{kind} ({len(names)})**: " if len(by_kind) > 1 else ""
            parts.append(head + ", ".join(names) + ".")
    else:
        if len(by_kind) > 1:
            parts.append(
                "Chia theo loại: "
                + " · ".join(f"{k} {len(v)}" for k, v in sorted(by_kind.items()))
            )
        sample = [n for names in by_kind.values() for n in names][:8]
        parts.append(f"Ví dụ: {', '.join(sample)}{'…' if total > len(sample) else '.'}")
    parts.append(
        "_Đếm tất định từ đồ thị tri thức (mỗi trang wiki một thực thể, loại lấy từ "
        "infobox) — không qua LLM. Con số phản ánh kho tài liệu đã cào, không nhất "
        "thiết bằng tổng số trong game._"
    )
    return "\n\n".join(parts)

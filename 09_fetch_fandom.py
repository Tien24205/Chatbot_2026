"""
BƯỚC 9 — Tải knowledge base từ wiki Fandom (MediaWiki API).

KHÔNG tốn hạn mức Gemini: đây thuần là tải và chuyển đổi văn bản.
KHÔNG ghi vào knowledge_base/: xuất ra thư mục riêng để soi trước đã.

  .venv\\Scripts\\python.exe 09_fetch_fandom.py --category "Nguyên Tố" --limit 20
  .venv\\Scripts\\python.exe 09_fetch_fandom.py --stage1
  .venv\\Scripts\\python.exe 09_fetch_fandom.py --list-categories

VÌ SAO PHẢI LỌC MẠNH TAY

Đo trên 300 bài lấy ngẫu nhiên của wiki Genshin tiếng Việt:
  - văn xuôi trung vị chỉ 477 ký tự/trang (tỉ lệ văn xuôi 24%),
  - 25% số trang là trang phụ (`Sub:` hoặc có dấu `/` trong tiêu đề),
  - phần lớn đuôi dài là trang phục, thẻ tên, công thức nấu ăn.

pipeline.TOP_K = 4, tức mỗi câu hỏi chỉ lấy được 4 đoạn. Nạp hàng nghìn trang
mỏng vào nghĩa là chúng CẠNH TRANH bốn chỗ đó với đoạn thật sự trả lời được câu
hỏi — kho to hơn mà chatbot trả lời tệ đi. Ba bộ lọc bên dưới chạy offline, nên
lọc trước khi tiêu bất kỳ request embedding nào là rẻ nhất.

TỰ VIẾT BỘ DỌN WIKITEXT, không dùng mwparserfromhell: giữ đúng nguyên tắc của
plan §8 là không thêm thư viện quá sớm. Nếu về sau gặp trang mà bộ dọn này làm
hỏng, đó là lúc đổi sang thư viện chuyên dụng.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import httpx

from rag import config
from rag.chunker import chunk_documents
from rag.loader import load_directory

BASE = "https://genshin-impact.fandom.com/vi"
USER_AGENT = "chatbot-rag/0.1 (du an hoc tap; https://github.com/Tien24205/Chatbot_2026)"

OUT_DIR = config.PROJECT_DIR / "wiki_raw"
MANIFEST = "_manifest.json"

# Thể loại theo giai đoạn. Số trang mỗi thể loại đã đếm thật bằng API, hạn mức
# đặt cao hơn một chút để còn chỗ khi wiki thêm bài.
#
# Chia giai đoạn vì bước nạp là bước KHÔNG quay lại được: xem chất lượng lõi hệ
# thống trước, rồi mới đổ thêm hàng trăm trang thực thể.
# Chọn thể loại bằng cách TRA NGƯỢC: lấy vài trang cốt lõi (Mondstadt, Phản Ứng
# Nguyên Tố, Nguyên Thạch) rồi hỏi API xem chúng thuộc thể loại nào. Bản đầu tôi
# đoán tên thể loại và trượt hoàn toàn — "Khu Vực" hoá ra chỉ chứa tiểu khu, còn
# sáu quốc gia chính nằm ở "Quốc Gia", mọi cơ chế game nằm ở "Hệ Thống Trò Chơi".
# Hậu quả: 895 trang mà không có Mondstadt lẫn Phản Ứng Nguyên Tố.
STAGE1 = [  # lõi hệ thống — thứ người chơi hỏi nhiều nhất
    ("Hệ Thống Trò Chơi", 150),
    ("Quốc Gia", 20),
    ("Nguyên Tố", 20),
    ("Thuật Ngữ", 300),
    ("Loại Vật Phẩm", 80),
]
STAGE2 = [("Nhân Vật", 500), ("Địa Điểm", 80)]  # thực thể chính
STAGE3 = [("Vũ Khí", 300), ("Thánh Di Vật", 300)]  # trang bị

# Mục không mang tri thức trả lời được câu hỏi: bảng dịch tên, nhật ký phiên bản,
# chú thích nguồn, chuyện bên lề. Giữ lại chỉ làm loãng ngữ cảnh.
DROP_SECTIONS = {
    "ngôn ngữ khác", "lịch sử cập nhật", "tham khảo", "chú thích",
    "bên lề", "thư viện", "video", "dẫn nguồn", "xem thêm",
}

# Dưới ngưỡng này thì trang không đủ nội dung để thành một đoạn tri thức.
# 300 nằm dưới trung vị 477 đo được, nên nó cắt đuôi dài mà không cắt vào phần thân.
MIN_PROSE_CHARS = 300

# Mục ngắn hơn ngưỡng này bị gộp vào mục trước.
#
# Trang wiki thường có rất nhiều mục `##` chỉ một hai dòng. chunker cắt theo mục
# TRƯỚC, nên mỗi mục tí hon thành một chunk tí hon — đo được 75 chunk dưới 80 ký
# tự ở lần chạy đầu. Chunk 60 ký tự vừa vô dụng khi truy hồi, vừa chiếm một trong
# bốn chỗ của TOP_K.
#
# Gộp dưới dạng `- Tiêu đề: nội dung` thay vì nối thẳng văn xuôi: giữ được tên mục
# làm ngữ cảnh, và đúng luôn khuôn mà rag/graph.py trích thực thể.
MIN_SECTION_CHARS = 200

_ILLEGAL = re.compile(r'[<>:"/\\|?*]')

# Template nội dòng CÓ NGHĨA — thay bằng chữ thay vì xoá trắng.
#
# Wiki dùng {{Hỏa}}, {{Đóng Băng}}... để hiện icon + tên nguyên tố/phản ứng ngay
# giữa câu và cả TRONG TIÊU ĐỀ MỤC. Bản đầu xoá trắng mọi template, gây hai hậu quả
# đo được trên trang Thuyết Định Lượng Nguyên Tố:
#   - tiêu đề "==={{Đóng Băng}} và {{Phá Băng}}===" thành "=== và ===" -> mục "## và";
#     "==={{Sinh Trưởng}}===" thành "======" -> regex bắt nhầm ra mục "## =".
#     Chunk mang tên mục "=" lọt vào top-6 truy hồi — rác chiếm chỗ đoạn thật.
#   - câu "tiêu hao với các nguyên tố {{Thảo}} và {{Lôi}}" thành "với các nguyên
#     tố và." — mất sạch tên nguyên tố ở 10 file.
#
# Chỉ đưa vào danh sách những tên tra được từ chính nội dung đã tải (tên nguyên tố,
# tên phản ứng) + tên tiếng Anh chuẩn của 7 nguyên tố. KHÔNG thay bừa mọi template
# không tham số: {{stub}}, {{Clr}}... mà thành chữ trần thì còn tệ hơn xoá.
INLINE_TEMPLATES = {
    t.lower(): t
    for t in (
        "Hỏa", "Thủy", "Phong", "Lôi", "Thảo", "Băng", "Nham", "Vật Lý",
        "Quá Tải", "Phá Băng", "Điện Cảm", "Siêu Dẫn", "Khuếch Tán", "Kết Tinh",
        "Thiêu Đốt", "Bốc Hơi", "Tan Chảy", "Đóng Băng", "Sinh Trưởng",
        "Tăng Cường", "Lan Tràn", "Nở Rộ", "Sum Suê", "Bung Tỏa", "Tăng Trưởng",
    )
} | {
    "pyro": "Hỏa", "hydro": "Thủy", "anemo": "Phong", "electro": "Lôi",
    "dendro": "Thảo", "cryo": "Băng", "geo": "Nham",
}


# Template CÓ THAM SỐ mang chữ hiển thị — vị trí chữ tuỳ template, tra từ
# wikitext thật (trang Thánh Di Vật, Thuyết Định Lượng Nguyên Tố):
#   {{Item|Hoa Sự Sống|30|type=Biểu Tượng}}  -> chữ ở tham số ĐẦU
#   {{Color|dendro|Sinh Trưởng}}             -> chữ ở tham số CUỐI
#   {{Color|Đóng Băng}}                      -> một tham số thì đầu = cuối
# Bản đầu chỉ biết template không tham số nên "Có 5 loại...: {{Item|...}}, ..."
# thành "Có 5 loại...:,,,, và." — mất sạch 5 tên vị trí Thánh Di Vật, và bốn
# tiêu đề mục Đóng Băng/Sinh Trưởng/Điện Cảm/Thiêu Đốt bị lưới an toàn gộp mất.
_ARG_TEMPLATES = {"item": "first", "tt": "first", "color": "last", "w": "last"}


def _split_args(body: str) -> list[str]:
    """Tách '|' ở tầng ngoài cùng — không tách bên trong template lồng nhau."""
    parts: list[str] = []
    depth, cur = 0, []
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _template_text(tpl: str) -> str:
    """Chữ thay thế cho một template, hoặc '' nếu template đáng xoá thật."""
    parts = _split_args(tpl[2:-2])
    key = parts[0].strip().lower()
    # Tham số ĐỊNH DANH: chỉ lấy tham số vị trí, bỏ dạng "type=Biểu Tượng".
    pos = [a.strip() for a in parts[1:] if "=" not in a]
    # Dạng bọc {{NT|Hỏa}} / {{Nguyên Tố|Cryo}}: tên thật nằm ở tham số đầu.
    if key in {"nt", "nguyên tố", "element"} and pos:
        key = pos[0].lower()
    if key in _ARG_TEMPLATES and pos:
        return pos[0] if _ARG_TEMPLATES[key] == "first" else pos[-1]
    return INLINE_TEMPLATES.get(key, "")


def _strip_templates(text: str) -> tuple[str, list[str]]:
    """
    Bỏ mọi {{...}}, khớp ngoặc lồng nhau — trừ template nội dòng có nghĩa (xem
    INLINE_TEMPLATES) được thay bằng chữ. Trả về (phần còn lại, các template gốc).

    Không dùng regex cho việc này được: template Fandom lồng nhau nhiều tầng
    ({{Seffect|{{Star}}}}), mà regex không đếm được ngoặc.
    """
    out: list[str] = []
    found: list[str] = []
    i, depth, start = 0, 0, 0
    while i < len(text):
        if text.startswith("{{", i):
            if depth == 0:
                start = i
            depth += 1
            i += 2
        elif text.startswith("}}", i) and depth > 0:
            depth -= 1
            i += 2
            if depth == 0:
                tpl = text[start:i]
                found.append(tpl)
                out.append(_template_text(tpl))
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out), found


def _parse_infobox(templates: list[str]) -> tuple[str, dict[str, str]]:
    """
    Lấy cặp khoá-giá trị từ template Infobox đầu tiên.

    Đây là món quà của wiki game: infobox vốn đã là `|khoá = giá trị`, tức đúng
    khuôn `- Tên: định nghĩa` mà rag/graph.py trích thực thể bằng regex. Với văn
    xuôi Wikipedia thì infobox là chướng ngại; ở đây nó là dữ liệu có cấu trúc sẵn.
    """
    for tpl in templates:
        head = tpl[2:40].lower()
        if "infobox" not in head:
            continue

        body = tpl[2:-2]
        # Chỉ tách ở dấu | thuộc tầng ngoài cùng, không tách trong template lồng.
        parts: list[str] = []
        depth, cur = 0, []
        for ch in body:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "|" and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        parts.append("".join(cur))

        name = parts[0].strip()
        fields: dict[str, str] = {}
        for p in parts[1:]:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            # Giá trị infobox phải đi qua ĐÚNG bộ dọn như phần văn xuôi. Bản đầu
            # bỏ qua bước này, nên `<ul><li>` và `[[Liên kết]]` lọt nguyên vào file.
            # Giữ lại xuống dòng (không gộp về một dòng) để trường dài còn cắt được.
            v = _clean_prose(v)[0]
            k = k.strip()
            v = "\n".join(" ".join(ln.split()) for ln in v.split("\n") if ln.strip())
            # Bỏ trường ảnh/gallery và trường rỗng — không có giá trị tra cứu.
            if not k or not v or k.lower() in {"image", "icon", "caption"}:
                continue
            if "<gallery" in v or v.endswith(".png") or v.endswith(".jpg"):
                continue
            fields[k] = v
        return name, fields
    return "", {}


def _clean_prose(text: str) -> tuple[str, list[str]]:
    """Bỏ bảng, thẻ HTML, chú thích; đổi liên kết wiki thành chữ. Trả về (text, liên kết)."""
    # Giải mã thực thể HTML trước mọi thứ: wiki dùng &mdash;, &nbsp;, &quot;...
    # Không giải mã thì chúng lọt nguyên vào chunk và cả người đọc lẫn tầng
    # embedding đều thấy chuỗi vô nghĩa.
    text = html.unescape(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<gallery.*?</gallery>", "", text, flags=re.S)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)  # bảng wiki

    # <br> PHẢI thành xuống dòng, không được xoá trắng. Wiki dùng nó thay dấu chấm
    # để ngắt ý; xoá trắng thì các ý dính thành một khối dài không có ranh giới câu
    # nào — đo được một đoạn 4.013 ký tự mà chunker không cắt nổi vì `_split_units`
    # chỉ cắt được ở chỗ có dấu kết câu.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:ul|ol|li|div|p|span)[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)

    # Liên kết thể loại và tập tin KHÔNG phải nội dung. Nếu để lọt, bước đổi
    # [[A]] -> A biến chúng thành chữ trần và chunk kết thúc bằng
    # "Thể loại:Chiến Đấu Thể loại:Nguyên Tố" — rác thuần tuý.
    text = re.sub(r"\[\[\s*(?:Thể loại|Category|Tập tin|File|Hình):[^\]]*\]\]", "", text)

    links = [m.group(1) for m in re.finditer(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", text)]
    text = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", text)  # [[A|B]] -> B
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)  # [[A]] -> A
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    # URL trần còn sót (link ngoài thiếu dấu đóng ngoặc, link YouTube trong mục
    # "Xem thêm"). Với chatbot tra cứu thì URL không phải tri thức trả lời được câu
    # hỏi, mà lại hay nằm cuối chunk khiến đoạn trông như bị cắt dở.
    text = re.sub(r"\[?https?://\S+", "", text)

    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^\s*\|.*$", "", text, flags=re.M)  # dòng bảng còn sót

    # Liên kết liên ngữ ở cuối bài: `en:Heroes of Natlan`. Không dọn thì nó dính
    # vào cuối chunk và người đọc thấy một dòng tiếng Anh lạc lõng trong nguồn.
    text = re.sub(r"^\s*[a-z][a-z-]{1,11}:[^\s].*$", "", text, flags=re.M)
    # Tàn dư navbox sau khi bỏ template: "Natlan — Điều Hướng".
    text = re.sub(r"^.*—\s*Điều Hướng\s*$", "", text, flags=re.M)

    # Gạch đầu dòng của wiki là `*` hoặc `**`, thường KHÔNG có dấu cách phía sau.
    # rag/text.py BULLET đòi `\s+` sau dấu, nên `*Vùng Đất` sẽ không được nhận là
    # mục danh sách và chunker có thể xé đôi nó. Chuẩn hoá về `- `.
    text = re.sub(r"^\s*\*+\s*", "- ", text, flags=re.M)
    text = re.sub(r"^\s*#\s+", "- ", text, flags=re.M)  # danh sách đánh số của wiki

    # Danh sách định nghĩa của wikitext: dòng ";Thuật ngữ" rồi dòng ":giải thích".
    # CHỈ bỏ dấu đầu dòng, KHÔNG đổi thành gạch đầu dòng: bản thử đổi thành `- X:`
    # làm hỏng các dòng hội thoại vốn đã có sẵn nhiều dấu hai chấm
    # ("(Khi đến gần lối vào): Tie Hong: Xin dừng bước!"), khiến ranh giới mục
    # danh sách lệch khỏi nguồn.
    text = re.sub(r"^\s*[;:]\s*", "", text, flags=re.M)

    # Ngoặc template mồ côi: nguồn có `{{` nằm trong <ref> vốn đã bị xoá trước đó,
    # nên bộ đếm ngoặc lệch và để sót `}}` cùng mảnh `refName=...` giữa văn bản.
    text = re.sub(r"\{\{|\}\}", "", text)
    text = re.sub(r"\brefName\s*=\s*\S*", "", text)

    # Bỏ dòng gạch đầu dòng rỗng do template bị lược để lại ("- - - - -").
    text = re.sub(r"^\s*-\s*$", "", text, flags=re.M)

    # Bỏ template icon giữa câu để lại khoảng trắng đôi ("như sau:  cho La Hoàn").
    text = re.sub(r"[ \t]{2,}", " ", text)
    # PHẢI là [ \t]+ chứ KHÔNG được \s+: \s khớp cả ký tự xuống dòng, nên bản dùng
    # \s+ đã nối dòng "==Giáo Trình==" với dòng kế tiếp bắt đầu bằng ";" thành
    # "==Giáo Trình==;Anh Đào Thần Bảo Vệ" — tiêu đề mục bị phá, không còn khớp
    # regex `^==...==$` nên lọt nguyên vào nội dung chunk.
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    return text, links


def to_markdown(title: str, wikitext: str) -> tuple[str, dict]:
    """
    Chuyển wikitext thành đúng khuôn mà rag/chunker.py mong đợi.

    Khuôn đó là: `# Tên tài liệu`, rồi các `## Mục`, trong mục có thể có dòng
    `- Tên: định nghĩa`. Giữ được `== Mục ==` của wiki là điều kiện sống còn —
    nếu để văn xuôi phẳng, cả bài thành một mục tên "Mở đầu" và cột `section`
    dùng để trích nguồn mất hết ý nghĩa.
    """
    body, templates = _strip_templates(wikitext)
    infobox_name, fields = _parse_infobox(templates)

    text, links = _clean_prose(body)

    # --- Tách thành các mục -------------------------------------------------
    # Infobox thành một mục có tên hẳn hoi, thay vì rơi vào mục mặc định "Mở đầu".
    sections: list[tuple[str, list[str]]] = []

    # Trường infobox NGẮN thành gạch đầu dòng; trường DÀI thành mục văn xuôi riêng.
    #
    # Bắt buộc tách như vậy: chunker coi mỗi gạch đầu dòng là một đơn vị KHÔNG
    # được xé, nên một `- description: <2000 ký tự lore>` sinh thẳng ra chunk vượt
    # ngưỡng. Đo được trên trang Thánh Di Vật: chunk 4.013 ký tự, gấp đôi mục tiêu
    # 1800. Để dưới dạng văn xuôi thì chunker cắt được theo câu như bình thường.
    LONG_FIELD = 400
    short = {k: v for k, v in fields.items() if len(v) <= LONG_FIELD}
    if short:
        # CỐ Ý không dùng gạch đầu dòng ở đây. rag/graph.py trích thực thể từ khuôn
        # `- Tên: định nghĩa`, nên `- type: Chơi Được` sẽ tạo ra một thực thể tên
        # "type" cho MỖI trang — 895 thực thể rác, và Lexicon khớp chữ "type" sẽ
        # trả về cả 895 cái. Thực thể của một trang wiki là TIÊU ĐỀ TRANG, không
        # phải tên trường infobox.
        sections.append(
            ("Thông Tin Cơ Bản", [f"{k}: {' '.join(v.split())}" for k, v in short.items()])
        )
    for k, v in fields.items():
        if len(v) > LONG_FIELD:
            sections.append((k[:1].upper() + k[1:], v.split("\n")))

    prose_chars = 0
    name, buf, keep = "Tổng Quan", [], True  # phần trước `==` đầu tiên là mở bài
    for raw in text.split("\n"):
        m = re.match(r"^==+\s*(.+?)\s*==+\s*$", raw)
        if m:
            # Chốt chặn cho template NGOÀI danh sách INLINE_TEMPLATES: tiêu đề mà
            # sau khi bỏ template chỉ còn "=", "và", dấu câu... thì KHÔNG mở mục
            # mới — để nội dung chảy tiếp vào mục hiện tại. Mục tên "=" từng lọt
            # vào top-6 truy hồi; gộp nhầm mục còn đỡ hại hơn sinh mục rác.
            cand = m.group(1).strip("= \t")
            if not re.search(r"\w", cand, re.UNICODE) or cand.lower() in {"và", "hoặc"}:
                continue
            if keep and any(x.strip() for x in buf):
                sections.append((name, buf))
            name = cand
            keep = name.lower() not in DROP_SECTIONS
            buf = []
            continue

        line = raw.rstrip()
        if not line.strip():
            if buf and buf[-1] != "":
                buf.append("")
            continue
        buf.append(line)
        prose_chars += len(line)
    if keep and any(x.strip() for x in buf):
        sections.append((name, buf))

    # --- Gộp mục quá ngắn (xem MIN_SECTION_CHARS) ---------------------------
    merged: list[tuple[str, list[str]]] = []
    for sec_name, sec_lines in sections:
        body_text = "\n".join(sec_lines).strip()
        if not body_text:
            continue
        if merged and len(body_text) < MIN_SECTION_CHARS:
            one_line = " ".join(body_text.split())
            # Dòng trống trước gạch đầu dòng là BẮT BUỘC: không có nó thì mục vừa
            # gộp dính vào đoạn văn phía trên, và `_split_units` thấy một đoạn chủ
            # yếu là văn xuôi nên xử lý cả cụm theo kiểu văn xuôi — cắt luôn ở dấu
            # hai chấm, làm ranh giới mục lệch khỏi nguồn.
            if merged[-1][1] and merged[-1][1][-1] != "":
                merged[-1][1].append("")
            merged[-1][1].append(
                one_line if one_line.startswith("- ") else f"- {sec_name}: {one_line}"
            )
        else:
            merged.append((sec_name, list(sec_lines)))

    # Mục ĐẦU quá ngắn thì không có mục trước để gộp vào — dồn xuống mục kế tiếp.
    if len(merged) > 1 and len("\n".join(merged[0][1]).strip()) < MIN_SECTION_CHARS:
        _, head_lines = merged.pop(0)
        merged[0][1][:0] = head_lines + [""]

    lines = [f"# {title}", ""]
    for sec_name, sec_lines in merged:
        lines += [f"## {sec_name}", ""] + sec_lines + [""]

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    meta = {
        "title": title,
        "url": f"{BASE}/wiki/{title.replace(' ', '_')}",
        "infobox": infobox_name,
        "infobox_type": fields.get("type", ""),
        "fields": len(fields),
        "links": sorted(set(links))[:60],
        "prose_chars": prose_chars,
    }
    return out, meta


# --- Gọi API ---------------------------------------------------------------


class Wiki:
    def __init__(self) -> None:
        self.c = httpx.Client(
            base_url=BASE, timeout=40, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )

    def api(self, **p) -> dict:
        p["format"] = "json"
        r = self.c.get("/api.php", params=p)
        r.raise_for_status()
        time.sleep(0.3)  # gọi thưa cho lịch sự, wiki này là dịch vụ miễn phí
        return r.json()

    def category(self, name: str, limit: int) -> list[str]:
        """Tiêu đề bài viết trong một thể loại (chỉ không gian chính)."""
        titles: list[str] = []
        cont: dict = {}
        while len(titles) < limit:
            d = self.api(
                action="query", list="categorymembers", cmtitle=f"Thể loại:{name}",
                cmnamespace=0, cmlimit=min(500, limit - len(titles)), **cont
            )
            titles += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
            if "continue" not in d:
                break
            cont = d["continue"]
        return titles[:limit]

    def contents(self, titles: list[str]) -> dict[str, str]:
        """Wikitext theo lô 50 trang — nhanh hơn nhiều so với gọi từng trang."""
        out: dict[str, str] = {}
        for i in range(0, len(titles), 50):
            d = self.api(
                action="query", prop="revisions", rvprop="content", rvslots="main",
                titles="|".join(titles[i : i + 50])
            )
            for p in d.get("query", {}).get("pages", {}).values():
                revs = p.get("revisions")
                if revs:
                    out[p["title"]] = revs[0]["slots"]["main"]["*"]
        return out


# --- Chương trình chính ----------------------------------------------------


def safe_name(title: str) -> str:
    return _ILLEGAL.sub("-", title).strip() + ".txt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Tải knowledge base từ wiki Fandom.")
    ap.add_argument("--category", action="append", default=[], help="tên thể loại (lặp được)")
    ap.add_argument("--limit", type=int, default=50, help="số trang tối đa mỗi thể loại")
    ap.add_argument("--stage1", action="store_true", help="lõi hệ thống (nguyên tố, khu vực, thuật ngữ)")
    ap.add_argument("--stage2", action="store_true", help="thực thể chính (nhân vật)")
    ap.add_argument("--stage3", action="store_true", help="trang bị (vũ khí, thánh di vật)")
    ap.add_argument("--list-categories", action="store_true", help="chỉ liệt kê thể loại lớn")
    ap.add_argument("--out", default=str(OUT_DIR), help="thư mục xuất (KHÔNG phải knowledge_base)")
    ap.add_argument("--min-prose", type=int, default=MIN_PROSE_CHARS)
    args = ap.parse_args()

    wiki = Wiki()

    if args.list_categories:
        d = wiki.api(action="query", list="allcategories", aclimit=500, acprop="size", acmin=20)
        cats = [x for x in d["query"]["allcategories"] if x.get("pages", 0) > x.get("files", 0)]
        print(f"{'trang':>7}  thể loại")
        for x in sorted(cats, key=lambda y: -y.get("pages", 0))[:40]:
            print(f"{x.get('pages', 0):>7}  {x['*']}")
        return

    # Các giai đoạn cộng dồn được: thư mục xuất bị dọn mỗi lần chạy, nên muốn có
    # cả ba giai đoạn thì phải tải trong CÙNG một lượt.
    plan: list[tuple[str, int]] = []
    for flag, stage in ((args.stage1, STAGE1), (args.stage2, STAGE2), (args.stage3, STAGE3)):
        if flag:
            plan += stage
    plan += [(c, args.limit) for c in args.category]
    if not plan:
        ap.error("cần --category hoặc --stage1/2/3 (xem --list-categories)")

    out_dir = Path(args.out)
    if out_dir.resolve() == config.KB_DIR.resolve():
        ap.error("KHÔNG ghi thẳng vào knowledge_base/. Xuất ra thư mục khác rồi tự chép sang.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Xoá file .txt cũ trước khi ghi. Không xoá thì lần chạy sau lọc chặt hơn sẽ để
    # lại file của lần trước, và bộ kiểm thử đọc cả file cũ lẫn mới — đã vấp đúng
    # lỗi này: 73 trang lần đầu, 65 lần sau, 8 file cũ ở lại gây báo lỗi ma.
    stale = list(out_dir.glob("*.txt"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"  (đã xoá {len(stale)} file .txt của lần chạy trước)\n")

    print("=" * 74)
    print("TẢI KNOWLEDGE BASE TỪ FANDOM")
    print("=" * 74)
    print(f"\n  nguồn      : {BASE}")
    print(f"  xuất ra    : {out_dir}")
    print(f"  ngưỡng văn xuôi: {args.min_prose} ký tự\n")

    # --- Thu thập tiêu đề, áp bộ lọc 1 ngay tại đây -------------------------
    titles: list[str] = []
    dropped_sub = 0
    for cat, limit in plan:
        got = wiki.category(cat, limit)
        # Bộ lọc 1: trang phụ. Đo được 25% số bài là dạng này (Sub:, Phiên Bản/1.0,
        # Giáo Trình/..., Đổi Bụi Ánh Sáng/2021-12-01) — nội dung vụn hoặc theo mốc
        # thời gian, mau lỗi thời.
        keep = [t for t in got if "/" not in t and not t.startswith("Sub:")]
        dropped_sub += len(got) - len(keep)
        titles += keep
        print(f"  {cat:<24} {len(got):>4} trang -> giữ {len(keep):>4}")

    titles = list(dict.fromkeys(titles))
    print(f"\n  Tổng {len(titles)} trang cần tải (đã bỏ {dropped_sub} trang phụ)\n")

    # --- Tải và chuyển đổi --------------------------------------------------
    pages = wiki.contents(titles)
    print(f"  Tải xong {len(pages)}/{len(titles)} trang\n")

    kept, thin, manifest = [], [], []
    for title, wt in sorted(pages.items()):
        md, meta = to_markdown(title, wt)
        # Bộ lọc 2: ngưỡng độ dài. Rẻ nhất và hiệu quả nhất — tự loại trang phục,
        # thẻ tên, công thức nấu ăn mà không phải liệt kê tên từng thể loại rác.
        if meta["prose_chars"] < args.min_prose and meta["fields"] < 5:
            thin.append((title, meta["prose_chars"]))
            continue
        (out_dir / safe_name(title)).write_text(md, encoding="utf-8")
        meta["file"] = safe_name(title)
        meta["chars"] = len(md)
        manifest.append(meta)
        kept.append(title)

    (out_dir / MANIFEST).write_text(
        json.dumps({"source": BASE, "license": "CC BY-SA", "pages": manifest},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Báo cáo ------------------------------------------------------------
    print("=" * 74)
    print("KẾT QUẢ")
    print("=" * 74)
    print(f"\n  Giữ  : {len(kept)} trang")
    print(f"  Bỏ   : {len(thin)} trang do dưới ngưỡng văn xuôi")
    for t, n in thin[:8]:
        print(f"         {n:>5} ký tự  {t[:56]}")
    if len(thin) > 8:
        print(f"         ... và {len(thin) - 8} trang nữa")

    if not manifest:
        print("\n  Không giữ được trang nào. Thử hạ --min-prose hoặc đổi thể loại.\n")
        sys.exit(1)

    total = sum(m["chars"] for m in manifest)
    prose = sorted(m["prose_chars"] for m in manifest)
    # Đếm bằng CHÍNH chunker sẽ chạy lúc nạp, không ước theo số ký tự: chunker cắt
    # theo mục `##` trước rồi mới theo kích thước, nên một trang nhiều mục ngắn ra
    # nhiều chunk hơn hẳn phép chia dung lượng. Ước sai làm hụt dự toán hạn mức.
    n_chunks = len(chunk_documents(load_directory(out_dir)))
    print(f"\n  Tổng dung lượng : {total:,} ký tự")
    print(f"  Văn xuôi trung vị: {prose[len(prose) // 2]:,} ký tự/trang")
    print(f"  Chunk thật       : {n_chunks} -> {-(-n_chunks // 20)} request embedding")

    kinds: dict[str, int] = {}
    for m in manifest:
        kinds[m["infobox_type"] or "(không có infobox)"] = (
            kinds.get(m["infobox_type"] or "(không có infobox)", 0) + 1
        )
    print("\n  Loại theo infobox (dùng cho thiết kế lại tầng đồ thị):")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1])[:10]:
        print(f"    {n:>4}  {k}")

    print(f"\n  Đã ghi {MANIFEST} (kèm URL gốc + liên kết wiki, phục vụ ghi nguồn CC BY-SA)")
    print("\n  BƯỚC TIẾP THEO — chưa tốn request nào:")
    print(f"    1. Đọc mắt vài file trong {out_dir.name}/")
    print("    2. Chép sang knowledge_base/ rồi chạy 01_test_chunking.py")
    print("    3. Chỉ khi test sạch mới chạy 02_ingest.py\n")


if __name__ == "__main__":
    main()

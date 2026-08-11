"""
GraphRAG — tầng đồ thị tri thức đặt cạnh tầng vector.

VÌ SAO CẦN, khi vector search đã đạt recall@1 = 100% (đo ở 03_eval_retrieval.py
trên bản knowledge base 4 tài liệu, chưa đo lại sau khi mở rộng lên 10)?

Vì recall 100% đó đo trên các câu hỏi ĐƠN, tự đủ nghĩa. Đo thực tế ở Milestone 5,
truy hồi thuần vector hỏng đúng ở hai chỗ:

  1. Câu nối tiếp: sau khi hỏi Baron Nashor, câu "Còn Liên Quân thì sao?" truy hồi
     ra Tổng quan / Xếp hạng / Giải đấu và TRƯỢT chunk "Mục tiêu trung lập" — đúng
     chỗ chứa Rồng Bạo Chúa. Cách chữa hiện tại là gọi LLM viết lại câu hỏi,
     tốn ~7 giây và một lượt gọi API mỗi lần.
  2. Câu so sánh chéo game: "Liên Quân có con nào giống Baron Nashor không?".
     Vector chỉ trả về các chunk GIỐNG câu hỏi về mặt từ ngữ, mà "Baron Nashor" và
     "Thần Rừng" không hề giống nhau về từ ngữ — chúng chỉ giống nhau về VAI TRÒ.

Cả hai đều là quan hệ giữa các thực thể, không phải độ tương đồng văn bản. Vector
không biểu diễn được quan hệ; đồ thị thì có.

Trích thực thể ở đây là TẤT ĐỊNH, không gọi LLM. Lý do: knowledge base viết theo
đúng một khuôn "- <Tên> (<bí danh>): <định nghĩa>", nên regex đọc được chính xác;
và hạn mức free tier đo được chỉ 20 request nên không thể tiêu vào việc mà regex
làm được. Nếu sau này KB là văn xuôi tự do thì mới cần LLM trích thực thể.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import psycopg

from .loader import Document
from .text import BULLET as _BULLET
from .text import HEADING as _HEADING

# --- Mô hình dữ liệu ------------------------------------------------------


@dataclass
class Entity:
    name: str
    kind: str  # suy ra từ tiêu đề mục chứa nó — xem _kind_of_section
    definition: str
    source_name: str
    section: str
    aliases: list[str] = field(default_factory=list)

    @property
    def surfaces(self) -> list[str]:
        """Mọi chuỗi có thể dùng để nhắc tới thực thể này."""
        return [self.name, *self.aliases]

    @property
    def key(self) -> str:
        """
        Định danh duy nhất. PHẢI kèm tên tài liệu, không được dùng riêng tên.

        "Đường Giữa" tồn tại trong CẢ Liên Minh lẫn Liên Quân, và là hai thực thể
        khác nhau (khác vai trò, khác định nghĩa, thuộc hai game). Gộp chúng theo
        tên sẽ nuốt mất một bên cùng toàn bộ cạnh của nó.
        """
        return f"{self.source_name}#{self.name.lower()}"


@dataclass(frozen=True)
class Edge:
    src: str  # khoá thực thể (Entity.key)
    dst: str
    relation: str  # thuộc_về | đồng_xuất_hiện | tương_tự
    weight: float = 1.0


# --- Trích thực thể (offline) ---------------------------------------------

# Dòng định nghĩa: "- Tên: nội dung" hoặc "3. Tên: nội dung".
# Dựng từ text.BULLET để ranh giới mục danh sách ở đây luôn khớp với ranh giới
# mà chunker dùng — hai tầng đọc lệch nhau thì thực thể sẽ được gán vào chunk sai.
# Giới hạn phần tên ở 60 ký tự để câu văn xuôi có dấu hai chấm giữa dòng
# không bị nhận nhầm thành định nghĩa.
_DEFINITION = re.compile(_BULLET.pattern + r"([^:\n]{2,60}?)\s*:\s*(.+)$")
# Bí danh trong ngoặc: "Sứ Giả Khe Nứt (Rift Herald)", "Xạ thủ (ADC / Bot)".
_PAREN = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")

# Phần văn xuôi bám đuôi tên: "Thánh Di Vật (Artifact), gồm năm vị trí: ..." —
# tên thật kết thúc ở dấu phẩy. Chỉ cắt khi sau dấu phẩy là chữ THƯỜNG, để không
# cắt nhầm tên vốn có dấu phẩy như "Bo3, Bo5".
_PROSE_TAIL = re.compile(r",\s+(?=[^A-ZĐÀ-Ỹ0-9])")

# Bí danh quá ngắn ("Bo3" thì được, "CC" cũng được, nhưng 1-2 ký tự thì
# bắt nhầm khắp nơi) — bỏ qua để tránh nhiễu.
MIN_SURFACE_LEN = 3

# Vai trò có quá nhiều thực thể thì không còn phân biệt được gì.
#
# Cạnh `tương_tự` nối ĐÔI MỘT mọi thực thể cùng vai trò khác tài liệu, nên chi phí
# là bậc hai. Với knowledge base cào từ wiki (mỗi bài một file), nhóm "NPC Nhiệm
# Vụ" có 185 thực thể -> ~17.000 cạnh, mà "cùng là NPC nhiệm vụ" chẳng nói lên
# điều gì về quan hệ giữa hai NPC cụ thể. Đây đúng là lý do "khái niệm" bị loại
# từ đầu, chỉ khác là giờ đo được bằng số.
#
# Đặt 60 sau khi ĐO trên knowledge base thật (895 trang wiki Genshin):
#   giữ  — Kiếm Đơn 53, Pháp Khí 52, Cung 47, Vũ Khí Cán Dài 40, Khu Vực 19,
#          nguyên tố 18  -> "hai vũ khí cùng loại" là quan hệ dùng được
#   loại — khái niệm 992, chủ đề 285, Chơi Được 118, NPC Nhiệm Vụ 78
#          -> "hai nhân vật cùng là nhân vật chơi được" không nói lên điều gì
MAX_SIMILAR_GROUP = 60

# Cạnh đồng xuất hiện chỉ giữ khi hai thực thể gặp nhau ở NHIỀU HƠN một chunk.
# Gặp nhau đúng một lần thường là trùng hợp; giữ lại thì số cạnh phình theo số
# chunk mà không thêm thông tin.
MIN_COOCCURRENCE = 2

# Dòng `type: Chơi Được` do 09_fetch_fandom.py sinh từ infobox. Với tài liệu wiki,
# đây là tín hiệu vai trò tốt hơn hẳn tiêu đề mục: tiêu đề mục của wiki là
# "Tổng Quan", "Mô Tả", "Câu Chuyện" — không nói gì về việc trang đó nói về cái gì.
_INFOBOX_TYPE = re.compile(r"^\s*type\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# Tiêu đề mục -> loại thực thể. Loại là thứ tạo ra cạnh `tương_tự` giữa các game,
# nên nó phải phản ánh VAI TRÒ chứ không phải tên riêng.
_KIND_RULES: tuple[tuple[str, str], ...] = (
    ("mục tiêu", "mục tiêu trung lập"),
    # PHẢI đứng trước hai luật "vị trí"/"bản đồ" bên dưới. Trong game bắn súng và
    # sinh tồn, "bản đồ" là cả một đấu trường (Ascent, Bermuda); trong MOBA, mục
    # bản đồ lại liệt kê các ĐƯỜNG (Đường giữa, Đường Rồng). Gộp hai thứ vào cùng
    # vai trò thì đồ thị sinh ra cạnh vô nghĩa — đo được: "Xạ thủ" bị nối
    # `tương_tự` với "Ascent" chỉ vì cả hai nằm dưới mục có chữ "bản đồ".
    ("đấu trường", "bản đồ thi đấu"),
    ("vị trí", "vị trí"),
    ("bản đồ", "vị trí"),
    ("thuật ngữ", "thuật ngữ"),
    ("nguyên tố", "nguyên tố"),
    ("vùng đất", "vùng đất"),
    ("giải đấu", "giải đấu"),
    ("tướng", "nhóm tướng"),
    ("gacha", "gacha"),
    ("trang bị", "trang bị"),
    ("ngọc", "trang bị"),
    ("chế độ", "chế độ chơi"),
    ("xếp hạng", "xếp hạng"),
    ("nâng cấp", "nâng cấp nhân vật"),
    ("hoạt động", "hoạt động thường ngày"),
)


def _kind_of_section(section: str) -> str:
    s = section.lower()
    for key, kind in _KIND_RULES:
        if key in s:
            return kind
    return "khái niệm"


def _split_alias(raw: str) -> tuple[str, list[str]]:
    """"Xạ thủ (ADC / Bot)" -> ("Xạ thủ", ["ADC", "Bot"])."""
    raw = _PROSE_TAIL.split(raw.strip(), maxsplit=1)[0]
    m = _PAREN.match(raw.strip())
    if not m:
        return raw.strip(), []
    name = m.group(1).strip()
    aliases = [a.strip() for a in re.split(r"[/,]", m.group(2)) if a.strip()]
    if not name:  # cả cụm nằm trong ngoặc -> không phải bí danh
        return raw.strip(), []
    return name, [a for a in aliases if len(a) >= MIN_SURFACE_LEN]


def extract(docs: list[Document]) -> tuple[list[Entity], list[Edge]]:
    """
    Đọc tài liệu thô -> (thực thể, cạnh). Thuần tính toán, không đụng database,
    không gọi API — nên test được offline.
    """
    entities: list[Entity] = []
    edges: list[Edge] = []
    by_kind: dict[str, list[Entity]] = defaultdict(list)

    for doc in docs:
        doc_title = ""
        section = "Mở đầu"
        doc_entity: Entity | None = None

        for line in doc.text.split("\n"):
            h = _HEADING.match(line)
            if h:
                level, title = len(h.group(1)), h.group(2).strip()
                if level == 1 and not doc_title:
                    doc_title = title
                    name, aliases = _split_alias(title)
                    # Ưu tiên vai trò lấy từ infobox (tài liệu wiki), sau đó mới
                    # tới luật cũ cho knowledge base viết tay.
                    m_type = _INFOBOX_TYPE.search(doc.text)
                    if m_type:
                        kind = m_type.group(1)
                    else:
                        # Tài liệu có mục "Tổng quan" là một game; file thuật ngữ thì không.
                        kind = "game" if "## Tổng quan" in doc.text else "chủ đề"
                    doc_entity = Entity(
                        name=name,
                        kind=kind,
                        definition="",
                        source_name=doc.source_name,
                        section=title,
                        aliases=aliases,
                    )
                    entities.append(doc_entity)
                else:
                    section = title
                continue

            m = _DEFINITION.match(line)
            if not m:
                continue

            name, aliases = _split_alias(m.group(1))
            if len(name) < MIN_SURFACE_LEN:
                continue

            ent = Entity(
                name=name,
                kind=_kind_of_section(section),
                definition=m.group(2).strip(),
                source_name=doc.source_name,
                section=section,
                aliases=aliases,
            )
            entities.append(ent)
            by_kind[ent.kind].append(ent)

            if doc_entity is not None:
                edges.append(Edge(ent.key, doc_entity.key, "thuộc_về"))

        # Với tài liệu wiki, thực thể đáng kể của cả trang chính là TIÊU ĐỀ TRANG
        # (Nahida, Bạch Hồ Đông Vũ). Phải đưa nó vào by_kind thì cạnh `tương_tự`
        # mới nối được "các Kiếm Đơn" hay "các nhân vật chơi được" với nhau.
        if doc_entity is not None:
            by_kind[doc_entity.kind].append(doc_entity)

    # Cạnh `tương_tự`: cùng vai trò nhưng khác tài liệu.
    # Đây là cạnh mà vector search KHÔNG thể thay thế — Baron Nashor và Thần Rừng
    # không giống nhau một chữ nào, chúng chỉ cùng là "mục tiêu trung lập".
    for kind, group in by_kind.items():
        if kind == "khái niệm" or len(group) > MAX_SIMILAR_GROUP:
            continue  # quá rộng, nối tất cả với tất cả thì thành nhiễu
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.source_name != b.source_name:
                    edges.append(Edge(a.key, b.key, "tương_tự"))

    # Cùng một tên xuất hiện hai lần TRONG CÙNG tài liệu -> gộp, gộp cả bí danh.
    # Khác tài liệu thì để nguyên: đó là hai thực thể khác nhau (xem Entity.key).
    unique: dict[str, Entity] = {}
    for e in entities:
        if e.key in unique:
            for a in e.aliases:
                if a not in unique[e.key].aliases:
                    unique[e.key].aliases.append(a)
        else:
            unique[e.key] = e

    return list(unique.values()), edges


# --- Tìm thực thể trong một đoạn text --------------------------------------


class Lexicon:
    """
    Bộ dò tên thực thể trong văn bản, khớp cụm DÀI TRƯỚC.

    Bắt buộc khớp dài trước: "Rồng" là một thực thể, nhưng "Rồng Ngàn Tuổi" và
    "Rồng Bạo Chúa" cũng vậy. Nếu quét theo thứ tự tuỳ ý thì câu về Rồng Ngàn Tuổi
    sẽ bị tính là nhắc tới Rồng (một thực thể khác hẳn), kéo theo cả cụm chunk sai.
    """

    def __init__(self, entities: list[Entity]):
        # Một chuỗi có thể trỏ tới NHIỀU thực thể: "Đường Giữa" có ở cả hai game.
        # Giữ tất cả, để bên gọi tự quyết định lọc theo game hay không.
        by_surface: dict[str, list[Entity]] = defaultdict(list)
        for e in entities:
            for s in e.surfaces:
                if len(s) >= MIN_SURFACE_LEN:
                    by_surface[s.lower()].append(e)

        self._entities = {e.key: e for e in entities}
        self._by_surface = by_surface
        self._patterns: list[tuple[re.Pattern, str]] = [
            # Biên từ theo Unicode: \w trong Python đã bao gồm chữ tiếng Việt.
            (re.compile(rf"(?<!\w){re.escape(s)}(?!\w)", re.IGNORECASE), s)
            for s in sorted(by_surface, key=len, reverse=True)
        ]

    def _matches(self, text: str) -> list[str]:
        """Các chuỗi khớp, theo thứ tự xuất hiện, đã loại cụm nằm trong cụm dài hơn."""
        taken: list[tuple[int, int]] = []
        found: list[tuple[int, str]] = []
        for pattern, surface in self._patterns:
            for m in pattern.finditer(text):
                if any(m.start() < end and start < m.end() for start, end in taken):
                    continue  # nằm trong một cụm dài hơn đã khớp
                taken.append((m.start(), m.end()))
                found.append((m.start(), surface))
        return [s for _, s in sorted(found)]

    def find(self, text: str) -> list[str]:
        """Tên thực thể được nhắc tới, không trùng lặp, giữ thứ tự xuất hiện."""
        seen, out = set(), []
        for surface in self._matches(text):
            for e in self._by_surface[surface]:
                if e.name not in seen:
                    seen.add(e.name)
                    out.append(e.name)
        return out

    def find_keys(self, text: str) -> list[str]:
        """Như find() nhưng trả khoá định danh — phân biệt được hai game."""
        seen, out = set(), []
        for surface in self._matches(text):
            for e in self._by_surface[surface]:
                if e.key not in seen:
                    seen.add(e.key)
                    out.append(e.key)
        return out


# --- Lưu trữ ---------------------------------------------------------------

# DDL chạy từ Python thay vì thêm vào init.sql, vì init.sql CHỈ chạy khi volume
# còn rỗng (cái bẫy đã vấp ở Milestone 2). Nếu chỉ sửa init.sql thì phải
# `docker compose down -v` rồi nạp lại toàn bộ — tức là embedding lại mọi chunk,
# tốn hạn mức API cho một thay đổi vốn không đụng gì tới vector.
SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_entities (
    id          BIGSERIAL PRIMARY KEY,
    -- khoá "<tài liệu>#<tên viết thường>": tên KHÔNG duy nhất giữa các game,
    -- "Đường Giữa" có ở cả Liên Minh lẫn Liên Quân và là hai thực thể khác nhau.
    key         TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    definition  TEXT NOT NULL DEFAULT '',
    source_name TEXT NOT NULL,
    section     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_graph_entities_name ON graph_entities (name);

CREATE TABLE IF NOT EXISTS graph_aliases (
    entity_id BIGINT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    alias     TEXT   NOT NULL,
    PRIMARY KEY (entity_id, alias)
);

-- Thực thể X được nhắc tới trong chunk Y. Đây là cầu nối giữa tầng đồ thị và
-- tầng vector: đi từ chunk -> thực thể -> thực thể hàng xóm -> chunk khác.
CREATE TABLE IF NOT EXISTS graph_mentions (
    entity_id BIGINT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    chunk_id  BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    src_id   BIGINT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    dst_id   BIGINT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    relation TEXT   NOT NULL,
    weight   REAL   NOT NULL DEFAULT 1,
    PRIMARY KEY (src_id, dst_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_graph_mentions_chunk ON graph_mentions (chunk_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges (src_id);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def build(conn: psycopg.Connection, docs: list[Document]) -> dict:
    """
    Dựng lại toàn bộ đồ thị từ tài liệu. Chạy lại được nhiều lần (xoá rồi dựng),
    không tốn một lệnh gọi API nào.
    """
    entities, edges = extract(docs)

    with conn.cursor() as cur:
        # Đồ thị là dữ liệu SUY RA hoàn toàn từ knowledge base, không có gì phải giữ.
        # Xoá rồi dựng lại thay vì migrate: vừa đơn giản, vừa tránh trường hợp schema
        # cũ còn sót lại sau khi sửa cấu trúc (cái bẫy init.sql ở Milestone 2).
        cur.execute(
            "DROP TABLE IF EXISTS graph_mentions, graph_aliases, graph_edges, graph_entities CASCADE"
        )
        cur.execute(SCHEMA)
        cur.executemany(
            """
            INSERT INTO graph_entities (key, name, kind, definition, source_name, section)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [(e.key, e.name, e.kind, e.definition, e.source_name, e.section) for e in entities],
        )
        ids = {key: eid for eid, key in cur.execute("SELECT id, key FROM graph_entities")}

        cur.executemany(
            "INSERT INTO graph_aliases (entity_id, alias) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            [(ids[e.key], a) for e in entities for a in e.aliases if e.key in ids],
        )

        # --- Nhắc tới: quét toàn bộ chunk trong bộ nhớ, rẻ hơn nhiều so với SQL LIKE ---
        lex = Lexicon(entities)
        rows = conn.execute("SELECT id, content FROM chunks").fetchall()
        mentions: list[tuple[int, int]] = []
        chunk_entities: dict[int, list[str]] = {}
        for chunk_id, content in rows:
            keys = lex.find_keys(content)
            chunk_entities[chunk_id] = keys
            mentions.extend((ids[k], chunk_id) for k in keys if k in ids)
        cur.executemany(
            "INSERT INTO graph_mentions (entity_id, chunk_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            mentions,
        )

        # --- Cạnh đồng xuất hiện, suy ra từ mentions ---------------------------
        cooc: dict[tuple[str, str], int] = defaultdict(int)
        for keys in chunk_entities.values():
            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    cooc[(a, b) if a < b else (b, a)] += 1

        all_edges = list(edges) + [
            Edge(a, b, "đồng_xuất_hiện", float(w))
            for (a, b), w in cooc.items()
            if w >= MIN_COOCCURRENCE
        ]
        cur.executemany(
            """
            INSERT INTO graph_edges (src_id, dst_id, relation, weight)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (src_id, dst_id, relation) DO UPDATE SET weight = EXCLUDED.weight
            """,
            [
                (ids[e.src], ids[e.dst], e.relation, e.weight)
                for e in all_edges
                if e.src in ids and e.dst in ids
            ],
        )

    conn.commit()
    return {
        "entities": len(entities),
        "aliases": sum(len(e.aliases) for e in entities),
        "mentions": len(mentions),
        "edges": len(all_edges),
    }


# --- Truy vấn đồ thị -------------------------------------------------------


def load_lexicon(conn: psycopg.Connection) -> Lexicon:
    """Nạp bộ dò tên từ database (dùng để soi câu hỏi của người dùng)."""
    rows = conn.execute(
        """
        SELECT e.name, e.kind, e.definition, e.source_name, e.section,
               coalesce(array_agg(a.alias) FILTER (WHERE a.alias IS NOT NULL), '{}')
        FROM graph_entities e
        LEFT JOIN graph_aliases a ON a.entity_id = e.id
        GROUP BY e.id, e.name, e.kind, e.definition, e.source_name, e.section
        """
    ).fetchall()
    return Lexicon(
        [Entity(name=r[0], kind=r[1], definition=r[2], source_name=r[3], section=r[4], aliases=list(r[5]))
         for r in rows]
    )


def neighbors(
    conn: psycopg.Connection, names: list[str], relations: tuple[str, ...] = ("tương_tự",)
) -> list[dict]:
    """
    Các thực thể kề. Cạnh coi như vô hướng nên phải hợp nhất cả hai chiều.

    Trả về cả `from_name` (đi từ thực thể nào) để bên gọi giải thích được vì sao
    một đoạn được kéo vào — "đoạn này vào ngữ cảnh vì X cùng vai trò với Y" là
    thông tin người dùng có quyền biết, khác hẳn việc lặng lẽ nhét thêm văn bản.
    """
    if not names:
        return []
    rows = conn.execute(
        """
        SELECT d.name, d.source_name, d.kind, g.relation, g.weight, s.name AS from_name
        FROM graph_edges g
        JOIN graph_entities s ON s.id = g.src_id
        JOIN graph_entities d ON d.id = g.dst_id
        WHERE s.name = ANY(%s) AND g.relation = ANY(%s)
        UNION
        SELECT s.name, s.source_name, s.kind, g.relation, g.weight, d.name AS from_name
        FROM graph_edges g
        JOIN graph_entities s ON s.id = g.src_id
        JOIN graph_entities d ON d.id = g.dst_id
        WHERE d.name = ANY(%s) AND g.relation = ANY(%s)
        """,
        (names, list(relations), names, list(relations)),
    ).fetchall()
    return [
        {
            "name": r[0],
            "source_name": r[1],
            "kind": r[2],
            "relation": r[3],
            "weight": r[4],
            "from_name": r[5],
        }
        for r in rows
        if r[0] not in names
    ]


def chunks_mentioning(conn: psycopg.Connection, names: list[str], limit: int = 3) -> list[dict]:
    """
    Các chunk có nhắc tới những thực thể này, chunk nào nhắc nhiều thực thể thì xếp trước.
    """
    if not names:
        return []
    rows = conn.execute(
        """
        SELECT c.id, c.content, c.source_name, c.section, count(*) AS n
        FROM graph_mentions m
        JOIN graph_entities e ON e.id = m.entity_id
        JOIN chunks c ON c.id = m.chunk_id
        WHERE e.name = ANY(%s)
        GROUP BY c.id, c.content, c.source_name, c.section
        ORDER BY n DESC, c.id
        LIMIT %s
        """,
        (names, limit),
    ).fetchall()
    return [
        {"id": r[0], "content": r[1], "source_name": r[2], "section": r[3], "hits": r[4]}
        for r in rows
    ]


def communities(conn: psycopg.Connection) -> list[dict]:
    """
    Tóm tắt cộng đồng: gom theo tài liệu nguồn.

    Ở KB này ranh giới cộng đồng trùng với ranh giới tài liệu (mỗi file một game),
    nên không cần chạy Louvain/Leiden cho vài chục chunk — làm vậy chỉ là trưng
    thuật toán. Dùng để trả lời "kho kiến thức có những gì" và để viết câu từ chối
    cho đúng phạm vi.
    """
    rows = conn.execute(
        """
        SELECT e.source_name,
               count(*) AS n,
               array_agg(DISTINCT e.kind) AS kinds,
               (array_agg(e.name ORDER BY length(e.definition) DESC))[1:6] AS samples
        FROM graph_entities e
        GROUP BY e.source_name
        ORDER BY n DESC
        """
    ).fetchall()
    return [
        {"source_name": r[0], "entities": r[1], "kinds": sorted(r[2]), "samples": list(r[3])}
        for r in rows
    ]


def is_built(conn: psycopg.Connection) -> bool:
    """Đồ thị đã dựng chưa. Chưa dựng thì pipeline chạy thuần vector như cũ."""
    try:
        return (conn.execute("SELECT count(*) FROM graph_entities").fetchone()[0] or 0) > 0
    except psycopg.Error:
        conn.rollback()
        return False

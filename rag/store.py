"""
BƯỚC 5 — Lưu trữ và truy vấn pgvector.

Tách riêng khỏi embedder để đúng nguyên tắc plan §8.7: ingestion pipeline và
query pipeline dùng chung tầng lưu trữ nhưng không dính vào nhau.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from . import config
from .chunker import Chunk


@dataclass(frozen=True)
class SearchHit:
    content: str
    source_name: str
    section: str
    similarity: float  # 1.0 = trùng khớp hoàn toàn
    id: int = 0  # id chunk trong database, dùng để nối sang tầng đồ thị
    via: str = "vector"  # "vector" = do similarity search; "graph" = do mở rộng đồ thị


# Khi Docker chưa chạy, cổng 5433 không có ai lắng nghe và psycopg có thể TREO
# thay vì báo lỗi (đo được: script đứng im quá 2 phút). Đặt hạn thời gian để
# hỏng NHANH và rõ ràng — mọi script gọi connect() đều đã có sẵn nhánh xử lý lỗi
# kèm hướng dẫn `docker compose up -d`, nhưng nhánh đó chỉ chạy nếu có lỗi ném ra.
CONNECT_TIMEOUT_SECONDS = 5


def connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    register_vector(conn)
    return conn


def connect_or_exit(require_chunks: bool = False) -> psycopg.Connection:
    """
    Kết nối dành cho các script dòng lệnh: hỏng thì in đúng cách chữa rồi thoát.

    Năm script từng chép cùng một khối try/except kèm gợi ý `docker compose up -d`,
    mỗi bản một cách diễn đạt khác nhau. Gom về đây để thông báo đồng nhất và chỉ
    còn một chỗ phải sửa.

    `require_chunks=True` dành cho script cần dữ liệu có sẵn (chat, đánh giá, dựng
    đồ thị): bảng rỗng thì hỏng ngay kèm hướng dẫn chạy 02_ingest.py, thay vì để
    người dùng nhận về kết quả rỗng rồi tự đoán vì sao.
    """
    try:
        conn = connect()
    except Exception as e:
        print(f"\n[THẤT BẠI] Không kết nối được database.\n  {e}", file=sys.stderr)
        print("\n  Kiểm tra   : docker compose ps", file=sys.stderr)
        print("  Nếu chưa chạy: docker compose up -d", file=sys.stderr)
        sys.exit(1)

    if require_chunks and count(conn) == 0:
        conn.close()
        print(
            "\n[THẤT BẠI] Bảng chunks rỗng — chưa nạp dữ liệu.\n"
            "  Chạy 02_ingest.py trước.",
            file=sys.stderr,
        )
        sys.exit(1)

    return conn


def assert_schema_matches(conn: psycopg.Connection) -> None:
    """
    Đối chiếu số chiều trong database với EMBED_DIMENSION đang cấu hình.

    Không có kiểm tra này, việc đổi model embedding sẽ chỉ lộ ra dưới dạng lỗi
    khó hiểu lúc insert, hoặc tệ hơn là kết quả tìm kiếm sai âm thầm.
    """
    row = conn.execute(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("Không tìm thấy bảng `chunks`. Container đã chạy init.sql chưa?")

    db_dim = row[0]
    if db_dim != config.EMBED_DIMENSION:
        raise RuntimeError(
            f"Lệch số chiều: database vector({db_dim}) nhưng .env đặt "
            f"EMBED_DIMENSION={config.EMBED_DIMENSION}.\n"
            "Sửa .env cho khớp, hoặc dựng lại database:\n"
            "  docker compose down -v && docker compose up -d"
        )


def upsert_chunks(
    conn: psycopg.Connection, chunks: list[Chunk], vectors: list[list[float]]
) -> int:
    """Nạp chunk kèm vector. Chạy lại sẽ ghi đè, không nhân bản."""
    if len(chunks) != len(vectors):
        raise ValueError(f"Số chunk ({len(chunks)}) khác số vector ({len(vectors)}).")

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (source_name, section, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source_name, chunk_index) DO UPDATE
                SET section    = EXCLUDED.section,
                    content    = EXCLUDED.content,
                    embedding  = EXCLUDED.embedding,
                    created_at = now()
            """,
            [
                (c.source_name, c.section, c.chunk_index, c.content, v)
                # strict=True: đã kiểm tra độ dài ở trên, giữ thêm chốt chặn ở đây
                # để một thay đổi sau này không lặng lẽ nạp thiếu vector.
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
    conn.commit()
    return len(chunks)


def search(conn: psycopg.Connection, query_vector: list[float], top_k: int = 5) -> list[SearchHit]:
    """
    Tìm top-k chunk gần nhất. `<=>` là khoảng cách cosine, similarity = 1 - distance.

    Bắt buộc ép kiểu `%s::vector`: list Python được psycopg gửi đi dưới dạng
    double precision[]. Lúc INSERT thì Postgres tự ép vì đã biết kiểu cột, nhưng
    toán tử <=> không có gì để suy ra kiểu nên sẽ báo "operator does not exist".
    """
    rows = conn.execute(
        """
        SELECT content, source_name, section, 1 - (embedding <=> %s::vector) AS similarity, id
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vector, query_vector, top_k),
    ).fetchall()
    return [
        SearchHit(content=r[0], source_name=r[1], section=r[2], similarity=r[3], id=r[4])
        for r in rows
    ]


def hits_for_ids(
    conn: psycopg.Connection, query_vector: list[float], ids: list[int]
) -> list[SearchHit]:
    """
    Lấy các chunk theo id, KÈM similarity thật so với câu hỏi.

    Dùng cho chunk do tầng đồ thị đề cử. Vẫn đo similarity dù chúng được chọn
    bằng quan hệ chứ không bằng độ tương đồng: có con số thật thì mới biết đồ thị
    đang bổ sung thứ vector bỏ sót (điểm thấp) hay chỉ lặp lại thứ vector đã có.
    """
    if not ids:
        return []
    rows = conn.execute(
        """
        SELECT content, source_name, section, 1 - (embedding <=> %s::vector) AS similarity, id
        FROM chunks
        WHERE id = ANY(%s)
        """,
        (query_vector, ids),
    ).fetchall()
    order = {cid: i for i, cid in enumerate(ids)}
    hits = [
        SearchHit(content=r[0], source_name=r[1], section=r[2], similarity=r[3], id=r[4], via="graph")
        for r in rows
    ]
    return sorted(hits, key=lambda h: order.get(h.id, 999))


def count(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def existing_keys(conn: psycopg.Connection) -> set[tuple[str, int]]:
    """
    Các chunk ĐÃ có vector trong database, theo khoá (tên file, thứ tự).

    Dùng để chạy lại việc nạp mà không embed lại thứ đã embed. Với knowledge base
    vài nghìn chunk thì đây là khác biệt giữa "mất 5 phút" và "mất cả hạn mức
    ngày": embedding đã trả tiền rồi, trả lần nữa cho cùng một đoạn là lãng phí.
    """
    rows = conn.execute("SELECT source_name, chunk_index FROM chunks").fetchall()
    return {(r[0], r[1]) for r in rows}

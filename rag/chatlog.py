"""
Lưu hội thoại vào PostgreSQL để xem lại được sau khi đóng trình duyệt.

VÌ SAO CẦN: st.session_state sống theo tab trình duyệt — F5 là mất sạch; dict
`_sessions` của api.py sống theo tiến trình — tắt server là mất. Cả hai đều không
phải "lịch sử". Database thì đã có sẵn (đang giữ chunks + đồ thị), thêm một bảng
là có lịch sử bền mà không thêm hạ tầng nào.

LƯU GÌ: lưu cả lượt BỊ TỪ CHỐI. Khác với pipeline.remember() (bỏ lượt từ chối để
LLM không học thói từ chối lây), ở đây mục đích là XEM LẠI — người dùng phải thấy
đúng những gì đã diễn ra, kể cả câu bị từ chối. Hai mục đích khác nhau thì hai
luật khác nhau, không phải trùng lặp.

DDL chạy từ Python thay vì thêm vào init.sql, cùng lý do với rag/graph.py:
init.sql CHỈ chạy khi volume còn rỗng, sửa nó là phải `docker compose down -v`
rồi embedding lại toàn bộ — quá đắt cho một bảng không liên quan tới vector.
"""

from __future__ import annotations

import json

import psycopg
from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_turns (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    question    TEXT NOT NULL,
    -- Toàn bộ dict `view` mà streamlit_app dùng để vẽ một lượt trả lời.
    -- Lưu nguyên khối JSONB thay vì tách cột: cấu trúc view còn đổi theo giao
    -- diện, mà lịch sử thì chỉ cần vẽ lại được đúng như lúc nó xảy ra.
    view        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_turns_session ON chat_turns (session_id, id);
"""


def _ensure(conn: psycopg.Connection) -> None:
    # IF NOT EXISTS nên gọi lặp vô hại; rẻ hơn hẳn việc bắt mọi nơi dùng module
    # này phải nhớ gọi riêng một hàm khởi tạo trước.
    conn.execute(SCHEMA)
    conn.commit()


def save_turn(conn: psycopg.Connection, session_id: str, question: str, view: dict) -> None:
    _ensure(conn)
    conn.execute(
        "INSERT INTO chat_turns (session_id, question, view) VALUES (%s, %s, %s)",
        (session_id, question, Jsonb(view)),
    )
    conn.commit()


def list_sessions(conn: psycopg.Connection, limit: int = 20) -> list[dict]:
    """Mỗi hội thoại một dòng, mới nhất trước. Tiêu đề = câu hỏi ĐẦU TIÊN."""
    _ensure(conn)
    rows = conn.execute(
        """
        SELECT session_id,
               min(question) FILTER (WHERE rn = 1)  AS title,
               count(*)                              AS turns,
               max(created_at)                       AS last_at
        FROM (
            SELECT session_id, question, created_at,
                   row_number() OVER (PARTITION BY session_id ORDER BY id) AS rn
            FROM chat_turns
        ) t
        GROUP BY session_id
        ORDER BY last_at DESC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    return [
        {"session_id": r[0], "title": r[1], "turns": r[2], "last_at": r[3]} for r in rows
    ]


def load_session(conn: psycopg.Connection, session_id: str) -> list[dict]:
    """Các lượt của một hội thoại, đúng thứ tự đã hỏi — khớp khuôn st.session_state.turns."""
    _ensure(conn)
    rows = conn.execute(
        "SELECT question, view FROM chat_turns WHERE session_id = %s ORDER BY id",
        (session_id,),
    ).fetchall()
    # psycopg trả JSONB về dict sẵn; phòng cả trường hợp driver trả chuỗi.
    return [
        {"q": q, "view": v if isinstance(v, dict) else json.loads(v)} for q, v in rows
    ]


def delete_session(conn: psycopg.Connection, session_id: str) -> None:
    _ensure(conn)
    conn.execute("DELETE FROM chat_turns WHERE session_id = %s", (session_id,))
    conn.commit()

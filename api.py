"""
BƯỚC 8 (backend) — REST API thuần.

Chạy:
  .venv\\Scripts\\python.exe -m uvicorn api:app --reload --port 8000
Tài liệu API tự sinh: http://localhost:8000/docs

KHÔNG phục vụ giao diện. Giao diện chat là streamlit_app.py, và nó gọi thẳng
rag.pipeline chứ không đi qua API này — nên file ở đây chỉ còn đúng hai việc:
lưu lịch sử theo phiên và gọi pipeline.

Toàn bộ logic RAG nằm trong rag/pipeline.py.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag import config, graph, pipeline, store
from rag.retry import RateLimited

app = FastAPI(title="Chatbot RAG — Kiến thức Game")

# Lịch sử hội thoại trong RAM, khoá theo session_id.
# Đủ cho POC nhưng MẤT KHI RESTART và không chia sẻ được giữa nhiều tiến trình.
# Muốn chạy thật thì chuyển sang Redis hoặc bảng trong Postgres.
_sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    source_name: str
    section: str
    similarity: float
    via: str  # "vector" hoặc "graph" — nguồn đến từ đâu, người đọc có quyền biết


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    refused_by: str  # "threshold" (lớp 2 chặn) | "model" (LLM nói không có) | ""
    top_similarity: float
    search_query: str
    rewrote_query: bool
    rewrite_ms: int
    retrieval_ms: int
    llm_ms: int
    verify_ms: int
    total_ms: int

    # --- Kết quả các lớp chống hallucination ---
    verified: bool  # sạch mọi cờ ở lớp 3-4-5
    flags: list[str]  # lớp 3-4 (tất định)
    unsupported: list[str]  # lớp 5 (LLM soát entailment)
    entailment_ran: bool
    graph_expanded: int  # số chunk do đồ thị bổ sung


@app.get("/")
def index() -> dict:
    """Chỉ đường, vì gốc "/" trước đây trả về giao diện chat nên có người còn mở."""
    return {
        "service": "Chatbot RAG — Kiến thức Game (REST API)",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/graph", "/api/chat"],
        "giao_dien": "streamlit run streamlit_app.py -> http://localhost:8501",
    }


@app.get("/api/health")
def health() -> dict:
    """Kiểm tra database và cấu hình. Dùng để chẩn đoán khi UI báo lỗi."""
    try:
        with store.connect() as conn:
            n = store.count(conn)
            graph_built = graph.is_built(conn)
            n_entities = (
                conn.execute("SELECT count(*) FROM graph_entities").fetchone()[0]
                if graph_built
                else 0
            )
        return {
            "status": "ok",
            "chunks": n,
            "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL,
            "embed_dimension": config.EMBED_DIMENSION,
            "similarity_threshold": config.SIMILARITY_THRESHOLD,
            "graph_built": graph_built,
            "graph_entities": n_entities,
            "entailment_check": config.ENTAILMENT_CHECK,
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.get("/api/graph")
def graph_summary() -> dict:
    """
    Tóm tắt cộng đồng trong đồ thị tri thức.

    Dùng để trả lời câu "kho kiến thức này biết những gì" mà không phải tốn một
    lệnh gọi LLM, và để người dùng biết trước phạm vi trước khi hỏi.
    """
    with store.connect() as conn:
        if not graph.is_built(conn):
            return {"built": False, "communities": []}
        return {"built": True, "communities": graph.communities(conn)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    history = _sessions.setdefault(req.session_id, [])

    # Hàm def thường (không phải async def): FastAPI tự đẩy sang threadpool, nên
    # lệnh gọi pipeline đồng bộ không chặn event loop.
    # Mở kết nối theo từng request cho đơn giản; tải cao thì chuyển sang psycopg_pool.
    try:
        with store.connect() as conn:
            ans = pipeline.ask(conn, req.message, history=history)
    except RateLimited as e:
        # 429 chứ không phải 500: đây không phải lỗi hệ thống mà là hạn mức nhà
        # cung cấp. Trả đúng mã để frontend hiển thị được thông báo hữu ích.
        raise HTTPException(status_code=429, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {e}") from e

    # Tự bỏ qua lượt bị từ chối — xem pipeline.remember.
    pipeline.remember(history, req.message, ans)

    cited = set(ans.citations)
    return ChatResponse(
        answer=ans.text,
        citations=[
            Citation(
                source_name=h.source_name,
                section=h.section,
                similarity=round(h.similarity, 4),
                via=h.via,
            )
            for h in ans.hits
            if f"{h.source_name} · {h.section}" in cited
        ],
        refused=ans.refused,
        refused_by=ans.refused_by,
        top_similarity=round(ans.top_similarity, 4),
        search_query=ans.search_query,
        rewrote_query=bool(ans.search_query and ans.search_query != req.message),
        rewrite_ms=ans.rewrite_ms,
        retrieval_ms=ans.retrieval_ms,
        llm_ms=ans.llm_ms,
        verify_ms=ans.verify_ms,
        total_ms=ans.total_ms,
        verified=ans.verified,
        flags=ans.flags,
        unsupported=ans.unsupported,
        entailment_ran=ans.entailment_ran,
        graph_expanded=ans.graph_expanded,
    )


@app.delete("/api/chat/{session_id}")
def reset(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"status": "ok"}

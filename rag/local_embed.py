"""
Embedding chạy trên máy — không hạn mức, không cần API key.

VÌ SAO CẦN, khi đã có Gemini embedding?

Vì hạn mức free tier là bức tường cứng, không phải chuyện chờ vài phút. Đo được
trong chính dự án này:

  - `EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier` = 1.000/ngày,
  - và nó tính theo **tài khoản Google**, không chỉ theo project — nên tạo project
    mới rồi lấy key mới KHÔNG cho thêm hạn mức,
  - nạp knowledge base 2.848 chunk cần ~350 request, tức mỗi lần nạp lại đã ăn
    một phần ba hạn mức ngày,
  - và mỗi câu hỏi của người dùng cũng tốn một request — hết hạn mức là chatbot
    câm hoàn toàn, kể cả khi database đã đầy đủ.

Chạy trên máy thì cả hai vấn đề biến mất cùng lúc.

VÌ SAO CHỌN multilingual-e5

Model mặc định của demo LangChain phổ biến (`all-MiniLM-L6-v2`) huấn luyện gần như
thuần tiếng Anh. Knowledge base ở đây là tiếng Việt, nên dùng nó là tự bắn vào
chân. `intfloat/multilingual-e5-base` huấn luyện đa ngữ, có tiếng Việt.

Họ e5 đòi TIỀN TỐ phân biệt câu hỏi với tài liệu:
    "query: <câu hỏi>"     khi tìm kiếm
    "passage: <tài liệu>"  khi nạp
Bỏ tiền tố đi thì chất lượng truy hồi giảm rõ. Đây đúng là khái niệm `task_type`
của Gemini (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT), chỉ khác cách diễn đạt — nên
tầng trên không phải biết mình đang dùng backend nào.
"""

from __future__ import annotations

from . import config

_model = None

# Tiền tố bắt buộc của họ e5. Model khác thì để rỗng.
_PREFIX_QUERY = "query: "
_PREFIX_DOC = "passage: "


def _uses_e5_prefix() -> bool:
    return "e5" in config.LOCAL_EMBED_MODEL.lower()


def model():
    """
    Nạp model một lần rồi dùng lại.

    Lần gọi đầu tiên sẽ TẢI model về (~1,1 GB cho bản base) và mất vài chục giây;
    những lần sau đọc từ cache trong thư mục người dùng.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.LOCAL_EMBED_MODEL, device=config.LOCAL_EMBED_DEVICE)
    return _model


def dimension() -> int:
    return int(model().get_sentence_embedding_dimension())


def _encode(texts: list[str], prefix: str, progress: bool = False) -> list[list[float]]:
    if _uses_e5_prefix():
        texts = [prefix + t for t in texts]
    vectors = model().encode(
        texts,
        batch_size=config.LOCAL_EMBED_BATCH,
        # Chuẩn hoá ngay tại đây: pgvector so bằng cosine, và store.py giả định
        # vector đã có độ dài 1 (xem embedder.normalize).
        normalize_embeddings=True,
        show_progress_bar=progress,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_documents(texts: list[str], progress: bool = False) -> list[list[float]]:
    return _encode(texts, _PREFIX_DOC, progress=progress)


def embed_query(text: str) -> list[float]:
    return _encode([text], _PREFIX_QUERY)[0]

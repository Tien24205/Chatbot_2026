"""
BƯỚC 4 — Embedding.

Hai điều bắt buộc, cả hai đều do 00_smoke_test.py phát hiện:

1. CHUẨN HOÁ LẠI VECTOR. `output_dimensionality` cắt vector 3072 chiều xuống
   1536 theo kiểu Matryoshka, và vector sau khi cắt KHÔNG còn độ dài 1 (đo được
   norm ≈ 0.69). Nếu lưu thẳng thì phép so cosine trong pgvector sẽ lệch.

2. PHÂN BIỆT task_type. Gemini sinh vector khác nhau cho tài liệu và cho câu hỏi:
   - RETRIEVAL_DOCUMENT khi nạp knowledge base,
   - RETRIEVAL_QUERY khi người dùng hỏi.
   Dùng lẫn hai loại sẽ làm giảm chất lượng truy hồi.
"""

from __future__ import annotations

import math

from google.genai import types

from . import config
from .gemini import client
from .retry import RateLimited, with_retry

# Gemini giới hạn số nội dung mỗi request. Chia lô nhỏ cho an toàn; nếu lô lỗi
# thì tự lùi về gọi từng cái một.
BATCH_SIZE = 20


def normalize(vector: list[float]) -> list[float]:
    """Đưa vector về độ dài 1. Bắt buộc sau khi cắt chiều (xem docstring module)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("Vector toàn số 0 — không chuẩn hoá được.")
    return [x / norm for x in vector]


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    cfg = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=config.EMBED_DIMENSION,
    )

    def call():
        return client().models.embed_content(
            model=config.EMBED_MODEL, contents=texts, config=cfg
        )

    resp = with_retry(call)
    return [normalize(e.values) for e in resp.embeddings]


def embed_documents(texts: list[str], progress: bool = False) -> list[list[float]]:
    """Sinh vector cho các chunk của knowledge base."""
    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        try:
            vectors = _embed(batch, "RETRIEVAL_DOCUMENT")
        except RateLimited:
            # KHÔNG được lùi về gọi lẻ khi nguyên nhân là hết hạn mức.
            #
            # Bản trước bắt `except Exception`, nên một lô bị 429 sẽ sinh ra thêm
            # 20 lệnh gọi lẻ — mỗi lệnh lại 429 và lại thử 3 lần. Đo được hậu quả
            # thật: một lần nạp 1.994 chunk đáng lẽ tốn 100 request đã đốt sạch
            # hạn mức 1.000 request/ngày của free tier.
            #
            # Hết hạn mức là tình trạng của TOÀN BỘ tài khoản, không phải lỗi của
            # một đoạn văn bản cụ thể, nên gọi lẻ không chẩn đoán được gì.
            raise
        except Exception:
            # Lô lỗi vì lý do khác -> gọi lẻ từng cái để biết chính xác cái nào hỏng.
            vectors = [_embed([t], "RETRIEVAL_DOCUMENT")[0] for t in batch]
        out.extend(vectors)
        if progress:
            print(f"    đã embed {len(out)}/{len(texts)} chunk")
    return out


def embed_query(text: str) -> list[float]:
    """Sinh vector cho câu hỏi của người dùng."""
    return _embed([text], "RETRIEVAL_QUERY")[0]

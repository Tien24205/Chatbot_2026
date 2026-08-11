-- BƯỚC 5 — Schema cho vector database.
--
-- Số chiều 1536 KHÔNG phải chép từ tutorial. Nó đến từ 00_smoke_test.py:
--   gemini-embedding-001 trả về 3072 chiều,
--   pgvector chỉ đánh index HNSW/IVFFlat được tối đa 2000 chiều,
--   nên rút xuống 1536 bằng output_dimensionality.
-- Nếu đổi model embedding thì phải chạy lại smoke test và sửa con số này.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    source_name  TEXT        NOT NULL,   -- tên file nguồn, dùng để trích dẫn
    section      TEXT        NOT NULL,   -- tiêu đề mục trong file
    chunk_index  INT         NOT NULL,   -- thứ tự chunk trong file
    content      TEXT        NOT NULL,   -- nội dung đã kèm dòng ngữ cảnh
    embedding    vector(1536) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Nạp lại cùng một file sẽ ghi đè thay vì nhân bản chunk.
    UNIQUE (source_name, chunk_index)
);

-- Index cosine. Vector được chuẩn hoá ở tầng Python trước khi lưu, nên
-- cosine và inner product tương đương; dùng cosine cho rõ nghĩa.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source_name);

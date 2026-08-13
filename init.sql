-- BƯỚC 5 — Schema cho vector database.
--
-- Số chiều PHẢI khớp với EMBED_DIMENSION trong .env. store.assert_schema_matches()
-- đối chiếu hai con số này và hỏng ngay nếu lệch, nên không sợ quên âm thầm.
--
-- Hiện đặt 768 vì backend embedding là `local` với intfloat/multilingual-e5-base
-- (768 chiều). Lịch sử con số này:
--   gemini-embedding-001 trả 3072 chiều -> rút xuống 1536 bằng output_dimensionality
--   vì pgvector chỉ đánh index HNSW/IVFFlat được tối đa 2000 chiều.
--   Chuyển sang embedding chạy trên máy (không hạn mức) -> 768.
--
-- Đổi model embedding thì phải sửa con số ở đây VÀ dựng lại volume:
--   docker compose down -v && docker compose up -d

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    source_name  TEXT        NOT NULL,   -- tên file nguồn, dùng để trích dẫn
    section      TEXT        NOT NULL,   -- tiêu đề mục trong file
    chunk_index  INT         NOT NULL,   -- thứ tự chunk trong file
    content      TEXT        NOT NULL,   -- nội dung đã kèm dòng ngữ cảnh
    embedding    vector(768) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Nạp lại cùng một file sẽ ghi đè thay vì nhân bản chunk.
    UNIQUE (source_name, chunk_index)
);

-- Index cosine. Vector được chuẩn hoá ở tầng Python trước khi lưu, nên
-- cosine và inner product tương đương; dùng cosine cho rõ nghĩa.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source_name);

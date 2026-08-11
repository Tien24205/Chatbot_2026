"""
BƯỚC 4-5 — Ingestion pipeline: đọc -> chunk -> embed -> nạp vào pgvector.

Yêu cầu: Docker đang chạy (`docker compose up -d`) và .env đã có GEMINI_API_KEY.

Chạy:
  .venv\\Scripts\\python.exe 02_ingest.py
"""

from __future__ import annotations

import time

from rag import config, embedder, store
from rag.chunker import chunk_documents
from rag.loader import load_directory


def main() -> None:
    print("=" * 66)
    print("INGESTION PIPELINE")
    print("=" * 66)
    print(f"\n  model embedding : {config.EMBED_MODEL}")
    print(f"  số chiều        : {config.EMBED_DIMENSION}")
    print(f"  database        : {config.DATABASE_URL.rsplit('@', 1)[-1]}")

    # --- 1. Đọc + chunk (offline, đã kiểm thử ở 01_test_chunking.py) ---------
    docs = load_directory(config.KB_DIR)
    chunks = chunk_documents(docs)
    print(f"\n  Đọc {len(docs)} tài liệu -> {len(chunks)} chunk")

    # --- 2. Kết nối database TRƯỚC khi embed --------------------------------
    # Cố tình kiểm tra database trước: nếu schema sai thì hỏng ngay từ đầu,
    # thay vì phát hiện sau khi đã đốt tiền embedding cho toàn bộ chunk.
    # Không đòi require_chunks: đây chính là script nạp dữ liệu vào bảng rỗng.
    with store.connect_or_exit() as conn:
        store.assert_schema_matches(conn)
        print(f"  Schema khớp: vector({config.EMBED_DIMENSION})")

        # --- 3. Embedding ---------------------------------------------------
        print(f"\n  Đang embed {len(chunks)} chunk...")
        t0 = time.perf_counter()
        vectors = embedder.embed_documents([c.content for c in chunks], progress=True)
        elapsed = time.perf_counter() - t0
        print(f"  Xong trong {elapsed:.1f}s ({elapsed / len(chunks):.2f}s/chunk)")

        # Kiểm chứng đầu ra thay vì tin tưởng mù quáng.
        assert all(len(v) == config.EMBED_DIMENSION for v in vectors), "Sai số chiều"
        norms = [sum(x * x for x in v) ** 0.5 for v in vectors]
        worst = max(abs(n - 1.0) for n in norms)
        print(f"  Mọi vector đã chuẩn hoá (sai lệch norm lớn nhất: {worst:.2e})")

        # --- 4. Nạp vào pgvector --------------------------------------------
        n = store.upsert_chunks(conn, chunks, vectors)
        print(f"\n  Đã nạp {n} chunk. Tổng trong database: {store.count(conn)}")

        # --- 5. Truy vấn thử ngay để xác nhận toàn tuyến chạy ---------------
        print("\n" + "=" * 66)
        print("TRUY VẤN THỬ")
        print("=" * 66)

        for q in [
            "Baron Nashor là gì?",
            "Phản ứng Bốc Hơi trong Genshin hoạt động thế nào?",
            "Cách nấu phở bò",  # ngoài phạm vi -> để xem điểm rơi tới đâu
        ]:
            hits = store.search(conn, embedder.embed_query(q), top_k=3)
            print(f"\n  Hỏi: {q}")
            for i, h in enumerate(hits, 1):
                print(f"    {i}. {h.similarity:.4f}  [{h.source_name} · {h.section}]")

    print("\n" + "=" * 66)
    print("HOÀN TẤT")
    print("=" * 66)
    print(
        "\nBước tiếp theo (Bước 6): hiệu chuẩn ngưỡng similarity bằng bộ câu hỏi test,\n"
        "rồi ghép prompt + LLM ở Bước 7-9.\n"
    )


if __name__ == "__main__":
    main()

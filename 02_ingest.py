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

        # --- 3. Bỏ qua chunk đã embed lần chạy trước -------------------------
        # Embedding là thứ ĐÃ TRẢ TIỀN (bằng hạn mức). Chạy lại mà embed lại từ
        # đầu là trả lần thứ hai cho cùng một đoạn văn — với KB vài nghìn chunk
        # thì đủ để đốt hết hạn mức ngày.
        done = store.existing_keys(conn)
        todo = [c for c in chunks if (c.source_name, c.chunk_index) not in done]
        if len(todo) < len(chunks):
            print(f"\n  Bỏ qua {len(chunks) - len(todo)} chunk đã có vector từ lần chạy trước")
        if not todo:
            print("  Không có gì để nạp thêm — database đã đầy đủ.")

        # --- 4. Embedding + ghi theo LÔ --------------------------------------
        # Ghi sau mỗi lô thay vì gom hết tới cuối: hỏng giữa chừng thì chỉ mất lô
        # cuối, phần trước vẫn nằm trong database và lần chạy sau bỏ qua được.
        WRITE_EVERY = 200
        t0 = time.perf_counter()
        worst = 0.0
        loaded = 0
        for start in range(0, len(todo), WRITE_EVERY):
            part = todo[start : start + WRITE_EVERY]
            print(f"\n  Đang embed lô {start + 1}-{start + len(part)} / {len(todo)}...")
            vectors = embedder.embed_documents([c.content for c in part], progress=True)

            # Kiểm chứng đầu ra thay vì tin tưởng mù quáng.
            assert all(len(v) == config.EMBED_DIMENSION for v in vectors), "Sai số chiều"
            worst = max(worst, max(abs(sum(x * x for x in v) ** 0.5 - 1.0) for v in vectors))

            loaded += store.upsert_chunks(conn, part, vectors)
            print(f"    đã ghi vào database: {loaded}/{len(todo)}")

        elapsed = time.perf_counter() - t0
        if todo:
            print(f"\n  Xong trong {elapsed:.1f}s ({elapsed / len(todo):.2f}s/chunk)")
            print(f"  Mọi vector đã chuẩn hoá (sai lệch norm lớn nhất: {worst:.2e})")
        print(f"  Đã nạp {loaded} chunk. Tổng trong database: {store.count(conn)}")

        # --- 5. Truy vấn thử ngay để xác nhận toàn tuyến chạy ---------------
        print("\n" + "=" * 66)
        print("TRUY VẤN THỬ")
        print("=" * 66)

        for q in [
            "Phản ứng Bốc Hơi hoạt động thế nào?",
            "Mondstadt là vùng đất của thần nào?",
            # Ngoài phạm vi. Chọn câu về GAME KHÁC chứ không phải nấu ăn: sau khi
            # thu hẹp kho về một game, câu hỏi cùng miền game mới là ca sát ngưỡng.
            "Baron Nashor trong Liên Minh Huyền Thoại là gì?",
            "Cách nấu phở bò",
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

"""
BƯỚC 7-9 — Chatbot dòng lệnh.

Chạy:  .venv\\Scripts\\python.exe 04_chat.py

Lệnh trong phiên chat:
  /nguon    xem chi tiết các chunk đã dùng cho câu trả lời vừa rồi
  /moi      xoá lịch sử hội thoại
  /thoat    kết thúc
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rag import config, pipeline, store

# Giới hạn lịch sử để prompt không phình vô hạn qua nhiều lượt.
MAX_HISTORY_TURNS = 6


def main() -> None:
    try:
        conn = store.connect()
    except Exception as e:
        print(f"[THẤT BẠI] Không kết nối được database: {e}")
        print("  Chạy: docker compose up -d")
        sys.exit(1)

    with conn:
        n = store.count(conn)
        if n == 0:
            print("[THẤT BẠI] Database rỗng. Chạy 02_ingest.py trước.")
            sys.exit(1)

        print("=" * 70)
        print("CHATBOT TRA CỨU KIẾN THỨC GAME")
        print("=" * 70)
        print(f"\n  {n} chunk · model {config.CHAT_MODEL} · ngưỡng {config.SIMILARITY_THRESHOLD}")
        print("  Chủ đề: Liên Minh Huyền Thoại, Genshin Impact, Liên Quân Mobile, thuật ngữ game")
        print("\n  Lệnh: /nguon  /moi  /thoat\n")

        history: list[dict] = []
        last = None

        while True:
            try:
                q = input("Bạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nTạm biệt.")
                break

            if not q:
                continue

            if q in ("/thoat", "/exit", "/quit"):
                print("Tạm biệt.")
                break

            if q == "/moi":
                history.clear()
                last = None
                print("  (đã xoá lịch sử hội thoại)\n")
                continue

            if q == "/nguon":
                if not last or not last.hits:
                    print("  (chưa có câu trả lời nào)\n")
                    continue
                print()
                for i, h in enumerate(last.hits, 1):
                    print(f"  [{i}] {h.similarity:.4f}  {h.source_name} · {h.section}")
                    preview = " ".join(h.content.split())[:160]
                    print(f"      {preview}...")
                print()
                continue

            try:
                ans = pipeline.ask(conn, q, history=history)
            except Exception as e:
                print(f"\n  [LỖI] {e}\n")
                continue

            last = ans

            # Câu hỏi nối tiếp được viết lại trước khi truy hồi — in ra cho minh bạch.
            if ans.search_query and ans.search_query != q:
                print(f'  (tìm kiếm với: "{ans.search_query}")')

            print(f"\nBot: {ans.text}\n")

            if ans.refused:
                print(f"  (từ chối — độ liên quan cao nhất {ans.top_similarity:.3f} "
                      f"< ngưỡng {config.SIMILARITY_THRESHOLD})")
            else:
                print(f"  Nguồn: {' | '.join(ans.citations)}")
                # Chỉ ghi lịch sử khi thực sự trả lời được. Nếu ghi cả lượt bị từ
                # chối, LLM sẽ thấy mẫu "hỏi -> từ chối" và dễ từ chối lây sang
                # các câu sau vốn trả lời được.
                history.append({"role": "user", "parts": [{"text": q}]})
                history.append({"role": "model", "parts": [{"text": ans.text}]})
                del history[: max(0, len(history) - MAX_HISTORY_TURNS * 2)]

            parts = [f"truy hồi {ans.retrieval_ms} ms"]
            if ans.rewrite_ms:
                parts.insert(0, f"viết lại câu hỏi {ans.rewrite_ms} ms")
            if ans.llm_ms:
                parts.append(f"LLM {ans.llm_ms} ms")
            print(f"  ({ans.total_ms} ms — {', '.join(parts)})\n")


if __name__ == "__main__":
    main()

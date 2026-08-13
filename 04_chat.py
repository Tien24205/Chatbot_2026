"""
BƯỚC 7-9 — Chatbot dòng lệnh.

Chạy:  .venv\\Scripts\\python.exe 04_chat.py

Lệnh trong phiên chat:
  /nguon    xem chi tiết các chunk đã dùng cho câu trả lời vừa rồi
  /moi      xoá lịch sử hội thoại
  /thoat    kết thúc
"""

from __future__ import annotations

from rag import config, pipeline, store


def main() -> None:
    with store.connect_or_exit(require_chunks=True) as conn:
        n = store.count(conn)

        print("=" * 70)
        print("CHATBOT TRA CỨU KIẾN THỨC GAME")
        print("=" * 70)
        print(f"\n  {n} chunk · model {config.CHAT_MODEL} · ngưỡng {config.SIMILARITY_THRESHOLD}")
        print("  Chủ đề: wiki Genshin Impact — nhân vật, vũ khí, thánh di vật, khu vực, thuật ngữ")
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

            if ans.refused and ans.refused_by == "model":
                print(f"  (có tài liệu liên quan {ans.top_similarity:.3f} nhưng không "
                      f"đoạn nào chứa câu trả lời — model tự nói, không phải ngưỡng)")
            elif ans.refused:
                print(f"  (từ chối — độ liên quan cao nhất {ans.top_similarity:.3f} "
                      f"< ngưỡng {config.SIMILARITY_THRESHOLD})")
            else:
                print(f"  Nguồn: {' | '.join(ans.citations)}")

            # Tự bỏ qua lượt bị từ chối — xem pipeline.remember.
            pipeline.remember(history, q, ans)

            parts = [f"truy hồi {ans.retrieval_ms} ms"]
            if ans.rewrite_ms:
                parts.insert(0, f"viết lại câu hỏi {ans.rewrite_ms} ms")
            if ans.llm_ms:
                parts.append(f"LLM {ans.llm_ms} ms")
            print(f"  ({ans.total_ms} ms — {', '.join(parts)})\n")


if __name__ == "__main__":
    main()

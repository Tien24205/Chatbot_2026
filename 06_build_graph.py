"""
GRAPHRAG — dựng đồ thị tri thức từ knowledge base.

Chạy SAU 02_ingest.py (cần bảng chunks đã có dữ liệu để nối mentions).
KHÔNG gọi API: trích thực thể bằng regex, dựng cạnh bằng SQL.

  .venv\\Scripts\\python.exe 06_build_graph.py

Script tự kiểm chứng luôn hai thứ mà truy hồi thuần vector làm không tốt:
  1. đi từ một thực thể sang thực thể cùng vai trò ở tài liệu khác (đa bước),
  2. tóm tắt cộng đồng để biết kho kiến thức thực sự phủ những gì.
"""

from __future__ import annotations

from rag import config, graph, store
from rag.loader import load_directory


def main() -> None:
    print("=" * 66)
    print("DỰNG ĐỒ THỊ TRI THỨC (GraphRAG)")
    print("=" * 66)

    docs = load_directory(config.KB_DIR)

    # Cần bảng chunks đã có dữ liệu: đồ thị nối thực thể vào chunk qua mentions.
    with store.connect_or_exit(require_chunks=True) as conn:
        n_chunks = store.count(conn)

        stats = graph.build(conn, docs)
        print(f"\n  {len(docs)} tài liệu, {n_chunks} chunk")
        print(f"  Thực thể      : {stats['entities']}")
        print(f"  Bí danh       : {stats['aliases']}")
        print(f"  Lượt nhắc tới : {stats['mentions']}")
        print(f"  Cạnh          : {stats['edges']}")

        # --- Phân bố theo loại -------------------------------------------------
        print("\n" + "-" * 66)
        print("THỰC THỂ THEO VAI TRÒ")
        print("-" * 66)
        rows = conn.execute(
            "SELECT kind, count(*) FROM graph_entities GROUP BY kind ORDER BY 2 DESC"
        ).fetchall()
        for kind, n in rows:
            print(f"  {n:>4}  {kind}")

        print("\n" + "-" * 66)
        print("CẠNH THEO QUAN HỆ")
        print("-" * 66)
        rows = conn.execute(
            "SELECT relation, count(*) FROM graph_edges GROUP BY relation ORDER BY 2 DESC"
        ).fetchall()
        for rel, n in rows:
            print(f"  {n:>4}  {rel}")

        # --- Kiểm chứng 1: đi đa bước ------------------------------------------
        # Đây chính là loại câu mà truy hồi thuần vector trượt (đo ở Milestone 5):
        # "Kiếm Sắt Đen" và "Ánh Trăng Xiphos" không giống nhau MỘT CHỮ NÀO, nên
        # vector không nối được. Đồ thị nối bằng vai trò (cùng type trong infobox).
        # "Mondstadt" cố ý giữ lại làm ca 0 hàng xóm: được nhận ra nhưng không có
        # thực thể cùng vai trò — để thấy nhánh đó của code cũng chạy.
        print("\n" + "-" * 66)
        print("KIỂM CHỨNG — ĐI ĐA BƯỚC GIỮA CÁC TÀI LIỆU")
        print("-" * 66)

        for start in ["Quá Tải", "Kiếm Sắt Đen", "Mondstadt"]:
            names = graph.load_lexicon(conn).find(start)
            if not names:
                print(f"\n  '{start}' — không nhận ra thực thể nào")
                continue
            nbrs = graph.neighbors(conn, names, ("tương_tự",))
            print(f"\n  '{start}' -> nhận ra: {', '.join(names)}")
            if not nbrs:
                print("      (không có thực thể cùng vai trò ở tài liệu khác)")
            for n in nbrs[:6]:
                print(
                    f"      {n['from_name']} --{n['relation']}--> {n['name']}"
                    f"  [{n['source_name']} · {n['kind']}]"
                )

            extra = graph.chunks_mentioning(conn, [n["name"] for n in nbrs], limit=2)
            for c in extra:
                print(f"      => chunk có thể bổ sung: {c['source_name']} · {c['section']}")

        # --- Kiểm chứng 2: tóm tắt cộng đồng -----------------------------------
        print("\n" + "-" * 66)
        print("TÓM TẮT CỘNG ĐỒNG (kho kiến thức phủ những gì)")
        print("-" * 66)
        for c in graph.communities(conn):
            print(f"\n  {c['source_name']} — {c['entities']} thực thể")
            print(f"    vai trò : {', '.join(c['kinds'])}")
            print(f"    ví dụ   : {', '.join(c['samples'][:5])}")

    print("\n" + "=" * 66)
    print("HOÀN TẤT — đồ thị sẵn sàng, pipeline sẽ tự dùng khi truy hồi")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()

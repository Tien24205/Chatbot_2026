"""
Kiểm thử tầng đồ thị (GraphRAG) và các lớp chống hallucination tất định.

Plan §Bước 10 yêu cầu test từng tầng. Đây là Test 8 (Graph) và Test 9 (Verification).

KHÔNG cần API key. Cần Docker cho phần đồ thị (đọc bảng chunks); phần kiểm chứng
chạy hoàn toàn offline. Chạy:

  .venv\\Scripts\\python.exe 07_test_graph_verify.py
"""

from __future__ import annotations

from rag import config, graph, store, verify
from rag.checks import check, report
from rag.loader import load_directory
from rag.store import SearchHit


def hit(content: str, similarity: float = 0.7, via: str = "vector", cid: int = 1) -> SearchHit:
    return SearchHit(
        content=content, source_name="test.txt", section="Mục thử", similarity=similarity,
        id=cid, via=via,
    )


# ===========================================================================
# TEST 8 — TRÍCH THỰC THỂ VÀ ĐỒ THỊ
# ===========================================================================


def test_extraction() -> None:
    print("=" * 66)
    print("TEST 8 — TRÍCH THỰC THỂ (offline, không cần database)")
    print("=" * 66 + "\n")

    docs = load_directory(config.KB_DIR)
    entities, edges = graph.extract(docs)
    by_name = {e.name: e for e in entities}

    check(len(entities) > 50, f"Trích được nhiều thực thể ({len(entities)})")

    check("Nahida" in by_name, "Bắt được thực thể là tiêu đề trang wiki")
    check("Phản Ứng Nguyên Tố" in by_name, "Bắt được tên nhiều chữ")

    # Vai trò lấy từ dòng `type:` của infobox chứ KHÔNG từ tiêu đề mục: tiêu đề mục
    # của wiki là "Tổng Quan", "Mô Tả" — không nói gì về việc trang đó nói về cái gì.
    # Phải tra theo CẢ tài liệu, không chỉ theo tên: "Nahida" xuất hiện ở cả
    # Nahida.txt lẫn NPC.txt, mà by_name chỉ giữ bản cuối cùng.
    nahida = next(
        (e for e in entities if e.name == "Nahida" and e.source_name == "Nahida.txt"), None
    )
    check(
        nahida is not None and nahida.kind == "Chơi Được",
        "Suy được vai trò từ infobox (dòng type:)",
        f"kind={nahida.kind if nahida else None}",
    )
    tdv = next(
        (e for e in entities
         if e.name == "Thánh Di Vật" and e.source_name == "Thánh Di Vật.txt"),
        None,
    )
    check(
        tdv is not None and tdv.kind == "Loại Vật Phẩm",
        "Trang Thánh Di Vật lấy đúng vai trò từ infobox",
        f"kind={tdv.kind if tdv else None}",
    )

    # Khoá infobox không được thành thực thể. Nếu 09_fetch_fandom.py xuất infobox
    # dưới dạng `- type: X` thì mỗi trang sinh một thực thể tên "type" — 1.100 thực
    # thể rác, và Lexicon khớp chữ "type" sẽ trả về toàn bộ chúng.
    junk = [n for n in by_name if n.lower() in {"type", "quality", "region", "element"}]
    check(not junk, "Khoá infobox không bị nhầm thành tên thực thể", f"{junk}")

    # Cùng một tên ở nhiều tài liệu phải là NHIỀU thực thể riêng, không được gộp.
    mond = [e for e in entities if e.name.lower() == "mondstadt"]
    sources = {e.source_name for e in mond}
    check(
        len(mond) >= 2 and len(sources) == len(mond),
        "Mỗi tài liệu giữ riêng một 'Mondstadt', không gộp theo tên",
        f"tìm thấy {len(mond)} bản từ {len(sources)} tài liệu",
    )

    # Cạnh tương_tự PHẢI nối chéo tài liệu, không bao giờ nối trong cùng một file.
    by_key = {e.key: e for e in entities}
    similar = [e for e in edges if e.relation == "tương_tự"]
    orphan = [e for e in edges if e.src not in by_key or e.dst not in by_key]
    check(not orphan, f"Mọi cạnh đều trỏ tới thực thể có thật ({len(orphan)} cạnh mồ côi)")

    same_file = [
        e for e in similar if by_key[e.src].source_name == by_key[e.dst].source_name
    ]
    check(len(similar) > 0, f"Có cạnh tương_tự giữa các tài liệu ({len(similar)} cạnh)")
    check(not same_file, "Cạnh tương_tự không bao giờ nối trong cùng một tài liệu")

    # Nhóm vai trò QUÁ LỚN phải bị loại, nếu không số cạnh phình theo bậc hai và
    # quan hệ mất hết ý nghĩa ("hai nhân vật cùng là nhân vật chơi được").
    sim_kinds = {by_key[e.src].kind for e in similar} | {by_key[e.dst].kind for e in similar}
    check(
        not ({"khái niệm", "chủ đề", "Chơi Được"} & sim_kinds),
        f"Nhóm vai trò lớn hơn {graph.MAX_SIMILAR_GROUP} bị loại khỏi cạnh tương_tự",
        f"lọt: {sorted({'khái niệm', 'chủ đề', 'Chơi Được'} & sim_kinds)}",
    )
    # ...nhưng nhóm vừa phải thì PHẢI được nối: hai vũ khí cùng loại là quan hệ
    # có thật mà vector search không nối được (tên chúng không giống nhau chữ nào).
    check(
        bool({"Kiếm Đơn", "Trọng Kiếm", "Cung", "Pháp Khí"} & sim_kinds),
        "Nối được các vũ khí cùng loại (thứ vector KHÔNG nối được)",
        f"vai trò có cạnh: {sorted(sim_kinds)[:8]}",
    )


def test_lexicon() -> None:
    print("\n" + "=" * 66)
    print("TEST 8b — DÒ TÊN TRONG VĂN BẢN")
    print("=" * 66 + "\n")

    docs = load_directory(config.KB_DIR)
    entities, _ = graph.extract(docs)
    lex = graph.Lexicon(entities)

    check("Mondstadt" in lex.find("Mondstadt là thành phố của tự do"), "Tìm được tên đầy đủ")

    # Khớp cụm DÀI trước: "Phản Ứng Nguyên Tố" không được cắt thành "Nguyên Tố".
    found = lex.find("Phản Ứng Nguyên Tố quyết định lối chơi")
    check(
        "Phản Ứng Nguyên Tố" in found,
        "Khớp cụm dài trước, không cắt thành thực thể ngắn hơn",
        f"tìm thấy: {found[:5]}",
    )

    check(bool(lex.find("Phản ứng Bốc Hơi rất mạnh")), "Tìm được tên nằm giữa câu")
    check(not lex.find("Hôm nay trời đẹp và tôi muốn nấu phở"), "Không bắt nhầm văn bản vô quan")


def test_graph_db() -> None:
    print("\n" + "=" * 66)
    print("TEST 8c — ĐỒ THỊ TRONG DATABASE")
    print("=" * 66 + "\n")

    try:
        conn = store.connect()
    except Exception as e:
        print(f"  [BỎ QUA] Không kết nối được database: {e}")
        print("           Chạy `docker compose up -d` rồi thử lại.\n")
        return

    with conn:
        if not graph.is_built(conn):
            print("  [BỎ QUA] Đồ thị chưa dựng. Chạy 06_build_graph.py trước.\n")
            return

        lex = graph.load_lexicon(conn)
        check(bool(lex.find("Mondstadt")), "Nạp được bộ dò tên từ database")

        # Akuoumaru là một Trọng Kiếm — nhóm vai trò vừa đủ nhỏ để có cạnh tương_tự.
        seed = "Akuoumaru"
        nbrs = graph.neighbors(conn, [seed], ("tương_tự",))
        names = [n["name"] for n in nbrs]
        check(bool(nbrs), "Đi được một bước sang thực thể cùng vai trò", f"{names[:5]}")
        check(seed not in names, "Không trả về chính nó làm hàng xóm")
        check(
            all(n["from_name"] == seed for n in nbrs),
            "Ghi lại được đi TỪ thực thể nào, để giải thích được vì sao đoạn được kéo vào",
        )
        check(
            len({n["kind"] for n in nbrs}) == 1,
            "Hàng xóm qua cạnh tương_tự luôn cùng vai trò",
            f"{sorted({n['kind'] for n in nbrs})}",
        )

        # Điều cần kiểm: đi qua cạnh `tương_tự` phải RA KHỎI tài liệu gốc. Không
        # chốt cứng tên file đích — top-3 rơi vào tài liệu nào là chuyện của trọng
        # số, không phải của bất biến.
        chunks = graph.chunks_mentioning(conn, names, limit=3)
        sources = {c["source_name"] for c in chunks}
        check(
            bool(sources - {f"{seed}.txt"}),
            "Từ thực thể hàng xóm lấy được chunk của tài liệu khác",
            f"chỉ lấy được chunk trong chính {seed}.txt: {sources}",
        )

        # Đếm theo số file thật trong knowledge_base/, KHÔNG chốt cứng con số:
        # thêm một tài liệu vào KB là chuyện bình thường, mà test hỏng theo thì
        # người ta sẽ sửa con số cho qua chứ không đọc xem có gì sai thật.
        n_docs = len(load_directory(config.KB_DIR))
        comms = graph.communities(conn)
        check(
            len(comms) == n_docs,
            f"Tóm tắt cộng đồng phủ đủ {n_docs} tài liệu ({len(comms)})",
            f"thiếu: {sorted({d.source_name for d in load_directory(config.KB_DIR)} - {c['source_name'] for c in comms})}",
        )
        check(
            all(c["entities"] > 0 and c["kinds"] for c in comms),
            "Mỗi cộng đồng đều có thực thể và vai trò",
        )


# ===========================================================================
# TEST 9 — CÁC LỚP CHỐNG HALLUCINATION TẤT ĐỊNH
# ===========================================================================


def test_citations() -> None:
    print("\n" + "=" * 66)
    print("TEST 9 — LỚP 3: TRÍCH DẪN CÓ CẤU TRÚC")
    print("=" * 66 + "\n")

    text, invalid = verify.strip_invalid_citations("Baron cho buff [1] và tiếp sức lính [7].", 4)
    check(invalid == [7], "Phát hiện chỉ số trích dẫn không tồn tại", f"invalid={invalid}")
    check("[7]" not in text, "Đã bỏ chỉ số không tồn tại khỏi câu trả lời", text)
    check("[1]" in text, "Giữ nguyên chỉ số hợp lệ", text)
    check(text.endswith("."), "Không để lại khoảng trắng trước dấu câu", repr(text))

    _, invalid = verify.strip_invalid_citations("Không trích gì cả.", 4)
    check(invalid == [], "Câu không trích dẫn thì không bị báo lỗi")

    r = verify.check("Ăn đủ bốn rồng nhận Linh Hồn Rồng [1]. Đây là lợi thế lớn.",
                     [hit("Ăn đủ bốn rồng sẽ nhận Linh Hồn Rồng, một lợi thế rất lớn.")])
    check(0 < r.citation_coverage < 1, f"Đo được tỉ lệ câu có trích dẫn ({r.citation_coverage:.0%})")


def test_grounding() -> None:
    print("\n" + "=" * 66)
    print("TEST 9b — LỚP 4: KIỂM CHỨNG GROUNDING TẤT ĐỊNH")
    print("=" * 66 + "\n")

    context = hit("Mỗi lượt cầu nguyện tốn 160 Nguyên Thạch. Bảo hiểm 90 lượt.")

    r = verify.check("Mỗi lượt tốn 160 Nguyên Thạch, bảo hiểm ở lượt 90.", [context])
    check(r.ok, "Câu trả lời đúng số liệu thì sạch cờ", str(r.flags))

    # Đây là lỗi mà system prompt CHỈ có thể khuyên, không thể chặn.
    r = verify.check("Mỗi lượt tốn 180 Nguyên Thạch.", [context])
    check(r.ungrounded_numbers == ["180"], "Bắt được số bịa", str(r.ungrounded_numbers))
    check(verify.escalate_needed(r), "Số bịa thì leo thang sang lớp 5")

    # Số do người dùng nêu trong câu hỏi không phải là bịa.
    r = verify.check("Bạn hỏi về 5 lượt thì tốn 800.", [context], question="Tôi quay 5 lượt?")
    check("5" not in r.ungrounded_numbers, "Số lấy từ câu hỏi không bị tính là bịa")
    check("800" in r.ungrounded_numbers, "Nhưng số tự tính ra thì vẫn bị bắt")

    # Số nhỏ trong lối diễn đạt thông thường không tính.
    r = verify.check("Có hai đội, mỗi đội 1 người dẫn.", [context])
    check(r.ok, "Số 0/1/2 trong lối nói không gây báo động giả", str(r.flags))

    # Tên riêng không có trong ngữ cảnh — model lôi từ kiến thức nền ra.
    r = verify.check("Cơ chế này giống Honkai Star Rail của HoYoverse.", [context])
    check(bool(r.ungrounded_names), "Bắt được tên riêng lạ", str(r.ungrounded_names))

    # Chữ đầu câu viết hoa KHÔNG phải tên riêng. Ca thật gặp khi chạy 08_eval_answers:
    # "Trong Liên Minh Huyền Thoại..." từng bị báo là bịa tên "Trong Liên".
    lmht = hit("Liên Minh Huyền Thoại là game MOBA do Riot Games phát triển.")
    r = verify.check("Trong Liên Minh Huyền Thoại, Riot Games là nhà phát triển.", [lmht])
    check(
        not r.ungrounded_names,
        "Chữ đầu câu viết hoa không bị tính là tên riêng bịa",
        str(r.ungrounded_names),
    )
    r = verify.check("Game này do Riot Games và Tencent Holdings phát triển.", [lmht])
    check(
        any("Tencent" in n for n in r.ungrounded_names),
        "Nhưng tên riêng thật sự lạ giữa câu vẫn bị bắt",
        str(r.ungrounded_names),
    )

    docs = load_directory(config.KB_DIR)
    entities, _ = graph.extract(docs)
    lex = graph.Lexicon(entities)

    # Thực thể CÓ THẬT trong knowledge base nhưng KHÔNG có trong đoạn được truy hồi:
    # thông tin nghe đúng nhưng không đến từ nguồn đã trích — rất khó thấy bằng mắt.
    r = verify.check(
        "Nguyên Thạch dùng cho gacha, giống như ở Mondstadt.", [context], lexicon=lex
    )
    check(
        "Mondstadt" in r.ungrounded_entities,
        "Bắt được thực thể có thật nhưng nằm ngoài ngữ cảnh đã truy hồi",
        str(r.ungrounded_entities),
    )


def main() -> None:
    test_extraction()
    test_lexicon()
    test_graph_db()
    test_citations()
    test_grounding()

    report()


if __name__ == "__main__":
    main()

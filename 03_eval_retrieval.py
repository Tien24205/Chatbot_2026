"""
BƯỚC 6 — Đánh giá retrieval và hiệu chuẩn ngưỡng similarity.

Trả lời hai câu hỏi tách bạch:

  1. TRUY HỒI CÓ ĐÚNG KHÔNG?  -> recall@k: chunk đúng có nằm trong top-k không.
     Đây là chất lượng embedding + chunking, không liên quan tới ngưỡng.

  2. NGƯỠNG τ NÊN ĐẶT Ở ĐÂU?  -> quét toàn dải τ, đo xem mức nào vừa trả lời
     đúng câu trong phạm vi, vừa từ chối được câu ngoài phạm vi.

Không chép τ từ tutorial. Mỗi model embedding có nền tương đồng khác nhau.

Chạy:  .venv\\Scripts\\python.exe 03_eval_retrieval.py
"""

from __future__ import annotations

import json

from rag import config, embedder, store

QUESTIONS_FILE = config.PROJECT_DIR / "eval_questions.json"
TOP_K = 5


def main() -> None:
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = data["questions"]

    print("=" * 74)
    print("BƯỚC 6 — ĐÁNH GIÁ RETRIEVAL")
    print("=" * 74)

    with store.connect_or_exit(require_chunks=True) as conn:
        total_chunks = store.count(conn)
        print(f"\n  {len(questions)} câu hỏi · {total_chunks} chunk trong database · top-{TOP_K}\n")

        results = []
        for item in questions:
            hits = store.search(conn, embedder.embed_query(item["q"]), top_k=TOP_K)
            results.append({"item": item, "hits": hits})
    print(f"    đã truy vấn {len(results)} câu\n")

    def acceptable(item: dict) -> set[str]:
        if item["type"] == "in_scope":
            return {item["expect_source"]}
        if item["type"] == "ambiguous":
            return set(item["expect_any_source"])
        return set()

    answerable = [r for r in results if r["item"]["type"] in ("in_scope", "ambiguous")]
    out_scope = [r for r in results if r["item"]["type"] == "out_of_scope"]

    # ---------------------------------------------------------------- recall
    print("=" * 74)
    print("1. RECALL — chunk đúng có nằm trong top-k không?")
    print("=" * 74)
    print("\n  (Chỉ tính chất lượng truy hồi, chưa áp ngưỡng nào.)\n")

    for k in (1, 3, 5):
        src_hit = sum(
            1 for r in answerable if any(h.source_name in acceptable(r["item"]) for h in r["hits"][:k])
        )
        sec_hit = sum(
            1
            for r in answerable
            if r["item"].get("expect_section")
            and any(
                h.source_name in acceptable(r["item"])
                and h.section == r["item"]["expect_section"]
                for h in r["hits"][:k]
            )
        )
        strict_total = sum(1 for r in answerable if r["item"].get("expect_section"))
        print(
            f"    recall@{k}:  đúng FILE {src_hit}/{len(answerable)} ({src_hit / len(answerable):.0%})"
            f"    ·  đúng cả MỤC {sec_hit}/{strict_total} ({sec_hit / strict_total:.0%})"
        )

    misses = [
        r
        for r in answerable
        if not any(h.source_name in acceptable(r["item"]) for h in r["hits"][:TOP_K])
    ]
    if misses:
        print(f"\n  Trượt hoàn toàn ({len(misses)} câu):")
        for r in misses:
            top = r["hits"][0]
            print(f"    - {r['item']['q']}")
            print(f"        mong đợi: {sorted(acceptable(r['item']))}")
            print(f"        nhận được: {top.source_name} · {top.section} ({top.similarity:.3f})")

    # ------------------------------------------------------- phân bố điểm số
    print("\n" + "=" * 74)
    print("2. PHÂN BỐ ĐIỂM SỐ — trong phạm vi vs ngoài phạm vi")
    print("=" * 74)

    in_top1 = sorted(r["hits"][0].similarity for r in answerable)
    out_top1 = sorted(r["hits"][0].similarity for r in out_scope)

    def stats(name: str, xs: list[float]) -> None:
        print(
            f"    {name:<22} thấp nhất {min(xs):.3f} · trung vị {xs[len(xs) // 2]:.3f} "
            f"· cao nhất {max(xs):.3f}"
        )

    print()
    stats("Trong phạm vi", in_top1)
    stats("Ngoài phạm vi", out_top1)

    gap = min(in_top1) - max(out_top1)
    print(
        f"\n    Khoảng cách: {gap:+.3f}  "
        + (
            "(hai nhóm tách rời — tồn tại ngưỡng chia đúng tuyệt đối)"
            if gap > 0
            else "(hai nhóm CHỒNG LẤN — không ngưỡng nào chia đúng 100%)"
        )
    )

    # Chỉ ra CHÍNH XÁC câu nào gây chồng lấn — đó là câu đáng đọc nhất trong cả bộ.
    if gap <= 0:
        worst_out = max(out_scope, key=lambda r: r["hits"][0].similarity)
        weakest_in = min(answerable, key=lambda r: r["hits"][0].similarity)
        print("\n    Cặp gây chồng lấn:")
        print(
            f"      ngoài phạm vi cao nhất  {worst_out['hits'][0].similarity:.3f}  "
            f"\"{worst_out['item']['q']}\""
        )
        print(
            f"      trong phạm vi thấp nhất {weakest_in['hits'][0].similarity:.3f}  "
            f"\"{weakest_in['item']['q']}\""
        )

    # ------------------------------------------------------------ quét ngưỡng
    print("\n" + "=" * 74)
    print("3. QUÉT NGƯỠNG τ")
    print("=" * 74)
    print(
        "\n    τ      trả lời đúng   từ chối đúng   sai (bịa)   bỏ sót   độ chính xác"
    )
    print("    " + "-" * 68)

    best = None
    rows = []
    tau = 0.40
    while tau <= 0.801:
        correct_answer = wrong_answer = missed = 0
        for r in answerable:
            top = r["hits"][0]
            if top.similarity >= tau:
                if top.source_name in acceptable(r["item"]):
                    correct_answer += 1
                else:
                    wrong_answer += 1
            else:
                missed += 1

        good_refusal = sum(1 for r in out_scope if r["hits"][0].similarity < tau)
        bad_answer = len(out_scope) - good_refusal

        accuracy = (correct_answer + good_refusal) / len(results)
        rows.append((tau, correct_answer, good_refusal, wrong_answer + bad_answer, missed, accuracy))
        if best is None or accuracy > best[5]:
            best = rows[-1]
        tau += 0.02

    for tau, ca, gr, wrong, missed, acc in rows:
        mark = " <<<" if best and abs(tau - best[0]) < 1e-9 else ""
        print(
            f"    {tau:.2f}   {ca:>6}/{len(answerable):<7} {gr:>6}/{len(out_scope):<8}"
            f" {wrong:>6}      {missed:>4}      {acc:>6.0%}{mark}"
        )

    # Độ chính xác coi "bịa" và "bỏ sót" nặng như nhau — với RAG thì KHÔNG.
    # Bỏ sót = trả lời "tôi không có thông tin", người dùng hỏi lại được.
    # Bịa    = trả lời sai bằng giọng tự tin, người dùng không biết mà kiểm chứng.
    # Nên chọn ngưỡng thấp nhất mà số câu bịa bằng 0, rồi mới tối đa số câu trả lời được.
    safe = [r for r in rows if r[3] == 0]
    best_safe = max(safe, key=lambda r: (r[1], -r[0])) if safe else None

    print("\n" + "=" * 74)
    print("KẾT LUẬN")
    print("=" * 74)

    tau, ca, gr, wrong, missed, acc = best
    print(f"\n  [A] Tối đa độ chính xác thô:  τ = {tau:.2f}  ({acc:.0%})")
    print(f"      trả lời đúng {ca}/{len(answerable)} · từ chối đúng {gr}/{len(out_scope)}"
          f" · BỊA {wrong} · bỏ sót {missed}")

    if best_safe:
        s_tau, s_ca, s_gr, _, s_missed, s_acc = best_safe
        print(f"\n  [B] Không bịa câu nào:        τ = {s_tau:.2f}  ({s_acc:.0%})")
        print(f"      trả lời đúng {s_ca}/{len(answerable)} · từ chối đúng {s_gr}/{len(out_scope)}"
              f" · BỊA 0 · bỏ sót {s_missed}")

        chosen = s_tau
        print(
            f"\n  -> Chọn [B] τ = {chosen:.2f}."
            "\n     Hai phương án có thể ngang nhau về độ chính xác, nhưng với chatbot"
            "\n     tra cứu thì bịa nguy hiểm hơn bỏ sót: bỏ sót thì người dùng hỏi lại,"
            "\n     còn bịa thì họ tin luôn mà không có cách kiểm chứng."
        )
    else:
        chosen = tau
        print("\n  -> Không ngưỡng nào loại hết câu bịa. Cần cải thiện KB hoặc chunking.")

    print(f"\n  Ghi vào .env:  SIMILARITY_THRESHOLD={chosen:.2f}\n")


if __name__ == "__main__":
    main()

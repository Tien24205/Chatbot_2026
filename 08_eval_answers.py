"""
MILESTONE 6 — Đánh giá CÂU TRẢ LỜI (không chỉ truy hồi).

03_eval_retrieval.py đo tầng truy hồi: chunk đúng có nằm trong top-k không.
Script này đo tầng cuối: câu trả lời có đúng, có từ chối đúng lúc, có bịa không.

Đây là hai phép đo khác nhau. Truy hồi đúng 100% vẫn có thể ra câu trả lời sai,
vì LLM có thể suy diễn từ chunk đúng.

CẢNH BÁO HẠN MỨC: mỗi câu hỏi tốn ít nhất 2 lệnh gọi API (embed + sinh câu trả
lời), cộng 1 lệnh nữa nếu lớp 5 phải leo thang. Hạn mức free tier đo được chỉ
khoảng 20 request/model, nên mặc định chỉ chạy 6 câu. Dùng --all khi cần bản đầy đủ
và chấp nhận chờ hạn mức hồi.

  .venv\\Scripts\\python.exe 08_eval_answers.py              # 6 câu, cân đối nhóm
  .venv\\Scripts\\python.exe 08_eval_answers.py --limit 12
  .venv\\Scripts\\python.exe 08_eval_answers.py --all
  .venv\\Scripts\\python.exe 08_eval_answers.py --graph      # riêng nhóm câu so sánh chéo game
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from rag import config, pipeline, store
from rag.retry import RateLimited

QUESTIONS_FILE = config.PROJECT_DIR / "eval_questions.json"
RESULTS_FILE = config.PROJECT_DIR / "eval_answers_results.json"


def pick(questions: list[dict], limit: int) -> list[dict]:
    """
    Chọn mẫu CÂN ĐỐI giữa các nhóm, không cắt 6 câu đầu.

    Cắt đầu danh sách sẽ toàn câu in_scope của Liên Minh — chạy xong thấy 100%
    mà không hề chạm tới nhóm out_of_scope, tức là không đo được thứ đáng lo nhất.
    """
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(q["type"], []).append(q)

    out: list[dict] = []
    i = 0
    while len(out) < limit and any(len(g) > i for g in groups.values()):
        for name in sorted(groups):
            if len(groups[name]) > i and len(out) < limit:
                out.append(groups[name][i])
        i += 1
    return out


def judge(q: dict, ans) -> tuple[bool, str]:
    """So kết quả với kỳ vọng đã ghi trong bộ câu hỏi. Trả (đạt, lý do)."""
    sources = {h.source_name for h in ans.hits if f"{h.source_name} · {h.section}" in set(ans.citations)}

    if q["type"] == "out_of_scope":
        if ans.refused:
            return True, "từ chối đúng"
        return False, "PHẢI từ chối nhưng đã trả lời"

    if ans.refused:
        # Phân biệt hai lối từ chối — in "0.824 < τ" khi τ = 0.82 là sai toán và
        # đổ lỗi nhầm cho ngưỡng trong khi chính model nói không tìm thấy.
        if ans.refused_by == "model":
            return False, f"bỏ sót (model tự từ chối dù điểm {ans.top_similarity:.3f} ≥ τ)"
        return False, f"bỏ sót (điểm cao nhất {ans.top_similarity:.3f} < τ)"

    if q["type"] == "suy_luan":
        # Chấm theo NỘI DUNG chứ không chỉ theo nguồn: đáp án suy luận (con số
        # đếm được, tên rút ra) phải xuất hiện trong câu trả lời. So sánh không
        # phân biệt hoa thường; con số thì so nguyên chuỗi ("13" khớp "13").
        missing = [s for s in q["expect_contains"] if s.lower() not in ans.text.lower()]
        if missing:
            return False, f"thiếu đáp án {missing} trong câu trả lời"
        expect = set(q.get("expect_any_source", []))
        if expect and not (sources & expect):
            return False, f"đáp án đúng nhưng trích {sorted(sources)}, kỳ vọng {sorted(expect)}"
        return True, f"đủ {q['expect_contains']}, trích {', '.join(sorted(sources)) or '—'}"

    if q["type"] == "ambiguous":
        expect = set(q.get("expect_any_source", []))
        if not expect or sources & expect:
            return True, f"trích {', '.join(sorted(sources)) or '—'}"
        return False, f"trích {sorted(sources)}, kỳ vọng một trong {sorted(expect)}"

    expect = q.get("expect_source")
    if expect and expect not in sources:
        return False, f"trích {sorted(sources)}, kỳ vọng {expect}"
    return True, f"trích {', '.join(sorted(sources))}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6, help="số câu hỏi (mặc định 6, giữ hạn mức)")
    ap.add_argument("--all", action="store_true", help="chạy toàn bộ bộ câu hỏi")
    ap.add_argument("--graph", action="store_true", help="chỉ chạy nhóm câu so sánh")
    ap.add_argument("--reasoning", action="store_true", help="chỉ chạy nhóm câu suy luận")
    args = ap.parse_args()

    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.graph:
        pool = data["questions_graph"]
    elif args.reasoning:
        pool = data["questions_reasoning"]
    else:
        pool = data["questions"]
    questions = pool if (args.all or args.graph or args.reasoning) else pick(pool, args.limit)

    print("=" * 74)
    print("MILESTONE 6 — ĐÁNH GIÁ CÂU TRẢ LỜI")
    print("=" * 74)
    print(f"\n  model            : {config.CHAT_MODEL}")
    print(f"  ngưỡng τ         : {config.SIMILARITY_THRESHOLD}")
    print(f"  entailment (lớp 5): {config.ENTAILMENT_CHECK}")
    print(f"  mở rộng đồ thị   : {'bật' if config.GRAPH_EXPANSION else 'tắt'}")
    print(f"  số câu           : {len(questions)}/{len(pool)}")
    print(f"\n  Ước tính {len(questions) * 2}-{len(questions) * 3} lệnh gọi API.\n")

    results: list[dict] = []
    stopped_early = ""

    with store.connect_or_exit(require_chunks=True) as conn:
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q['q']}")
            t0 = time.perf_counter()
            try:
                ans = pipeline.ask(conn, q["q"])
            except RateLimited as e:
                stopped_early = f"chạm hạn mức sau {i - 1} câu: {e}"
                print(f"\n  [DỪNG] {stopped_early}\n")
                break
            except Exception as e:
                print(f"      LỖI: {e}")
                results.append({**q, "ok": False, "reason": f"lỗi: {e}"})
                continue

            ok, reason = judge(q, ans)
            wall = int((time.perf_counter() - t0) * 1000)

            mark = "ĐẠT " if ok else "HỎNG"
            print(f"      [{mark}] {reason}  ({wall} ms)")
            if ans.graph_expanded:
                print(f"             đồ thị bổ sung {ans.graph_expanded} đoạn")
            if ans.doc_expanded:
                print(f"             nạp trọn trang +{ans.doc_expanded} đoạn")
            for f in ans.flags:
                print(f"             ⚠ lớp 3-4: {f}")
            for u in ans.unsupported:
                print(f"             ⚠ lớp 5: {u}")

            results.append(
                {
                    "q": q["q"],
                    "type": q["type"],
                    "ok": ok,
                    "reason": reason,
                    "refused": ans.refused,
                    "top_similarity": round(ans.top_similarity, 4),
                    "flags": ans.flags,
                    "unsupported": ans.unsupported,
                    "entailment_ran": ans.entailment_ran,
                    "graph_expanded": ans.graph_expanded,
                    "doc_expanded": ans.doc_expanded,
                    "rewrite_ms": ans.rewrite_ms,
                    "retrieval_ms": ans.retrieval_ms,
                    "llm_ms": ans.llm_ms,
                    "verify_ms": ans.verify_ms,
                    "total_ms": ans.total_ms,
                    "answer": ans.text,
                }
            )

    if not results:
        print("\n  Không có kết quả nào (hết hạn mức ngay từ câu đầu).\n")
        sys.exit(1)

    # --- Tổng hợp ---------------------------------------------------------
    print("\n" + "=" * 74)
    print("TỔNG HỢP")
    print("=" * 74)

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    print(f"\n  {'Nhóm':<16}{'Đạt':>8}{'Tổng':>8}")
    for t in sorted(by_type):
        rs = by_type[t]
        print(f"  {t:<16}{sum(r['ok'] for r in rs):>8}{len(rs):>8}")
    print(f"  {'TẤT CẢ':<16}{sum(r['ok'] for r in results):>8}{len(results):>8}")

    # --- Hallucination ----------------------------------------------------
    answered = [r for r in results if not r["refused"]]
    flagged = [r for r in answered if r["flags"] or r["unsupported"]]
    fabricated = [
        r for r in results if r["type"] == "out_of_scope" and not r["refused"]
    ]

    print("\n" + "-" * 74)
    print("HALLUCINATION")
    print("-" * 74)
    print(f"  Câu ngoài phạm vi bị trả lời (bịa thẳng) : {len(fabricated)}")
    print(f"  Câu có cờ kiểm chứng                      : {len(flagged)}/{len(answered)}")
    for r in flagged:
        print(f"    - {r['q']}")
        for f in r["flags"] + r["unsupported"]:
            print(f"        {f}")
    if not flagged:
        print("    (không câu nào bị gắn cờ)")

    # --- Latency ----------------------------------------------------------
    if answered:
        print("\n" + "-" * 74)
        print("LATENCY (chỉ tính câu thực sự gọi LLM)")
        print("-" * 74)
        for label, key in [
            ("truy hồi", "retrieval_ms"),
            ("sinh câu trả lời", "llm_ms"),
            ("kiểm chứng lớp 5", "verify_ms"),
            ("tổng", "total_ms"),
        ]:
            vals = [r[key] for r in answered]
            if not any(vals):
                continue
            print(
                f"  {label:<20} trung vị {statistics.median(vals):>6.0f} ms"
                f"   thấp nhất {min(vals):>6} ms   cao nhất {max(vals):>6} ms"
            )

        n_entail = sum(r["entailment_ran"] for r in answered)
        n_graph = sum(bool(r["graph_expanded"]) for r in answered)
        print(f"\n  Lớp 5 chạy    : {n_entail}/{len(answered)} câu (chế độ '{config.ENTAILMENT_CHECK}')")
        print(f"  Đồ thị bổ sung: {n_graph}/{len(answered)} câu")

    RESULTS_FILE.write_text(
        json.dumps(
            {
                "model": config.CHAT_MODEL,
                "threshold": config.SIMILARITY_THRESHOLD,
                "entailment_check": config.ENTAILMENT_CHECK,
                "graph_expansion": config.GRAPH_EXPANSION,
                "stopped_early": stopped_early,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Chi tiết đã ghi vào {RESULTS_FILE.name}")

    if stopped_early:
        print(f"\n  LƯU Ý: {stopped_early}")
        print("  Con số trên chỉ tính trên phần đã chạy được.")
    print()


if __name__ == "__main__":
    main()

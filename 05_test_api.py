"""
Kiểm thử API bằng HTTP thật (không dùng TestClient) — xác nhận server đang chạy
phục vụ đúng, kể cả phần mã hoá UTF-8 cho tiếng Việt.

Yêu cầu server đang chạy:
  .venv\\Scripts\\python.exe -m uvicorn api:app --port 8000

Chạy:  .venv\\Scripts\\python.exe 05_test_api.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
SESSION = "test-session"

failures: list[str] = []


def call(path: str, payload: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Chi tiết lỗi nằm trong body, không nằm trong status line. Không đọc body
        # thì chỉ thấy "HTTP Error 500" và không biết vì sao.
        if e.code >= 500:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} tại {path}: {body[:600]}") from None
        raise


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        if detail:
            print(f"         {detail}")
        failures.append(label)


def main() -> None:
    print("=" * 70)
    print("KIỂM THỬ API")
    print("=" * 70)

    try:
        health = call("/api/health")
    except urllib.error.URLError as e:
        print(f"\n[THẤT BẠI] Không gọi được {BASE}/api/health — server chưa chạy?\n  {e}")
        sys.exit(1)

    print(f"\n  health: {health}\n")
    check(health.get("status") == "ok", "GET /api/health trả về ok")
    check(health.get("chunks", 0) > 0, "Database có dữ liệu")

    call(f"/api/chat/{SESSION}", method="DELETE")  # bắt đầu từ hội thoại sạch

    # --- 1. Câu trong phạm vi ------------------------------------------------
    print("\n  --- Câu trong phạm vi ---")
    r = call("/api/chat", {"session_id": SESSION, "message": "Baron Nashor là gì?"}, "POST")
    print(f"    trả lời: {r['answer'][:90]}...")
    print(f"    nguồn  : {[c['source_name'] for c in r['citations']]}")
    check(not r["refused"], "Không bị từ chối")
    check("Baron" in r["answer"], "Câu trả lời nhắc tới Baron")
    check(
        any(c["source_name"] == "lien_minh_huyen_thoai.txt" for c in r["citations"]),
        "Trích đúng nguồn Liên Minh Huyền Thoại",
    )
    check(len(r["citations"]) <= 2, f"Không trích thừa nguồn (có {len(r['citations'])})")

    # --- 2. Câu nối tiếp — kiểm tra query rewriting -------------------------
    print("\n  --- Câu nối tiếp ---")
    r2 = call("/api/chat", {"session_id": SESSION, "message": "Còn Liên Quân thì sao?"}, "POST")
    print(f"    viết lại: {r2['search_query']}")
    print(f"    trả lời : {r2['answer'][:90]}...")
    check(r2["rewrote_query"], "Câu hỏi phụ thuộc ngữ cảnh đã được viết lại")
    check(not r2["refused"], "Trả lời được câu nối tiếp")
    check(
        any(c["source_name"] == "lien_quan_mobile.txt" for c in r2["citations"]),
        "Truy hồi đúng sang tài liệu Liên Quân",
    )

    # --- 3. Câu ngoài phạm vi -> phải từ chối --------------------------------
    print("\n  --- Câu ngoài phạm vi ---")
    r3 = call("/api/chat", {"session_id": SESSION, "message": "Cách nấu phở bò Hà Nội"}, "POST")
    print(f"    top_similarity: {r3['top_similarity']}")
    check(r3["refused"], "Từ chối câu ngoài phạm vi")
    check(r3["llm_ms"] == 0, "Không gọi LLM khi đã bị ngưỡng chặn (tiết kiệm chi phí)")
    check(r3["citations"] == [], "Không trích nguồn nào khi từ chối")

    # --- 4. Câu đúng chủ đề nhưng KB không có câu trả lời -------------------
    print("\n  --- Đúng chủ đề nhưng không có dữ liệu ---")
    r4 = call(
        "/api/chat",
        {"session_id": SESSION, "message": "Ai là nhân vật mạnh nhất trong game?"},
        "POST",
    )
    print(f"    top_similarity: {r4['top_similarity']}  refused={r4['refused']}")
    check(
        r4["refused"] or "không tìm thấy" in r4["answer"].lower(),
        "Không bịa xếp hạng sức mạnh (KB không có thông tin này)",
        f"trả lời: {r4['answer'][:120]}",
    )

    # --- 5. Kiểm tra tính hợp lệ đầu vào ------------------------------------
    print("\n  --- Kiểm tra đầu vào ---")
    try:
        call("/api/chat", {"session_id": "x", "message": ""}, "POST")
        check(False, "Từ chối tin nhắn rỗng")
    except urllib.error.HTTPError as e:
        check(e.code == 422, "Từ chối tin nhắn rỗng (HTTP 422)", f"nhận được {e.code}")

    # --- 6. Xoá lịch sử ------------------------------------------------------
    print("\n  --- Xoá hội thoại ---")
    d = call(f"/api/chat/{SESSION}", method="DELETE")
    check(d.get("status") == "ok", "DELETE /api/chat/{session_id} hoạt động")

    print("\n" + "=" * 70)
    if failures:
        print(f"CÓ {len(failures)} KIỂM TRA THẤT BẠI:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)
    print("TẤT CẢ KIỂM TRA API ĐỀU PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()

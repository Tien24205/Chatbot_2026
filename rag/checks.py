"""
Bộ khung kiểm thử dùng chung cho 01_test_chunking.py, 05_test_api.py và
07_test_graph_verify.py.

Ba script đó từng chép nguyên si cùng một hàm check(), cùng một danh sách
`failures` và cùng một khối tổng kết + sys.exit — ba bản sao phải sửa cùng lúc
mỗi khi đổi cách in.

Cố tình KHÔNG dùng pytest: các script được đánh số 00→08 theo đúng thứ tự trong
chatbot_rag_plan.md và mọi tài liệu đều tham chiếu tới tên file, nên chúng phải
chạy được bằng `python 01_test_chunking.py` như mọi bước khác.
"""

from __future__ import annotations

import sys

_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    """
    Ghi nhận một phép kiểm tra và in kết quả ngay.

    In ngay thay vì gom lại cuối cùng: khi một phép kiểm tra treo (gọi API, gọi
    database) thì dòng PASS cuối cùng in ra chính là chỗ đang treo.

    Trả về chính `ok` để bên gọi rẽ nhánh được nếu cần.
    """
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        if detail:
            print(f"         {detail}")
        _failures.append(label)
    return ok


def report(width: int = 66, success_note: str = "") -> None:
    """
    In tổng kết rồi thoát với mã 1 nếu có phép kiểm tra nào hỏng.

    Mã thoát khác 0 là thứ bắt buộc: không có nó thì một script test hỏng vẫn
    được CI (và cả người chạy tay nối lệnh bằng `&&`) coi là chạy thành công.
    """
    print("\n" + "=" * width)
    if _failures:
        print(f"CÓ {len(_failures)} KIỂM TRA THẤT BẠI:")
        for f in _failures:
            print(f"  - {f}")
        print("=" * width)
        sys.exit(1)

    print("TẤT CẢ KIỂM TRA ĐỀU PASS")
    print("=" * width)
    if success_note:
        print(success_note)

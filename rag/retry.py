"""
Thử lại khi bị giới hạn tần suất (429).

Free tier của Gemini giới hạn số request mỗi phút (đo được: 20 RPM với
gemini-3.6-flash). Một lượt hỏi nối tiếp tốn tới 3 lệnh gọi — viết lại câu hỏi,
embed, sinh câu trả lời — nên chạm trần rất dễ.

Google trả kèm thời gian chờ đề nghị ngay trong thông báo lỗi ("Please retry in
42.2s"). Đọc đúng con số đó tốt hơn nhiều so với backoff đoán mò.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRY_AFTER = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
MAX_ATTEMPTS = 3
MAX_WAIT_SECONDS = 65.0


class RateLimited(RuntimeError):
    """Hết lượt thử mà vẫn bị giới hạn tần suất."""


def is_rate_limit(e: Exception) -> bool:
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def suggested_wait(e: Exception, default: float = 20.0) -> float:
    m = _RETRY_AFTER.search(str(e))
    wait = float(m.group(1)) + 1.0 if m else default
    return min(wait, MAX_WAIT_SECONDS)


def with_retry(
    fn: Callable[[], T],
    on_wait: Callable[[float, int], None] | None = None,
    attempts: int = MAX_ATTEMPTS,
) -> T:
    """
    Chạy fn(), thử lại khi gặp 429. Lỗi khác thì ném ra ngay.

    `attempts` để bên gọi tự chọn độ kiên nhẫn, vì hai tình huống khác hẳn nhau:

      - Lượt chat: người dùng đang ngồi chờ, thử 3 lần rồi báo lỗi là đúng.
      - Nạp knowledge base: chạy hàng chục phút, không ai ngồi nhìn. Bỏ cuộc sớm
        làm hỏng cả tác vụ dài. Đo được thật: nạp 2.848 chunk chết ở chunk 400
        chỉ vì 3 lần thử không đủ vượt qua một đợt giới hạn theo phút.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if not is_rate_limit(e):
                raise
            last = e
            if attempt == attempts:
                break
            wait = suggested_wait(e)
            if on_wait:
                on_wait(wait, attempt)
            time.sleep(wait)

    raise RateLimited(
        "Đã chạm giới hạn tần suất của Gemini (free tier ~20 request/phút) "
        f"sau {attempts} lần thử. Chờ khoảng một phút rồi hỏi lại."
    ) from last

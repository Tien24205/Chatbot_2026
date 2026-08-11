"""
Client Gemini dùng chung.

embedder.py và pipeline.py trước đây giữ MỖI FILE MỘT singleton riêng với cùng
một đoạn mã. Hệ quả không chỉ là mã chép đôi: một tiến trình tạo hai đối tượng
Client và hai pool HTTP cho cùng một API key. Gom về đây thì vừa bỏ được đoạn
lặp, vừa còn đúng một client thật.
"""

from __future__ import annotations

from google import genai

from . import config

_client: genai.Client | None = None


def client() -> genai.Client:
    """Tạo ở lần gọi đầu, sau đó dùng lại. Thiếu API key thì dừng ngay tại đây."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_api_key())
    return _client

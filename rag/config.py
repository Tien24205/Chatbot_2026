"""
Cấu hình tập trung — đọc từ .env, neo theo vị trí file.

Mọi giá trị ở đây đều đã được 00_smoke_test.py xác minh bằng lệnh gọi thật.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_DIR / "knowledge_base"

load_dotenv(PROJECT_DIR / ".env")

def env(name: str, default: str) -> str:
    """
    Đọc biến môi trường, coi CHUỖI RỖNG là chưa đặt.

    File .env sinh từ .env.example có sẵn các dòng `KEY=` để trống. Với những
    dòng đó os.getenv trả về "" chứ không phải None, nên tham số default của
    os.getenv không bao giờ được dùng — int("") sẽ ném ValueError.
    """
    return (os.getenv(name) or "").strip() or default


GEMINI_API_KEY = env("GEMINI_API_KEY", "")
# Đo được trên tài khoản free tier:
#   gemini-flash-latest       -> 20 request rồi hết hạn mức
#   gemini-2.0-flash(-lite)   -> limit 0, free tier không được dùng
#   gemini-flash-lite-latest  -> còn dùng được
# Nên đặt bản lite làm mặc định, giữ bản đầy đủ làm dự phòng khi lite hết hạn mức.
CHAT_MODEL = env("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest")
CHAT_MODEL_FALLBACK = env("GEMINI_CHAT_MODEL_FALLBACK", "gemini-flash-latest")
EMBED_MODEL = env("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMENSION = int(env("EMBED_DIMENSION", "1536"))

# Đo bằng 03_eval_retrieval.py và 08_eval_answers.py, KHÔNG chép từ tutorial.
# Dưới ngưỡng này chatbot từ chối ngay, không gọi LLM.
#
# 0.66 -> 0.60 sau Milestone 6. Lý do là số đo, không phải cảm tính: khi ngưỡng
# là lớp phòng thủ DUY NHẤT thì phải đặt cao (0.66) và trả giá bằng việc loại oan
# câu trả lời được — đo được 2 câu (0.617 và 0.631) bị từ chối oan. Khi đã có đủ
# 5 lớp, hạ xuống 0.60 lấy lại được 2 câu đó mà vẫn 0/6 câu ngoài phạm vi bị bịa:
# câu "Ai là nhân vật mạnh nhất" (0.657) lọt qua ngưỡng nhưng bị lớp 1 chặn.
# Chạy lại hai script trên mỗi khi đổi model embedding hoặc sửa knowledge base.
SIMILARITY_THRESHOLD = float(env("SIMILARITY_THRESHOLD", "0.60"))

DATABASE_URL = env(
    "DATABASE_URL", "postgresql://postgres:postgrespassword@localhost:5433/chatbot_rag"
)

# --- Chống hallucination lớp 5: vòng kiểm chứng entailment ------------------
# Tốn thêm MỘT lệnh gọi API cho mỗi lần chạy, mà hạn mức free tier đo được chỉ
# 20 request — nên mặc định là "auto": chỉ chạy khi lớp 3-4 (tất định, miễn phí)
# đã thấy dấu hiệu khả nghi.
#   auto | on | off
ENTAILMENT_CHECK = env("ENTAILMENT_CHECK", "auto").lower()

# --- GraphRAG: mở rộng truy hồi bằng quan hệ giữa các thực thể --------------
# Chỉ kích hoạt với câu hỏi so sánh / nối tiếp. Câu hỏi đơn đã đạt recall@1 =
# 100% bằng vector, thêm chunk vào đó chỉ làm loãng ngữ cảnh.
GRAPH_EXPANSION = env("GRAPH_EXPANSION", "on").lower() != "off"
GRAPH_MAX_EXTRA_CHUNKS = int(env("GRAPH_MAX_EXTRA_CHUNKS", "2"))


def require_api_key() -> str:
    if not GEMINI_API_KEY:
        print(
            f"\n[THẤT BẠI] Chưa có GEMINI_API_KEY trong {PROJECT_DIR / '.env'}",
            file=sys.stderr,
        )
        sys.exit(1)
    return GEMINI_API_KEY

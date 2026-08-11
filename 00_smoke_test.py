"""
BƯỚC 0 — Smoke test Gemini API.

Mục đích: trả lời 3 câu hỏi TRƯỚC KHI viết bất kỳ dòng code RAG nào.
  1. Key có hoạt động không?
  2. Key này truy cập được model nào (tên thật, không phải tên đoán)?
  3. Vector embedding có bao nhiêu chiều? -> con số này khoá schema database ở Bước 5.

Chạy:  python 00_smoke_test.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"

# Ứng viên model, thử theo thứ tự. Script chỉ dùng tên nào THỰC SỰ có trong
# danh sách API trả về, nên danh sách này lỗi thời cũng không sao.
CHAT_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
]

# Model có trong models.list() KHÔNG đảm bảo gọi được: Google chặn model cũ với
# tài khoản mới và chỉ báo 404 lúc gọi thật. Loại sẵn các model không phải chat.
NOT_CHAT = ("tts", "image", "imagen", "veo", "embedding", "aqa", "audio", "live", "robotics")
EMBED_CANDIDATES = [
    "gemini-embedding-001",
    "text-embedding-004",
    "embedding-001",
]


def die(msg: str) -> None:
    print(f"\n[THẤT BẠI] {msg}")
    sys.exit(1)


def require_deps():
    """
    Nạp thư viện, báo lỗi rõ ràng nếu chạy nhầm trình thông dịch.

    Thư viện được cài trong venv của thư mục này, không phải Python toàn cục.
    Chạy `python 00_smoke_test.py` bằng Python hệ thống sẽ thiếu module.
    """
    try:
        from dotenv import load_dotenv
        from google import genai
    except ImportError as e:
        hint = (
            f'"{VENV_PY}" "{Path(__file__).resolve()}"'
            if VENV_PY.exists()
            else f'python -m venv .venv && .venv\\Scripts\\python.exe -m pip install -r "{HERE / "requirements.txt"}"'
        )
        die(
            f"Thiếu thư viện ({e.name}).\n\n"
            f"  Bạn đang chạy: {sys.executable}\n"
            f"  Thư viện nằm trong venv của Chatbox, không phải Python toàn cục.\n\n"
            f"  Chạy lại bằng lệnh này:\n"
            f"    {hint}"
        )
    return load_dotenv, genai


def short(name: str) -> str:
    """'models/gemini-2.5-flash' -> 'gemini-2.5-flash'"""
    return name.split("/", 1)[-1]


def main() -> None:
    load_dotenv, genai = require_deps()

    # Neo theo vị trí FILE, không theo thư mục đang đứng: nếu chạy từ thư mục cha
    # (ví dụ FPT/) thì load_dotenv() mặc định sẽ không tìm thấy Chatbox/.env.
    env_path = HERE / ".env"
    if not env_path.exists():
        die(
            f"Không tìm thấy {env_path}\n"
            f"  1. Copy .env.example thành .env trong {HERE}\n"
            "  2. Lấy key tại https://aistudio.google.com/apikey\n"
            "  3. Dán vào dòng GEMINI_API_KEY="
        )
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        die(
            f"File {env_path} có tồn tại nhưng dòng GEMINI_API_KEY còn trống.\n"
            "  Lấy key tại https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    # --- 1. Model nào thực sự dùng được? -------------------------------------
    print("=" * 62)
    print("1. LIỆT KÊ MODEL KEY NÀY TRUY CẬP ĐƯỢC")
    print("=" * 62)

    try:
        models = list(client.models.list())
    except Exception as e:
        die(f"Không gọi được API. Key sai hoặc mất mạng.\n  Chi tiết: {e}")

    chat_models, embed_models = [], []
    for m in models:
        actions = getattr(m, "supported_actions", None) or []
        name = short(m.name)
        if "embedContent" in actions:
            embed_models.append(name)
        if "generateContent" in actions:
            chat_models.append(name)

    print(f"\n  Chat   ({len(chat_models)} model):")
    for n in chat_models[:8]:
        print(f"    - {n}")
    if len(chat_models) > 8:
        print(f"    ... và {len(chat_models) - 8} model khác")

    print(f"\n  Embedding ({len(embed_models)} model):")
    for n in embed_models:
        print(f"    - {n}")

    if not chat_models:
        die("Key không truy cập được model chat nào.")
    if not embed_models:
        die("Key không truy cập được model embedding nào.")

    def pick(env_var: str, candidates: list[str], available: list[str]) -> str:
        chosen = os.getenv(env_var, "").strip()
        if chosen:
            if chosen not in available:
                die(f"{env_var}={chosen} không có trong danh sách khả dụng ở trên.")
            return chosen
        for c in candidates:
            if c in available:
                return c
        return available[0]

    embed_model = pick("GEMINI_EMBED_MODEL", EMBED_CANDIDATES, embed_models)
    print(f"\n  -> Sẽ dùng embedding model: {embed_model}")

    # --- 2. Embedding: vector có bao nhiêu chiều? ----------------------------
    print("\n" + "=" * 62)
    print("2. TEST EMBEDDING (đo số chiều vector thật)")
    print("=" * 62)

    sample = "Baron Nashor là quái trung lập mạnh nhất trong Liên Minh Huyền Thoại."
    try:
        resp = client.models.embed_content(model=embed_model, contents=sample)
        vector = resp.embeddings[0].values
    except Exception as e:
        die(f"Gọi embedding thất bại.\n  Chi tiết: {e}")

    dimension = len(vector)
    print(f"\n  Câu thử : {sample}")
    print(f"  SỐ CHIỀU: {dimension}")
    print(f"  5 giá trị đầu: {[round(v, 5) for v in vector[:5]]}")

    # Kiểm tra tiếng Việt có thực sự được hiểu về mặt ngữ nghĩa hay không.
    print("\n  Kiểm tra ngữ nghĩa tiếng Việt (cosine similarity):")
    pairs = [
        ("Cách ăn Rồng Bạo Chúa trong Liên Quân", "Mục tiêu trung lập của game MOBA", "nên CAO"),
        ("Cách ăn Rồng Bạo Chúa trong Liên Quân", "Công thức nấu phở bò Hà Nội", "nên THẤP"),
    ]
    for a, b, expect in pairs:
        va = client.models.embed_content(model=embed_model, contents=a).embeddings[0].values
        vb = client.models.embed_content(model=embed_model, contents=b).embeddings[0].values
        dot = sum(x * y for x, y in zip(va, vb, strict=True))
        na = sum(x * x for x in va) ** 0.5
        nb = sum(x * x for x in vb) ** 0.5
        print(f"    {dot / (na * nb):.4f}  ({expect})  {a[:38]}... <-> {b[:38]}...")

    # --- 2b. Số chiều có vừa giới hạn index của pgvector không? --------------
    PGVECTOR_INDEX_LIMIT = 2000  # giới hạn của index hnsw / ivfflat trên kiểu `vector`
    chosen_dim = dimension

    if dimension > PGVECTOR_INDEX_LIMIT:
        print(f"\n  [CẢNH BÁO] {dimension} chiều > giới hạn index {PGVECTOR_INDEX_LIMIT} của pgvector.")
        print("  `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` sẽ LỖI.")
        print("  Thử rút gọn số chiều bằng output_dimensionality (Matryoshka):\n")

        from google.genai import types

        for d in (1536, 768):
            try:
                r = client.models.embed_content(
                    model=embed_model,
                    contents=sample,
                    config=types.EmbedContentConfig(output_dimensionality=d),
                )
                v = r.embeddings[0].values
                norm = sum(x * x for x in v) ** 0.5
                print(f"    {d:>4} chiều: OK (norm = {norm:.4f})")
                if chosen_dim > PGVECTOR_INDEX_LIMIT:
                    chosen_dim = d
            except Exception as e:
                print(f"    {d:>4} chiều: KHÔNG hỗ trợ ({str(e).splitlines()[0][:50]})")

        print(f"\n  -> Chọn {chosen_dim} chiều để vừa index pgvector.")
        print("     Lưu ý: vector rút gọn KHÔNG còn chuẩn hoá, phải tự chia cho norm")
        print("     trước khi lưu, nếu không cosine similarity sẽ sai.")

    # --- 3. Chat: model trả lời tiếng Việt được không? -----------------------
    print("\n" + "=" * 62)
    print("3. TEST CHAT (kiểm tra khả năng trả lời tiếng Việt)")
    print("=" * 62)

    prompt = "Giải thích 'gank' trong game MOBA bằng đúng 2 câu tiếng Việt."

    # DÒ THẬT, không tin danh sách: models.list() vẫn liệt kê những model mà tài
    # khoản mới bị chặn (404 "no longer available to new users"). Gọi thử tới khi
    # có model chạy được.
    forced = os.getenv("GEMINI_CHAT_MODEL", "").strip()
    if forced:
        order = [forced]
    else:
        usable = [m for m in chat_models if not any(k in m for k in NOT_CHAT)]
        preferred = [m for m in CHAT_CANDIDATES if m in usable]
        rest = [m for m in usable if m not in preferred]
        # Ưu tiên bản chính thức (không 'preview'/'exp') và bản 'flash' cho rẻ/nhanh.
        rest.sort(key=lambda m: ("preview" in m or "exp" in m, "flash" not in m, m))
        order = preferred + rest

    chat_model, answer, tried = None, None, []
    for candidate in order[:12]:
        try:
            answer = client.models.generate_content(model=candidate, contents=prompt).text
            chat_model = candidate
            break
        except Exception as e:
            tried.append(f"{candidate}: {str(e).splitlines()[0][:70]}")

    if not chat_model:
        die("Không model chat nào gọi được.\n  Đã thử:\n    " + "\n    ".join(tried))

    if tried:
        print(f"\n  (Bỏ qua {len(tried)} model bị chặn: {', '.join(t.split(':')[0] for t in tried)})")
    print(f"\n  Model dùng được: {chat_model}")
    print(f"  Câu hỏi : {prompt}")
    print(f"  Trả lời : {answer.strip()}")

    # --- Kết luận ------------------------------------------------------------
    print("\n" + "=" * 62)
    print("THÀNH CÔNG — chép 3 dòng này vào file .env của bạn:")
    print("=" * 62)
    print(f"\nGEMINI_CHAT_MODEL={chat_model}")
    print(f"GEMINI_EMBED_MODEL={embed_model}")
    print(f"EMBED_DIMENSION={chosen_dim}")
    print(f"\nSố {chosen_dim} sẽ đi vào `vector({chosen_dim})` khi tạo bảng ở Bước 5.")
    if chosen_dim != dimension:
        print(f"(Model trả về {dimension} chiều, rút gọn xuống {chosen_dim} để vừa index pgvector.)")
    print()


if __name__ == "__main__":
    main()

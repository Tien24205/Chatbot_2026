# Chatbot RAG với PostgreSQL + pgvector

Chatbot hỏi đáp tiếng Việt trên knowledge base riêng, xây từ đầu bằng Python
thuần — **không dùng LangChain hay LlamaIndex** — để mỗi tầng đều nhìn thấy được
và đo được.

Điểm khác biệt so với một RAG demo thông thường:

- **Năm lớp chống hallucination**, mỗi lớp chặn một loại lỗi khác nhau, ba trong
  số đó chạy tất định và không tốn lệnh gọi API.
- **Tầng đồ thị tri thức (GraphRAG)** đặt cạnh tầng vector, xử lý được câu hỏi so
  sánh chéo tài liệu mà similarity search về bản chất không làm được.
- **Mọi hằng số đều đo mà ra**, không chép từ tutorial — kèm số liệu và lý do
  trong tài liệu.

Chi tiết kỹ thuật: [CHONG_HALLUCINATION_VA_GRAPH.md](CHONG_HALLUCINATION_VA_GRAPH.md)
Nhật ký thiết kế và các bẫy đã vấp: [chatbot_rag_plan.md](chatbot_rag_plan.md)

---

## Kiến trúc

```text
                     KNOWLEDGE BASE (10 file .txt)
                               │
        ┌──────────────────────┴──────────────────────┐
        ↓                                             ↓
   Chunk theo tiêu đề `##`                    Trích thực thể (regex)
        ↓                                             ↓
   Embedding (Gemini)                          Đồ thị: 189 thực thể
        ↓                                      2352 cạnh, 3 loại quan hệ
   PostgreSQL + pgvector                              │
   (HNSW, cosine, 1536 chiều)                         │
        │                                             │
        └──────────────────┬──────────────────────────┘
                           ↓
Câu hỏi ──→ (viết lại nếu là câu nối tiếp) ──→ Similarity search
                           ↓
                  LỚP 2: điểm < τ ? ──→ TỪ CHỐI, không gọi LLM
                           ↓
              câu so sánh ? ──→ đồ thị bổ sung đoạn + quan hệ
                           ↓
              LỚP 1: system prompt + ngữ cảnh ──→ Gemini
                           ↓
              LỚP 3-4: soi trích dẫn, số liệu, tên riêng (miễn phí)
                           ↓
              có cờ ? ──→ LỚP 5: vòng entailment bằng LLM
                           ↓
              Câu trả lời + trích nguồn + kết quả kiểm chứng
```

## Năm lớp chống hallucination

| # | Lớp | Chặn loại lỗi | Chi phí |
|---|---|---|---|
| 1 | System prompt ràng buộc phạm vi | LLM dùng kiến thức nền | 0 |
| 2 | Ngưỡng similarity hiệu chuẩn bằng đo đạc | Không đoạn nào đủ liên quan | 0 — còn **tiết kiệm** 1 lệnh gọi |
| 3 | Trích dẫn có cấu trúc `[n]`, kiểm tra chỉ số | Bịa nguồn | 0 |
| 4 | Kiểm chứng grounding tất định | Bịa số liệu, bịa tên riêng | 0 |
| 5 | Vòng kiểm chứng entailment bằng LLM | Suy diễn sai từ ngữ cảnh đúng | 1 lệnh gọi, leo thang có điều kiện |

Lớp 5 mặc định chỉ chạy khi lớp 3-4 đã thấy dấu hiệu khả nghi: thứ regex bắt được
thì rẻ, thứ nó không bắt được mới đáng tiêu hạn mức API.

## Kết quả đo được

> ⚠ Các số ở mục này đo trên knowledge base **4 tài liệu / 25 chunk**. Kho hiện đã
> mở rộng lên **10 tài liệu / 65 chunk**, và `eval_questions.json` chưa phủ 6 tài
> liệu mới. Phải chạy lại `03_eval_retrieval.py` và `08_eval_answers.py` rồi mới
> được trích dẫn con số ở đây như kết quả hiện tại.

Bộ câu hỏi so sánh chéo game (8 câu, τ = 0.60):

| Nhóm | Đạt | Tổng |
|---|---|---|
| ambiguous (so sánh chéo game) | 4 | 4 |
| in_scope | 3 | 3 |
| out_of_scope | 1 | 1 |
| **Tất cả** | **8** | **8** |

Câu ngoài phạm vi bị bịa: **0**. Câu bị gắn cờ kiểm chứng: **0/7**.
Latency trung vị: truy hồi 413 ms · sinh câu trả lời 1269 ms · **tổng 1718 ms**.

Truy hồi (25 chunk): **recall@1 = 100%** cả về đúng file lẫn đúng mục.

**Phát hiện đáng nói nhất: ngưỡng similarity là hàm của số lớp phòng thủ, không
phải hằng số của model.**

| | τ = 0.66 | τ = 0.60 |
|---|---|---|
| Câu trả lời được nhưng bị từ chối oan | **2** | **0** |
| Câu ngoài phạm vi bị bịa | 0 | **0** |

Khi ngưỡng phải gánh một mình, nó buộc phải đặt cao và trả giá bằng recall. Có đủ
năm lớp phía sau thì hạ được ngưỡng mà không mất an toàn.

> Để so sánh: ngưỡng **0.45** hay gặp trong tutorial sẽ để lọt **6/6** câu ngoài
> phạm vi — chatbot trả lời cả câu hỏi về cách nấu phở bằng tài liệu game. Nền
> tương đồng của `gemini-embedding-001` rất cao: hai câu hoàn toàn không liên quan
> vẫn đạt 0.513.

## Cài đặt

Cần: Python 3.11+, Docker Desktop, và một API key Gemini
(miễn phí tại <https://aistudio.google.com/apikey>).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

copy .env.example .env      # rồi điền GEMINI_API_KEY vào .env
docker compose up -d        # PostgreSQL + pgvector, cổng 5433
```

## Chạy

Theo đúng thứ tự — mỗi bước kiểm chứng bước trước:

```powershell
.venv\Scripts\python.exe 00_smoke_test.py        # xác nhận key, tên model, SỐ CHIỀU THẬT
.venv\Scripts\python.exe 01_test_chunking.py     # test loader + chunking (offline)
.venv\Scripts\python.exe 02_ingest.py            # embed + nạp vào pgvector
.venv\Scripts\python.exe 03_eval_retrieval.py    # đo recall, hiệu chuẩn ngưỡng τ
.venv\Scripts\python.exe 06_build_graph.py       # dựng đồ thị (offline, không tốn API)
.venv\Scripts\python.exe 07_test_graph_verify.py # 41 kiểm thử đồ thị + kiểm chứng
.venv\Scripts\python.exe 08_eval_answers.py      # đánh giá câu trả lời đầu-cuối

.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Rồi mở <http://localhost:8501>.

Hai giao diện khác cho cùng một pipeline:

```powershell
.venv\Scripts\python.exe 04_chat.py                               # dòng lệnh
.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000  # REST API thuần
```

`streamlit_app.py` gọi thẳng `rag.pipeline`, không cần uvicorn chạy kèm. `api.py`
chỉ còn là REST API (tài liệu tự sinh ở `/docs`), không phục vụ giao diện nữa.

## Cấu trúc

```text
rag/
├── config.py     Đọc .env, neo mọi đường dẫn theo vị trí file
├── loader.py     Đọc + làm sạch (NFC, LF, BOM, zero-width)
├── chunker.py    Cắt theo tiêu đề `##` trước, kích thước sau
├── embedder.py   Gemini embedding — CHUẨN HOÁ LẠI vector + tách task_type
├── store.py      pgvector: lưu, tìm, kiểm tra khớp số chiều
├── graph.py      Đồ thị tri thức: trích thực thể, cạnh, đi đa bước
├── prompt.py     System prompt (8 quy tắc) + ghép ngữ cảnh + trích dẫn
├── verify.py     Lớp 3-4: kiểm chứng tất định, không gọi API
├── retry.py      Thử lại khi 429, đọc đúng thời gian chờ Google đề nghị
├── pipeline.py   Ghép toàn tuyến: truy hồi → ngưỡng → LLM → kiểm chứng
│
│  Dùng chung, gom về đây để không còn bản chép:
├── gemini.py     Một client Gemini duy nhất cho cả embedder lẫn pipeline
├── text.py       Regex/hằng số đọc tiếng Việt (tiêu đề, gạch đầu dòng, chữ HOA)
└── checks.py     Khung check()/report() cho 01, 05, 07

streamlit_app.py  Giao diện chat (Streamlit), gọi thẳng rag.pipeline
api.py            REST API thuần: /api/chat, /api/health, /api/graph
knowledge_base/   10 tài liệu tiếng Việt về game (9 game + 1 file thuật ngữ)
init.sql          Schema pgvector — số chiều 1536 có lý do, đọc chú thích
pyproject.toml    Chỉ chứa cấu hình ruff, dự án không đóng gói
```

Kiểm tra mã trước khi commit (cần `requirements-dev.txt`):

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m ruff check .
```

## Ba điều học được, đáng ghi lại

**1. `models.list()` nói dối.** Model có trong danh sách vẫn có thể trả 404 khi
gọi thật. Luôn gọi thử trước khi tin.

**2. Vector rút gọn chiều không còn chuẩn hoá.** `output_dimensionality` cắt
3072 → 1536 theo kiểu Matryoshka, và vector sau khi cắt có norm ≈ 0.69 chứ không
phải 1. Lưu thẳng vào pgvector thì cosine similarity sẽ sai. Phải tự chia lại cho
norm.

**3. Ngưỡng similarity phải tự đo.** Xem bảng ở trên — chép 0.45 từ tutorial thì
chatbot trả lời mọi thứ, kể cả câu hỏi về phở.

## Giới hạn đã biết

- Trích thực thể dựa vào khuôn viết `- Tên: định nghĩa` của knowledge base. Với
  văn xuôi tự do sẽ cần LLM trích thực thể, và chi phí dựng đồ thị không còn bằng 0.
- Viết lại câu hỏi nối tiếp có thể nối đại từ sai khi hội thoại dài.
- Lịch sử hội thoại lưu trong RAM, mất khi restart. Chạy thật cần Redis hoặc bảng
  trong Postgres.
- Hạn mức free tier của Gemini rất chặt (đo được ~20 request với một số model),
  nên các script đánh giá đều mặc định chạy số câu nhỏ.

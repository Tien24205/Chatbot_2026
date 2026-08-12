# Chatbot RAG với PostgreSQL + pgvector

> **Nguồn dữ liệu.** Knowledge base được cào từ
> [Wiki Genshin Impact tiếng Việt](https://genshin-impact.fandom.com/vi) bằng
> `09_fetch_fandom.py`, qua MediaWiki API. Nội dung gốc thuộc giấy phép
> **CC BY-SA**; bản phái sinh trong `knowledge_base/` giữ nguyên giấy phép đó.
> URL từng trang được ghi trong `knowledge_base/_manifest.json`.

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

Chi tiết kỹ thuật: [docs/CHONG_HALLUCINATION_VA_GRAPH.md](docs/CHONG_HALLUCINATION_VA_GRAPH.md)
Nhật ký thiết kế và các bẫy đã vấp: [docs/chatbot_rag_plan.md](docs/chatbot_rag_plan.md)

---

## Kiến trúc

```text
                   KNOWLEDGE BASE (1.123 file .txt)
                               │
        ┌──────────────────────┴──────────────────────┐
        ↓                                             ↓
   Chunk theo tiêu đề `##`                    Trích thực thể (regex)
        ↓                                             ↓
   Embedding (Gemini)                         Đồ thị: 2.590 thực thể
        ↓                                      9.549 cạnh, 3 loại quan hệ
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

Đo trên knowledge base wiki Genshin (1.123 tài liệu / 2.848 chunk),
embedding `intfloat/multilingual-e5-base` chạy trên máy, bộ 25 câu hỏi:

| | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| Đúng file nguồn | 53% | 82% | **88%** |

**Phân bố điểm số — đây mới là phát hiện đáng nói:**

| | thấp nhất | trung vị | cao nhất |
|---|---|---|---|
| Trong phạm vi | 0.824 | 0.865 | 0.902 |
| Ngoài phạm vi | 0.785 | 0.823 | 0.863 |

Hai nhóm **chồng lấn 0.039**. Câu gây chồng lấn: *"Doanh thu Genshin Impact 2024
là bao nhiêu?"* đạt 0.863 — vì nó **thật sự** nói về Genshin, chỉ là kho không có
câu trả lời. Ngưỡng đơn thuần không phân biệt được "đúng chủ đề" với "có câu trả
lời", và không model embedding nào sửa được điều đó.

`03_eval_retrieval.py` đề xuất τ = 0.891 để đạt 0 câu bịa — nhưng nó **chỉ nhìn
ngưỡng**, coi như không có lớp nào khác, và trả giá bằng 16/17 câu bị từ chối oan.
Dự án này có năm lớp, nên chọn **τ = 0.82** và để bốn lớp còn lại làm việc của
chúng. Đúng nguyên tắc đã ghi bên dưới: ngưỡng là hàm của số lớp phòng thủ.

**Đo đầu-cuối bằng `08_eval_answers.py` (25 câu) chứng minh lựa chọn đó:**

| Nhóm | Đạt | Tổng |
|---|---|---|
| in_scope | 10 | 13 |
| ambiguous | 4 | 4 |
| out_of_scope | 8 | 8 |
| **Tất cả** | **22** | **25** |

| | Chỉ nhìn ngưỡng (`03`) | Đầu-cuối (`08`) |
|---|---|---|
| Câu ngoài phạm vi bị trả lời | dự báo **9** | thực tế **0** |

Chín câu đó bị các lớp phía sau chặn hết. Đây là con số nói rõ nhất vì sao hiệu
chuẩn ngưỡng một mình là chưa đủ: `03` đo tầng truy hồi, `08` đo cái người dùng
thực sự nhận được.

Câu bị gắn cờ kiểm chứng: **2/14**. Lớp 5 chỉ chạy **2/14** lần (chế độ `auto`).
Latency trung vị: truy hồi **91 ms** · sinh câu trả lời **1044 ms** · tổng
**1174 ms**. Một lần lớp 5 mất 87 giây vì chạm giới hạn tần suất của model chat —
truy hồi thì không, vì embedding đã chạy trên máy.

---

<details>
<summary>Số liệu cũ — knowledge base 4 tài liệu game viết tay, Gemini embedding</summary>

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

</details>

## Cài đặt

Cần: Python 3.11+, Docker Desktop, và một API key Gemini cho phần **sinh câu trả
lời** (miễn phí tại <https://aistudio.google.com/apikey>).

Phần **embedding chạy trên máy**, không cần key và không có hạn mức — xem
`EMBED_BACKEND` trong `.env.example`. Lý do: hạn mức embedding của free tier là
1.000 request/ngày tính theo **tài khoản Google**, mà nạp knowledge base 2.848
chunk đã ăn một phần ba, và mỗi câu hỏi của người dùng cũng tốn một request.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# torch bản CPU (~120 MB) thay vì bản CUDA (~2,5 GB):
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

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

Thư mục gốc chỉ chứa **script đánh số theo bước** (chạy trực tiếp) và cấu hình.
Mọi thứ khác đã gom vào thư mục con.

```text
00_smoke_test.py … 09_fetch_fandom.py   Mười bước, chạy theo thứ tự
streamlit_app.py    Giao diện chat, gọi thẳng rag.pipeline
api.py              REST API thuần: /api/chat, /api/health, /api/graph

rag/                Thư viện — không script nào ở đây chạy trực tiếp
├── config.py       Đọc .env, neo mọi đường dẫn theo vị trí file
├── loader.py       Đọc + làm sạch (NFC, LF, BOM), quét cả thư mục con
├── chunker.py      Cắt theo tiêu đề `##` trước, kích thước sau
├── embedder.py     Chọn backend: Gemini API hoặc chạy trên máy
├── local_embed.py  multilingual-e5-base — không hạn mức, không cần key
├── store.py        pgvector: lưu, tìm, kiểm tra khớp số chiều
├── graph.py        Đồ thị tri thức: trích thực thể, cạnh, đi đa bước
├── prompt.py       System prompt (8 quy tắc) + ghép ngữ cảnh + trích dẫn
├── verify.py       Lớp 3-4: kiểm chứng tất định, không gọi API
├── retry.py        Thử lại khi 429, đọc đúng thời gian chờ Google đề nghị
├── pipeline.py     Ghép toàn tuyến: truy hồi → ngưỡng → LLM → kiểm chứng
│
│   Dùng chung, gom về đây để không còn bản chép:
├── gemini.py       Một client Gemini duy nhất cho cả embedder lẫn pipeline
├── text.py         Regex/hằng số đọc tiếng Việt (tiêu đề, gạch đầu dòng, chữ HOA)
└── checks.py       Khung check()/report() cho 01, 05, 07

knowledge_base/     1.123 tài liệu wiki Genshin → 2.848 chunk
├── thanh_di_vat/   274      ├── nhan_vat/    129      ├── cot_truyen/  33
├── vu_khi/         232      ├── dia_diem/     47      ├── khac/        14
├── npc/            204      ├── he_thong/    183      └── nguyen_to/    7
└── _manifest.json  URL gốc + liên kết wiki của từng trang (ghi nguồn CC BY-SA)

docs/               chatbot_rag_plan.md, CHONG_HALLUCINATION_VA_GRAPH.md
init.sql            Schema pgvector — số chiều phải khớp EMBED_DIMENSION
pyproject.toml      Chỉ chứa cấu hình ruff, dự án không đóng gói
```

Thư mục con của `knowledge_base/` **không đổi `source_name`** — nó vẫn là tên file,
nên khoá `(source_name, chunk_index)` giữ nguyên và sắp xếp lại **không bắt phải
embed lại**. Đổi lại, tên file phải duy nhất trên toàn cây; `loader.py` kiểm tra
và báo lỗi nếu trùng, vì trùng thì hai tài liệu sẽ ghi đè nhau âm thầm.

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

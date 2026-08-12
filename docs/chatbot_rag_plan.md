# Kế hoạch xây dựng Chatbot RAG với PostgreSQL + pgvector

> **Ghi chú (branch GI_Rag):** đây là bản kế hoạch gốc, viết cho knowledge base
> game tự soạn (còn trên branch `main`). Branch này đã thay KB bằng wiki Genshin
> Impact tiếng Việt và chuyển embedding sang chạy máy — xem README cho hiện trạng.

## 1. Mục tiêu

Xây dựng một chatbot hỏi đáp dựa trên Knowledge Base, sử dụng mô hình RAG (Retrieval-Augmented Generation), PostgreSQL/pgvector để lưu trữ và tìm kiếm vector, kết hợp với LLM API để sinh câu trả lời.

---

## 2. Kiến trúc tổng quan

```text
                    ┌─────────────────┐
                    │   Knowledge     │
                    │      Base       │
                    └────────┬────────┘
                             ↓
                       Document Loader
                             ↓
                         Chunking
                             ↓
                        Embedding
                             ↓
                  ┌─────────────────────┐
                  │ PostgreSQL +        │
                  │      pgvector       │
                  └──────────┬──────────┘
                             │
                             │
User ──→ Query ──→ Embedding ──→ Similarity Search
                                      ↓
                                Top-K Chunks
                                      ↓
                              Retrieved Context
                                      ↓
                           ┌──────────────────┐
                           │  System Prompt   │
                           │        +         │
                           │     Context      │
                           │        +         │
                           │      Query       │
                           └────────┬─────────┘
                                    ↓
                                LLM API
                                    ↓
                                  Answer
```

---

## 3. Các giai đoạn thực hiện

### Bước 0 — Chốt provider và smoke test (LÀM ĐẦU TIÊN)

> Bổ sung sau khi rà soát: bản kế hoạch gốc đặt LLM ở Bước 8 và embedding ở Bước 4.
> Cả hai đều cần API key, nên nếu đến giữa tuần mới phát hiện chưa có key thì hỏng cả tuần.
> Xác thực provider **trước**, rồi mới xây pipeline.

Provider đã chọn: **Google Gemini** (một key dùng cho cả chat lẫn embedding).

Việc cần làm:

- [x] Tạo `requirements.txt`, `.env.example`, `.gitignore`.
- [x] Lấy API key tại <https://aistudio.google.com/apikey>, copy `.env.example` → `.env`.
- [x] Chạy `python 00_smoke_test.py` để xác nhận:
  - Key hoạt động;
  - Tên model chat và embedding **thật** mà key truy cập được (không đoán);
  - **Số chiều vector thật** — con số này khoá schema ở Bước 5;
  - Embedding có phân biệt đúng ngữ nghĩa tiếng Việt (cosine similarity).

**Output:** 3 dòng `GEMINI_CHAT_MODEL`, `GEMINI_EMBED_MODEL`, `EMBED_DIMENSION` trong `.env`.

Chỉ khi bước này xanh mới sang Bước 1.

---

### Bước 1 — Xây dựng Knowledge Base

Xác định và chuẩn bị nguồn dữ liệu mà chatbot được phép sử dụng.

Ví dụ:

```text
knowledge_base/
├── company_policy.pdf
├── employee_handbook.pdf
├── faq.txt
└── product_documentation.pdf
```

Công việc:

- Xác định phạm vi kiến thức.
- Thu thập tài liệu.
- Kiểm tra chất lượng tài liệu.
- Loại bỏ dữ liệu không cần thiết.
- Chuẩn hóa nội dung.
- Xác định metadata cần lưu.

Metadata có thể bao gồm:

```text
document_name
page
section
source
created_at
```

**Output:** Bộ tài liệu sạch và có cấu trúc.

---

### Bước 2 — Data Ingestion & Cleaning

Đọc dữ liệu từ PDF, TXT, DOCX hoặc các nguồn khác và chuyển thành dạng text có thể xử lý.

Pipeline:

```text
Document
   ↓
Document Loader
   ↓
Raw Text
   ↓
Cleaning / Normalization
   ↓
Clean Text
```

Cần xử lý các vấn đề như:

- Header/footer lặp lại.
- Ký tự đặc biệt.
- Khoảng trắng thừa.
- Nội dung bị xuống dòng sai.
- Encoding.
- Trang hoặc section bị trộn lẫn.

**Output:** Text sạch.

---

### Bước 3 — Chunking

Chia tài liệu thành các đoạn nhỏ để embedding và retrieval hiệu quả.

Ví dụ:

```text
Document
   ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

Có thể bắt đầu với:

- Chunk size: khoảng 500–1000 tokens.
- Chunk overlap: khoảng 50–150 tokens.

Đây chỉ là giá trị khởi đầu. Sau khi test retrieval cần điều chỉnh dựa trên dữ liệu thực tế.

Mỗi chunk nên giữ metadata liên quan:

```text
chunk_id
content
document_name
page
section
```

**Output:** Danh sách các chunks có metadata.

---

### Bước 4 — Embedding

Chuyển mỗi chunk thành một vector bằng embedding model.

```text
Chunk
  ↓
Embedding Model
  ↓
Vector
```

Ví dụ:

```text
[0.021, -0.184, 0.732, ...]
```

Cần lưu ý:

- Chọn embedding model phù hợp với ngôn ngữ của Knowledge Base.
- Xác định số chiều (dimension) của vector.
- Query sau này phải sử dụng cùng loại embedding space với document chunks.

**Output:** Mỗi chunk có một embedding vector.

---

### Bước 5 — Lưu trữ và Index với PostgreSQL + pgvector

Thiết kế database để lưu nội dung, metadata và embedding.

Ví dụ:

```text
documents
├── id
├── name
└── metadata

chunks
├── id
├── document_id
├── content
├── embedding
├── page
├── section
└── metadata
```

`pgvector` được sử dụng để lưu vector và thực hiện similarity search.

Pipeline:

```text
Chunk
   ↓
Embedding
   ↓
PostgreSQL + pgvector
   ↓
Vector Index
```

Cần kiểm tra:

- PostgreSQL hoạt động.
- Extension `vector` được bật.
- Dimension của vector chính xác.
- Vector được insert thành công.
- Index được tạo đúng.

**Output:** Vector database có thể truy vấn.

---

### Bước 6 — Xây dựng Retrieval

Đây là phần quan trọng của RAG.

Khi người dùng đặt câu hỏi:

```text
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Top-K relevant chunks
```

Ví dụ:

```text
Query:
"Nhân viên được nghỉ phép bao nhiêu ngày?"

Top-K results:
────────────────────────
Chunk 21: similarity = 0.91
Chunk 08: similarity = 0.87
Chunk 35: similarity = 0.82
────────────────────────
```

Các thông số cần thử nghiệm:

- `top_k`
- similarity metric
- similarity threshold
- vector index
- metadata filtering

**Output:** Các chunks liên quan nhất đến câu hỏi.

---

### Bước 7 — Xây dựng RAG Pipeline

Kết hợp Query, Retrieval và Context.

Pipeline:

```text
User Question
      ↓
Query Embedding
      ↓
Vector Search
      ↓
Top-K Chunks
      ↓
Retrieved Context
      ↓
Context + Question
```

Ví dụ:

```text
Question:
"Nhân viên được nghỉ phép bao nhiêu ngày?"

Context:
[Retrieved Chunk 1]
[Retrieved Chunk 2]
[Retrieved Chunk 3]
```

**Output:** Context phù hợp để cung cấp cho LLM.

---

### Bước 8 — Call LLM API

Sau khi Retrieval hoạt động ổn định, kết nối LLM API.

Luồng xử lý:

```text
User Question
      +
Retrieved Context
      +
System Prompt
      ↓
LLM API
      ↓
Generated Answer
```

Mục tiêu ban đầu:

- Gửi request thành công.
- Nhận response.
- Hiển thị câu trả lời.
- Xử lý lỗi API.
- Quản lý API key an toàn.

**Output:** Chatbot có thể sinh câu trả lời.

---

### Bước 9 — Xây dựng System Prompt

System Prompt quy định cách LLM sử dụng context và trả lời.

Ví dụ:

```text
You are a company knowledge assistant.

Rules:
1. Answer only using the provided context.
2. If the answer is not in the context, say you don't know.
3. Do not invent information.
4. Answer in Vietnamese.
5. Keep answers concise.

Context:
{retrieved_context}

User:
{question}
```

Các nguyên tắc quan trọng:

- Không cho phép model tự bịa thông tin.
- Ưu tiên thông tin từ Retrieved Context.
- Quy định ngôn ngữ trả lời.
- Quy định format câu trả lời.
- Xử lý trường hợp không tìm thấy thông tin.

**Output:** LLM có hành vi phù hợp với Knowledge Base.

---

### Bước 10 — Test & Evaluation

Không nên chỉ test sau khi hoàn thành toàn bộ hệ thống. Nên test từng tầng.

#### Test 1 — Document Processing

```text
PDF → Text
```

Kiểm tra:

- Text có bị mất không?
- Thứ tự nội dung đúng không?
- Metadata có chính xác không?

#### Test 2 — Chunking

```text
Text → Chunks
```

Kiểm tra:

- Chunk có quá dài không?
- Chunk có quá ngắn không?
- Nội dung có bị cắt giữa câu hoặc section không?

#### Test 3 — Embedding

```text
Chunk → Embedding
```

Kiểm tra:

- Vector có được tạo không?
- Dimension có đúng không?

#### Test 4 — Database

```text
Embedding → pgvector
```

Kiểm tra:

- Insert thành công.
- Query thành công.
- Index hoạt động.

#### Test 5 — Retrieval

```text
Question → Top-K Chunks
```

Kiểm tra:

- Các chunks trả về có liên quan không?
- Chunk quan trọng có nằm trong Top-K không?

#### Test 6 — LLM

```text
Context + Question → Answer
```

Kiểm tra:

- Model có sử dụng context không?
- Có hallucination không?
- Câu trả lời có đúng không?

#### Test 7 — Full RAG

```text
Question
   ↓
Retrieval
   ↓
Context
   ↓
LLM
   ↓
Answer
```

---

## 4. Các lỗi cần có khả năng phân biệt

Khi chatbot trả lời sai, cần xác định lỗi nằm ở tầng nào:

```text
                    Wrong Answer
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      Chunking       Retrieval        Prompt
          │              │              │
          ↓              ↓              ↓
     Embedding       pgvector          LLM
```

Không nên ngay lập tức thay đổi LLM nếu vấn đề thực tế nằm ở Retrieval.

---

## 5. Thứ tự triển khai thực tế

Nên triển khai theo các milestone sau:

### Milestone 0 — Provider ✅

- [x] Chốt provider: Google Gemini.
- [x] Lấy API key, điền `.env`.
- [x] Smoke test chạy xanh.

Cấu hình đã xác minh bằng lệnh gọi thật (không phải đọc tài liệu):

| Khoá | Giá trị | Ghi chú |
|---|---|---|
| `GEMINI_CHAT_MODEL` | `gemini-flash-latest` | `gemini-2.5-flash` bị chặn với tài khoản mới (404) dù vẫn nằm trong `models.list()` |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | |
| `EMBED_DIMENSION` | **1536** | Model trả 3072 chiều; rút xuống 1536 vì pgvector chỉ index được ≤ 2000 chiều |

**Ba cái bẫy đã phát hiện, phải nhớ khi code Bước 4-5:**

1. `models.list()` **nói dối** — model có trong danh sách vẫn có thể 404 lúc gọi thật.
   Luôn gọi thử, đừng tin danh sách.
2. Vector rút gọn bằng `output_dimensionality` **không còn chuẩn hoá**
   (norm ≈ 0.69 ở 1536 chiều). Phải tự chia cho norm trước khi lưu, nếu không
   cosine similarity sẽ sai.
3. **Ngưỡng similarity phải tự đo, không chép từ tutorial.** Đo thực tế trên
   `gemini-embedding-001`: câu liên quan 0.655, câu hoàn toàn không liên quan
   (game vs. công thức nấu phở) vẫn **0.513**. Nền tương đồng của model này rất
   cao — ngưỡng τ = 0.45 kiểu tutorial sẽ cho qua *mọi thứ*, kể cả phở.
   Phải hiệu chuẩn τ bằng bộ câu hỏi test ở Bước 6.

### Milestone 1 — Knowledge Base ✅

- [x] Chuẩn bị tài liệu — 4 file về game trong `knowledge_base/`.
- [x] Load tài liệu — `rag/loader.py`.
- [x] Cleaning — chuẩn hoá NFC, LF, bỏ BOM/zero-width, gom dòng trống.
- [x] Chuẩn hóa metadata — mỗi chunk có `source_name`, `section`, `chunk_index`.
- [x] Test tầng 1-2 — `01_test_chunking.py`, 12/12 PASS, 25 chunk.

Ghi chú kỹ thuật:
- `CHARS_PER_TOKEN = 3.0` mới chỉ là **ước lượng** cho tiếng Việt. Sau khi có API key,
  kiểm chứng bằng `client.models.count_tokens()` rồi chỉnh lại nếu lệch.
- Mỗi mục `##` của KB hiện tại gọn hơn ngưỡng nên ra đúng 1 chunk/mục — nhánh overlap
  được kiểm thử riêng bằng dữ liệu tổng hợp (Test 3), không hạ ngưỡng để ép test pass.

### Milestone 2 — Vector Database ✅

- [x] Cài PostgreSQL — `docker-compose.yml`, image `pgvector/pgvector:pg17`, cổng **5433**
      (5432 đã bị dự án CRM chiếm).
- [x] Cài/bật pgvector — `CREATE EXTENSION vector` trong `init.sql`.
- [x] Tạo database schema — bảng `chunks`, cột `vector(1536)`.
- [x] Implement chunking — `rag/chunker.py`, cắt theo tiêu đề `##` trước, kích thước sau.
- [x] Implement embedding — `rag/embedder.py`, có chuẩn hoá lại + tách `task_type`.
- [x] Insert chunks + vectors — `rag/store.py` + `02_ingest.py`, 25 chunk trong 3.0s.
- [x] Tạo vector index — HNSW `vector_cosine_ops`.

**Bốn cái bẫy đã vấp và sửa khi làm milestone này:**

1. `os.getenv("X", default)` trả về `""` chứ không phải default khi `.env` có dòng
   `X=` để trống → `int("")` nổ. Đã thêm hàm `env()` coi chuỗi rỗng là chưa đặt.
2. `.env` giữ cổng 5432 cũ trong khi container chuyển sang 5433 → gõ nhầm vào
   database khác, báo lỗi "password authentication failed" gây hiểu nhầm.
3. `psycopg` gửi list Python thành `double precision[]`. INSERT thì Postgres tự ép
   kiểu (đã biết kiểu cột), nhưng toán tử `<=>` thì không → phải viết `%s::vector`.
4. `init.sql` chỉ chạy khi volume còn rỗng. Sửa schema sau đó phải
   `docker compose down -v` rồi dựng lại, nếu không thay đổi sẽ bị bỏ qua âm thầm.

### Milestone 3 — Retrieval ✅

- [x] Nhận user query, sinh query embedding, similarity search, trả top-k — `rag/store.py`.
- [x] Bộ câu hỏi test cố định — `eval_questions.json`, 25 câu / 3 nhóm.
- [x] Đo chất lượng retrieval + hiệu chuẩn ngưỡng — `03_eval_retrieval.py`.

**Recall (25 chunk, `gemini-embedding-001` 1536 chiều):**

| | đúng file | đúng cả mục |
|---|---|---|
| recall@1 | 19/19 (100%) | 16/16 (100%) |

Chunk đúng luôn đứng **hạng 1**. Chiến lược cắt theo tiêu đề `##` cho kết quả tốt
trên KB này; chưa cần đụng tới chunk size hay top-k.

**Phân bố điểm số:**

| Nhóm | thấp nhất | trung vị | cao nhất |
|---|---|---|---|
| Trong phạm vi | 0.617 | 0.718 | 0.783 |
| Ngoài phạm vi | 0.478 | 0.510 | **0.657** |

Hai nhóm **chồng lấn** (khoảng cách −0.040) → không ngưỡng nào đúng 100%.
Cặp gây chồng lấn đáng đọc:

- ngoài phạm vi cao nhất: *"Ai là nhân vật mạnh nhất trong game?"* → 0.657
- trong phạm vi thấp nhất: *"Thánh Di Vật gồm những vị trí nào?"* → 0.617

Câu về "nhân vật mạnh nhất" **đúng chủ đề game nhưng KB không hề xếp hạng sức mạnh**.
Đây chính là dạng câu khiến LLM bịa: đủ liên quan để lọt qua ngưỡng thấp, nhưng
không có dữ liệu để trả lời. Tương đồng chủ đề ≠ có câu trả lời.

**Ngưỡng đã chốt: τ = 0.66** (`SIMILARITY_THRESHOLD` trong `.env`)

Hai ngưỡng cùng đạt 96%, chọn cái an toàn hơn:

| | τ = 0.52 | **τ = 0.66** |
|---|---|---|
| Trả lời đúng | 19/19 | 18/19 |
| Từ chối đúng | 5/6 | **6/6** |
| **Bịa** | **1** | **0** |
| Bỏ sót | 0 | 1 |

Độ chính xác thô coi "bịa" và "bỏ sót" nặng ngang nhau, nhưng với chatbot tra cứu
thì không: bỏ sót thì người dùng hỏi lại được, còn bịa thì họ tin luôn mà không có
cách kiểm chứng. Chọn τ = 0.66.

> So sánh: **τ = 0.45** trong tài liệu CRM sẽ để lọt **6/6** câu ngoài phạm vi —
> tức là chatbot trả lời cả câu hỏi về phở bằng chunk thuật ngữ game.

> **Cập nhật ở Milestone 6: τ hạ xuống 0.60.** Con số 0.66 đúng cho hoàn cảnh lúc
> đó — ngưỡng là lớp phòng thủ gần như duy nhất nên phải đặt cao. Sau khi có đủ 5
> lớp chống hallucination, hạ xuống 0.60 lấy lại được 2 câu bị từ chối oan mà vẫn
> 0/6 câu ngoài phạm vi bị bịa. Ngưỡng không phải hằng số của model, nó là hàm của
> số lớp phòng thủ đang có.

### Milestone 4 — RAG ✅

- [x] Ghép context — `rag/prompt.py`, mỗi chunk gắn nhãn nguồn + độ liên quan.
- [x] Xây dựng prompt — system prompt 6 quy tắc ràng buộc.
- [x] Call LLM API — `rag/pipeline.py`, `gemini-flash-latest`, temperature 0.2.
- [x] Parse response + trích nguồn.
- [x] Trả câu trả lời cho user — `04_chat.py`.

**Hai lớp chống bịa, chặn hai loại lỗi khác nhau:**

| Lớp | Chặn gì | Cơ chế |
|---|---|---|
| 1. Ngưỡng τ = 0.66 | Không có chunk nào đủ liên quan | Từ chối TRƯỚC khi gọi LLM — vừa chắc chắn không bịa, vừa không tốn tiền API |
| 2. System prompt | Chunk đúng chủ đề nhưng không chứa câu trả lời | Quy tắc #3: "liên quan chủ đề ≠ có câu trả lời" |

Cần cả hai. Câu *"Ai là nhân vật mạnh nhất trong game?"* (0.657) bị lớp 1 chặn,
nhưng nếu KB lớn hơn và câu đó vượt ngưỡng thì chỉ còn lớp 2 giữ.

**Trích nguồn chỉ liệt kê chunk vượt ngưỡng**, không liệt kê toàn bộ top-k. Toàn
bộ top-k vẫn vào ngữ cảnh cho LLM (thừa ngữ cảnh không hại), nhưng ghi tất cả làm
"nguồn" là sai — câu trả lời về Baron Nashor từng ghi thêm nguồn Liên Quân và
thuật ngữ game, khiến người đọc tưởng đã đối chiếu chéo.

### Milestone 5 — Chatbot ✅

- [x] Xử lý conversation history — giữ 6 lượt gần nhất, chỉ ghi lượt trả lời được.
- [x] Hiển thị câu trả lời + trích nguồn + số liệu latency.
- [x] Tạo API backend — `api.py` (FastAPI): `POST /api/chat`, `GET /api/health`,
      `DELETE /api/chat/{session_id}`.
- [x] Tạo giao diện chat web — ban đầu là `web/`, HTML/CSS/JS thuần, không framework.
- [x] Kết nối frontend → backend — cùng origin, không cần CORS.
- [x] Kiểm thử API — `05_test_api.py`, 13/13 PASS qua HTTP thật.
- [x] **Làm lại giao diện bằng Streamlit** — `streamlit_app.py` thay hẳn `web/`.
      Nó gọi THẲNG `rag.pipeline` chứ không qua HTTP: Streamlit đã là một tiến
      trình Python đầy đủ, đi vòng qua API của chính mình chỉ tổ phải chạy hai
      server và nhân đôi chỗ lưu lịch sử hội thoại. `web/` đã xoá; `api.py` giữ
      lại làm REST API thuần và vẫn được `05_test_api.py` kiểm thử.

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
# mở http://localhost:8501

.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000  # REST API, /docs
```

**Hạn mức Gemini free tier — đo được, không phải đọc tài liệu:**

| Model | Kết quả |
|---|---|
| `gemini-flash-latest` (→ `gemini-3.6-flash`) | 20 request rồi hết hạn mức |
| `gemini-2.0-flash`, `gemini-2.0-flash-lite` | **limit 0** — free tier không được dùng |
| `gemini-flash-lite-latest` | Còn dùng được → đặt làm mặc định |
| `gemini-2.5-flash`, `gemini-2.5-flash-lite` | 404 với tài khoản mới |

Một lượt hỏi nối tiếp tốn tới **3 lệnh gọi** (viết lại câu hỏi + embed + sinh câu
trả lời) nên chạm trần rất nhanh. Đã xử lý ba lớp:

1. `rag/retry.py` — thử lại khi gặp 429, đọc đúng thời gian chờ Google đề nghị
   trong thông báo lỗi thay vì backoff đoán mò.
2. Model dự phòng — hạn mức tính riêng theo từng model, nên khi model chính hết
   lượt thì tự chuyển sang model dự phòng thay vì báo lỗi.
3. API trả **429** kèm thông báo tiếng Việt, không phải 500. Đây là hạn mức nhà
   cung cấp, không phải hệ thống hỏng — người dùng cần biết chỉ phải chờ.

**Query rewriting.** Retrieval chỉ nhìn thấy văn bản câu hỏi, không thấy hội thoại.
Đo thực tế: sau khi hỏi Baron Nashor, câu *"Còn Liên Quân thì sao?"* truy hồi ra
Tổng quan / Xếp hạng / Giải đấu và **trượt** chunk "Mục tiêu trung lập" — đúng chỗ
chứa Rồng Bạo Chúa. Đã thêm bước viết lại câu hỏi thành dạng độc lập trước khi tìm.

Chỉ chạy khi câu hỏi có dấu hiệu phụ thuộc ngữ cảnh (đại từ, "còn... thì sao", câu
ngắn), vì mỗi lần viết lại tốn thêm ~7 giây.

| Loại câu | Latency |
|---|---|
| Tự đủ nghĩa | ~2.5 s |
| Cần viết lại | ~12 s |

**Hạn chế đã biết sau Milestone 5:**

1. **Đại từ có thể nối sai.** Sau hai lượt, câu *"Nó xuất hiện từ phút thứ mấy?"*
   được viết lại thành *"Baron Nashor... xuất hiện từ phút thứ mấy?"* trong khi
   "nó" đáng lẽ trỏ Thần Rừng ở câu trả lời ngay trước. Chatbot không bịa (nói
   thẳng là không tìm thấy), nhưng đã hiểu sai ý. → **vẫn chưa xử lý.**
2. **Biên ngưỡng rất mỏng.** Câu khó nhất đạt 0.657, ngưỡng 0.66 — cách nhau
   0.003. → **đã xử lý ở Milestone 6**: thêm câu test, hạ ngưỡng xuống 0.60 và
   chuyển gánh nặng sang các lớp phòng thủ khác. Biên hiện tại là 0.017.

### Milestone 6 — Evaluation & Optimization ✅

- [x] Tạo bộ câu hỏi test — thêm nhóm `questions_graph` (8 câu so sánh chéo game
      và câu sát biên ngưỡng), tách khỏi bộ 25 câu gốc để bộ gốc còn dùng làm mốc.
- [x] Đánh giá retrieval — `03_eval_retrieval.py` (từ Milestone 3).
- [x] Đánh giá câu trả lời — `08_eval_answers.py`, đây là phép đo MỚI: truy hồi
      đúng 100% vẫn có thể ra câu trả lời sai vì LLM suy diễn từ chunk đúng.
- [x] Điều chỉnh Top-K — giữ 4, thêm cơ chế đồ thị bổ sung tối đa 2 đoạn.
- [x] Điều chỉnh prompt — thêm quy tắc 4 (ngoại lệ so sánh) và quy tắc 8 (trích
      dẫn theo chỉ số).
- [x] Kiểm tra hallucination — 5 lớp phòng thủ, đo được 0 câu bịa.
- [x] Tối ưu latency và chi phí — lớp 5 leo thang có điều kiện, chi phí thực tế 0.
- [ ] Điều chỉnh chunk size / overlap — **cố ý không đụng.** Chunk hiện tại đạt
      recall@1 = 100%; sửa nó sẽ phải embedding lại toàn bộ (tốn hạn mức) để chữa
      một thứ chưa hỏng. Nguyên tắc plan §8.3: kiểm tra retrieval trước, đừng tối
      ưu cái đang chạy tốt.

**Kết quả trên bộ câu hỏi đồ thị (8 câu, τ = 0.60):**

| Nhóm | Đạt | Tổng |
|---|---|---|
| ambiguous (so sánh chéo game) | 4 | 4 |
| in_scope | 3 | 3 |
| out_of_scope | 1 | 1 |
| **Tất cả** | **8** | **8** |

Câu ngoài phạm vi bị bịa: **0**. Câu bị gắn cờ kiểm chứng: **0/7**.
Latency trung vị: truy hồi 413 ms · sinh câu trả lời 1269 ms · **tổng 1718 ms**.

**Phát hiện chính: ngưỡng τ là hàm của số lớp phòng thủ, không phải hằng số.**

| | τ = 0.66 | τ = 0.60 |
|---|---|---|
| Câu in_scope bị từ chối oan | **2** (0.617 và 0.631) | **0** |
| Câu ngoài phạm vi bị bịa | 0 | **0** |
| Câu "ai mạnh nhất" (0.657) | chặn bởi lớp 2 (ngưỡng) | chặn bởi **lớp 1** (prompt) |

Khi ngưỡng phải gánh một mình, nó buộc phải đặt cao và trả giá bằng recall. Có đủ
5 lớp phía sau thì hạ được ngưỡng mà không mất an toàn.

**Hai lỗi báo động giả đã tìm ra và sửa nhờ chạy đánh giá thật:**

1. Lớp ký tự `[A-ZĐÀ-Ỹ]` không phải "chữ hoa tiếng Việt" — khoảng `À-Ỹ` chứa xen kẽ
   cả chữ thường, nên "mỗi đội" bị nhận là danh từ riêng bịa.
2. Chữ đầu câu viết hoa bị tính là tên riêng: "Trong Liên Minh Huyền Thoại…" sinh
   ra cặp "Trong Liên" và bị báo là bịa. Đo được 2/5 câu đúng bị gắn cờ oan.

### Milestone 7 — Chống hallucination chuyên sâu + GraphRAG ✅

Chi tiết đầy đủ: [CHONG_HALLUCINATION_VA_GRAPH.md](CHONG_HALLUCINATION_VA_GRAPH.md)

**Năm lớp chống hallucination** — mỗi lớp chặn một loại lỗi khác nhau:

| # | Lớp | Chặn gì | Chi phí |
|---|---|---|---|
| 1 | System prompt ràng buộc phạm vi | Dùng kiến thức nền | 0 |
| 2 | Ngưỡng similarity hiệu chuẩn | Không có đoạn nào đủ liên quan | 0 (tiết kiệm 1 lệnh gọi) |
| 3 | Trích dẫn có cấu trúc `[n]` | Bịa nguồn | 0 |
| 4 | Kiểm chứng grounding tất định | Bịa số liệu, bịa tên riêng | 0 |
| 5 | Vòng entailment bằng LLM | Suy diễn sai từ ngữ cảnh đúng | 1 lệnh gọi, leo thang có điều kiện |

Ba trong năm lớp miễn phí và tất định — bắt buộc phải vậy, vì hạn mức free tier
chỉ khoảng 20 request, nếu lớp nào cũng gọi LLM thì không lớp nào chạy được thật.

**GraphRAG** — `rag/graph.py`, 82 thực thể / 571 cạnh, dựng bằng regex nên **không
tốn một lệnh gọi API nào**:

- Cạnh `tương_tự` (52) nối các thực thể **cùng vai trò nhưng khác game** — đây là
  thứ vector không thể thay thế: "Baron Nashor" và "Thần Rừng" không giống nhau
  một chữ nào, chúng chỉ giống nhau về vai trò.
- Đồ thị đóng góp cả **chunk bổ sung** lẫn **chính quan hệ đó viết thành câu** đưa
  vào prompt — không có phần thứ hai thì model từ chối trả lời câu so sánh.
- Ràng buộc an toàn: đồ thị chỉ chạy **sau** khi ngưỡng đã cho qua, nên không bao
  giờ cứu được một câu đáng bị từ chối. Đã kiểm chứng bằng câu bẫy Genshin.

**Đối chiếu Microsoft Agent Framework / Semantic Kernel**: bảng ánh xạ đầy đủ
(`ITextSearch`, `TextSearchProvider`, `TextSearchStore`, `AIContextProviders`,
`ContextFormatter`, `RagBehavior`…) nằm ở Phần 3 của tài liệu trên.

**Kiểm thử:** `07_test_graph_verify.py` — 41 kiểm thử, chạy offline, không cần API key.

---

## 6. Pipeline hoàn chỉnh

```text
                 KNOWLEDGE BASE
                       │
                       ↓
              Document Ingestion
                       │
                       ↓
                    Cleaning
                       │
                       ↓
                   Chunking
                       │
                       ↓
                   Embedding
                       │
                       ↓
              PostgreSQL + pgvector
                       │
                       │
                       │
User ──→ Query ──→ Query Embedding
                       │
                       ↓
                Similarity Search
                       │
                       ↓
                    Top-K
                       │
                       ↓
         ┌─── LỚP 2: điểm cao nhất < τ ? ──→ TỪ CHỐI (không gọi LLM)
                       │ không
                       ↓
         câu hỏi so sánh ? ──→ ĐỒ THỊ: cạnh tương_tự
                       │              → chunk bổ sung + quan hệ
                       ↓
              Retrieved Context
                       │
                       ↓
          ┌─────────────────────────┐
          │  System Prompt (LỚP 1)  │
          │          +              │
          │  Retrieved Context      │
          │          +              │
          │  Quan hệ từ đồ thị      │
          │          +              │
          │     User Question       │
          └────────────┬────────────┘
                       ↓
                    LLM API
                       │
                       ↓
                    Answer
                       │
                       ↓
         LỚP 3: kiểm tra chỉ số trích dẫn   ─┐
         LỚP 4: soi số liệu + tên riêng      ├─ tất định, miễn phí
                       │                     ─┘
                       ↓
         có cờ ? ──→ LỚP 5: vòng entailment bằng LLM
                       │
                       ↓
              Answer + trích nguồn + cờ kiểm chứng
                       │
                       ↓
                  Chatbot UI
```

Chi tiết năm lớp và tầng đồ thị: [CHONG_HALLUCINATION_VA_GRAPH.md](CHONG_HALLUCINATION_VA_GRAPH.md)

---

## 7. Công nghệ đề xuất

| Thành phần | Công nghệ |
|---|---|
| Backend | Python + FastAPI/Flask |
| Database | PostgreSQL |
| Vector Store | pgvector |
| Embedding | Embedding Model phù hợp |
| LLM | LLM API |
| Document Processing | Python |
| Frontend | HTML/CSS/JavaScript |
| API Communication | REST API |
| Testing | Python / Postman |
| Version Control | Git + GitHub |

---

## 8. Nguyên tắc triển khai

1. **Làm từng tầng và test ngay sau mỗi tầng.**
2. **Không thêm framework RAG quá sớm** nếu mục tiêu là hiểu rõ hệ thống.
3. **Ưu tiên kiểm tra Retrieval trước khi tối ưu Prompt.**
4. **Không để LLM tự suy diễn khi Knowledge Base không có thông tin.**
5. **Lưu metadata cùng với chunk** để có thể truy xuất nguồn và debug.
6. **Tạo một bộ câu hỏi test cố định** để so sánh kết quả sau mỗi lần thay đổi.
7. **Tách ingestion pipeline và query pipeline** để dễ bảo trì.

---

## 9. Kết quả mong đợi

Sau khi hoàn thành, hệ thống có thể xử lý:

```text
User:
"Nhân viên được nghỉ phép bao nhiêu ngày?"

        ↓

Query Embedding

        ↓

pgvector Similarity Search

        ↓

Relevant Chunks

        ↓

RAG Context

        ↓

System Prompt + Context + Question

        ↓

LLM API

        ↓

Chatbot:

"Theo chính sách hiện tại, nhân viên
được nghỉ phép ... ngày."
```

Hệ thống cuối cùng sẽ có hai pipeline chính:

### Offline / Indexing Pipeline

```text
Documents
→ Cleaning
→ Chunking
→ Embedding
→ PostgreSQL + pgvector
```

### Online / Query Pipeline

```text
User Query
→ Query Embedding
→ Retrieval
→ Context
→ System Prompt
→ LLM API
→ Answer
```

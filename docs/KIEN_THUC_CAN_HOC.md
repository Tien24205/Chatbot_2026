# Kiến thức cần học để hiểu và làm được dự án này

Tài liệu này liệt kê các mảng kiến thức theo đúng thứ tự nên học, mỗi mảng gồm
ba phần: **cần nắm gì**, **nó nằm ở đâu trong dự án**, và **bài học đã trả giá**
— những lỗi có thật đã vấp trong quá trình làm, học từ chúng nhanh hơn nhiều so
với đọc lý thuyết suông.

Cách dùng hiệu quả nhất: đọc mục nào thì mở đúng file code của mục đó bên cạnh.
Toàn bộ dự án được đánh số `00 → 09` theo thứ tự xây dựng — đọc code theo đúng
thứ tự đó chính là lộ trình học.

---

## 1. Python trung cấp

**Cần nắm:**
- `dataclass` và type hint (`list[dict]`, `str | None`) — toàn bộ cấu trúc dữ
  liệu của dự án (`SearchHit`, `Answer`, `Entity`, `Report`) là dataclass.
- Regex ở mức thành thạo: nhóm bắt `(...)`, non-greedy `+?`, lookbehind,
  cờ `re.M`/`re.S`/`re.I`. Dự án dùng regex cho việc nặng: tách câu, trích
  thực thể, dọn wikitext, nhận diện câu đếm.
- Tổ chức package: `rag/` là thư viện dùng chung, các script `0*.py` là chương
  trình dòng lệnh gọi vào nó. Hiểu vì sao logic phải nằm MỘT chỗ.
- `pathlib`, context manager (`with`), generator, xử lý Unicode tiếng Việt.

**Trong dự án:** mọi file. Bắt đầu từ `rag/store.py` (ngắn, đủ khái niệm).

**Bài học đã trả giá:** hai bản sao của một quy tắc sẽ lệch nhau một cách âm
thầm — `04_chat.py` và `api.py` từng chép tay cùng đoạn quản lý lịch sử, sau
phải gom về `pipeline.remember()` duy nhất. Nguyên tắc: quy tắc nghiệp vụ chỉ
được tồn tại ở một chỗ.

---

## 2. RAG — Retrieval-Augmented Generation (xương sống của dự án)

**Cần nắm:**
- Vì sao không hỏi thẳng LLM: kiến thức nền lỗi thời, bịa không kiểm soát được,
  không trích được nguồn.
- Chuỗi chuẩn: **tài liệu → cắt chunk → embedding → lưu vector DB** (nạp một
  lần), rồi **câu hỏi → embedding → tìm k chunk gần nhất → ngưỡng → ghép vào
  prompt → LLM → kiểm chứng** (mỗi lượt hỏi).
- Khái niệm top-k và hệ quả của nó: ngữ cảnh chỉ chứa được k đoạn — mọi giới
  hạn "trả lời thiếu" đều bắt nguồn từ đây.
- recall@k: chunk đúng có nằm trong k kết quả đầu không.

**Trong dự án:** `rag/pipeline.py` là bản đồ toàn tuyến — đọc hàm `ask()` từ
trên xuống dưới là thấy đủ mọi tầng. `02_ingest.py` là tuyến nạp.

**Bài học đã trả giá:** truy hồi đúng 100% vẫn ra câu trả lời sai được, vì LLM
suy diễn từ chunk đúng. Đo tầng truy hồi (`03`) và đo đầu-cuối (`08`) là hai
phép đo KHÁC NHAU: `03` từng dự báo 12 câu bịa, thực tế `08` đo được 0.

---

## 3. Embedding và tìm kiếm ngữ nghĩa

**Cần nắm:**
- Embedding là gì: câu → vector số trăm chiều; câu gần nghĩa thì vector gần
  nhau. Cosine similarity và vì sao phải chuẩn hoá vector.
- Model đa ngữ vs thuần Anh: `all-MiniLM-L6-v2` phổ biến trong tutorial nhưng
  gần như thuần tiếng Anh — dùng cho tiếng Việt là hỏng từ gốc.
- Họ e5 cần tiền tố `"query: "` / `"passage: "` — quên là điểm số sai lệch.
- Chạy local (sentence-transformers, CPU) vs gọi API: đánh đổi chất lượng /
  hạn mức / độ trễ / chi phí.

**Trong dự án:** `rag/local_embed.py`, `rag/embedder.py` (dispatch hai backend),
`.env.example` ghi lý do chọn từng thứ.

**Bài học đã trả giá:** hạn mức embedding free của Gemini là 1.000 request/ngày
tính theo TÀI KHOẢN (không phải project — tạo key mới không thêm hạn mức), và
mỗi câu hỏi của người dùng cũng tốn một request embed. Kho vài nghìn chunk thì
embedding local là lối thoát duy nhất. Ngoài ra: giới hạn 429 hoá ra tính theo
TOKEN mỗi request chứ không phải số request — phải đo batch 1/2/4/8/16 mới tìm
ra, đoán mò hai lần đều sai.

---

## 4. PostgreSQL + pgvector

**Cần nắm:**
- SQL căn bản: JOIN, GROUP BY, window function (`row_number() OVER`), UPSERT
  (`ON CONFLICT DO UPDATE`), transaction/commit.
- pgvector: kiểu cột `vector(N)`, toán tử khoảng cách cosine `<=>`, index HNSW,
  và trần 2000 chiều khi đánh index (lý do vector Gemini phải cắt 3072 → 1536).
- Ép kiểu `%s::vector` khi truy vấn — psycopg gửi list Python thành mảng số,
  toán tử `<=>` không tự suy ra kiểu.
- Docker volume: `init.sql` CHỈ chạy khi volume còn rỗng — sửa schema sau đó
  phải làm bằng cách khác.

**Trong dự án:** `init.sql`, `rag/store.py`, `docker-compose.yml`,
`rag/graph.py` (DDL từ Python — chính vì cái bẫy volume ở trên),
`rag/chatlog.py` (bảng lưu hội thoại).

**Bài học đã trả giá:** đổi `EMBED_DIMENSION` là phải `docker compose down -v`
và embedding lại toàn bộ. Khoá resume `(source_name, chunk_index)` cho phép nạp
lại CHỌN LỌC: chỉ xoá chunk của file đổi rồi chạy `02` — tiết kiệm cả giờ CPU.

---

## 5. Chunking — cắt tài liệu

**Cần nắm:**
- Vì sao phải cắt: embedding có trần token, và chunk càng đúng ranh giới ngữ
  nghĩa thì truy hồi càng sạch.
- Chiến lược cắt theo cấu trúc (tiêu đề `##`) TRƯỚC, theo kích thước SAU —
  và vì sao cột `section` sống còn cho việc trích nguồn.
- Ranh giới không được xé: gạch đầu dòng, câu.

**Trong dự án:** `rag/chunker.py`, `rag/text.py` (regex dùng chung),
`01_test_chunking.py` — bộ test miễn phí, chạy TRƯỚC khi tiêu tiền embedding.

**Bài học đã trả giá:** một mục `- description: <2000 ký tự lore>` sinh chunk
4.013 ký tự vì gạch đầu dòng là đơn vị không được xé — phải tách trường dài
của infobox thành mục văn xuôi riêng ngay từ bước cào.

---

## 6. Prompt engineering (kiểu đo được, không phải kiểu cầu may)

**Cần nắm:**
- System prompt là YÊU CẦU, không phải RÀNG BUỘC — model có thể lờ. Mọi quy
  tắc quan trọng phải có tầng kiểm chứng đứng sau (xem mục 7).
- Cấu trúc quy tắc đánh số + NGOẠI LỆ tường minh. Model nhỏ tuân "quy tắc 3:
  đừng suy ra" chặt đến mức phải viết hẳn "NGOẠI LỆ của quy tắc 3" thì câu so
  sánh và câu đếm mới hoạt động — chữ "NGOẠI LỆ" là thứ làm nên khác biệt,
  đã đo cả hai chiều.
- Khuôn đầu ra bắt buộc (`"Suy ra từ [1][2]: ..."`) để tầng sau nhận diện được
  bằng regex.
- Temperature: tra cứu để 0 — ở 0.2, cùng ngữ cảnh mà lúc trả lời lúc từ chối.
- Nhét chỉ dẫn vào USER message khi system prompt không đủ lực với model nhỏ.

**Trong dự án:** `rag/prompt.py` — đọc từng quy tắc kèm comment lịch sử của nó.

**Bài học đã trả giá:** ví dụ trong prompt phải là dữ liệu THẬT trong kho
(Kiếm Sắt Đen, không phải Baron Nashor của KB cũ); câu từ chối bắt buộc phải
đúng khuôn để nhận diện — model viết chệch "thông tin CỤ THỂ" thay vì "thông
tin này" là cả tầng phân loại từ chối hỏng theo.

---

## 7. Chống hallucination nhiều tầng — phần "ra tiền" nhất của dự án

**Cần nắm:**
- Không tầng nào đủ một mình. Năm lớp, mỗi lớp chặn một loại lỗi khác nhau:
  1. System prompt (chặn dùng kiến thức nền) — chi phí 0
  2. Ngưỡng similarity hiệu chuẩn bằng đo (chặn câu ngoài phạm vi) — còn tiết
     kiệm được lệnh gọi LLM
  3. Trích dẫn `[n]` có cấu trúc + kiểm chỉ số (chặn bịa nguồn)
  4. Kiểm grounding tất định bằng regex: số, tên riêng, thực thể phải có trong
     ngữ cảnh (chặn bịa dữ kiện) — chi phí 0
  5. Vòng entailment bằng LLM thứ hai (chặn suy diễn sai) — tốn API nên chỉ
     leo thang khi lớp 3-4 nghi ngờ
- Precision/recall trade-off của ngưỡng: τ cao thì không bịa nhưng từ chối oan
  hàng loạt. Ngưỡng là HÀM CỦA SỐ LỚP phòng thủ: có 5 lớp thì τ được phép thấp.
- Nguyên tắc "gắn cờ, không tự sửa": máy kiểm chứng cũng có thể sai — báo cho
  người đọc tự đối chiếu trung thực hơn là lặng lẽ xoá.
- Mở suy luận có kiểm soát: miễn kiểm số cho câu suy ra CÓ ĐÁNH DẤU, đổi lại
  ép lớp 5 kiểm phép tính — nới một lớp thì phải siết lớp khác.

**Trong dự án:** `rag/verify.py` (đọc docstring đầu file trước),
`rag/pipeline.py` phần sau của `ask()`, `docs/CHONG_HALLUCINATION_VA_GRAPH.md`.

**Bài học đã trả giá:** lớp 4 từng gắn cờ oan câu đếm ĐÚNG ("số 9 không có
trong ngữ cảnh" — vì 9 là kết quả đếm, làm sao có nguyên văn được). Kiểm chứng
mù quáng bóp chết cả suy luận hợp lệ — phải thiết kế lối thoát có điều kiện.

---

## 8. GraphRAG — đồ thị tri thức bên cạnh vector

**Cần nắm:**
- Vector search chỉ thấy ĐỘ GIỐNG CÂU CHỮ; quan hệ giữa các thực thể ("cùng là
  Kiếm Đơn") thì nó mù. Đồ thị lấp đúng chỗ đó.
- Trích thực thể TẤT ĐỊNH bằng regex từ khuôn `- Tên: định nghĩa` và infobox —
  không tốn API. Khi nào cách này đủ, khi nào phải cần LLM.
- Khoá thực thể phải kèm nguồn (`trang#tên`): "Tinh Luyện" có ở hàng trăm trang
  vũ khí và mỗi trang là một thực thể khác nhau.
- Khớp tên dài trước ("Đóng Băng" trước "Băng") — sai thứ tự là gán nhầm hàng
  loạt.
- Các kiểu mở rộng ngữ cảnh và điều kiện kích hoạt từng kiểu: theo quan hệ
  `tương_tự` (câu so sánh), theo thực thể được nhắc (mọi câu), nạp trọn trang
  (câu đếm/liệt kê), và nguyên tắc thép: chỉ chạy SAU ngưỡng — đồ thị không
  bao giờ được cứu câu đáng bị từ chối.
- Đường tất định: câu "bao nhiêu X / liệt kê X" trả lời bằng SQL đếm thực thể
  cấp trang — 0 API, không thể bịa. Và giới hạn của nó: chỉ đúng với loại
  danh-mục (một trang = một cá thể), sai với loại khái niệm.

**Trong dự án:** `rag/graph.py`, `rag/counting.py`, `06_build_graph.py`,
`07_test_graph_verify.py`.

**Bài học đã trả giá:** phiên bản đầu nối 880 nghìn cạnh vì mỗi lượt nhắc tên
sinh một mention cho MỌI thực thể trùng tên — đồ thị nổ tung. Và kind
"Phản Ứng Nguyên Tố" có thật trong infobox nhưng chỉ 2 trang — đếm trang mà
trả lời "có 2 phản ứng" là sai tự tin; whitelist sinh ra từ vụ đó.

---

## 9. Làm việc với LLM API (Gemini)

**Cần nắm:**
- Cấu trúc request: system instruction, contents theo lượt (`role`/`parts`),
  GenerateContentConfig, đọc hạn mức từ mã lỗi 429.
- Retry có backoff, model dự phòng khi model chính hết lượt.
- Hạn mức free tier THỰC TẾ (phải tự đo, tài liệu chính thức không đủ): giới
  hạn theo ngày, theo phút, theo token mỗi request — ba thứ khác nhau.
- Viết lại câu hỏi nối tiếp (query rewriting) bằng LLM: khi nào cần, tốn gì.

**Trong dự án:** `rag/gemini.py` (một client duy nhất), `rag/retry.py`,
`rag/config.py`, `00_smoke_test.py` (đo model nào còn sống trước khi dùng).

**Bài học đã trả giá:** khi hết hạn mức mà code fallback về gọi lẻ từng item,
một batch lỗi đốt 20 lần hạn mức thay vì 1 — nhánh xử lý lỗi cũng phải hiểu
NGUYÊN NHÂN lỗi (`except RateLimited: raise` đứng trước `except Exception`).

---

## 10. Đánh giá có kỷ luật — thứ phân biệt dự án nghiêm túc với demo

**Cần nắm:**
- Thiết kế bộ câu hỏi theo nhóm: trong phạm vi / mơ hồ / NGOÀI phạm vi (nhóm
  quan trọng nhất — đo khả năng TỪ CHỐI), suy luận, câu khó.
- Câu ngoài phạm vi khó nhất là câu CÙNG MIỀN (hỏi về game khác, hỏi doanh thu
  Genshin) — không phải câu nấu phở.
- Kỳ vọng phải ĐỐI CHIẾU TAY với nguồn trước khi ghi — và đôi khi kỳ vọng sai
  chứ không phải hệ thống sai (câu "13 phản ứng": trang nguồn không hề nói ba
  nhóm là tất cả, model từ chối mới là đúng).
- Quy trình bất di bất dịch: đo MỐC trước khi sửa → sửa MỘT thứ → đo lại →
  chạy hồi quy toàn bộ. Số xấu đi thì ghi thật, không giữ số đẹp cũ.
- Thước bão hoà (25/25) không còn chỉ ra điểm yếu — phải làm bộ khó hơn.

**Trong dự án:** `eval_questions.json` (đọc các khối `_ghi_chu_*` — chúng là
nhật ký phát hiện), `03_eval_retrieval.py`, `08_eval_answers.py`.

---

## 11. Cào dữ liệu web thực chiến

**Cần nắm:**
- MediaWiki API: `action=query`, `list=categorymembers`, `prop=revisions`,
  gọi theo lô 50 trang, phân trang bằng `continue`. TRA NGƯỢC thể loại bằng
  `prop=categories` thay vì đoán tên.
- Wikitext: template `{{...}}` lồng nhau (regex không đếm được ngoặc — phải
  viết parser đếm depth), bảng `{|...|}`, liên kết `[[A|B]]`, infobox là dữ
  liệu có cấu trúc miễn phí.
- Template có tham số mang chữ hiển thị ở vị trí KHÁC NHAU tuỳ template
  (`{{Item|Tên|30}}` → đầu; `{{Color|dendro|Tên}}` → cuối) — xoá trắng là mất
  chữ giữa câu lẫn tiêu đề mục.
- Cloudflare đời mới: chặn theo vân tay TLS (cookie đúng vẫn 403 nếu client
  không phải trình duyệt thật), phát hiện được cả Chromium của Playwright.
  Đường đi được: Chrome THẬT mở bằng lệnh thường + cổng CDP + `fetch()` chạy
  TRONG trang. Lịch sự: sleep giữa các request, ghi rõ User-Agent, tôn trọng
  giấy phép nguồn (CC BY-SA — giữ URL gốc trong manifest).
- Lọc dữ liệu TRƯỚC khi tiêu tiền embedding: ngưỡng văn xuôi, bỏ trang phụ.

**Trong dự án:** `09_fetch_fandom.py` (file dài nhất, mỗi bước lọc có số liệu
lý do), `requirements-dev.txt` (ghi nguyên đường vượt Cloudflare).

---

## 12. Streamlit và trạng thái phiên

**Cần nắm:**
- Mô hình chạy-lại-toàn-script sau MỖI thao tác — mọi trạng thái phải nằm
  trong `st.session_state`, mọi thứ đắt phải cache.
- `st.cache_data` (dữ liệu, có TTL) vs `st.cache_resource` — và vì sao dự án
  này KHÔNG cache connection (psycopg không dùng chung giữa các luồng).
- Bẫy module caching: Streamlit chạy lại script nhưng GIỮ module đã import —
  sửa bất kỳ file nào trong `rag/` là phải khởi động lại server, tải lại trang
  không đủ. Đã vấp hai lần.
- Kiểm thử headless bằng `AppTest` — hỏi thật, bấm nút thật, không cần mở
  trình duyệt.
- Lưu bền vào PostgreSQL thay vì session_state khi dữ liệu cần sống qua F5.

**Trong dự án:** `streamlit_app.py`, `rag/chatlog.py`.

---

## 13. Git, bí mật, và vận hành

**Cần nắm:**
- Nhánh theo mạch việc (`main` giữ bản cũ, `GI_Rag` là bản Genshin), commit
  message ghi VÌ SAO và số liệu kiểm chứng — đọc `git log` của repo này như
  đọc nhật ký kỹ thuật.
- Bí mật: `.env` gitignore tuyệt đối, `.env.example` commit nhưng RỖNG; lộ key
  vào hội thoại/file commit là phải thu hồi, và thứ tự thu hồi an toàn (tạo
  key mới → xác nhận chạy → mới revoke key cũ).
- `.gitignore` cho sản phẩm sinh ra được (`wiki_raw/`, kết quả eval) — kèm lý
  do ghi ngay trong file.

---

## Lộ trình học đề xuất

| Giai đoạn | Học gì | Làm gì trong repo |
|---|---|---|
| 1 | Python + SQL căn bản | Đọc `rag/store.py`, chạy `docker compose up -d`, nghịch SQL trực tiếp |
| 2 | Embedding + RAG tối giản | Đọc `rag/local_embed.py`, `rag/chunker.py`; chạy `01`, `02`, xem `03` đo gì |
| 3 | Pipeline + prompt | Đọc `rag/pipeline.py` hàm `ask()` và `rag/prompt.py`; chạy `04_chat.py` |
| 4 | Chống hallucination | Đọc `rag/verify.py`; cố tình sửa prompt cho model bịa rồi xem lớp nào bắt được |
| 5 | GraphRAG + đếm | Đọc `rag/graph.py`, `rag/counting.py`; chạy `06`, `07` |
| 6 | Đánh giá | Đọc `eval_questions.json` + `08`; tự thêm một câu khó và đoán trước kết quả |
| 7 | Cào dữ liệu | Đọc `09_fetch_fandom.py`; thử cào một thể loại nhỏ |

Nguyên tắc xuyên suốt toàn dự án, cũng là thứ đáng học nhất: **không sửa gì mà
không đo trước — đo mốc, sửa một thứ, đo lại, chạy hồi quy; số xấu thì ghi số
xấu.** Mọi con số trong README đều tái tạo được bằng script trong repo.

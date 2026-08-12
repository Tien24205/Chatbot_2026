# Chống hallucination (5 lớp) và GraphRAG

Tài liệu này mô tả hai phần kỹ thuật đi sâu của chatbot RAG: **năm lớp chống
hallucination** và **tầng đồ thị tri thức**. Mọi con số trong đây đều đo được
bằng script trong repo, không lấy từ tài liệu tham khảo.

Chạy lại để kiểm chứng:

```powershell
.venv\Scripts\python.exe 07_test_graph_verify.py     # 41 kiểm thử, không cần API key
.venv\Scripts\python.exe 08_eval_answers.py --graph  # đánh giá câu trả lời, cần API
```

---

## Phần 1 — Năm lớp chống hallucination

### Vì sao phải nhiều lớp

Câu hỏi thật sự không phải "làm sao chặn hallucination" mà là **"hallucination có
mấy loại"**. Mỗi loại lỗi lọt qua một chỗ khác nhau, nên một lớp phòng thủ dù mạnh
đến đâu cũng chỉ bịt được một loại:

```text
                        Câu trả lời sai
                               │
        ┌──────────────┬───────┴───────┬──────────────┐
        ↓              ↓               ↓              ↓
   Không có tài     Tài liệu đúng   Bịa số liệu   Suy diễn từ
   liệu liên quan   chủ đề nhưng    / bịa tên     dữ kiện đúng
                    không chứa
                    câu trả lời
        ↓              ↓               ↓              ↓
     Lớp 2          Lớp 1 + 3        Lớp 4          Lớp 5
```

### Bảng năm lớp

| # | Lớp | Chặn loại lỗi nào | Chi phí | Ở đâu |
|---|---|---|---|---|
| 1 | Ràng buộc phạm vi bằng system prompt | LLM dùng kiến thức nền thay vì ngữ cảnh | 0 | `rag/prompt.py` — `SYSTEM_PROMPT`, 8 quy tắc |
| 2 | Ngưỡng similarity hiệu chuẩn bằng đo đạc | Không có đoạn nào đủ liên quan | 0, còn **tiết kiệm** 1 lệnh gọi API | `rag/pipeline.py` — chặn trước khi gọi LLM |
| 3 | Trích dẫn có cấu trúc, kiểm tra chỉ số | Bịa nguồn: trích `[7]` khi chỉ có 4 đoạn | 0 | `rag/verify.py` — `strip_invalid_citations` |
| 4 | Kiểm chứng grounding tất định | Bịa số liệu, bịa tên riêng | 0 | `rag/verify.py` — `check` |
| 5 | Vòng kiểm chứng entailment bằng LLM | Suy diễn sai từ ngữ cảnh đúng | 1 lệnh gọi API | `rag/pipeline.py` — `entailment_check` |

Ba trong năm lớp **hoàn toàn miễn phí** và tất định. Điều này quan trọng vì hạn
mức free tier đo được chỉ khoảng 20 request: nếu mọi lớp đều phải gọi LLM thì
không lớp nào chạy được trong thực tế.

### Lớp 1 — Ràng buộc phạm vi

Tám quy tắc trong `SYSTEM_PROMPT`. Đáng chú ý nhất là hai quy tắc **đá nhau và
phải hoà giải tường minh**:

- Quy tắc 3: *"liên quan chủ đề KHÔNG có nghĩa là chứa câu trả lời"* — cần để chặn
  câu kiểu "ai mạnh nhất" khi tài liệu không hề xếp hạng.
- Quy tắc 4: **ngoại lệ của quy tắc 3** cho câu hỏi so sánh.

Đo được: khi chỉ có quy tắc 3, câu *"Liên Quân có mục tiêu nào giống Baron Nashor
không?"* bị **từ chối** dù ngữ cảnh đã có đủ cả Baron Nashor lẫn Thần Rừng — model
coi việc đặt hai dữ kiện cạnh nhau là suy diễn. Phải viết ngoại lệ ra thành quy
tắc riêng, kèm ví dụ, mới trả lời được. Bài học: prompt chống bịa quá chặt sẽ ăn
mất khả năng tổng hợp chéo tài liệu, và ranh giới đó phải được viết ra rõ ràng
chứ không thể hy vọng model tự cân.

### Lớp 2 — Ngưỡng similarity, và vì sao 0.60 chứ không phải 0.45

Ngưỡng **phải đo, không được chép**. Đo trên `gemini-embedding-001`:

| Nhóm câu hỏi | thấp nhất | trung vị | cao nhất |
|---|---|---|---|
| Trong phạm vi | 0.617 | 0.718 | 0.783 |
| Ngoài phạm vi | 0.478 | 0.510 | **0.657** |

Nền tương đồng của model này rất cao: câu *"cách nấu phở"* so với chunk thuật ngữ
game vẫn đạt 0.478. **Ngưỡng 0.45 kiểu tutorial sẽ cho qua 6/6 câu ngoài phạm vi**
— chatbot sẽ trả lời câu hỏi về phở bằng tài liệu game.

Hai nhóm **chồng lấn** (−0.040), nên không ngưỡng nào đúng 100%.

**Ngưỡng thay đổi theo số lớp phòng thủ đang có.** Đây là kết quả chính của
Milestone 6:

| | τ = 0.66 (khi ngưỡng là lớp gần như duy nhất) | τ = 0.60 (khi đã đủ 5 lớp) |
|---|---|---|
| Câu in_scope bị từ chối oan | **2** (0.617 và 0.631) | **0** |
| Câu ngoài phạm vi bị bịa | 0 | **0** |
| Câu 0.657 ("ai mạnh nhất") | chặn bởi lớp 2 | chặn bởi **lớp 1** |

Khi ngưỡng phải gánh một mình, nó buộc phải đặt cao và trả giá bằng recall. Có đủ
lớp phía sau thì hạ được ngưỡng mà không mất an toàn — **đây chính là giá trị đo
được của phòng thủ nhiều lớp**, không phải lý thuyết.

### Lớp 3 — Trích dẫn có cấu trúc

Quy tắc 8 buộc mỗi câu nêu thông tin phải kèm `[n]`. Sau khi LLM trả lời,
`strip_invalid_citations` đối chiếu mọi chỉ số với số đoạn thực sự được cung cấp:
chỉ số vượt phạm vi bị **xoá khỏi câu trả lời** và ghi cờ.

Hiển thị `[7]` cho người dùng khi chỉ có 4 nguồn còn tệ hơn không trích gì:
người đọc tưởng có một nguồn thứ 7 nào đó đã được đối chiếu.

Cùng nguyên tắc đó, danh sách nguồn **chỉ liệt kê đoạn vượt ngưỡng**, không liệt
kê toàn bộ top-k. Toàn bộ top-k vẫn được đưa vào ngữ cảnh (thừa ngữ cảnh không
hại), nhưng ghi tất cả làm "nguồn" là sai: câu trả lời về Baron Nashor từng ghi
kèm nguồn Liên Quân và thuật ngữ game, khiến người đọc tưởng đã đối chiếu chéo.

### Lớp 4 — Kiểm chứng grounding tất định

System prompt là **yêu cầu**, không phải **ràng buộc**. Quy tắc 6 ghi "không bịa
số liệu", nhưng không có gì bảo đảm model tuân theo, và khi nó không tuân thì hệ
thống hoàn toàn không biết. Lớp 4 biến yêu cầu đó thành phép kiểm tra chạy được:

- **4a — số liệu**: mọi con số trong câu trả lời phải tìm thấy trong ngữ cảnh
  (hoặc trong chính câu hỏi). Bỏ qua 0/1/2 vì chúng thường là lối diễn đạt
  ("hai đội", "một trong những") chứ không phải số liệu trích dẫn.
- **4b — thực thể**: thực thể **có thật trong knowledge base** nhưng **không có
  trong đoạn đã truy hồi**. Đây là dạng lỗi rất khó thấy bằng mắt — thông tin nghe
  đúng, chỉ là không đến từ nguồn được trích. Dựa vào tầng đồ thị (Phần 2).
- **4c — tên riêng lạ**: cụm từ hai chữ viết hoa liên tiếp trở lên không có trong
  ngữ cảnh, tức model lôi từ kiến thức nền ra.

Hai cái bẫy đã vấp khi làm lớp này, cả hai đều gây **báo động giả**:

1. Lớp ký tự `[A-ZĐÀ-Ỹ]` **không phải** là "chữ hoa tiếng Việt". Khoảng `À-Ỹ`
   trong Unicode chứa xen kẽ cả chữ thường (ỗ, ộ, ề… đều nằm giữa À và Ỹ), nên
   `"mỗi đội"` bị nhận là danh từ riêng. Phải liệt kê tường minh: khối
   U+1EA0..U+1EF8 xếp xen kẽ hoa/thường, mã **chẵn** là chữ hoa.
2. Chữ đầu câu viết hoa vì nó đứng đầu câu, không phải vì là tên riêng. Không loại
   nó ra thì `"Trong Liên Minh Huyền Thoại…"` sinh ra cặp `"Trong Liên"` và bị báo
   là tên bịa — đo được **2/5 câu trả lời đúng bị gắn cờ oan** vì lỗi này.

### Lớp 5 — Vòng kiểm chứng entailment

Gọi LLM lần thứ hai, chỉ với một việc: đối chiếu từng khẳng định trong câu trả lời
với ngữ cảnh. Prompt kiểm chứng **cố ý không kèm câu hỏi gốc** — nếu đưa vào,
model dễ chuyển sang chế độ "giúp người dùng" và bênh vực câu trả lời.

Kết quả đo thật (ngữ cảnh: chunk "Mục tiêu trung lập" của Liên Quân):

| Câu trả lời đưa vào soát | Kết quả |
|---|---|
| "Thần Rừng xuất hiện ở đường trên từ phút thứ 8 [1]." | ĐẠT |
| "…Vì vậy các trận đấu thường kết thúc ngay sau phút thứ 10, và đội nào ăn Thần Rừng chắc chắn thắng." | Bác bỏ 2 khẳng định |

Câu *"đội nào ăn Thần Rừng chắc chắn thắng"* **không chứa số nào, không chứa tên
lạ nào** — lớp 4 không thể thấy. Chỉ đọc hiểu mới thấy đó là suy diễn. Đó là lý
do lớp 5 tồn tại dù tốn thêm một lệnh gọi API.

**Leo thang có điều kiện.** Chế độ mặc định `auto`: lớp 5 chỉ chạy khi lớp 3-4
(miễn phí) đã thấy dấu hiệu khả nghi. Thứ regex bắt được thì rẻ; thứ nó không bắt
được mới đáng tiêu hạn mức. Đo trên bộ 8 câu: lớp 5 chạy **0/7** lần, tức chi phí
thực tế bằng 0 mà vẫn sẵn sàng khi cần.

**Không tự ý xoá câu.** Khi lớp 5 bác bỏ, hệ thống **ghi rõ chỗ đáng ngờ** vào
cuối câu trả lời chứ không lặng lẽ cắt bỏ. Xoá câu là tự sửa nội dung dựa trên
phán đoán của một LLM khác — vốn cũng có thể sai. Ghi ra để người đọc tự đối chiếu
với nguồn đã trích thì trung thực hơn, và giữ được phần đúng.

---

## Phần 2 — GraphRAG

### Vì sao cần đồ thị khi vector đã đạt recall@1 = 100%

Con số 100% đó đo trên các câu hỏi **đơn, tự đủ nghĩa**. Truy hồi thuần vector
hỏng ở đúng hai chỗ, cả hai đều đo được:

1. **Câu nối tiếp**: sau khi hỏi Baron Nashor, câu *"Còn Liên Quân thì sao?"* truy
   hồi ra Tổng quan / Xếp hạng / Giải đấu và **trượt** chunk "Mục tiêu trung lập".
2. **Câu so sánh chéo game**: *"Liên Quân có con nào giống Baron Nashor không?"*.
   Vector chỉ trả về chunk **giống câu hỏi về mặt từ ngữ**, mà "Baron Nashor" và
   "Thần Rừng" **không giống nhau một chữ nào** — chúng chỉ giống nhau về **vai trò**.

Cả hai đều là **quan hệ giữa các thực thể**, không phải độ tương đồng văn bản.
Vector không biểu diễn được quan hệ; đồ thị thì có.

### Trích thực thể tất định, không gọi LLM

Knowledge base viết theo đúng một khuôn `- <Tên> (<bí danh>): <định nghĩa>`, nên
regex đọc được chính xác. Không tiêu hạn mức API vào việc mà regex làm được — nếu
sau này KB là văn xuôi tự do thì mới cần LLM trích thực thể.

Kết quả trên 4 tài liệu / 25 chunk:

| | Số lượng |
|---|---|
| Thực thể | 82 |
| Bí danh | 40 |
| Lượt nhắc tới (thực thể ↔ chunk) | 145 |
| Cạnh | 571 |

Phân bố vai trò: 35 thuật ngữ, 9 vị trí, 7 mục tiêu trung lập, 7 nguyên tố,
7 vùng đất, 4 hoạt động, 4 nhóm tướng, 3 chế độ chơi, 3 game, 2 nâng cấp nhân vật.

### Ba loại cạnh

| Quan hệ | Số cạnh | Suy ra từ |
|---|---|---|
| `đồng_xuất_hiện` | 427 | hai thực thể cùng nằm trong một chunk |
| `thuộc_về` | 79 | thực thể → game/tài liệu chứa nó |
| `tương_tự` | 52 | **cùng vai trò nhưng khác tài liệu** |

`tương_tự` là cạnh mà vector **không thể thay thế**. Vai trò được suy từ tiêu đề
mục chứa thực thể ("## Mục tiêu trung lập" → vai trò `mục tiêu trung lập`), nên nó
phản ánh cách chính tài liệu phân loại, không phải phán đoán của ai.

```text
Baron Nashor  ──tương_tự──> Thần Rừng          (lien_quan_mobile · mục tiêu trung lập)
[lien_minh]   ──tương_tự──> Rồng Bạo Chúa      (lien_quan_mobile · mục tiêu trung lập)
              ──tương_tự──> Bùa Đỏ và Bùa Xanh (lien_quan_mobile · mục tiêu trung lập)
                                  ↓
                    chunk "Mục tiêu trung lập" của Liên Quân
                    — đúng chunk mà vector bỏ sót
```

### Đồ thị đóng góp gì vào ngữ cảnh

Hai thứ, và thứ thứ hai quan trọng không kém:

1. **Chunk bổ sung** mà vector bỏ sót, gắn nhãn `via="graph"` để phân biệt.
2. **Chính quan hệ đó**, viết thành câu và đưa vào prompt:
   *"Baron Nashor cùng vai trò 'mục tiêu trung lập' với: Thần Rừng (lien_quan_mobile),
   Rồng Bạo Chúa (lien_quan_mobile)…"*

Khối quan hệ **không phải kiến thức từ bên ngoài** — nó rút ra từ cách chính tài
liệu phân mục. Nói rõ điều đó cho model là cần thiết: nếu không, gặp câu so sánh
nó sẽ từ chối vì không đoạn nào viết sẵn câu so sánh, dù mọi dữ kiện đã nằm trong
ngữ cảnh.

### Hai ràng buộc an toàn

**1. Đồ thị không bao giờ được cứu một câu đáng bị từ chối.**
Mở rộng đồ thị chỉ chạy **sau** khi ngưỡng (lớp 2) đã cho qua. Nếu không, mọi câu
ngoài phạm vi chỉ cần nhắc đúng một tên thực thể là lọt lưới.

Kiểm chứng bằng câu bẫy trong bộ test: *"Genshin Impact có mục tiêu trung lập
giống Baron Nashor không?"* — có dạng so sánh nên đồ thị **có** mở rộng (bổ sung 2
đoạn), nhưng Genshin không hề có mục tiêu trung lập, và hệ thống vẫn **từ chối**.

**2. Chỉ gieo mầm từ thực thể người dùng nhắc trong câu hỏi**, không gieo từ thực
thể nằm trong các chunk đã truy hồi. Đo được: chunk "Mục tiêu trung lập" của Liên
Quân có nhắc cả "đường dưới", "đường trên" — gieo từ đó thì đồ thị kéo về toàn
chunk vị trí, làm loãng hẳn ngữ cảnh của câu hỏi về mục tiêu.

### Tóm tắt cộng đồng

Gom theo tài liệu nguồn. Ở KB này ranh giới cộng đồng trùng ranh giới tài liệu
(mỗi file một game) nên **không chạy Louvain/Leiden cho 25 chunk** — làm vậy chỉ
là trưng thuật toán. Dùng để trả lời "kho kiến thức có những gì" (`GET /api/graph`)
mà không tốn lệnh gọi LLM nào.

### Kết quả đo trên bộ câu hỏi đồ thị (8 câu)

```text
Nhóm                 Đạt    Tổng
ambiguous              4       4
in_scope               3       3
out_of_scope           1       1
TẤT CẢ                 8       8

Câu ngoài phạm vi bị trả lời (bịa thẳng) : 0
Câu có cờ kiểm chứng                      : 0/7
Latency (trung vị)  : truy hồi 413 ms · sinh câu trả lời 1269 ms · tổng 1718 ms
```

---

## Phần 3 — Đối chiếu với Microsoft Agent Framework / Semantic Kernel

Tham chiếu: [Adding RAG to Semantic Kernel Agents](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-rag?pivots=programming-language-csharp)

Dự án này viết bằng Python và **không dùng framework RAG** (plan §8 nguyên tắc 2:
không thêm framework quá sớm nếu mục tiêu là hiểu rõ hệ thống). Bảng dưới đối chiếu
từng khái niệm của Semantic Kernel với chỗ tương ứng trong repo.

| Semantic Kernel (C#) | Ở đây | Ghi chú |
|---|---|---|
| `ITextSearch` | `rag/store.py` → `search()` | Similarity search trên pgvector |
| `TextSearchStore<T>` | bảng `chunks` + `init.sql` | SK có schema dựng sẵn; ở đây tự định nghĩa schema, tương đương nhánh `VectorStoreTextSearch` của SK |
| `TextSearchProvider` | `rag/pipeline.py` → `ask()` | Truy hồi rồi bơm vào ngữ cảnh trước khi gọi model |
| `TextSearchProviderOptions.Top` | `pipeline.TOP_K = 4` | |
| `SearchTime = BeforeAIInvoke` | mặc định và duy nhất | Luôn truy hồi trước khi gọi LLM |
| `SearchTime = OnDemandFunctionCalling` | **chưa làm** | Sẽ để model tự quyết định lúc nào cần tìm; hiện chưa cần vì mỗi lượt đều là câu hỏi tra cứu |
| `ContextPrompt` | `prompt.build_user_message()` | |
| `IncludeCitationsPrompt` | quy tắc 8 trong `SYSTEM_PROMPT` | |
| `ContextFormatter` | `prompt.build_context()` | Đánh số `[1] [2]` + nhãn nguồn + điểm liên quan |
| `TextSearchDocument.SourceName` / `SourceLink` | `Chunk.source_name` / `section` | KB là file cục bộ nên không có link |
| `Namespaces` + `SearchNamespace` | **chưa dùng** | Lọc theo phạm vi; ở đây vai trò đó do đồ thị đảm nhiệm (lọc theo thực thể/game) |
| `AIContextProviders` (ghép nhiều nguồn ngữ cảnh) | `pipeline.ask()` ghép: vector + đồ thị + lịch sử hội thoại | |
| `WhiteboardProvider` / `mem0` (bộ nhớ) | lịch sử hội thoại trong `api.py` + viết lại câu hỏi | Giữ 6 lượt gần nhất |

**Những thứ SK không cung cấp sẵn và dự án này tự thêm:** ngưỡng similarity hiệu
chuẩn bằng đo đạc (lớp 2), kiểm chứng trích dẫn (lớp 3), kiểm chứng grounding tất
định (lớp 4), vòng entailment (lớp 5), và toàn bộ tầng đồ thị. SK lo phần *đưa
được tài liệu vào ngữ cảnh*; phần *bảo đảm câu trả lời không vượt quá tài liệu*
vẫn là việc phải tự làm.

### Đối chiếu với repo demo `Trung1234/book-store-agent`

| | book-store-agent | Dự án này |
|---|---|---|
| Framework | LangChain (Tool-Calling Agent, GraphCypherQAChain) | không framework RAG |
| Lưu trữ | Neo4j (vừa graph vừa vector index) | PostgreSQL + pgvector, đồ thị lưu bằng bảng quan hệ |
| Embedding | `all-MiniLM-L6-v2` cục bộ, 384 chiều | `gemini-embedding-001`, 1536 chiều |
| Cách dùng graph | agent tự chọn công cụ: Cypher hay semantic search | vector luôn chạy trước, đồ thị **bổ sung** khi câu hỏi cần so sánh |
| Chống hallucination | không có cơ chế riêng | 5 lớp |

Điểm khác đáng nói nhất là **vai trò của đồ thị**. Ở demo, agent tự quyết định
dùng graph hay vector — linh hoạt nhưng khó đoán, và khi agent chọn sai công cụ thì
không có gì đỡ. Ở đây vector luôn chạy trước và chịu trách nhiệm về ngưỡng từ
chối; đồ thị chỉ được phép **thêm** ngữ cảnh, không được phép **mở cổng** cho một
câu hỏi vốn phải bị từ chối. Đánh đổi: kém linh hoạt hơn, nhưng ranh giới an toàn
nằm ở một chỗ duy nhất và kiểm chứng được.

---

## Hạn chế đã biết, chưa xử lý

1. **Trích thực thể phụ thuộc khuôn viết của KB.** Regex đọc được vì knowledge
   base viết theo dạng `- Tên: định nghĩa`. Với văn xuôi tự do sẽ cần LLM trích
   thực thể, và khi đó chi phí dựng đồ thị không còn bằng 0.
2. **Đại từ có thể nối sai khi viết lại câu hỏi.** Sau hai lượt, *"Nó xuất hiện từ
   phút thứ mấy?"* được viết lại thành *"Baron Nashor… xuất hiện từ phút thứ mấy?"*
   trong khi "nó" đáng lẽ trỏ Thần Rừng. Chatbot không bịa (nói thẳng là không tìm
   thấy), nhưng đã hiểu sai ý.
3. **Cạnh `tương_tự` nối đầy đủ trong cùng một vai trò.** Baron Nashor nối cả sang
   "Bùa Đỏ và Bùa Xanh" — cùng mục "mục tiêu trung lập" nhưng vai trò thực tế khác
   hẳn. Với KB lớn hơn cần thêm trọng số theo độ giống của định nghĩa.
4. **Lớp 4c chỉ bắt được tên riêng viết hoa.** Tên riêng viết thường hoặc khái
   niệm bịa không có tên riêng thì lọt xuống lớp 5.
5. **Chưa đo lại toàn bộ 25 câu gốc ở τ = 0.60.** Đã đo nhóm out_of_scope (6/6 an
   toàn) và các câu sát biên; nhóm in_scope hạ ngưỡng chỉ có thể tốt lên nên chưa
   chạy lại để tiết kiệm hạn mức.

# Hỏi – đáp về dự án

Tổng hợp các câu hỏi thường gặp khi tìm hiểu hoặc trình bày dự án, kèm câu trả
lời ngắn gọn dựa trên số liệu ĐÃ ĐO trong repo. Mọi con số đều tái tạo được
bằng script tương ứng.

---

## A. Tổng quan

**H: Dự án này làm gì?**
Đ: Chatbot hỏi đáp tiếng Việt trên knowledge base riêng (wiki Genshin Impact,
1.125 tài liệu / 2.859 chunk) theo kiến trúc RAG, với trọng tâm là **chống
hallucination bằng năm lớp** và một **tầng đồ thị tri thức** đặt cạnh tầng
vector. Đo đầu-cuối: 25/25 câu chuẩn, 0 câu ngoài phạm vi bị bịa.

**H: Vì sao không dùng LangChain / LlamaIndex?**
Đ: Để mỗi tầng đều nhìn thấy được và đo được. Framework che mất các quyết định
quan trọng (cắt chunk thế nào, ngưỡng bao nhiêu, prompt ra sao) sau các lớp
trừu tượng — mà chính các quyết định đó là nội dung của dự án. Toàn bộ pipeline
viết bằng Python thuần, mỗi hằng số có số liệu lý do đi kèm.

**H: Công nghệ chính?**
Đ: Python 3.11 · PostgreSQL + pgvector (Docker) · sentence-transformers với
`intfloat/multilingual-e5-base` chạy trên máy (embedding, 768 chiều) · Gemini
`flash-lite` (sinh câu trả lời) · Streamlit (giao diện) · FastAPI (REST tuỳ
chọn).

**H: Vì sao chọn dữ liệu Genshin Impact?**
Đ: Cần một kho tri thức tiếng Việt đủ lớn, có cấu trúc (infobox, thể loại),
giấy phép rõ ràng (CC BY-SA), và người làm đủ hiểu miền để tự thẩm định câu
trả lời. Wiki Fandom tiếng Việt đáp ứng cả bốn.

---

## B. Dữ liệu

**H: Dữ liệu lấy về bằng cách nào?**
Đ: `09_fetch_fandom.py` gọi MediaWiki API theo lô 50 trang, chọn thể loại bằng
cách TRA NGƯỢC (lấy vài trang cốt lõi rồi hỏi API chúng thuộc thể loại nào —
đoán tên thể loại đã từng trượt cả Mondstadt lẫn Phản Ứng Nguyên Tố). Tải
1.412 trang, ba bộ lọc offline giữ lại 1.123 (bỏ trang phụ, trang dưới 300 ký
tự văn xuôi).

**H: Wikitext bẩn thế nào và xử lý ra sao?**
Đ: Template lồng nhau phải bóc bằng parser đếm ngoặc (regex không đếm được);
template có nghĩa (`{{Hỏa}}`, `{{Item|Hoa Sự Sống|30}}`, `{{Color|dendro|Sinh
Trưởng}}`) được thay bằng chữ theo VỊ TRÍ THAM SỐ đúng của từng loại — xoá
trắng như bản đầu làm mất 5 tên vị trí Thánh Di Vật và 4 tiêu đề mục. Bảng
`{|...|}` hiện bị lược — đây là giới hạn đã biết (xem mục F).

**H: Cào lại dữ liệu có khó không?**
Đ: Cloudflare của Fandom chặn theo vân tay TLS — httpx/curl 403 mọi kiểu
header, cookie đúng cũng vô dụng, Chromium của Playwright bị thách thức vĩnh
viễn. Đường duy nhất đo được: mở Chrome THẬT bằng lệnh thường với cổng CDP,
rồi chạy `fetch()` ngay trong trang. Chỉ dẫn đầy đủ nằm trong
`requirements-dev.txt`.

**H: Bản quyền dữ liệu?**
Đ: Nội dung wiki thuộc CC BY-SA; bản phái sinh trong `knowledge_base/` giữ
nguyên giấy phép, URL từng trang gốc ghi trong `_manifest.json`.

---

## C. Truy hồi (retrieval)

**H: Vì sao embedding chạy trên máy thay vì gọi API?**
Đ: Hạn mức embedding free của Gemini là 1.000 request/NGÀY tính theo tài khoản
(key mới không thêm hạn mức), trong khi kho có 2.859 chunk và MỖI câu hỏi cũng
tốn một request embed. Chuyển sang e5 local: nạp kho và hỏi đều 0 đồng, truy
hồi ~100 ms.

**H: Vì sao là `multilingual-e5-base` mà không phải model phổ biến trong
tutorial?**
Đ: `all-MiniLM-L6-v2` gần như thuần tiếng Anh — với tiếng Việt là hỏng từ gốc.
e5-base đa ngữ, 768 chiều (dưới trần index 2000 của pgvector), và cần nhớ tiền
tố `query:`/`passage:` của họ e5.

**H: TOP_K = 6 lấy đâu ra?**
Đ: Đo bằng `03_eval_retrieval.py`: recall@1 = 53%, recall@5 = 88% — chunk đúng
thường CÓ trong kho nhưng không phải hạng nhất, lấy 4 đoạn là vứt đi phần lớn
khoảng cách đó. KB cũ 65 chunk thì k=4 đủ vì recall@1 đã 100%; kho 2.859 chunk
thì không.

**H: Ngưỡng 0.82 lấy đâu ra?**
Đ: Cũng đo. Phân bố điểm hai nhóm trong/ngoài phạm vi chồng lấn 0.028 — ngưỡng
đơn thuần không tách nổi "đúng chủ đề" khỏi "có câu trả lời". `03` đề xuất
τ=0.89 cho 0 câu bịa nhưng độ chính xác toàn cục rơi còn 36% vì từ chối oan.
Vì có 5 lớp phòng thủ nên chọn 0.82 và để các lớp sau làm việc: `03` dự báo 12
câu bịa lọt, `08` đo thực tế **0**.

---

## D. Chống hallucination

**H: Năm lớp là những gì?**
Đ: (1) System prompt ràng buộc phạm vi; (2) ngưỡng similarity — chặn TRƯỚC khi
gọi LLM, còn tiết kiệm tiền; (3) trích dẫn `[n]` có cấu trúc + kiểm chỉ số;
(4) kiểm grounding tất định bằng regex — số, tên riêng, thực thể phải có trong
ngữ cảnh; (5) vòng entailment bằng LLM thứ hai, chỉ leo thang khi lớp 3-4 nghi
ngờ. Ba lớp giữa chạy tất định, 0 chi phí.

**H: Vì sao cần cả 5? Một ngưỡng tốt không đủ sao?**
Đ: Không — mỗi lớp chặn một LOẠI lỗi khác nhau. Ví dụ đo được: "Ai là nhân vật
mạnh nhất?" đạt điểm trên ngưỡng (đúng chủ đề!) nhưng kho không xếp hạng — chỉ
prompt + lớp 4-5 mới chặn được kiểu này. Ngược lại câu ngoài miền thì ngưỡng
chặn rẻ nhất. Nguyên tắc rút ra: ngưỡng là hàm của số lớp phòng thủ.

**H: Khi máy kiểm chứng nghi ngờ thì nó sửa câu trả lời à?**
Đ: Không — chỉ GẮN CỜ và hiển thị cho người đọc tự đối chiếu. Máy kiểm chứng
cũng có thể sai; lặng lẽ xoá là tự ý sửa nội dung dựa trên phán đoán của một
LLM khác.

**H: Chatbot có phân biệt được các kiểu từ chối không?**
Đ: Có, và bắt buộc phải phân biệt: từ chối do NGƯỠNG (không tài liệu nào đủ
liên quan) khác từ chối do MODEL (tài liệu liên quan nhưng không chứa đáp án).
Giao diện ghi rõ từng loại — từng có bug hiển thị "0.872 < ngưỡng 0.82" vừa
sai số học vừa đổ lỗi nhầm.

---

## E. GraphRAG, suy luận và đếm

**H: Đồ thị tri thức để làm gì khi đã có vector search?**
Đ: Vector chỉ thấy độ giống câu chữ. "Kiếm Sắt Đen" và "Ánh Trăng Xiphos"
không giống nhau một chữ nào nhưng cùng là Kiếm Đơn — quan hệ đó chỉ đồ thị
biết. Đồ thị được trích TẤT ĐỊNH bằng regex từ khuôn tài liệu (2.600 thực thể,
24.079 cạnh, 0 lệnh gọi API) và dùng cho: mở rộng câu so sánh, kéo về chunk
nhắc tới thực thể trong câu hỏi, và trả lời thẳng câu đếm/liệt kê.

**H: "Có bao nhiêu Kiếm Đơn?" được trả lời thế nào?**
Đ: Không qua LLM. Câu khớp đúng một loại trong whitelist thì đếm bằng một câu
SQL trên thực thể cấp-trang: 53 Kiếm Đơn, trả lời trong 2 ms, không thể bịa.
"Liệt kê các nhân vật chơi được" cùng cơ chế — ra đủ 120 tên. Whitelist chỉ
nhận loại danh-mục (một trang = một cá thể); loại khái niệm như "phản ứng
nguyên tố" cố tình KHÔNG vào whitelist vì đếm trang sẽ ra 2 (chỉ 2 phản ứng có
trang riêng) — sai một cách tự tin.

**H: RAG có suy luận được không (đếm, cộng, đa bước)?**
Đ: Có, theo hợp đồng "suy luận có đánh dấu": model được đếm/cộng từ dữ kiện
trong ngữ cảnh nhưng phải lộ phép tính theo khuôn `"Suy ra từ [1][2]: 9 + 2 +
2 = 13 loại"`. Lớp 4 miễn kiểm số cho câu đó, ĐỔI LẠI lớp 5 bắt buộc chạy để
kiểm phép tính — nới lớp này thì siết lớp khác. Câu hỏi đa bước ("Mondstadt
thuộc nguyên tố nào, nguyên tố đó + Hỏa ra phản ứng gì?" → Phong, Khuếch Tán)
trả lời được nhờ ngữ cảnh gom từ nhiều trang.

**H: Vì sao "có tổng cộng bao nhiêu phản ứng nguyên tố?" bị từ chối — đó là
lỗi à?**
Đ: Không, và đây là câu đáng tự hào nhất: trang nguồn liệt kê ba nhóm (9+2+2)
nhưng KHÔNG HỀ nói đó là tất cả — Kết Tinh, Đóng Băng nằm ngoài cả ba nhóm.
Model từ chối vì "danh sách chưa trọn vẹn" là tuân đúng luật. Kỳ vọng "13" của
chính bộ eval mới là chỗ đáng ngờ. Câu này được giữ làm thước đo tính thận
trọng.

---

## F. Đánh giá và giới hạn

**H: Kết quả đo cuối cùng?**
Đ: Bộ chính 25 câu: **25/25** (trong phạm vi 13/13, mơ hồ 4/4, ngoài phạm vi
từ chối 8/8). Nhóm suy luận: 6/7. Bộ câu khó: 10/11. **0 câu bịa ở cả ba bộ.**
Latency trung vị ~1,3 giây; câu đếm 2 ms.

**H: 25/25 nghĩa là hoàn hảo?**
Đ: Không — nghĩa là THƯỚC ĐÃ BÃO HOÀ. Vì vậy mới có bộ câu khó (`--stress`):
nó tìm ra đúng hai điểm yếu (so sánh ba bên thiếu một bên trong ngữ cảnh — đã
sửa bằng nguyên tắc "đủ mặt"; thuật ngữ sâu "Ấn Ngầm" điểm 0.791 dưới ngưỡng —
giữ làm giới hạn đã biết).

**H: Quy trình sửa lỗi của dự án?**
Đ: Bất di bất dịch: đo MỐC trước khi sửa → sửa MỘT thứ → đo lại → chạy hồi quy
toàn bộ → số xấu thì ghi số xấu. Ví dụ thật: sau refetch điểm rơi 22→21, ghi
thẳng vào commit thay vì giữ số đẹp cũ; rồi chẩn đoán từng câu nâng lên 24 và
25 bằng bốn bản vá riêng biệt, mỗi bản vá một lần đo.

**H: Giới hạn hiện tại?**
Đ: (1) Bảng wiki `{|...|}` bị lược khi cào — chỉ số theo cấp, hiệu ứng bộ
Thánh Di Vật dạng bảng chưa có trong kho; (2) thuật ngữ quá sâu có thể rơi
dưới ngưỡng ("Ấn Ngầm" 0.791); (3) con số đếm phản ánh kho tài liệu đã cào,
không nhất thiết bằng tổng trong game; (4) model chat free (flash-lite) đôi
khi tóm tắt mạnh tay — các lớp kiểm chứng bắt được bịa chứ không bắt được bỏ
sót ý.

**H: Từ điển tiếng lóng là gì và vì sao cần?**
Đ: Người chơi nói "bảo hiểm", wiki viết "Đảm Bảo 5: lần thứ 90 chắc chắn sẽ
ra" — lệch từ vựng làm điểm truy hồi rơi sát ngưỡng và câu trả lời chập chờn.
`_SLANG` nối chú giải vào câu truy vấn (điểm 0.824 → 0.847, ổn định 3/3 lần).
Kỷ luật: từng mục phải grep thấy vế phải trong kho; "banner"/"quay"/"gacha"
cố tình không thêm vì wiki tự dùng các từ đó.

---

## G. Vận hành

**H: Chạy chatbot thế nào?**
Đ: `docker compose up -d` (database) rồi
`.venv\Scripts\python.exe -m streamlit run streamlit_app.py` → mở
http://localhost:8501. Câu đầu tiên chậm ~20 giây (nạp model e5), các câu sau
~1 giây.

**H: Cần API key gì?**
Đ: Chỉ một key Gemini trong `.env` cho phần SINH câu trả lời. Truy hồi, đếm,
liệt kê chạy hoàn toàn trên máy. `.env` nằm trong `.gitignore`; `.env.example`
được commit nhưng để trống key.

**H: Lịch sử hội thoại lưu ở đâu?**
Đ: Bảng `chat_turns` trong PostgreSQL (`rag/chatlog.py`) — sống qua F5 và cả
khi tắt server. Thanh bên liệt kê phiên cũ, mở lại chat tiếp được. Lưu CẢ lượt
bị từ chối (để xem lại trung thực), khác với lịch sử cho LLM (bỏ lượt từ chối
để model không "từ chối lây").

**H: Sửa code xong thấy giao diện vẫn như cũ?**
Đ: Bẫy kinh điển của Streamlit: nó chạy lại script sau mỗi thao tác nhưng GIỮ
module đã import. Sửa bất kỳ file nào trong `rag/` là phải khởi động lại
server, tải lại trang không đủ.

**H: Cấu trúc repo đọc từ đâu?**
Đ: Các script đánh số `00 → 09` theo đúng thứ tự xây dựng — đọc theo số là
theo lộ trình. Logic dùng chung nằm trong `rag/`. Ba tài liệu trong `docs/`:
kỹ thuật chống hallucination, kế hoạch gốc, và kiến thức cần học.

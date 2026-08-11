"""
BƯỚC 9 — System prompt và lắp ghép ngữ cảnh.

Đây là LỚP CHỐNG BỊA THỨ HAI. Lớp thứ nhất là ngưỡng similarity (chặn trước khi
gọi LLM). Cần cả hai vì chúng chặn hai loại lỗi khác nhau:

  - Ngưỡng chặn câu hỏi KHÔNG CÓ chunk nào đủ liên quan.
  - System prompt chặn trường hợp chunk đủ liên quan về CHỦ ĐỀ nhưng không chứa
    câu trả lời. Bước 6 đã đo được đúng trường hợp này: "Ai là nhân vật mạnh nhất
    trong game?" đạt 0.657 — đúng chủ đề game, nhưng knowledge base không hề xếp
    hạng sức mạnh. Ngưỡng đơn thuần không phân biệt được.
"""

from __future__ import annotations

from .store import SearchHit

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu kiến thức về game, trả lời bằng tiếng Việt.

QUY TẮC BẮT BUỘC:

1. Chỉ trả lời dựa trên phần NGỮ CẢNH được cung cấp trong mỗi câu hỏi. Không dùng
   kiến thức có sẵn của bạn về game, kể cả khi bạn chắc chắn là đúng.

2. Nếu ngữ cảnh không chứa thông tin để trả lời, hãy nói thẳng:
   "Tôi không tìm thấy thông tin này trong tài liệu hiện có."
   Sau đó có thể gợi ý người dùng hỏi lại theo hướng khác. Tuyệt đối không suy đoán,
   không lấp chỗ trống bằng kiến thức bên ngoài.

3. Ngữ cảnh liên quan tới chủ đề KHÔNG có nghĩa là nó chứa câu trả lời. Ví dụ, nếu
   được hỏi "nhân vật nào mạnh nhất" mà ngữ cảnh chỉ mô tả cơ chế game chứ không xếp
   hạng sức mạnh, hãy nói rằng tài liệu không xếp hạng — đừng tự suy ra.

4. NGOẠI LỆ của quy tắc 3 — câu hỏi SO SÁNH, ĐỐI CHIẾU, "có gì giống/khác":
   Đặt hai thông tin đều CÓ SẴN trong ngữ cảnh cạnh nhau không phải là suy diễn.
   Nếu ngữ cảnh có dữ liệu của cả hai bên thì phải trả lời: nêu dữ liệu từng bên
   kèm chỉ số, rồi chỉ ra điểm giống/khác rút ra từ chính những dữ liệu đó.
   Chỉ từ chối khi ngữ cảnh THIẾU HẲN một bên — khi đó nói rõ tài liệu chỉ có
   bên nào, không đoán bên còn lại.
   Ví dụ: hỏi "Liên Quân có mục tiêu nào giống Baron Nashor không", mà ngữ cảnh có
   cả Baron Nashor lẫn Thần Rừng, thì phải mô tả cả hai và chỉ ra chỗ giống nhau —
   dù không đoạn nào viết sẵn câu so sánh đó.

5. Trả lời ngắn gọn, đúng trọng tâm. Không lặp lại câu hỏi, không mở đầu bằng
   "Dựa trên ngữ cảnh...". Đi thẳng vào câu trả lời.

6. Không bịa số liệu. Nếu ngữ cảnh ghi "160 Nguyên Thạch" thì dùng đúng con số đó;
   nếu không có con số, đừng đưa ra con số nào.

7. Nếu ngữ cảnh có nhiều game cùng khớp (ví dụ cả Liên Minh lẫn Liên Quân đều có
   rồng), hãy nêu rõ từng game riêng thay vì gộp chung thành một câu trả lời.

8. Mỗi câu nêu thông tin phải kèm chỉ số đoạn đã lấy thông tin đó, dạng [1], [2].
   Chỉ dùng đúng những chỉ số có trong ngữ cảnh. Không gộp kiểu [1-3], không bịa
   thêm chỉ số. Câu dẫn dắt hoặc câu kết luận chung thì không cần chỉ số."""


# Câu mở đầu của lời từ chối, dùng để NHẬN RA khi chính LLM tự từ chối (quy tắc 2
# bắt nó nói đúng câu này). Cần nhận ra vì hai lối từ chối phải được đối xử như
# nhau: không trích nguồn, không ghi vào lịch sử hội thoại.
REFUSAL_PREFIX = "Tôi không tìm thấy thông tin này"

REFUSAL_MESSAGE = (
    "Tôi không tìm thấy thông tin này trong tài liệu hiện có.\n\n"
    "Kho kiến thức hiện tại chỉ gồm: Liên Minh Huyền Thoại, Genshin Impact, "
    "Liên Quân Mobile và các thuật ngữ game thông dụng."
)


# --- LỚP CHỐNG BỊA THỨ NĂM: vòng kiểm chứng entailment ---------------------
# Prompt riêng, KHÔNG kèm câu hỏi gốc và không kèm system prompt trả lời. Cố ý:
# nếu đưa cả câu hỏi vào, model dễ chuyển sang chế độ "giúp người dùng" và bênh
# vực câu trả lời. Ở đây nó chỉ có đúng một việc: đối chiếu văn bản với văn bản.
ENTAILMENT_PROMPT = """Bạn là người soát lỗi. Nhiệm vụ: kiểm tra từng khẳng định
trong CÂU TRẢ LỜI có được NGỮ CẢNH chứng minh hay không.

Quy tắc:
- Chỉ dựa vào NGỮ CẢNH. Kiến thức riêng của bạn không có giá trị ở đây.
- Một khẳng định chỉ "được chứng minh" khi ngữ cảnh nói đúng điều đó, không phải
  khi ngữ cảnh nói điều gần giống hoặc cho phép suy ra.
- Diễn đạt lại bằng từ khác vẫn tính là được chứng minh.

NGỮ CẢNH:
{context}

CÂU TRẢ LỜI CẦN SOÁT:
{answer}

Nếu MỌI khẳng định đều được chứng minh, xuất đúng một từ: ĐẠT
Nếu không, liệt kê từng khẳng định chưa được chứng minh, mỗi dòng bắt đầu bằng "- ".
Không giải thích gì thêm."""


def build_context(hits: list[SearchHit]) -> str:
    """
    Ghép các chunk thành khối ngữ cảnh, mỗi đoạn có nhãn nguồn.

    Đánh số [1], [2]... để câu trả lời có thể tham chiếu, và để khi đọc log ta biết
    LLM đã được đưa những gì.
    """
    blocks = []
    for i, h in enumerate(hits, 1):
        # Ghi rõ đoạn nào đến từ đồ thị: nó có điểm similarity thấp một cách bình
        # thường (được chọn vì quan hệ, không vì giống câu hỏi), nên nếu không chú
        # thích thì con số 0.4x nằm cạnh các đoạn 0.7x trông như lỗi.
        via = " · bổ sung qua đồ thị tri thức" if h.via == "graph" else ""
        blocks.append(
            f"[{i}] (nguồn: {h.source_name} · mục: {h.section} · "
            f"độ liên quan: {h.similarity:.3f}{via})\n"
            f"{h.content}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_message(
    question: str, hits: list[SearchHit], graph_notes: list[str] | None = None
) -> str:
    """
    Ghép ngữ cảnh + quan hệ từ đồ thị + câu hỏi.

    Khối quan hệ KHÔNG phải là kiến thức thêm từ bên ngoài: nó rút ra từ cách
    chính tài liệu phân mục (Baron Nashor và Thần Rừng đều nằm dưới mục "mục tiêu
    trung lập" của game mình). Nói rõ điều đó cho model là cần thiết — nếu không,
    gặp câu so sánh nó sẽ từ chối vì không đoạn nào viết sẵn câu so sánh, dù mọi
    dữ kiện để so sánh đều đã nằm trong ngữ cảnh. Đo được đúng như vậy.
    """
    parts = [f"NGỮ CẢNH:\n\n{build_context(hits)}"]
    if graph_notes:
        parts.append(
            "QUAN HỆ RÚT TỪ CẤU TRÚC TÀI LIỆU (dùng để đối chiếu, không phải nguồn "
            "để trích dẫn):\n" + "\n".join(f"- {n}" for n in graph_notes)
        )
    parts.append(f"{'=' * 60}\n\nCÂU HỎI: {question}")
    return "\n\n".join(parts)


def citations(hits: list[SearchHit], min_similarity: float) -> list[str]:
    """
    Danh sách nguồn duy nhất, giữ nguyên thứ tự độ liên quan.

    CHỈ trích những chunk vượt ngưỡng tin cậy, không trích toàn bộ top-k. Toàn bộ
    top-k vẫn được đưa vào ngữ cảnh cho LLM (thừa ngữ cảnh không hại), nhưng liệt
    kê tất cả làm "nguồn" thì sai: câu trả lời về Baron Nashor lấy từ tài liệu Liên
    Minh mà lại ghi thêm nguồn Liên Quân và thuật ngữ game, khiến người đọc tưởng
    thông tin đã được đối chiếu chéo. Trích nguồn sai còn tệ hơn không trích.

    Luôn trả về ít nhất chunk hạng 1 — hàm này chỉ được gọi khi đã qua ngưỡng.

    Ngoại lệ: chunk do tầng đồ thị đề cử (via="graph") luôn được trích dù điểm
    similarity thấp — điểm thấp chính là LÝ DO vector bỏ sót nó. Nó được chọn vì
    một quan hệ tường minh trong đồ thị, và quan hệ đó cũng là một căn cứ.
    """
    seen, out = set(), []
    for h in hits:
        if h.via != "graph" and h.similarity < min_similarity and out:
            break
        label = f"{h.source_name} · {h.section}"
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out

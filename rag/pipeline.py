"""
BƯỚC 7-8 — Query pipeline: câu hỏi -> embedding -> retrieval -> ngưỡng -> LLM -> câu trả lời.

Tách hẳn khỏi ingestion pipeline (plan §8.7). Ingestion chạy một lần khi nạp dữ
liệu; query pipeline chạy mỗi lượt hỏi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import psycopg
from google.genai import types

from . import config, counting, embedder, graph, prompt, verify
from .gemini import client
from .retry import RateLimited, with_retry
from .store import SearchHit, chunks_of_source, hits_for_ids, search

_lexicon: graph.Lexicon | None = None

# Nâng 4 -> 6 khi knowledge base lên 2.848 chunk.
#
# Đo bằng 03_eval_retrieval.py trên KB wiki Genshin:
#   recall@1 = 53%   recall@3 = 82%   recall@5 = 88%
# Chunk đúng thường CÓ trong kho nhưng không phải hạng nhất — lấy 4 đoạn là tự
# vứt đi phần lớn khoảng cách giữa 53% và 88%. Với KB viết tay 65 chunk thì
# recall@1 đã là 100% nên 4 là đủ; kho lớn hơn thì không.
TOP_K = 6

# Số lượt hỏi-đáp giữ lại trong lịch sử. Giới hạn để prompt không phình vô hạn
# qua nhiều lượt, mà vẫn đủ để viết lại được câu hỏi có đại từ.
MAX_HISTORY_TURNS = 6

# Câu ĐẾM/TỔNG HỢP: đáp án nằm rải trong nhiều mục của MỘT trang, mà top-k chỉ
# vớt được vài mục. Đo được: hỏi "có bao nhiêu phản ứng nguyên tố?", mục "Phản
# Ứng Chuyển Hóa" (chứa 9/13 cái tên) xếp hạng 15 — không bao giờ vào top-6.
# Gặp dạng câu này thì nạp thêm TRỌN TRANG của hit hạng 1 vào ngữ cảnh.
_AGGREGATE_MARKERS = ("bao nhiêu", "tổng cộng", "tất cả", "liệt kê", "gồm những", "có những")
# Trần mở rộng: trang wiki hiện tại trung bình ~2,5 chunk nhưng trang hệ thống
# lớn có thể 16 chunk (Thuyết Định Lượng Nguyên Tố). 12 chunk ~ 14k ký tự, đủ
# trọn mọi trang trừ vài trang ngoại cỡ, mà không làm prompt phình gấp ba.
MAX_DOC_CHUNKS = 12


def needs_doc_expansion(question: str) -> bool:
    q = question.lower()
    return any(m in q for m in _AGGREGATE_MARKERS)


def expand_with_document(
    conn: psycopg.Connection, hits: list[SearchHit], query: str
) -> list[SearchHit]:
    """
    Nạp phần còn lại của MỘT tài liệu — xem _AGGREGATE_MARKERS.

    Chọn tài liệu theo TIÊU ĐỀ TRANG nằm trong câu hỏi trước, hit hạng 1 sau.
    Đo được vì sao phải vậy: hỏi "có bao nhiêu phản ứng nguyên tố?", hạng 1 là
    trang Hỏa (nói về phản ứng CỦA Hỏa) — nạp trọn trang Hỏa thì vẫn thiếu ba
    danh sách nằm ở trang "Phản Ứng Nguyên Tố", đúng trang mang tên câu hỏi.
    """
    q = query.lower()
    target = next(
        (h.source_name for h in hits if h.source_name.rsplit(".", 1)[0].lower() in q),
        hits[0].source_name,
    )
    have = {h.id for h in hits}
    return chunks_of_source(conn, target, have, MAX_DOC_CHUNKS)


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    refused: bool = False
    # AI đã từ chối: "threshold" = lớp 2 chặn trước khi gọi LLM;
    #                "model"     = ngữ cảnh đủ liên quan nhưng LLM nói không có
    #                              câu trả lời trong đó (quy tắc 2 của system prompt).
    # Phải phân biệt, nếu không giao diện sẽ báo "0.872 < ngưỡng 0.82" — một câu
    # vừa sai số học vừa đổ lỗi nhầm cho ngưỡng.
    refused_by: str = ""
    top_similarity: float = 0.0
    rewrite_ms: int = 0  # gọi LLM viết lại câu hỏi (chỉ khi có lịch sử)
    retrieval_ms: int = 0  # embed câu hỏi + truy vấn pgvector
    llm_ms: int = 0  # gọi LLM sinh câu trả lời
    verify_ms: int = 0  # lớp 5 — chỉ khác 0 khi vòng entailment thực sự chạy
    search_query: str = ""  # câu thực sự dùng để tìm kiếm (có thể đã viết lại)
    model_used: str = ""  # có thể là model dự phòng nếu model chính hết hạn mức

    # --- Kết quả kiểm chứng (lớp 3-4-5) ---
    flags: list[str] = field(default_factory=list)  # lớp 3-4, tất định
    unsupported: list[str] = field(default_factory=list)  # lớp 5, do LLM soát
    entailment_ran: bool = False
    graph_expanded: int = 0  # số chunk do đồ thị bổ sung
    doc_expanded: int = 0  # số chunk nạp thêm khi mở rộng trọn trang (câu đếm/tổng hợp)
    counted: bool = False  # trả lời bằng đường đếm tất định (rag/counting.py), không qua LLM

    @property
    def verified(self) -> bool:
        """Không cờ nào và không khẳng định nào bị bác — chỉ khi đó mới nói là sạch."""
        return not self.flags and not self.unsupported

    @property
    def total_ms(self) -> int:
        return self.rewrite_ms + self.retrieval_ms + self.llm_ms + self.verify_ms


def remember(history: list[dict], question: str, answer: Answer) -> None:
    """
    Ghi một lượt vào lịch sử hội thoại rồi cắt bớt phần cũ.

    KHÔNG ghi lượt bị từ chối. Nếu ghi, LLM sẽ thấy mẫu "hỏi -> từ chối" trong
    lịch sử và dễ từ chối lây sang các câu sau vốn trả lời được.

    04_chat.py và api.py từng chép cùng đoạn này kèm cùng hằng số MAX_HISTORY_TURNS
    — hai bản sao của một quy tắc mà lệch nhau thì rất khó phát hiện, vì cả hai
    đều "chạy được", chỉ khác nhau ở chỗ chatbot nhớ được bao xa.
    """
    if answer.refused:
        return
    history.append({"role": "user", "parts": [{"text": question}]})
    history.append({"role": "model", "parts": [{"text": answer.text}]})
    del history[: max(0, len(history) - MAX_HISTORY_TURNS * 2)]


REWRITE_PROMPT = """Viết lại câu hỏi cuối thành một câu hỏi ĐỘC LẬP, tự nó đủ nghĩa
khi tách khỏi hội thoại.

Quy tắc:
- Thay đại từ và cách nói tắt ("nó", "cái đó", "còn X thì sao") bằng danh từ cụ thể
  lấy từ hội thoại phía trên.
- Giữ nguyên ý định của người hỏi, không thêm thông tin mới, không trả lời.
- Nếu câu hỏi vốn đã độc lập, chép lại y nguyên.
- Chỉ xuất ra đúng câu hỏi đã viết lại, không giải thích, không dấu ngoặc kép.

HỘI THOẠI:
{history}

CÂU HỎI CUỐI: {question}

CÂU HỎI ĐỘC LẬP:"""


# Dấu hiệu câu hỏi PHỤ THUỘC ngữ cảnh trước đó. Chỉ khi thấy một trong số này
# (hoặc câu quá ngắn) mới tốn thêm lượt gọi LLM để viết lại.
# Cố tình để rộng: bỏ sót một câu cần viết lại (truy hồi trượt) tệ hơn nhiều so
# với viết lại thừa một câu vốn đã đủ nghĩa (chỉ tốn thêm vài giây).
_DEPENDENT_MARKERS = (
    "nó", "đó", "này", "kia", "ấy", "vậy", "thế", "họ", "chúng",
    "còn", "thì sao", "ra sao", "tương tự", "cái đó", "cái này",
    "loại đó", "game đó", "game này", "trên", "vừa rồi", "ở đâu",
)


def needs_rewrite(question: str) -> bool:
    q = question.lower()
    if len(q) < 30:  # câu ngắn thường lược bỏ chủ ngữ
        return True
    return any(m in q for m in _DEPENDENT_MARKERS)


def rewrite_query(question: str, history: list[dict]) -> str:
    """
    Viết lại câu hỏi nối tiếp thành câu đứng độc lập trước khi truy hồi.

    Lý do: retrieval chỉ nhìn thấy văn bản câu hỏi, không thấy hội thoại. Câu
    "Còn Kết Tinh thì sao?" đứng một mình không chứa từ khoá nào của chủ đề đang
    nói, nên truy hồi trả về chunk lạc đề và trượt đúng trang Kết Tinh (đo được
    hiện tượng này trên KB cũ, cơ chế không phụ thuộc dữ liệu).

    Đánh đổi: tốn thêm một lượt gọi LLM mỗi khi có lịch sử. Chỉ chạy khi cần.
    """
    if not history or not needs_rewrite(question):
        return question

    convo = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['parts'][0]['text']}"
        for m in history[-4:]  # 2 lượt gần nhất là đủ để giải nghĩa đại từ
    )
    try:
        resp = client().models.generate_content(
            model=config.CHAT_MODEL,
            contents=REWRITE_PROMPT.format(history=convo, question=question),
            config=types.GenerateContentConfig(temperature=0.0),
        )
        rewritten = (resp.text or "").strip().strip('"')
        # Kết quả bất thường (rỗng, hoặc dài bất hợp lý) thì dùng lại câu gốc còn hơn.
        if rewritten and len(rewritten) < len(question) + 200:
            return rewritten
    except Exception:
        pass
    return question


# --- GraphRAG: mở rộng truy hồi bằng quan hệ -------------------------------

# Dấu hiệu câu hỏi cần NHIỀU HƠN một chủ thể để trả lời. Chỉ khi đó mới mở rộng
# bằng đồ thị: câu hỏi đơn đã đạt recall@1 = 100% bằng vector (đo ở Bước 6 trên
# KB 4 tài liệu), thêm chunk vào đó chỉ làm loãng ngữ cảnh và tăng nguy cơ LLM
# trộn nguồn.
_COMPARE_MARKERS = (
    "so sánh", "giống", "tương tự", "khác nhau", "khác gì", "đối chiếu",
    "cả hai", "hai game", "game khác", "bên nào", "còn", "thì sao", "tương đương",
)


def needs_graph_expansion(question: str) -> bool:
    q = question.lower()
    return any(m in q for m in _COMPARE_MARKERS)


def lexicon(conn: psycopg.Connection) -> graph.Lexicon | None:
    """Bộ dò tên thực thể, nạp một lần rồi dùng lại — nhỏ, không đáng nạp lại mỗi lượt."""
    global _lexicon
    if _lexicon is None:
        if not graph.is_built(conn):
            return None
        _lexicon = graph.load_lexicon(conn)
    return _lexicon


def expand_with_graph(
    conn: psycopg.Connection,
    question: str,
    query_vector: list[float],
    hits: list[SearchHit],
    limit: int,
) -> tuple[list[SearchHit], list[str]]:
    """
    Bổ sung chunk mà vector bỏ sót, đi qua cạnh `tương_tự` trong đồ thị.

    Trả về (chunk bổ sung, các quan hệ đã đi qua). Phần quan hệ quan trọng ngang
    phần chunk: bản thân "Kiếm Sắt Đen và Ánh Trăng Xiphos cùng vai trò Kiếm Đơn"
    đã là một dữ kiện rút từ cách tài liệu phân loại, và nó chính là thứ LLM
    cần để dám so sánh mà không phải tự suy diễn.

    HAI RÀNG BUỘC:

    1. Chỉ gieo mầm từ thực thể NGƯỜI DÙNG NHẮC TRONG CÂU HỎI, không gieo từ thực
       thể nằm trong các chunk đã truy hồi. Lý do (đo trên KB cũ): chunk trả lời
       thường nhắc kèm hàng loạt thực thể phụ — gieo từ chúng thì đồ thị kéo về
       toàn chunk lạc đề, loãng hẳn ngữ cảnh của chính câu hỏi.

    2. Chỉ chạy SAU khi ngưỡng similarity (lớp 2) đã cho qua. Đồ thị không bao giờ
       được cứu một câu vốn phải bị từ chối, nếu không thì mọi câu ngoài phạm vi
       chỉ cần nhắc đúng một tên thực thể là lọt lưới.
    """
    lex = lexicon(conn)
    if lex is None:
        return [], []

    seeds = lex.find(question)
    if not seeds:
        return [], []

    nbrs = graph.neighbors(conn, seeds, ("tương_tự",))
    if not nbrs:
        return [], []

    # Gom theo thực thể gốc để câu giải thích đọc được, thay vì liệt kê từng cạnh.
    grouped: dict[tuple[str, str], list[str]] = {}
    for n in nbrs:
        grouped.setdefault((n["from_name"], n["kind"]), []).append(
            f"{n['name']} ({n['source_name'].removesuffix('.txt')})"
        )
    notes = [
        f"{src} cùng vai trò \"{kind}\" với: {', '.join(sorted(set(dsts))[:5])}"
        for (src, kind), dsts in grouped.items()
    ]

    have = {h.id for h in hits}
    candidates = [
        c["id"]
        for c in graph.chunks_mentioning(conn, [n["name"] for n in nbrs], limit=limit + len(hits))
        if c["id"] not in have
    ][:limit]

    return hits_for_ids(conn, query_vector, candidates), notes


# --- Lớp chống bịa 5: vòng kiểm chứng entailment ---------------------------


def entailment_check(answer_text: str, hits: list[SearchHit]) -> list[str]:
    """
    Hỏi lại LLM: từng khẳng định trong câu trả lời có được ngữ cảnh chứng minh không.

    Trả về danh sách khẳng định KHÔNG được chứng minh (rỗng = đạt).

    Đây là lớp duy nhất bắt được lỗi SUY DIỄN: ngữ cảnh nói "phản ứng Bốc Hơi
    nhân sát thương 1,5 lần", câu trả lời viết "nên Bốc Hơi luôn là lựa chọn mạnh
    nhất" — mọi con số đều có thật, mọi tên đều có thật, lớp 4 không thấy gì;
    chỉ có đọc hiểu mới thấy đó là suy diễn.

    Lỗi ở lớp này KHÔNG được phép làm hỏng câu trả lời: nếu lệnh gọi kiểm chứng
    thất bại thì coi như chưa kiểm chứng, chứ không chặn câu trả lời đã sinh xong.
    """
    try:
        resp = with_retry(
            lambda: client().models.generate_content(
                model=config.CHAT_MODEL,
                contents=prompt.ENTAILMENT_PROMPT.format(
                    context=prompt.build_context(hits), answer=answer_text
                ),
                config=types.GenerateContentConfig(temperature=0.0),
            )
        )
    except Exception:
        return []

    text = (resp.text or "").strip()
    if not text or text.upper().startswith("ĐẠT"):
        return []
    return [line.lstrip("- ").strip() for line in text.split("\n") if line.strip().startswith("-")]


def ask(
    conn: psycopg.Connection,
    question: str,
    history: list[dict] | None = None,
    top_k: int = TOP_K,
) -> Answer:
    """
    Trả lời một câu hỏi.

    `history` là các lượt trước dạng [{"role": "user"|"model", "parts":[{"text":...}]}].
    Khi có lịch sử, câu hỏi được viết lại thành dạng độc lập trước khi truy hồi
    (xem rewrite_query).
    """
    # Đo tách bạch: viết lại câu hỏi là một lượt gọi LLM, không phải chi phí truy hồi.
    t0 = time.perf_counter()
    search_query = rewrite_query(question, history or [])
    rewrite_ms = int((time.perf_counter() - t0) * 1000)

    # --- Đường đếm tất định (hướng B) ---------------------------------------
    # Chạy TRƯỚC cả embedding: câu "có bao nhiêu <loại>?" khớp đúng một kind
    # trong đồ thị thì đếm bằng SQL — 0 lệnh gọi API, không thể bịa. Câu không
    # khớp (kể cả "bao nhiêu phản ứng nguyên tố" — không phải kind nào) rơi
    # xuống pipeline thường, nơi quy tắc 8 + mở rộng trọn trang xử lý tiếp.
    t1 = time.perf_counter()
    counted_text = counting.try_count(conn, search_query)
    if counted_text is not None:
        return Answer(
            text=counted_text,
            counted=True,
            top_similarity=1.0,  # khớp kind chính xác — trong phạm vi theo định nghĩa
            rewrite_ms=rewrite_ms,
            retrieval_ms=int((time.perf_counter() - t1) * 1000),
            search_query=search_query,
            model_used="đồ thị tri thức",
        )

    query_vector = embedder.embed_query(search_query)
    hits = search(conn, query_vector, top_k=top_k)

    top = hits[0].similarity if hits else 0.0

    # --- Lớp chống bịa 2: ngưỡng similarity ---------------------------------
    # Chặn TRƯỚC khi gọi LLM: vừa chắc chắn không bịa, vừa không tốn tiền gọi API.
    # Ngưỡng đo trên hit VECTOR, trước khi đồ thị bổ sung gì — xem expand_with_graph.
    if not hits or top < config.SIMILARITY_THRESHOLD:
        return Answer(
            text=prompt.REFUSAL_MESSAGE,
            refused=True,
            refused_by="threshold",
            top_similarity=top,
            hits=hits,
            rewrite_ms=rewrite_ms,
            retrieval_ms=int((time.perf_counter() - t1) * 1000),
            search_query=search_query,
        )

    # --- GraphRAG: bổ sung chunk mà vector bỏ sót ---------------------------
    graph_hits: list[SearchHit] = []
    graph_notes: list[str] = []
    if config.GRAPH_EXPANSION and needs_graph_expansion(search_query):
        graph_hits, graph_notes = expand_with_graph(
            conn, search_query, query_vector, hits, config.GRAPH_MAX_EXTRA_CHUNKS
        )
        hits = hits + graph_hits

    # --- Mở rộng trọn trang cho câu đếm/tổng hợp ----------------------------
    # Chạy SAU ngưỡng (lớp 2), cùng nguyên tắc với đồ thị: chỉ bổ sung ngữ cảnh
    # cho câu đã đáng trả lời, không bao giờ cứu câu đáng bị từ chối.
    doc_hits: list[SearchHit] = []
    doc_note = ""
    if needs_doc_expansion(search_query):
        doc_hits = expand_with_document(conn, hits, search_query)
        hits = hits + doc_hits
        if doc_hits:
            doc_note = (
                f"LƯU Ý: ngữ cảnh trên đã gồm TOÀN BỘ trang "
                f"\"{doc_hits[0].source_name.rsplit('.', 1)[0]}\". Nếu câu hỏi là đếm "
                "hay tổng hợp, hãy đếm trực tiếp từ các danh sách trong ngữ cảnh theo "
                "quy tắc 8 — đừng từ chối chỉ vì không có con số tổng viết sẵn."
            )

    retrieval_ms = int((time.perf_counter() - t1) * 1000)

    # --- Lớp chống bịa 1: system prompt ràng buộc ---------------------------
    contents = list(history or [])
    contents.append(
        {
            "role": "user",
            "parts": [{"text": prompt.build_user_message(question, hits, graph_notes, doc_note)}],
        }
    )

    gen_cfg = types.GenerateContentConfig(
        system_instruction=prompt.SYSTEM_PROMPT,
        # Nhiệt độ thấp: đây là tác vụ tra cứu, không phải sáng tác.
        temperature=0.2,
    )

    t2 = time.perf_counter()
    try:
        resp = with_retry(
            lambda: client().models.generate_content(
                model=config.CHAT_MODEL, contents=contents, config=gen_cfg
            )
        )
        model_used = config.CHAT_MODEL
    except RateLimited:
        # Hạn mức free tier tính riêng cho từng model, nên khi model chính hết
        # lượt thì model dự phòng thường vẫn còn. Thà trả lời bằng model yếu hơn
        # còn hơn báo lỗi cho người dùng.
        if not config.CHAT_MODEL_FALLBACK or config.CHAT_MODEL_FALLBACK == config.CHAT_MODEL:
            raise
        resp = with_retry(
            lambda: client().models.generate_content(
                model=config.CHAT_MODEL_FALLBACK, contents=contents, config=gen_cfg
            )
        )
        model_used = config.CHAT_MODEL_FALLBACK
    llm_ms = int((time.perf_counter() - t2) * 1000)

    text = (resp.text or "").strip() or prompt.REFUSAL_MESSAGE

    # --- Lớp chống bịa 3: trích dẫn có cấu trúc ------------------------------
    text, invalid = verify.strip_invalid_citations(text, len(hits))

    # LLM tự từ chối (quy tắc 2) tuy khác lớp 2 về cơ chế nhưng giống hệt về hệ quả:
    # không có câu trả lời nào cả. Phải trả về refused=True, nếu không thì API sẽ
    # ghi lượt này vào lịch sử và giao diện sẽ liệt kê "nguồn" cho một câu từ chối —
    # người đọc tưởng đã có tài liệu chứng minh điều gì đó.
    if text.startswith(prompt.REFUSAL_PREFIX):
        return Answer(
            text=text,
            refused=True,
            refused_by="model",
            top_similarity=top,
            hits=hits,
            rewrite_ms=rewrite_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            search_query=search_query,
            model_used=model_used,
            graph_expanded=len(graph_hits),
            doc_expanded=len(doc_hits),
        )

    # --- Lớp chống bịa 4: kiểm chứng grounding tất định ----------------------
    report = verify.check(text, hits, question=question, lexicon=lexicon(conn))
    report.invalid_citations = sorted(set(report.invalid_citations) | set(invalid))

    # --- Lớp chống bịa 5: vòng entailment (tốn API, nên leo thang có điều kiện) ---
    unsupported: list[str] = []
    entailment_ran = False
    verify_ms = 0
    if config.ENTAILMENT_CHECK == "on" or (
        config.ENTAILMENT_CHECK == "auto" and verify.escalate_needed(report)
    ):
        t3 = time.perf_counter()
        unsupported = entailment_check(text, hits)
        verify_ms = int((time.perf_counter() - t3) * 1000)
        entailment_ran = True

    # Có khẳng định không chứng minh được thì NÓI RA, không lặng lẽ xoá.
    # Xoá câu là tự ý sửa nội dung dựa trên phán đoán của chính một LLM khác —
    # vốn cũng có thể sai. Ghi rõ chỗ đáng ngờ để người đọc tự đối chiếu với
    # nguồn đã trích thì trung thực hơn, và vẫn giữ được phần đúng.
    if unsupported:
        text += "\n\n⚠ Chưa đối chiếu được với tài liệu:\n" + "\n".join(
            f"  • {c}" for c in unsupported
        )

    return Answer(
        text=text,
        citations=prompt.citations(hits, config.SIMILARITY_THRESHOLD),
        hits=hits,
        refused=False,
        top_similarity=top,
        rewrite_ms=rewrite_ms,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        verify_ms=verify_ms,
        search_query=search_query,
        model_used=model_used,
        flags=report.flags,
        unsupported=unsupported,
        entailment_ran=entailment_ran,
        graph_expanded=len(graph_hits),
        doc_expanded=len(doc_hits),
    )

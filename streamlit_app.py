"""
BƯỚC 8 (frontend) — Giao diện chat bằng Streamlit.

Chạy:
  .venv\\Scripts\\python.exe -m streamlit run streamlit_app.py

GỌI THẲNG rag.pipeline, KHÔNG đi qua HTTP tới api.py. Lý do: Streamlit tự nó đã
là một tiến trình Python đầy đủ. Bắt nó gọi HTTP về chính máy mình thì phải chạy
hai server, và phần lưu lịch sử hội thoại bị nhân đôi — api.py giữ trong dict
`_sessions`, Streamlit giữ trong `st.session_state`. Hai bản sao của cùng một
trạng thái là chỗ dễ lệch nhau nhất.

Phần dùng chung vẫn dùng chung: lịch sử hội thoại đi qua đúng pipeline.remember()
mà 04_chat.py và api.py đang dùng, nên quy tắc "không ghi lượt bị từ chối" chỉ tồn
tại ở một chỗ duy nhất.

Đây là giao diện DUY NHẤT. Bản HTML/CSS/JS cũ trong web/ đã bỏ; api.py giờ chỉ
còn là REST API thuần, vẫn chạy và vẫn được 05_test_api.py kiểm thử.
"""

from __future__ import annotations

import streamlit as st

from rag import config, graph, pipeline, store
from rag.retry import RateLimited

SUGGESTIONS = [
    "Baron Nashor là gì?",
    "Cơ chế pity trong Genshin ra sao?",
    "Thần Rừng Liên Quân xuất hiện phút mấy?",
    "Nerf nghĩa là gì?",
]

st.set_page_config(
    page_title="Chatbot Kiến Thức Game",
    page_icon="🎮",
    layout="centered",
)


# --- Trạng thái backend ----------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict:
    """
    Đếm dữ liệu đang có trong database.

    Cache 30 giây: con số này chỉ đổi khi chạy lại 02_ingest.py hoặc
    06_build_graph.py, mà Streamlit thì chạy lại toàn bộ script sau MỖI thao tác
    của người dùng — không cache thì mỗi lần gõ phím là thêm hai truy vấn.
    """
    with store.connect() as conn:
        built = graph.is_built(conn)
        return {
            "chunks": store.count(conn),
            "graph_built": built,
            "graph_entities": (
                conn.execute("SELECT count(*) FROM graph_entities").fetchone()[0] if built else 0
            ),
        }


def to_view(ans) -> dict:
    """
    Rút từ Answer ra đúng những gì cần để vẽ lại một lượt trả lời.

    Streamlit chạy lại script từ đầu sau mỗi thao tác, nên mọi lượt cũ phải vẽ
    lại từ session_state. Giữ dict thay vì giữ nguyên Answer để phần hiển thị
    không phụ thuộc vào cấu trúc bên trong của pipeline.
    """
    cited = set(ans.citations)
    return {
        "text": ans.text,
        "refused": ans.refused,
        "top_similarity": ans.top_similarity,
        "search_query": ans.search_query,
        "citations": [
            {
                "source_name": h.source_name,
                "section": h.section,
                "similarity": round(h.similarity, 4),
                "via": h.via,
            }
            for h in ans.hits
            if f"{h.source_name} · {h.section}" in cited
        ],
        "flags": ans.flags,
        "unsupported": ans.unsupported,
        "entailment_ran": ans.entailment_ran,
        "graph_expanded": ans.graph_expanded,
        "rewrite_ms": ans.rewrite_ms,
        "retrieval_ms": ans.retrieval_ms,
        "llm_ms": ans.llm_ms,
        "verify_ms": ans.verify_ms,
        "total_ms": ans.total_ms,
    }


# --- Vẽ một lượt trả lời ---------------------------------------------------


def draw_answer(view: dict, question: str) -> None:
    # Câu hỏi nối tiếp được viết lại trước khi truy hồi — nói ra cho minh bạch,
    # nếu không người dùng không hiểu vì sao "Còn Liên Quân thì sao?" lại ra
    # nguyên một đoạn về Rồng Bạo Chúa.
    if view["search_query"] and view["search_query"] != question:
        st.caption(f'Đã diễn giải câu hỏi thành: "{view["search_query"]}"')

    st.markdown(view["text"])

    if view["refused"]:
        st.caption(
            f"Không nguồn nào vượt ngưỡng tin cậy "
            f"(liên quan cao nhất {view['top_similarity']:.4f} "
            f"< ngưỡng {config.SIMILARITY_THRESHOLD})"
        )
    elif view["citations"]:
        lines = []
        for c in view["citations"]:
            # Nguồn do đồ thị bổ sung có điểm similarity thấp một cách bình thường:
            # nó được chọn vì quan hệ giữa các thực thể, không vì giống câu hỏi.
            # Không ghi rõ thì con số 0.4x nằm cạnh 0.7x trông như lỗi.
            via = " · `qua đồ thị`" if c["via"] == "graph" else ""
            lines.append(f"- {c['source_name']} · {c['section']} ({c['similarity']}){via}")
        st.caption("Nguồn:\n" + "\n".join(lines))

    # Kết quả kiểm chứng hiện CẢ KHI ĐẠT. Nếu chỉ hiện lúc có lỗi thì người đọc
    # không phân biệt được "đã soát, sạch" với "chưa soát gì".
    if not view["refused"]:
        if view["flags"] or view["unsupported"]:
            items = "\n".join(f"- {f}" for f in view["flags"] + view["unsupported"])
            st.warning("Kiểm chứng phát hiện dấu hiệu đáng ngờ:\n" + items)
        else:
            extra = " + đối chiếu entailment" if view["entailment_ran"] else ""
            st.success(f"Đã soát trích dẫn, số liệu và tên riêng{extra}")

    t = []
    if view["rewrite_ms"]:
        t.append(f"diễn giải {view['rewrite_ms']} ms")
    t.append(f"truy hồi {view['retrieval_ms']} ms")
    if view["llm_ms"]:
        t.append(f"sinh câu trả lời {view['llm_ms']} ms")
    if view["verify_ms"]:
        t.append(f"kiểm chứng {view['verify_ms']} ms")
    if view["graph_expanded"]:
        t.append(f"đồ thị bổ sung {view['graph_expanded']} đoạn")
    st.caption(f"{view['total_ms']} ms — " + " · ".join(t))


# --- Trạng thái phiên ------------------------------------------------------

st.session_state.setdefault("turns", [])  # để vẽ lại: [{"q":..., "view":...}]
st.session_state.setdefault("history", [])  # định dạng của pipeline (Gemini contents)
st.session_state.setdefault("pending", None)  # câu hỏi bấm từ nút gợi ý


# --- Thanh bên -------------------------------------------------------------

with st.sidebar:
    st.subheader("Chatbot Kiến Thức Game")

    try:
        h = health()
    except Exception as e:
        st.error(f"Không kết nối được database.\n\n{e}")
        st.code("docker compose up -d", language="powershell")
        st.stop()

    if h["chunks"] == 0:
        st.error("Database rỗng — chưa nạp dữ liệu.")
        st.code(".venv\\Scripts\\python.exe 02_ingest.py", language="powershell")
        st.stop()

    st.caption(
        f"{h['chunks']} đoạn tri thức"
        + (f" · đồ thị {h['graph_entities']} thực thể" if h["graph_built"] else "")
        + f"\n\n{config.CHAT_MODEL} · ngưỡng {config.SIMILARITY_THRESHOLD}"
    )

    if st.button("Hội thoại mới", use_container_width=True):
        st.session_state.turns = []
        st.session_state.history = []
        st.session_state.pending = None
        st.rerun()

    st.divider()
    st.caption("Thử hỏi:")
    for i, s in enumerate(SUGGESTIONS):
        if st.button(s, key=f"sug{i}", use_container_width=True):
            st.session_state.pending = s
            st.rerun()


# --- Khung chat ------------------------------------------------------------

if not st.session_state.turns:
    with st.chat_message("assistant"):
        st.markdown(
            "Xin chào. Tôi trả lời dựa trên tài liệu về **Liên Minh Huyền Thoại**, "
            "**Genshin Impact**, **Liên Quân Mobile** và thuật ngữ game.\n\n"
            "Câu nào ngoài phạm vi đó tôi sẽ nói thẳng là không có thông tin, thay vì đoán."
        )

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        draw_answer(turn["view"], turn["q"])

question = st.chat_input("Nhập câu hỏi về game…", max_chars=2000)
if st.session_state.pending:
    question, st.session_state.pending = st.session_state.pending, None

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang truy hồi và kiểm chứng…"):
                # Mở kết nối theo từng câu hỏi, giống api.py. Không dùng
                # st.cache_resource để giữ sẵn một connection: Streamlit chạy mỗi
                # phiên trên một luồng riêng, mà connection của psycopg không dùng
                # chung được giữa các luồng.
                with store.connect() as conn:
                    ans = pipeline.ask(conn, question, history=st.session_state.history)
        except RateLimited as e:
            # Chạm hạn mức nhà cung cấp, không phải hỏng hệ thống. Nói rõ để người
            # dùng biết chỉ cần chờ chứ không đi sửa gì.
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Không xử lý được câu hỏi.\n\n{e}")
            st.stop()

        view = to_view(ans)
        draw_answer(view, question)

    # Tự bỏ qua lượt bị từ chối — xem pipeline.remember.
    pipeline.remember(st.session_state.history, question, ans)
    st.session_state.turns.append({"q": question, "view": view})

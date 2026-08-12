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

Hội thoại được lưu bền trong PostgreSQL (rag/chatlog.py): thanh bên liệt kê các
hội thoại cũ, mở lại xem được và chat tiếp được — lịch sử cho LLM dựng lại qua
đúng pipeline.remember() nên quy tắc bỏ-lượt-từ-chối vẫn chỉ có một chỗ.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import streamlit as st

from rag import chatlog, config, graph, pipeline, store
from rag.retry import RateLimited

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
        "refused_by": ans.refused_by,
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
        "doc_expanded": ans.doc_expanded,
        "rewrite_ms": ans.rewrite_ms,
        "retrieval_ms": ans.retrieval_ms,
        "llm_ms": ans.llm_ms,
        "verify_ms": ans.verify_ms,
        "total_ms": ans.total_ms,
    }


# --- Vẽ một lượt trả lời ---------------------------------------------------


def draw_answer(view: dict, question: str) -> None:
    # Câu hỏi nối tiếp được viết lại trước khi truy hồi — nói ra cho minh bạch,
    # nếu không người dùng không hiểu vì sao "Còn Kết Tinh thì sao?" lại ra
    # nguyên một đoạn về phản ứng Kết Tinh.
    if view["search_query"] and view["search_query"] != question:
        st.caption(f'Đã diễn giải câu hỏi thành: "{view["search_query"]}"')

    st.markdown(view["text"])

    if view["refused"]:
        # Hai lối từ chối rất khác nhau, và nói nhầm thì người đọc hiểu sai hoàn
        # toàn vì sao không có câu trả lời.
        if view["refused_by"] == "model":
            st.caption(
                f"Có tài liệu liên quan (cao nhất {view['top_similarity']:.4f}, "
                f"trên ngưỡng {config.SIMILARITY_THRESHOLD}) nhưng không đoạn nào "
                "chứa câu trả lời — chính model nói vậy, không phải ngưỡng chặn."
            )
        else:
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
            # Tách hai nhóm: lớp 3-4 là phép kiểm TẤT ĐỊNH (trích dẫn, số liệu,
            # tên riêng), lớp 5 là nhận định của một LLM khác. Gộp chung thành một
            # danh sách khiến người đọc tưởng chúng cùng độ tin cậy.
            parts = []
            if view["flags"]:
                parts.append(
                    "**Lớp 3-4 — kiểm tra tất định:**\n"
                    + "\n".join(f"- {f}" for f in view["flags"])
                )
            if view["unsupported"]:
                parts.append(
                    "**Lớp 5 — khẳng định chưa đối chiếu được với tài liệu:**\n"
                    + "\n".join(f"- {u}" for u in view["unsupported"])
                )
            st.warning("\n\n".join(parts))
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
    # .get: các lượt lưu trong chat_turns TRƯỚC khi có mở rộng trọn trang không
    # mang khoá này — vẽ lại lịch sử cũ không được phép vỡ.
    if view.get("doc_expanded"):
        t.append(f"nạp trọn trang +{view['doc_expanded']} đoạn")
    st.caption(f"{view['total_ms']} ms — " + " · ".join(t))


# --- Trạng thái phiên ------------------------------------------------------

st.session_state.setdefault("turns", [])  # để vẽ lại: [{"q":..., "view":...}]
st.session_state.setdefault("history", [])  # định dạng của pipeline (Gemini contents)
# Mã hội thoại dùng làm khoá lưu trong chat_turns. Sinh sẵn từ lượt đầu: câu hỏi
# đầu tiên phải được lưu ngay dưới đúng mã này, không chờ tới "Hội thoại mới".
st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])


def open_session(sid: str, turns: list[dict]) -> None:
    """
    Mở lại một hội thoại đã lưu: vẽ lại các lượt và DỰNG LẠI lịch sử cho LLM.

    Phần dựng lại đi qua đúng pipeline.remember() — nó là nơi duy nhất giữ quy
    tắc "không ghi lượt bị từ chối vào lịch sử LLM". remember() chỉ đọc hai
    trường refused/text nên một SimpleNamespace là đủ đóng vai Answer.
    """
    st.session_state.turns = turns
    st.session_state.history = []
    for t in turns:
        pipeline.remember(
            st.session_state.history,
            t["q"],
            SimpleNamespace(refused=t["view"]["refused"], text=t["view"]["text"]),
        )
    st.session_state.session_id = sid


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
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()

    st.divider()
    st.caption("Hội thoại đã lưu:")

    # Không cache như health(): danh sách này phải thấy ngay lượt vừa lưu xong.
    # Một truy vấn nhỏ mỗi thao tác là cái giá chấp nhận được cho sự đúng.
    with store.connect() as conn:
        sessions = chatlog.list_sessions(conn)

    if not sessions:
        st.caption("_Chưa có — hỏi câu đầu tiên là hội thoại được lưu tự động._")

    for s in sessions:
        col_open, col_del = st.columns([5, 1])
        title = s["title"] if len(s["title"]) <= 40 else s["title"][:40] + "…"
        active = s["session_id"] == st.session_state.session_id
        with col_open:
            if st.button(
                title,
                key=f"open{s['session_id']}",
                use_container_width=True,
                type="primary" if active else "secondary",
                help=f"{s['turns']} lượt · lần cuối {s['last_at']:%d/%m %H:%M}",
            ):
                with store.connect() as conn:
                    open_session(s["session_id"], chatlog.load_session(conn, s["session_id"]))
                st.rerun()
        with col_del:
            if st.button("✕", key=f"del{s['session_id']}", help="Xoá hội thoại này"):
                with store.connect() as conn:
                    chatlog.delete_session(conn, s["session_id"])
                if active:
                    st.session_state.turns = []
                    st.session_state.history = []
                    st.session_state.session_id = uuid.uuid4().hex[:12]
                st.rerun()


# --- Khung chat ------------------------------------------------------------

if not st.session_state.turns:
    with st.chat_message("assistant"):
        st.markdown(
            "Xin chào. Tôi trả lời dựa trên **wiki Genshin Impact tiếng Việt**: "
            "nhân vật, vũ khí, thánh di vật, nguyên tố, khu vực và thuật ngữ trong game.\n\n"
            "Câu nào ngoài phạm vi đó tôi sẽ nói thẳng là không có thông tin, thay vì đoán."
        )

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        draw_answer(turn["view"], turn["q"])

question = st.chat_input("Nhập câu hỏi về game…", max_chars=2000)

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

    # Lưu bền MỌI lượt, kể cả lượt bị từ chối — mục đích là xem lại đúng những
    # gì đã diễn ra, khác với lịch sử LLM ở trên (xem rag/chatlog.py).
    with store.connect() as conn:
        chatlog.save_turn(conn, st.session_state.session_id, question, view)
    # Vẽ lại để hội thoại vừa lưu hiện ngay trong danh sách ở thanh bên.
    st.rerun()

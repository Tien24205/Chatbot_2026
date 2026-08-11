const messages = document.getElementById("messages");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");
const resetBtn = document.getElementById("reset");

let sessionId = "web-" + Math.random().toString(36).slice(2, 10);
let threshold = null; // lấy từ /api/health, không hardcode ở frontend

// --- Kiểm tra backend ngay khi mở trang ------------------------------------
// Nếu database chưa chạy hoặc chưa nạp dữ liệu, người dùng biết ngay thay vì
// phải gõ một câu hỏi rồi mới thấy lỗi.
fetch("/api/health")
  .then((r) => r.json())
  .then((d) => {
    threshold = d.similarity_threshold ?? null;
    const g = d.graph_built ? ` · đồ thị ${d.graph_entities} thực thể` : "";
    statusEl.textContent =
      d.status === "ok"
        ? `${d.chunks} đoạn tri thức${g} · ${d.chat_model} · ngưỡng ${d.similarity_threshold}`
        : `Lỗi: ${d.detail}`;
  })
  .catch(() => (statusEl.textContent = "Không kết nối được backend"));

// --- Hiển thị ---------------------------------------------------------------

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/** Markdown tối giản: chỉ **đậm**, *nghiêng* và gạch đầu dòng.
 *  Escape TRƯỚC rồi mới thay thẻ, để nội dung từ LLM không chèn được HTML. */
function render(text) {
  const html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  const out = [];
  let list = null;
  for (const line of html.split("\n")) {
    const m = line.match(/^\s*[-*]\s+(.*)$/);
    if (m) {
      list ??= [];
      list.push(`<li>${m[1]}</li>`);
    } else {
      if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
      if (line.trim()) out.push(`<p>${line}</p>`);
    }
  }
  if (list) out.push(`<ul>${list.join("")}</ul>`);
  return out.join("");
}

function addMessage(who, html, extraClass = "") {
  const wrap = document.createElement("div");
  wrap.className = `msg ${who} ${extraClass}`.trim();
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap;
}

function metaBlock(data) {
  const bits = [];

  if (data.rewrote_query) {
    bits.push(
      `<div class="rewrite">Đã diễn giải câu hỏi thành: “${escapeHtml(data.search_query)}”</div>`
    );
  }

  if (data.refused) {
    const vs = threshold === null ? "" : ` &lt; ngưỡng ${threshold}`;
    bits.push(
      `<div>Không nguồn nào vượt ngưỡng tin cậy ` +
        `(liên quan cao nhất ${data.top_similarity}${vs})</div>`
    );
  } else if (data.citations.length) {
    bits.push("<div>Nguồn:</div>");
    for (const c of data.citations) {
      // Nguồn do đồ thị bổ sung có điểm similarity thấp một cách bình thường —
      // nó được chọn vì quan hệ giữa các thực thể, không vì giống câu hỏi.
      // Không ghi rõ thì con số 0.4x nằm cạnh 0.7x trông như lỗi.
      const via = c.via === "graph" ? `<span class="via">qua đồ thị</span>` : "";
      bits.push(
        `<span class="cite">• ${escapeHtml(c.source_name)} · ${escapeHtml(c.section)} ` +
          `<span class="score">(${c.similarity})</span> ${via}</span>`
      );
    }
  }

  // --- Kết quả kiểm chứng ---------------------------------------------------
  // Hiển thị cả khi ĐẠT: người dùng cần biết câu trả lời đã được soát, nếu chỉ
  // hiện lúc có lỗi thì họ không phân biệt được "đã soát, sạch" với "chưa soát".
  if (!data.refused) {
    if (data.flags.length || data.unsupported.length) {
      const items = [...data.flags, ...data.unsupported].map(
        (f) => `<li>${escapeHtml(f)}</li>`
      );
      bits.push(
        `<div class="checks warn">Kiểm chứng phát hiện dấu hiệu đáng ngờ:` +
          `<ul>${items.join("")}</ul></div>`
      );
    } else {
      const l5 = data.entailment_ran ? " + đối chiếu entailment" : "";
      bits.push(`<div class="checks ok">Đã soát trích dẫn, số liệu và tên riêng${l5}</div>`);
    }
  }

  const t = [];
  if (data.rewrite_ms) t.push(`diễn giải ${data.rewrite_ms} ms`);
  t.push(`truy hồi ${data.retrieval_ms} ms`);
  if (data.llm_ms) t.push(`sinh câu trả lời ${data.llm_ms} ms`);
  if (data.verify_ms) t.push(`kiểm chứng ${data.verify_ms} ms`);
  if (data.graph_expanded) t.push(`đồ thị bổ sung ${data.graph_expanded} đoạn`);
  bits.push(`<div class="timing">${data.total_ms} ms — ${t.join(" · ")}</div>`);

  return `<div class="meta">${bits.join("")}</div>`;
}

// --- Gửi câu hỏi ------------------------------------------------------------

async function ask(text) {
  addMessage("user", `<p>${escapeHtml(text)}</p>`);

  const pending = addMessage(
    "bot",
    `<div class="typing"><span></span><span></span><span></span></div>`
  );
  send.disabled = true;
  input.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });

    if (!res.ok) {
      let detail = await res.text();
      try { detail = JSON.parse(detail).detail ?? detail; } catch {}
      // 429 = chạm hạn mức Gemini, không phải hỏng hệ thống. Nói rõ để người
      // dùng biết chỉ cần chờ chứ không phải báo lỗi.
      throw new Error(res.status === 429 ? detail : `HTTP ${res.status}: ${String(detail).slice(0, 200)}`);
    }

    const data = await res.json();
    pending.remove();
    addMessage("bot", render(data.answer) + metaBlock(data), data.refused ? "refused" : "");
  } catch (err) {
    pending.remove();
    addMessage("bot", `<p>Không gửi được câu hỏi.</p><p>${escapeHtml(String(err))}</p>`, "refused");
  } finally {
    send.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  ask(text);
});

document.querySelectorAll("#suggestions button").forEach((b) =>
  b.addEventListener("click", () => ask(b.textContent.trim()))
);

resetBtn.addEventListener("click", async () => {
  await fetch(`/api/chat/${sessionId}`, { method: "DELETE" }).catch(() => {});
  sessionId = "web-" + Math.random().toString(36).slice(2, 10);
  messages.innerHTML = "";
  addMessage("bot", "<p>Đã bắt đầu hội thoại mới.</p>");
  input.focus();
});

input.focus();

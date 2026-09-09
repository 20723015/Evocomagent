/* 运营工作台:转人工工单看板(/v1/handoffs)+ 对话全文检索(/v1/messages/search)。
   两端点为 ops 面;开发环境 auth 关闭时直通,生产需 ops scope 的 JWT。 */
"use strict";

const STATUS_LABELS = { pending: "待处理", resolved: "已处理" };
const ROLE_LABELS = { user: "用户", assistant: "客服", tool: "工具" };
const HUMAN_STATUS_LABELS = {
  pending_review: "待审核", publish_queued: "发布中", published: "已发布",
  rejected: "已拒绝", superseded: "已过期", retired: "已下架",
  queued: "排队中", running: "进行中", retry_wait: "等待重试",
  completed: "已完成", blocked: "已阻塞",
};
const EVIDENCE_STATE_LABELS = {
  ok: "证据链完好",
  legacy_evidence_missing: "证据链缺失（禁止批准，需回填）",
};

let opsUser = localStorage.getItem("xiaoxi.userId") || "web-user";
let currentStatus = "pending";

/* ---------- 工单看板 ---------- */

async function loadTickets() {
  const box = $("#ticketBox");
  box.innerHTML = "";
  box.appendChild(loadingHint("正在加载工单…"));
  try {
    const data = await api(`/v1/handoffs?status=${encodeURIComponent(currentStatus)}`);
    const tickets = data.tickets || [];
    box.innerHTML = "";
    if (!tickets.length) {
      box.appendChild(emptyState("📭", `暂无${STATUS_LABELS[currentStatus] || ""}工单`,
        "对话中要求转人工 / guardrail 命中降级时自动生成"));
      announce("暂无工单");
      return;
    }
    for (const t of tickets.slice().reverse()) box.appendChild(renderTicket(t));
    announce(`已加载 ${tickets.length} 张工单`);
  } catch (e) {
    box.innerHTML = "";
    const b = document.createElement("div");
    b.className = "banner err";
    b.textContent = `加载工单失败:${e.message}`;
    box.appendChild(b);
    announce("工单加载失败");
  }
}

function renderTicket(t) {
  const card = document.createElement("div");
  card.className = "ticket-card";
  const intent = INTENT_LABELS[t.intent] || t.intent || "—";
  const head = `
    <div class="ticket-top">
      <span class="chip ${t.status === "pending" ? "warn" : "ok"}">${STATUS_LABELS[t.status] || t.status}</span>
      <span class="chip">${escapeHtml(intent)}</span>
      <span class="tid">${escapeHtml((t.ticket_id || "").slice(0, 12))}</span>
      <span class="t-time">${escapeHtml(fmtTime(t.created_at))}</span>
    </div>
    <div class="ticket-fields">
      <div class="t-field"><span class="f-label">用户/会话</span><span class="f-val">${escapeHtml(t.user_id)} / ${escapeHtml(t.session_id || "(默认)")}</span></div>
      ${t.question ? `<div class="t-field q"><span class="f-label">用户问题</span><span class="f-val">${renderMd(t.question)}</span></div>` : ""}
      ${t.reply ? `<div class="t-field"><span class="f-label">机器人话术</span><span class="f-val">${renderMd(t.reply)}</span></div>` : ""}
      ${t.summary ? `<div class="t-field"><span class="f-label">对话摘要</span><span class="f-val">${renderMd(t.summary)}</span></div>` : ""}
    </div>`;
  card.innerHTML = head;

  if (t.suggested_actions?.length) {
    const chips = document.createElement("div");
    chips.className = "actions-chips";
    for (const a of t.suggested_actions) {
      chips.insertAdjacentHTML("beforeend", `<span class="chip">建议:${escapeHtml(a)}</span>`);
    }
    card.appendChild(chips);
  }

  if (t.status === "resolved" && t.resolution) {
    card.insertAdjacentHTML("beforeend", `
      <div class="ticket-fields" style="margin-top:8px">
        <div class="t-field"><span class="f-label">坐席结论</span>
          <span class="f-val">${renderMd(typeof t.resolution === "string" ? t.resolution : prettyJson(t.resolution))}</span></div>
      </div>`);
  } else {
    const form = document.createElement("div");
    form.className = "ticket-resolve";
    form.style.display = "none";
    // 人工知识沉淀已切换为「外部会话批量接入 → MySQL 评审 → 人工批量发布」；
    // 工单勾选沉淀已下线（服务端对旧勾选字段返回 410 弃用错误）。
    form.innerHTML = `
      <textarea aria-label="人工处理结论" placeholder="填写人工处理结论（回写后可在对话页继续该会话）"></textarea>
      <div class="resolve-row">
        <label><input type="checkbox" class="reclaim-toggle" checked> 重建会话（坐席处理后用户可继续对话）</label>
        <span class="spacer"></span>
        <button class="btn btn-primary do-resolve">提交结论</button>
      </div>`;
    card.insertAdjacentHTML("beforeend",
      `<button class="btn btn-outline toggle-resolve">处理工单</button>`);
    card.appendChild(form);
    card.querySelector(".toggle-resolve").addEventListener("click", () => {
      const show = form.style.display === "none";
      form.style.display = show ? "flex" : "none";
      card.querySelector(".toggle-resolve").setAttribute("aria-expanded", String(show));
      if (show) form.querySelector("textarea").focus();
    });
    form.querySelector(".do-resolve").addEventListener("click", async () => {
      const note = form.querySelector("textarea").value.trim();
      if (!note) { toast("请先填写处理结论", "err"); return; }
      const reclaim = form.querySelector(".reclaim-toggle").checked;
      const btn = form.querySelector(".do-resolve");
      btn.disabled = true;
      try {
        await api(`/v1/handoffs/${encodeURIComponent(t.ticket_id)}/resolve`, {
          method: "POST",
          body: JSON.stringify({
            user_id: t.user_id,
            resolution: { note },
            reclaim,
          }),
        });
        toast("工单已处理", "ok");
        announce("工单已处理");
        loadTickets();
      } catch (e) {
        toast("处理失败:" + e.message, "err");
        announce("工单处理失败");
        btn.disabled = false;
      }
    });
  }
  return card;
}

/* ---------- 人工客服知识审核 ---------- */

function selectedHumanCandidates() {
  return Array.from(document.querySelectorAll("[data-human-select]:checked")).map(el => ({
    candidate_id: Number(el.dataset.humanSelect),
    revision: Number(el.dataset.revision),
  }));
}

function syncHumanPublishButton() {
  $("#publishHumanKnowledge").disabled = selectedHumanCandidates().length === 0;
}

function renderHumanCandidate(candidate) {
  const card = document.createElement("div");
  card.className = "ticket-card";
  // 证据链缺失或评分过期 → 禁止勾选并显示原因
  const evidenceBad = candidate.evidence_state && candidate.evidence_state !== "ok";
  const disabledReason = candidate.score_stale
    ? "编辑后评分已过期，等待重评完成"
    : evidenceBad ? (EVIDENCE_STATE_LABELS[candidate.evidence_state] || candidate.evidence_state)
    : "";
  const selectable = candidate.status === "pending_review" && !disabledReason;
  const dedupPath = candidate.dedup_target_path
    ? `<div class="t-field"><span class="f-label">去重目标</span><span class="f-val">${escapeHtml(candidate.dedup_target_path)}</span></div>` : "";
  const reasonHtml = candidate.value_reason
    ? `<div class="t-field"><span class="f-label">抽取理由</span><span class="f-val">${escapeHtml(candidate.value_reason)}</span></div>` : "";
  const evalHtml = (candidate.eval_model || candidate.eval_kb_generation)
    ? `<div class="t-field"><span class="f-label">评审</span><span class="f-val">${escapeHtml(candidate.eval_model || "—")} · 代际 ${escapeHtml(candidate.eval_kb_generation || "—")}</span></div>` : "";
  const disabledHtml = disabledReason
    ? `<div class="t-field"><span class="f-label">不可批准</span><span class="f-val err">${escapeHtml(disabledReason)}</span></div>` : "";
  card.innerHTML = `
    <div class="ticket-top">
      ${selectable ? `<input type="checkbox" data-human-select="${candidate.id}" data-revision="${candidate.revision}" aria-label="选择候选 ${candidate.id}">` : ""}
      <span class="chip ${candidate.status === "pending_review" ? "warn" : "ok"}">${escapeHtml(HUMAN_STATUS_LABELS[candidate.status] || candidate.status)}</span>
      <span class="chip">${escapeHtml(candidate.classification || "未分类")}</span>
      <span class="tid">#${candidate.id} · r${candidate.revision} · lc${candidate.lifecycle_revision ?? 0}</span>
    </div>
    <div class="ticket-fields">
      <div class="t-field q"><span class="f-label">规范问题</span><span class="f-val">${renderMd(candidate.question)}</span></div>
      <div class="t-field"><span class="f-label">标准答案</span><span class="f-val">${renderMd(candidate.answer)}</span></div>
      <div class="t-field"><span class="f-label">评分</span><span class="f-val">价值 ${candidate.value_score ?? "—"} · 新颖 ${candidate.novelty_score ?? "—"} · 综合 ${candidate.composite_score ?? "—"}</span></div>
      ${reasonHtml}
      ${dedupPath}
      ${evalHtml}
      ${disabledHtml}
    </div>
    <details class="evidence-details" data-evidence-for="${candidate.id}">
      <summary>证据快照（${(candidate.evidence_message_ids || []).length} 条引用）</summary>
      <div class="evidence-body" hidden></div>
    </details>
    ${selectable ? `<div class="doc-card-actions"><button class="btn btn-outline btn-sm" data-human-edit="${candidate.id}">编辑并重评</button><button class="btn btn-outline btn-sm" data-human-reject="${candidate.id}">拒绝</button></div>` : ""}
    ${candidate.status === "published" ? `<div class="doc-card-actions"><button class="btn btn-outline btn-sm" data-human-retire="${candidate.id}" data-lifecycle-revision="${candidate.lifecycle_revision ?? 0}">下架</button></div>` : ""}
    ${candidate.status === "pending_review" && hasBlockedJob(candidate) ? `<div class="doc-card-actions"><button class="btn btn-outline btn-sm" data-human-retry="${candidate.id}">重试评审</button></div>` : ""}`;
  card._candidate = candidate;
  return card;
}

function hasBlockedJob(candidate) {
  // 列表 DTO 不带任务明细；recent_evaluation_jobs 仅在详情展开时加载。
  return Array.isArray(candidate.recent_evaluation_jobs)
    && candidate.recent_evaluation_jobs.some(j => j.status === "blocked");
}

async function toggleEvidenceDetails(detailsEl, candidateId) {
  const body = detailsEl.querySelector(".evidence-body");
  if (!detailsEl.open || body.dataset.loaded) return;
  try {
    const detail = await api(`/v1/human-knowledge/candidates/${candidateId}`);
    const msgs = detail.evidence_snapshot || [];
    const source = detail.source_snapshot || {};
    const lines = [`来源：${escapeHtml(source.source || "—")} / ${escapeHtml(source.external_conversation_id || "—")} v${source.source_version ?? "—"}`];
    for (const m of msgs) {
      lines.push(`<div class="t-field"><span class="f-label">${escapeHtml(m.message_id)} · ${escapeHtml(m.sent_at || "")}</span><span class="f-val">${renderMd(m.content || "")}</span></div>`);
      if (m.prev_customer) lines.push(`<div class="t-field"><span class="f-label">前文(客户)</span><span class="f-val">${escapeHtml(String(m.prev_customer.content || "").slice(0, 200))}</span></div>`);
      if (m.next_customer) lines.push(`<div class="t-field"><span class="f-label">后文(客户)</span><span class="f-val">${escapeHtml(String(m.next_customer.content || "").slice(0, 200))}</span></div>`);
    }
    body.innerHTML = lines.join("");
    const jobs = detail.recent_evaluation_jobs || [];
    if (jobs.length) {
      const blocked = jobs.some(j => j.status === "blocked");
      const retryBtn = blocked
        ? ` <button class="btn btn-outline btn-sm" data-human-retry="${candidateId}">重试评审</button>` : "";
      body.insertAdjacentHTML("beforeend",
        `<div class="t-field"><span class="f-label">最近评审</span><span class="f-val">${jobs.map(j => `#${j.id} ${escapeHtml(HUMAN_STATUS_LABELS[j.status] || j.status)}`).join(" · ")}${retryBtn}</span></div>`);
    }
    body.dataset.loaded = "1";
    body.hidden = false;
  } catch (e) {
    body.innerHTML = `<span class="err">证据加载失败：${escapeHtml(e.message)}</span>`;
    body.hidden = false;
  }
}

async function retireHumanCandidate(candidateId, lifecycleRevision) {
  const reason = prompt("下架原因", "manual_retire");
  if (reason === null) return;
  try {
    const result = await api(`/v1/human-knowledge/candidates/${candidateId}/retire`, {
      method: "POST",
      body: JSON.stringify({ reason, expected_lifecycle_revision: Number(lifecycleRevision) }),
    });
    toast(`下架批次 #${result.batch_id} 已入队`, "ok");
    await loadHumanKnowledge();
  } catch (e) { toast(`下架失败：${e.message}`, "err"); }
}

async function retryHumanEvaluation(candidateId) {
  try {
    await api(`/v1/human-knowledge/candidates/${candidateId}/retry-evaluation`, { method: "POST" });
    toast("已重新排队评审", "ok");
    await loadHumanKnowledge();
  } catch (e) { toast(`重试失败：${e.message}`, "err"); }
}

async function loadHumanKnowledge() {
  const box = $("#humanKnowledgeBox");
  box.innerHTML = "";
  box.appendChild(loadingHint("正在加载知识候选…"));
  try {
    const status = $("#humanKnowledgeStatus").value;
    const data = await api(`/v1/human-knowledge/candidates?status=${encodeURIComponent(status)}&limit=200`);
    box.innerHTML = "";
    for (const candidate of data.candidates || []) box.appendChild(renderHumanCandidate(candidate));
    if (!(data.candidates || []).length) box.appendChild(emptyState("🧠", "暂无候选", "每日评审完成后，新的高价值知识会出现在这里"));
    $("#humanKnowledgeCount").textContent = `共 ${data.total || 0} 条`;
    syncHumanPublishButton();
  } catch (e) {
    box.innerHTML = `<div class="banner err">加载候选失败：${escapeHtml(e.message)}</div>`;
  }
}

async function publishSelectedHumanKnowledge() {
  const items = selectedHumanCandidates();
  if (!items.length) return;
  try {
    const result = await api("/v1/human-knowledge/publish-batches", {
      method: "POST", body: JSON.stringify({ items }),
    });
    toast(`发布批次 #${result.batch_id} 已入队`, "ok");
    await loadHumanKnowledge();
  } catch (e) { toast(`批准失败：${e.message}`, "err"); }
}

async function editHumanCandidate(candidate) {
  const question = prompt("规范问题", candidate.question);
  if (question === null) return;
  const answer = prompt("标准答案", candidate.answer);
  if (answer === null) return;
  try {
    await api(`/v1/human-knowledge/candidates/${candidate.id}/edit`, {
      method: "POST",
      body: JSON.stringify({ question, answer, expected_revision: candidate.revision }),
    });
    toast("已保存，等待重新评审", "ok");
    await loadHumanKnowledge();
  } catch (e) { toast(`编辑失败：${e.message}`, "err"); }
}

async function rejectHumanCandidate(candidateId) {
  if (!confirm("确认拒绝该知识候选？")) return;
  try {
    await api(`/v1/human-knowledge/candidates/${candidateId}/reject`, {
      method: "POST", body: JSON.stringify({ reason: "manual_review_reject" }),
    });
    await loadHumanKnowledge();
  } catch (e) { toast(`拒绝失败：${e.message}`, "err"); }
}

/* ---------- 消息检索 ---------- */

async function searchMessages() {
  const q = $("#searchQ").value.trim();
  const box = $("#searchBox");
  const session = $("#searchSession").value.trim();
  const btn = $("#doSearch");
  if (!q) { toast("请输入检索关键词", "err"); return; }
  box.innerHTML = "";
  box.appendChild(loadingHint("检索中…"));
  btn.disabled = true;
  try {
    const params = new URLSearchParams({
      user_id: opsUser, q, limit: String($("#searchLimit").value),
    });
    if (session) params.set("session_id", session);
    const data = await api(`/v1/messages/search?${params}`);
    box.innerHTML = "";
    if (data.degraded) {
      const b = document.createElement("div");
      b.className = "banner";
      b.textContent = `检索降级：${data.reason || "ES 不可用"}。对话全文检索依赖 MySQL→ES outbox 同步（阶段八），本地开发未配置 ES 时显示此提示。`;
      box.appendChild(b);
      announce("检索降级,ES 不可用");
      return;
    }
    const hits = data.hits || [];
    if (!hits.length) {
      box.appendChild(emptyState("🔍", "无匹配消息", `未找到包含「${q}」的对话`));
      announce("检索完成,无匹配消息");
      return;
    }
    for (const h of hits) box.appendChild(renderHit(h));
    announce(`检索完成,共 ${hits.length} 条结果`);
  } catch (e) {
    box.innerHTML = "";
    const b = document.createElement("div");
    b.className = "banner err";
    b.textContent = `检索失败：${e.message}`;
    box.appendChild(b);
    announce("检索失败");
  } finally {
    btn.disabled = false;
  }
}

function renderHit(h) {
  const role = (h.role || "").toLowerCase();
  const card = document.createElement("div");
  card.className = "hit-card";
  card.innerHTML = `
    <div class="hit-main">
      <span class="hit-role ${escapeHtml(role)}">${ROLE_LABELS[role] || role || "消息"}</span>
      <div class="hit-content">${renderMd(h.content || "")}</div>
    </div>
    <div class="hit-sub">会话 ${escapeHtml(h.session_id || "")} · #${escapeHtml(h.seq ?? "?")} · ${escapeHtml(fmtTime(h.ts))}</div>`;
  return card;
}

/* ---------- 标签页与事件(roving tabindex + 方向键/Home/End) ---------- */

function switchTab(name) {
  $$(".tab").forEach(b => {
    const selected = b.dataset.tab === name;
    b.setAttribute("aria-selected", String(selected));
    b.tabIndex = selected ? 0 : -1;
  });
  $("#tab-handoffs").hidden = name !== "handoffs";
  $("#tab-search").hidden = name !== "search";
  $("#tab-upload").hidden = name !== "upload";
  $("#tab-human-knowledge").hidden = name !== "human-knowledge";
  if (name === "handoffs") loadTickets();
  if (name === "upload") loadDocs();
  if (name === "human-knowledge") loadHumanKnowledge();
}

function initTabKeyboard() {
  const tabs = $$(".tab");
  tabs.forEach(tab => {
    tab.addEventListener("keydown", e => {
      const i = tabs.indexOf(tab);
      let next = null;
      if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
      else if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
      else if (e.key === "Home") next = tabs[0];
      else if (e.key === "End") next = tabs[tabs.length - 1];
      if (!next) return;
      e.preventDefault();
      switchTab(next.dataset.tab);
      next.focus();
    });
  });
}

$$(".tab").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));
initTabKeyboard();
$("#statusFilter").addEventListener("change", e => {
  currentStatus = e.target.value;
  loadTickets();
});
$("#refreshTickets").addEventListener("click", loadTickets);
$("#refreshHumanKnowledge").addEventListener("click", loadHumanKnowledge);
$("#humanKnowledgeStatus").addEventListener("change", loadHumanKnowledge);
$("#publishHumanKnowledge").addEventListener("click", publishSelectedHumanKnowledge);
$("#humanKnowledgeBox").addEventListener("change", e => {
  if (e.target.matches("[data-human-select]")) syncHumanPublishButton();
});
$("#humanKnowledgeBox").addEventListener("click", e => {
  const edit = e.target.closest("[data-human-edit]");
  const reject = e.target.closest("[data-human-reject]");
  const retire = e.target.closest("[data-human-retire]");
  const retry = e.target.closest("[data-human-retry]");
  if (edit) editHumanCandidate(edit.closest(".ticket-card")._candidate);
  if (reject) rejectHumanCandidate(Number(reject.dataset.humanReject));
  if (retire) retireHumanCandidate(Number(retire.dataset.humanRetire), retire.dataset.lifecycleRevision);
  if (retry) retryHumanEvaluation(Number(retry.dataset.humanRetry));
});
$("#humanKnowledgeBox").addEventListener("toggle", e => {
  const details = e.target.closest("[data-evidence-for]");
  if (details && details.open) toggleEvidenceDetails(details, Number(details.dataset.evidenceFor));
}, true);
$("#doSearch").addEventListener("click", searchMessages);
$("#searchQ").addEventListener("keydown", e => {
  if (e.key === "Enter") searchMessages();
});
$("#opsUser").value = opsUser;
$("#opsUser").addEventListener("change", e => {
  opsUser = e.target.value.trim() || "web-user";
  localStorage.setItem("xiaoxi.userId", opsUser);
});

switchTab("handoffs");

/* ======================================================
 * 知识库文档：分片上传（断点续传）+ 文档列表 + 下架
 * ====================================================== */
const KB_MIN_CHUNK = 64 * 1024;      // 服务端钳制下限（64KiB）
const KB_CHUNK_DEFAULT = 1024 * 1024;
let kbBusy = false;

function kbUploaderUser() {
  return opsUser || "web-user";
}

async function kbPutChunk(uploadId, seq, blob) {
  // 二进制分片：不能带 Content-Type（浏览器自动 boundary/二进制头）
  const resp = await fetch(`/v1/kb/uploads/${encodeURIComponent(uploadId)}/chunks/${seq}`, {
    method: "PUT", body: blob,
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const j = await resp.json();
      if (j.detail) detail = j.detail;
    } catch { /* 非 JSON 错误体 */ }
    throw new Error(detail);
  }
  return resp.json();
}

function setKbProgress(ratio, text) {
  const box = $("#kbProgress");
  box.hidden = false;
  $("#kbProgressFill").style.width = `${Math.round(ratio * 100)}%`;
  $("#kbProgressText").textContent = text;
}

function kbFmtBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KiB`;
  return `${(n / 1024 / 1024).toFixed(2)}MiB`;
}

function kbDocRow(d) {
  const badge = `<span class="badge badge-${escapeHtml(d.status)}">${escapeHtml(d.status)}</span>`;
  const gen = d.generation_id ? escapeHtml(d.generation_id) : "—";
  const size = kbFmtBytes(d.size_bytes);
  const when = fmtTime(d.created_at);
  const del = d.status === "indexed"
    ? `<button class="btn btn-outline btn-sm" data-del-doc="${escapeHtml(d.doc_id)}">下架</button>`
    : "";
  return `<div class="doc-card">
    <div class="doc-card-head">${badge} <strong>${escapeHtml(d.filename)}</strong>
      <span class="hint">${escapeHtml(d.format)} · ${size} · ${when}</span></div>
    <div class="doc-card-sub">doc=${escapeHtml(d.doc_id)} · gen=${gen} · chunks=${d.indexed_chunk_count ?? d.upload_chunk_count ?? "?"}
      ${d.error ? ` · <span class="err">${escapeHtml(d.error)}</span>` : ""}</div>
    <div class="doc-card-actions">${del}</div>
  </div>`;
}

async function loadDocs() {
  const box = $("#kbDocs");
  try {
    const data = await api("/v1/kb/documents");
    const docs = data.documents || [];
    if (!docs.length) {
      box.innerHTML = `<div class="empty-state"><span class="es-icon">📄</span>
        <div class="es-title">暂无知识库文档</div></div>`;
      return;
    }
    box.innerHTML = docs.map(kbDocRow).join("");
  } catch (e) {
    box.innerHTML = `<div class="empty-state"><span class="es-icon">⚠️</span>
      <div class="es-title">文档列表加载失败</div><div class="es-hint">${escapeHtml(e.message)}</div></div>`;
  }
}

async function uploadKbDocument() {
  if (kbBusy) return;
  const file = $("#kbFile").files[0];
  if (!file) { toast("请先选择文件", "warn"); return; }
  kbBusy = true;
  $("#kbUploadBtn").disabled = true;
  try {
    // 分片大小：默认 1MiB，大文件自动放大到 ≤ 500 片（服务端上限 1000）
    let chunkSize = KB_CHUNK_DEFAULT;
    if (file.size / chunkSize > 500) {
      chunkSize = Math.ceil(file.size / 500);
      const max = 4 * 1024 * 1024;
      const min = KB_MIN_CHUNK;
      chunkSize = Math.max(min, Math.min(max, chunkSize));
    }
    const up = await api("/v1/kb/uploads", {
      method: "POST",
      body: JSON.stringify({
        uploader: kbUploaderUser(), filename: file.name,
        size_bytes: file.size, chunk_size: chunkSize,
      }),
    });
    const total = up.total_chunks;
    const received = new Set(up.received || []);
    for (let seq = 0; seq < total; seq++) {
      if (received.has(seq)) continue;
      setKbProgress(seq / total, `分片 ${seq + 1}/${total}`);
      const blob = file.slice(seq * chunkSize, (seq + 1) * chunkSize);
      await kbPutChunk(up.upload_id, seq, blob);
    }
    setKbProgress(1, "合并校验并入库（版本化重建，最长约 30 秒）…");
    const done = await api(`/v1/kb/uploads/${encodeURIComponent(up.upload_id)}/complete`, {
      method: "POST",
      body: JSON.stringify({ uploader: kbUploaderUser() }),
    });
    setKbProgress(1, `✅ 入库完成：${done.doc_id}（generation ${done.generation_id}）`);
    toast("文档已入库，Agent 检索已热更新", "ok", 3200);
    loadDocs();
  } catch (e) {
    setKbProgress(1, `❌ 失败：${e.message}`);
    toast(e.message, "error", 5000);
  } finally {
    kbBusy = false;
    $("#kbUploadBtn").disabled = false;
  }
}

async function deleteKbDocument(docId) {
  if (!confirm("确认下架该文档？下架会触发一次全量重建（失败自动回滚）。")) return;
  try {
    await api(`/v1/kb/documents/${encodeURIComponent(docId)}?uploader=${encodeURIComponent(kbUploaderUser())}`,
      { method: "DELETE" });
    toast("已下架", "ok");
    loadDocs();
  } catch (e) {
    toast(`下架失败：${e.message}`, "error", 5000);
  }
}

$("#kbFile").addEventListener("change", () => {
  $("#kbUploadBtn").disabled = !$("#kbFile").files.length;
});
$("#kbUploadBtn").addEventListener("click", uploadKbDocument);
$("#kbDocs").addEventListener("click", e => {
  const btn = e.target.closest("[data-del-doc]");
  if (btn) deleteKbDocument(btn.dataset.delDoc);
});

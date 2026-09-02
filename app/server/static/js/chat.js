/* 对话页逻辑:POST /v1/chat/stream SSE 流式渲染 ReAct 过程,
   会话列表/上下文缓存存 localStorage(服务端无历史查询接口)。 */
"use strict";

const USER_KEY = "xiaoxi.userId";
const SESSIONS_KEY = "xiaoxi.sessions.v1";
const ACTIVE_KEY = "xiaoxi.active.v1";

const EXAMPLES = [
  "我的订单还没发货，怎么回事？",
  "有没有宽松透气的裤子推荐？",
  "这件衣服质量有问题，我要退货",
  "满99包邮是真的吗？偏远地区也包邮吗？",
  "会员积分怎么获得？",
];

const state = {
  userId: localStorage.getItem(USER_KEY) || "web-user",
  sessions: loadJson(SESSIONS_KEY, []),
  activeId: "",
  streaming: false,
  abortCtrl: null,
};

function loadJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function saveJson(key, val) {
  try { localStorage.setItem(key, JSON.stringify(val)); } catch { /* 容量满等,不影响对话 */ }
}

/* ---------- 会话列表(localStorage 索引;服务端会话以 session_id 为准) ---------- */

function sessionsOf(userId) {
  return state.sessions
    .filter(s => s.userId === userId)
    .sort((a, b) => (b.updatedAt || "").localeCompare(a.updatedAt || ""));
}

function newSessionObj(userId) {
  const id = "web-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const s = { id, userId, title: "新会话", updatedAt: new Date().toISOString() };
  state.sessions.push(s);
  saveJson(SESSIONS_KEY, state.sessions);
  return s;
}

function activeSession() {
  return state.sessions.find(s => s.id === state.activeId) || null;
}

function bindActive(userId) {
  const map = loadJson(ACTIVE_KEY, {});
  const wanted = map[userId];
  if (wanted && state.sessions.some(s => s.id === wanted && s.userId === userId)) {
    state.activeId = wanted;
  } else {
    const list = sessionsOf(userId);
    state.activeId = list.length ? list[0].id : "";
  }
}

function setActive(id) {
  state.activeId = id;
  const map = loadJson(ACTIVE_KEY, {});
  map[state.userId] = id;
  saveJson(ACTIVE_KEY, map);
}

/* ---------- 转写缓存(刷新/切换后可见;服务端无读回接口) ---------- */

function transcriptKey(sid) { return `xiaoxi.transcript.${state.userId}.${sid}`; }

function loadTranscript(sid) { return loadJson(transcriptKey(sid), []); }
function saveTranscript(sid, turns) {
  saveJson(transcriptKey(sid), turns.slice(-80));
}

/* ---------- DOM 引用 ---------- */

const chatEl = $("#chat");
const innerEl = $("#inner");
const inputEl = $("#input");
const sendBtn = $("#send");
const sessionListEl = $("#sessionList");
const titleEl = $("#chatTitle");

function nearBottom() {
  return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 90;
}
function scrollBottom(force) {
  if (force || nearBottom()) chatEl.scrollTop = chatEl.scrollHeight;
}

/* ---------- 渲染:会话列表 ---------- */

function renderSessions() {
  sessionListEl.innerHTML = "";
  const list = sessionsOf(state.userId);
  if (!list.length) {
    const d = document.createElement("div");
    d.className = "side-empty";
    d.textContent = "暂无历史会话";
    sessionListEl.appendChild(d);
    return;
  }
  for (const s of list) {
    // 选择与删除是并列按钮:键盘删除不会误触会话切换
    const item = document.createElement("div");
    item.className = "session-item";
    if (s.id === state.activeId) item.setAttribute("aria-current", "true");
    const main = document.createElement("button");
    main.type = "button";
    main.className = "s-main";
    main.innerHTML = `<span class="s-title"></span><span class="s-time"></span>`;
    main.querySelector(".s-title").textContent = s.title;
    main.querySelector(".s-time").textContent = fmtTime(s.updatedAt);
    main.addEventListener("click", () => switchSession(s.id));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "s-del";
    del.setAttribute("aria-label", "删除会话记录：" + s.title);
    del.textContent = "✕";
    del.addEventListener("click", e => {
      e.stopPropagation();
      removeSession(s.id);
    });
    item.append(main, del);
    sessionListEl.appendChild(item);
  }
}

/* ---------- 渲染:消息轮次 ---------- */

function addUserTurn(text, ts) {
  const row = document.createElement("div");
  row.className = "row user";
  row.innerHTML = `<div class="msg-avatar">🧑</div>
    <div class="bubble">${renderMd(text)}</div>`;
  innerEl.appendChild(row);
  scrollBottom(true);
}

function makeTraceEl() {
  const el = document.createElement("div");
  el.className = "trace";
  el.innerHTML = `<button type="button" class="trace-head" aria-expanded="true">
      <span aria-hidden="true">⚙️</span><span class="trace-title">推理与工具调用</span>
      <span class="chev" aria-hidden="true">▼</span></button>
    <div class="trace-body"></div>`;
  el.querySelector(".trace-head").addEventListener("click", () => toggleTrace(el));
  return el;
}

function toggleTrace(el) {
  el.classList.toggle("collapsed");
  el.querySelector(".trace-head").setAttribute(
    "aria-expanded", el.classList.contains("collapsed") ? "false" : "true");
}

function collapseTrace(el) {
  el.classList.add("collapsed");
  el.querySelector(".trace-head").setAttribute("aria-expanded", "false");
}

function traceSummary(counts) {
  const parts = [];
  if (counts.thoughts) parts.push(`${counts.thoughts} 步推理`);
  if (counts.tools) parts.push(`${counts.tools} 次工具调用`);
  return parts.length ? parts.join(" · ") : "推理与工具调用";
}

/** 创建一轮 bot 回答的骨架;trace 事件经 applyTraceEvent 增量填充。 */
function beginBotTurn() {
  const row = document.createElement("div");
  row.className = "row bot";
  row.innerHTML = `<div class="msg-avatar">🤖</div>`;
  const turn = document.createElement("div");
  turn.className = "turn";
  row.appendChild(turn);
  innerEl.appendChild(row);
  scrollBottom(true);

  const dom = {
    row, turn,
    trace: null,
    traceBody: null,
    traceTitle: null,
    bubble: null,
    meta: null,
    counts: { thoughts: 0, tools: 0 },
    toolCards: new Map(),   // tool_call_id -> {card, statusEl, resultPre, sec}
    gotReply: false,
    errorText: "",
  };
  // 打字中占位
  dom.bubble = document.createElement("div");
  dom.bubble.className = "bubble";
  dom.bubble.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>`;
  turn.appendChild(dom.bubble);
  return dom;
}

function ensureTrace(dom) {
  if (dom.trace) return;
  dom.trace = makeTraceEl();
  dom.traceBody = dom.trace.querySelector(".trace-body");
  dom.traceTitle = dom.trace.querySelector(".trace-title");
  dom.turn.insertBefore(dom.trace, dom.bubble);
}

function addThought(dom, text) {
  ensureTrace(dom);
  dom.counts.thoughts++;
  const line = document.createElement("div");
  line.className = "trace-line";
  line.innerHTML = `<span class="t-ico">💭</span><span>${renderMd(text)}</span>`;
  dom.traceBody.appendChild(line);
  scrollBottom();
}

function addRoute(dom, target) {
  ensureTrace(dom);
  const line = document.createElement("div");
  line.className = "trace-line";
  line.innerHTML = `<span class="t-ico">🧭</span><span class="route-chip">转交「${escapeHtml(target)}」</span>`;
  dom.traceBody.appendChild(line);
  scrollBottom();
}

function addToolCall(dom, ev) {
  ensureTrace(dom);
  dom.counts.tools++;
  const card = document.createElement("div");
  card.className = "tool-card";
  const label = TOOL_LABELS[ev.name] || ev.name;
  card.innerHTML = `
    <div class="tool-head">🔧 <span class="tool-name">${escapeHtml(ev.name)}</span>
      <span class="tool-desc">${escapeHtml(label)}</span>
      <span class="tool-status running">运行中…</span></div>
    <div class="tool-sec"><div class="sec-label">调用参数</div><pre>${escapeHtml(prettyJson(ev.arguments ?? {}))}</pre></div>`;
  const resultSec = document.createElement("div");
  resultSec.className = "tool-sec";
  resultSec.innerHTML = `<div class="sec-label">返回结果</div><pre>…</pre>`;
  card.appendChild(resultSec);
  dom.traceBody.appendChild(card);
  dom.toolCards.set(ev.tool_call_id || ev.name, {
    card,
    statusEl: card.querySelector(".tool-status"),
    resultPre: resultSec.querySelector("pre"),
  });
  scrollBottom();
}

function fillToolResult(dom, ev) {
  const entry = dom.toolCards.get(ev.tool_call_id) ||
    [...dom.toolCards.values()].pop(); // 老服务端无 id 时兜底取最后一张
  if (!entry) return;
  let result = ev.result ?? "";
  let failed = false;
  try {
    const parsed = JSON.parse(result);
    if (parsed && typeof parsed === "object" && parsed.error) failed = true;
  } catch { /* 非 JSON 结果 */ }
  entry.resultPre.textContent = prettyJson(result);
  entry.statusEl.textContent = failed ? "✗ 异常" : "✓ 完成";
  entry.statusEl.className = "tool-status " + (failed ? "fail" : "done");
  scrollBottom();
}

function applyTraceEvent(dom, type, data) {
  if (type === "route") addRoute(dom, data.target || "");
  else if (type === "thought") addThought(dom, data.text || "");
  else if (type === "tool_call") addToolCall(dom, data);
  else if (type === "tool_result") fillToolResult(dom, data);
  else if (type === "error") dom.errorText = data.detail || "本轮处理失败";
}

function addMetaChips(dom, r) {
  dom.meta = document.createElement("div");
  dom.meta.className = "meta";
  const intent = INTENT_LABELS[r.intent] || r.intent || "—";
  dom.meta.innerHTML =
    `<span class="chip">意图:${escapeHtml(intent)}</span>
     <span class="chip">置信度:${Math.round((r.confidence || 0) * 100)}%</span>` +
    (r.requires_human ? `<span class="chip warn">⚠ 已转人工</span>` : "");
  dom.turn.appendChild(dom.meta);
  if (r.follow_up_question) {
    const fu = document.createElement("button");
    fu.className = "followup";
    fu.title = "点击填入输入框";
    fu.innerHTML = `💡 <span class="fu-text">${escapeHtml(r.follow_up_question)}</span>`;
    fu.addEventListener("click", () => {
      inputEl.value = r.follow_up_question;
      autoGrow();
      inputEl.focus();
    });
    dom.meta.appendChild(fu);
  }
}

/** 回复到达:填正文、挂 chips、收起过程卡。 */
function finalizeBotTurn(dom, r) {
  dom.gotReply = true;
  dom.bubble.innerHTML = renderMd(r.reply || "(空回复)");
  addMetaChips(dom, r);
  if (dom.trace) {
    dom.traceTitle.textContent = traceSummary(dom.counts);
    collapseTrace(dom.trace);
  }
  scrollBottom();
}

function showErrorInTurn(dom, msg) {
  dom.bubble.innerHTML = `<span class="err">⚠️ ${escapeHtml(msg)}</span>`;
  if (dom.trace) {
    dom.traceTitle.textContent = traceSummary(dom.counts);
  }
  scrollBottom();
}

/* ---------- 转写恢复(刷新/切换会话) ---------- */

function renderTranscript(sid) {
  innerEl.innerHTML = "";
  const turns = sid ? loadTranscript(sid) : [];
  if (!turns.length) {
    renderGreeting();
    return;
  }
  for (const t of turns) {
    if (t.role === "user") { addUserTurn(t.text, t.ts); continue; }
    const dom = beginBotTurn();
    for (const ev of t.trace || []) {
      if (ev.t === "route") applyTraceEvent(dom, "route", { target: ev.target });
      else if (ev.t === "thought") applyTraceEvent(dom, "thought", { text: ev.text });
      else if (ev.t === "tool_call") applyTraceEvent(dom, "tool_call", ev);
      else if (ev.t === "tool_result") applyTraceEvent(dom, "tool_result", ev);
      else if (ev.t === "error") dom.errorText = ev.detail || "";
    }
    if (t.error && !t.text) { showErrorInTurn(dom, t.error); continue; }
    finalizeBotTurn(dom, {
      reply: t.text, intent: t.intent, confidence: t.confidence,
      requires_human: t.requiresHuman, follow_up_question: t.followUp,
    });
  }
  scrollBottom(true);
}

function renderGreeting() {
  const dom = beginBotTurn();
  finalizeBotTurn(dom, {
    reply: "您好，我是并夕夕智能客服小夕 👋\n查订单、催发货、退换货、商品推荐、优惠活动都可以问我～",
    intent: "greeting", confidence: 1,
  });
  const wrap = document.createElement("div");
  wrap.className = "examples";
  for (const q of EXAMPLES) {
    const b = document.createElement("button");
    b.textContent = q;
    b.addEventListener("click", () => { wrap.remove(); send(q); });
    wrap.appendChild(b);
  }
  innerEl.appendChild(wrap);
}

/* ---------- SSE 流式 ---------- */

function parseSseFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  if (!dataLines.length) return null;
  let data;
  try { data = JSON.parse(dataLines.join("\n")); } catch { data = { raw: dataLines.join("\n") }; }
  return { event, data };
}

async function streamChat(body, signal, onEvent) {
  const resp = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const j = await resp.json();
      if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* 保留状态码 */ }
    const e = new Error(detail);
    e.status = resp.status;
    throw e;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseSseFrame(frame);
      if (ev) onEvent(ev.event, ev.data);
    }
  }
}

/* ---------- 发送主流程 ---------- */

function currentTranscript() {
  return state.activeId ? loadTranscript(state.activeId) : [];
}
function appendTurn(t) {
  if (!state.activeId) return;
  const turns = loadTranscript(state.activeId);
  turns.push(t);
  saveTranscript(state.activeId, turns);
}

async function send(preset) {
  const text = (preset !== undefined ? preset : inputEl.value).trim();
  if (!text) return;
  if (state.streaming) { toast("小夕正在回复中,请稍候…"); return; }
  inputEl.value = "";
  autoGrow();

  if (!state.activeId) {
    const s = newSessionObj(state.userId);
    setActive(s.id);
    renderSessions();
    innerEl.innerHTML = "";
  }
  // 首条消息作为会话标题
  const sess = activeSession();
  if (sess && sess.title === "新会话") {
    sess.title = text.length > 24 ? text.slice(0, 24) + "…" : text;
  }
  sess.updatedAt = new Date().toISOString();
  saveJson(SESSIONS_KEY, state.sessions);
  renderSessions();
  titleEl.textContent = sess.title;

  addUserTurn(text);
  appendTurn({ role: "user", text, ts: new Date().toISOString() });

  const dom = beginBotTurn();
  const traceLog = [];   // 过程事件快照:落转写缓存,刷新后可还原
  state.streaming = true;
  state.abortCtrl = new AbortController();
  setSendMode("stop");
  announce("正在处理您的消息");
  let replyData = null;

  try {
    await streamChat(
      { user_id: state.userId, session_id: state.activeId, message: text },
      state.abortCtrl.signal,
      (type, data) => {
        if (type === "meta") {
          if (data.session_id && data.session_id !== state.activeId) {
            // 服务端归一化后的 session_id(当前实现保持传入值)
            state.activeId = data.session_id;
          }
          return;
        }
        if (type === "reply") { replyData = data; finalizeBotTurn(dom, data); return; }
        if (type === "end") return;
        traceLog.push(normalizeTrace(type, data));
        applyTraceEvent(dom, type, data);
      },
    );
    if (replyData) {
      appendTurn({
        role: "bot", text: replyData.reply, intent: replyData.intent,
        confidence: replyData.confidence, requiresHuman: !!replyData.requires_human,
        followUp: replyData.follow_up_question || "", trace: traceLog,
        ts: new Date().toISOString(),
      });
      announce("小夕已回复");
    } else if (dom.errorText) {
      showErrorInTurn(dom, dom.errorText);
      appendTurn({ role: "bot", error: dom.errorText, trace: traceLog, ts: new Date().toISOString() });
      announce("本轮回复失败");
    } else {
      showErrorInTurn(dom, "连接中断,未收到回复");
      announce("连接中断,未收到回复");
    }
  } catch (e) {
    if (e.name === "AbortError") {
      dom.bubble.innerHTML = `<span class="err">⏹ 已停止等待回复(本轮服务端仍会完成并入账)</span>`;
      appendTurn({ role: "bot", error: "已停止等待回复", trace: traceLog, ts: new Date().toISOString() });
      announce("已停止等待回复");
    } else {
      showErrorInTurn(dom, e.message || String(e));
      appendTurn({ role: "bot", error: e.message || String(e), trace: traceLog, ts: new Date().toISOString() });
      announce("回复失败");
    }
  } finally {
    state.streaming = false;
    state.abortCtrl = null;
    setSendMode("send");
    inputEl.focus();
  }
}

/* 过程事件 → 可缓存的精简结构(转写还原用)。 */
function normalizeTrace(type, data) {
  if (type === "route") return { t: "route", target: data.target || "" };
  if (type === "thought") return { t: "thought", text: data.text || "" };
  if (type === "tool_call") {
    return { t: "tool_call", tool_call_id: data.tool_call_id, name: data.name, arguments: data.arguments };
  }
  if (type === "tool_result") {
    return { t: "tool_result", tool_call_id: data.tool_call_id, tool_name: data.tool_name, result: data.result };
  }
  if (type === "error") return { t: "error", detail: data.detail || "" };
  return null;
}

function setSendMode(mode) {
  if (mode === "stop") {
    sendBtn.textContent = "停止";
    sendBtn.classList.add("stop");
    sendBtn.disabled = false;
  } else {
    sendBtn.textContent = "发送";
    sendBtn.classList.remove("stop");
    sendBtn.disabled = false;
  }
}

/* ---------- 会话操作 ---------- */

function switchSession(id) {
  if (state.streaming) { toast("小夕正在回复中,请稍候…"); return; }
  setActive(id);
  const sess = activeSession();
  renderSessions();
  titleEl.textContent = sess ? sess.title : "新会话";
  renderTranscript(id);
  setSidebar(false);
}

function removeSession(id) {
  if (state.streaming) { toast("小夕正在回复中,请稍候…"); return; }
  state.sessions = state.sessions.filter(s => s.id !== id);
  saveJson(SESSIONS_KEY, state.sessions);
  try { localStorage.removeItem(transcriptKey(id)); } catch {}
  if (state.activeId === id) {
    const list = sessionsOf(state.userId);
    setActive(list.length ? list[0].id : "");
    const sess = activeSession();
    titleEl.textContent = sess ? sess.title : "新会话";
    renderTranscript(state.activeId);
  }
  renderSessions();
  toast("已移除会话记录(仅本地)");
}

async function resetSession() {
  if (state.streaming) { toast("小夕正在回复中,请稍候…"); return; }
  if (!state.activeId) return;
  try {
    await api("/v1/sessions/reset", {
      method: "POST",
      body: JSON.stringify({ user_id: state.userId, session_id: state.activeId }),
    });
    saveTranscript(state.activeId, []);
    renderTranscript(state.activeId);
    toast("已清空该会话的服务端上下文", "ok");
  } catch (e) {
    toast("重置失败:" + e.message, "err");
  }
}

/* ---------- 健康探针 ---------- */

async function pollHealth() {
  const dot = $("#healthDot");
  const label = $("#healthText");
  try {
    const resp = await fetch("/readyz");
    dot.className = "dot " + (resp.ok ? "ok" : "bad");
    label.textContent = resp.ok ? "服务就绪" : "服务异常";
  } catch {
    dot.className = "dot bad";
    label.textContent = "服务未连接";
  }
  dot.setAttribute("aria-label", "服务状态:" + label.textContent);
}

/* ---------- 输入框 ---------- */

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 132) + "px";
  // 内容未超过上限时不显示内部滚动条(避免移动端空态出现滚动箭头)
  inputEl.style.overflowY = inputEl.scrollHeight > 132 ? "auto" : "hidden";
}

/* ---------- 事件绑定与启动 ---------- */

sendBtn.addEventListener("click", () => {
  if (state.streaming) state.abortCtrl?.abort();
  else send();
});
inputEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
inputEl.addEventListener("input", autoGrow);
$("#btnNew").addEventListener("click", () => {
  if (state.streaming) { toast("小夕正在回复中,请稍候…"); return; }
  const s = newSessionObj(state.userId);
  setActive(s.id);
  renderSessions();
  titleEl.textContent = s.title;
  innerEl.innerHTML = "";
  renderGreeting();
  setSidebar(false);
});
$("#btnReset").addEventListener("click", resetSession);
$("#userIdInput").value = state.userId;
$("#userIdInput").addEventListener("change", e => {
  const v = e.target.value.trim() || "web-user";
  state.userId = v;
  localStorage.setItem(USER_KEY, v);
  bindActive(v);
  const sess = activeSession();
  titleEl.textContent = sess ? sess.title : "新会话";
  renderSessions();
  renderTranscript(state.activeId);
  setSidebar(false);
});

/* 移动端侧栏抽屉:遮罩点击 / Escape 关闭,打开时锁定页面滚动;
   关闭时侧栏不可聚焦(inert),打开时主内容 inert,焦点不进背景 */
const sidebarEl = $("#sidebar");
const maskEl = $("#sidebarMask");
const openSidebarBtn = $("#openSidebar");
const mainEl = $(".main");
const mobileMq = window.matchMedia("(max-width: 760px)");

function syncDrawerInert() {
  const open = sidebarEl.classList.contains("open");
  sidebarEl.inert = mobileMq.matches && !open;
  mainEl.inert = mobileMq.matches && open;
}

function setSidebar(open) {
  const wasOpen = sidebarEl.classList.contains("open");
  sidebarEl.classList.toggle("open", open);
  maskEl.hidden = !open;
  document.body.classList.toggle("modal-open", open);
  openSidebarBtn.setAttribute("aria-expanded", String(open));
  syncDrawerInert();
  if (!wasOpen && open) {
    $("#closeSidebar").focus();
  } else if (wasOpen && !open && mobileMq.matches) {
    // 侧栏转 inert 会丢焦点,还给菜单按钮
    const ae = document.activeElement;
    if (!ae || ae === document.body || sidebarEl.contains(ae)) openSidebarBtn.focus();
  }
}
openSidebarBtn.addEventListener("click", () => setSidebar(true));
$("#closeSidebar").addEventListener("click", () => setSidebar(false));
maskEl.addEventListener("click", () => setSidebar(false));
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && sidebarEl.classList.contains("open")) setSidebar(false);
});
mobileMq.addEventListener("change", syncDrawerInert);
syncDrawerInert();

bindActive(state.userId);
const initial = activeSession();
titleEl.textContent = initial ? initial.title : "新会话";
renderSessions();
renderTranscript(state.activeId);
pollHealth();
setInterval(pollHealth, 20000);

/* 共享工具(对话页 index.html 与运营台 ops.html 共用)。无依赖、无构建链。 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/** 轻量富文本:转义 HTML → `code` → **粗体** → 换行。
 * 服务端回复视为半可信,一律先转义再叠加标记。 */
function renderMd(t) {
  let s = escapeHtml(t);
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");
  return s;
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return String(iso);
  const now = new Date();
  const pad = n => String(n).padStart(2, "0");
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (d.toDateString() === now.toDateString()) return hm;
  const md = `${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  return d.getFullYear() === now.getFullYear() ? `${md} ${hm}` : `${d.getFullYear()}-${md} ${hm}`;
}

/** JSON 美化;超长截断,避免 process 卡片被单条超大结果撑爆。 */
function prettyJson(v, maxLen = 4000) {
  let s;
  if (typeof v === "string") {
    try { s = JSON.stringify(JSON.parse(v), null, 2); } catch { s = v; }
  } else {
    try { s = JSON.stringify(v, null, 2); } catch { s = String(v); }
  }
  if (s.length > maxLen) s = s.slice(0, maxLen) + "\n…(已截断)";
  return s;
}

/** 统一 fetch:非 2xx 抛 Error(优先取 body.detail),2xx 解析 JSON。 */
async function api(path, opts = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const j = await resp.json();
      if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* 非 JSON 错误体,保留 HTTP 状态码 */ }
    const e = new Error(detail);
    e.status = resp.status;
    throw e;
  }
  return resp.json();
}

function toast(msg, kind = "info", ms = 2600) {
  let box = $("#toast-box");
  if (!box) {
    box = document.createElement("div");
    box.id = "toast-box";
    document.body.appendChild(box);
  }
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  box.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

/** 向屏幕阅读器播报状态变化(页面上的 sr-only status 区域)。 */
function announce(msg) {
  const el = $("#srStatus");
  if (!el) return;
  el.textContent = "";
  setTimeout(() => { el.textContent = msg; }, 30);
}

/** 空态/加载态组件(对话页与运营台共用)。 */
function emptyState(icon, title, hint = "") {
  const d = document.createElement("div");
  d.className = "empty-state";
  d.innerHTML = `<span class="es-icon" aria-hidden="true">${icon}</span>
    <div class="es-title">${escapeHtml(title)}</div>` +
    (hint ? `<div class="es-hint">${escapeHtml(hint)}</div>` : "");
  return d;
}

function loadingHint(text = "加载中…") {
  const d = document.createElement("div");
  d.className = "loading-hint";
  d.setAttribute("role", "status");
  d.textContent = text;
  return d;
}

const INTENT_LABELS = {
  order_query: "订单查询", return_request: "退换货", product_consult: "商品咨询",
  complaint: "投诉", after_sale: "售后服务", promotion: "优惠活动",
  account: "账户问题", greeting: "打招呼", other: "其他",
};

const TOOL_LABELS = {
  query_order: "查询订单",
  query_product: "查询商品",
  query_logistics: "查询物流",
  apply_refund: "申请退款",
  search_knowledge: "知识库检索",
  list_user_orders: "查询用户订单列表",
  recall_user_memory: "召回用户记忆",
  load_skill: "加载技能",
};

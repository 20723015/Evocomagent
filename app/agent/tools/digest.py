"""工具结果结构化精简（记忆/摘要转录层 · 单一事实源）。

动机：extraction.py 与 summarizer.py 此前各复制一份 `content[:200]` 前缀截断——
同一份工具结果信息量极低（订单第 8 项、物流最新事件都被切掉），且两处
曾因复制漂移。本模块把「工具结果 → 转录片段」收拢为一处：

- `digest_tool_result`：按工具投影表降维（三层降级，确定性，绝不抛异常）；
- `render_tool_result_line`：转录行渲染（`[工具结果] {tool_name} {digest}`，
  tool_digest_enabled=False 时整体回退旧版前缀截断——golden 兼容）；
- `tool_call_name_map`：tool_call_id → 工具名（tool 行自足，不依赖前一行）。

投影表字段**逐一对照真实返回结构核实**（见 docs/工具结果结构化精简计划.md）：
query_product 的字段在 `products[]` 列表内；物流字段名是 `carrier`。

明确不动：ReAct 循环内模型看到的完整结果、会话持久化、TurnRecorder。
"""

from __future__ import annotations

import json
from typing import Optional

from app.config.settings import settings

# 字级截断常量（钉死）：单条 60 / 短字段 40
_DETAIL_CUT = 60
_TEXT_CUT = 40
_LIST_MAX = 10   # list_user_orders 投影项数上限
_PRODUCT_MAX = 2  # query_product 投影项数上限
_DOC_MAX = 5     # search_knowledge doc 名上限


def digest_tool_result(tool_name: str, content: str, budget: int | None = None) -> str:
    """把工具返回的 JSON 字符串压缩成转录片段（三层降级，绝不抛异常）。

    1. JSON 解析成功且在投影表 → 按真实字段投影；
    2. JSON 解析成功但不在表内（含未来新增工具）→ 通用压缩：仅保留顶层
       标量（str 截 40 字），丢弃嵌套对象/长文本；
    3. 解析失败或任何内部异常 → 旧版前缀截断 content[:budget]。
    """
    budget = _budget(budget)
    try:
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("工具结果不是 JSON 对象")
        if tool_name == "recall_user_memory":
            return _clamp(content, budget)  # 原样：本身已短，仅预算兜底
        projector = _PROJECTORS.get(tool_name)
        if projector is not None:
            text = projector(payload, budget)
        else:
            text = _generic_compress(payload)
        return _clamp(text, budget)
    except Exception:  # noqa: BLE001 —— 三层降级兜底：绝不抛异常
        return _prefix(content, budget)


def render_tool_result_line(
    tool_name: str, content: str, budget: int | None = None,
) -> str:
    """转录行渲染。开关关闭时与旧版输出逐字节一致（无工具名、前缀截断）。"""
    if not settings.tool_digest_enabled:
        return f"[工具结果] {_prefix(content, budget)}"
    return f"[工具结果] {tool_name} {digest_tool_result(tool_name, content, budget=budget)}"


def tool_call_name_map(messages: list[dict]) -> dict[str, str]:
    """tool_call_id → 工具名（从 assistant 消息的 tool_calls 反查）。"""
    call_map: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            call_id = tc.get("id")
            if not call_id:
                continue
            func = tc.get("function") or {}
            call_map[call_id] = str(func.get("name", "") or "?")
    return call_map


# ============================================================
# 投影表（字段逐一对照真实返回结构）
# ============================================================
def _project_order(payload: dict, budget: int = 0) -> str:
    order = payload.get("order") or {}
    items = order.get("items") or []
    name = items[0].get("name") if items and isinstance(items[0], dict) else ""
    parts = [
        f"order_id={_cut(str(order.get('order_id', '')), _TEXT_CUT)}",
        f"status={_cut(str(order.get('status', '')), _TEXT_CUT)}",
        f"total={order.get('total', '')}",
        f"tracking_number={_cut(str(order.get('tracking_number', '')), _TEXT_CUT)}",
    ]
    if name:
        parts.append(f"商品={_cut(str(name), _TEXT_CUT)}")
    return " ".join(parts)


def _project_product(payload: dict, budget: int = 0) -> str:
    items = payload.get("products") or []
    out: list[str] = []
    for item in items[:_PRODUCT_MAX]:
        if not isinstance(item, dict):
            continue
        out.append(
            "商品={} price={} stock={} product_id={}".format(
                _cut(str(item.get("name", "")), _TEXT_CUT),
                item.get("price", ""), item.get("stock", ""), item.get("product_id", ""),
            )
        )
    extra = len(items) - _PRODUCT_MAX
    if extra > 0:
        out.append(f"…另有{extra}项")
    return "; ".join(out)


def _project_logistics(payload: dict, budget: int = 0) -> str:
    logi = payload.get("logistics") or {}
    events = logi.get("events") or []
    parts = [
        f"tracking_number={_cut(str(logi.get('tracking_number', '')), _TEXT_CUT)}",
        f"carrier={_cut(str(logi.get('carrier', '')), _TEXT_CUT)}",
        f"status={_cut(str(logi.get('status', '')), _TEXT_CUT)}",
    ]
    if events and isinstance(events[-1], dict):
        last = events[-1]
        parts.append(
            f"最新={_cut(str(last.get('time', '')), _TEXT_CUT)} "
            f"{_cut(str(last.get('description', '')), _TEXT_CUT)}"
        )
    return " ".join(parts)


def _project_user_orders(payload: dict, budget: int) -> str:
    """逐项投影：order_id + status 恒保（预算内最大化 items_summary）。

    执行修正（钉死事实时发现计划矛盾：10 项 × 60 字 = 600 ≫ 200 总预算，
    "每项截 60"与"≤200"与"第 8 项保留"数学上不可兼得）——按动机场景
    优先：摘要从 60 字档逐级压缩（60→30→16→8→0），预算不足时放弃摘要
    段；每项 order_id + status 全保，相关订单在第 8 个也不会被切掉。
    """
    items = payload.get("orders") or []
    visible = [o for o in items[:_LIST_MAX] if isinstance(o, dict)]
    extra = int(payload.get("count") or len(items)) - len(visible)
    tail = f"…另有{extra}单" if extra > 0 else ""

    for trunc in (_DETAIL_CUT, 30, 16, 8, 0):
        parts = []
        for o in visible:
            entry = f"{o.get('order_id', '')} {o.get('status', '')}".strip()
            summary = _cut(str(o.get("items_summary", "") or ""), trunc)
            if summary:
                entry = f"{entry} {summary}"
            parts.append(entry)
        body = " | ".join(parts)
        if tail:
            body = f"{body} {tail}"
        if len(body) <= budget:
            return body
        # 0 档仍超：分隔符收紧（10 项 + 尾标与预算只差数字符时保尾标/保项）
        if trunc == 0:
            parts = [p.replace("  ", " ") for p in parts]
            body = "|".join(parts)
            if tail:
                body = f"{body} {tail}"
            if len(body) <= budget:
                return body
    return body  # 理论不可达


def _project_knowledge(payload: dict, budget: int = 0) -> str:
    docs: list[str] = []
    for r in payload.get("results") or []:
        if not isinstance(r, dict):
            continue
        doc = str(r.get("doc", "") or "").strip()
        if doc and doc not in docs:
            docs.append(doc)  # 只取 doc 名，不碰 text（提取目标是用户事实）
    if not docs:
        return "命中0篇"
    names = "、".join(docs[:_DOC_MAX])
    if len(docs) > _DOC_MAX:
        names += f"…等{len(docs)}篇"
    return f"命中{len(docs)}篇: {names}"


def _project_refund(payload: dict, budget: int = 0) -> str:
    parts = [f"success={payload.get('success', '')}"]
    for key in ("status", "message", "error"):
        value = payload.get(key)
        if value:
            parts.append(f"{key}={_cut(str(value), _DETAIL_CUT)}")
    return " ".join(parts)


def _project_skill(payload: dict, budget: int = 0) -> str:
    if payload.get("success") and payload.get("skill_name"):
        return f"已加载技能: {payload['skill_name']}（指令略）"
    return _generic_compress(payload)  # 失败：走通用压缩（含 error）


_PROJECTORS = {
    "query_order": _project_order,
    "query_product": _project_product,
    "query_logistics": _project_logistics,
    "list_user_orders": _project_user_orders,
    "search_knowledge": _project_knowledge,
    "apply_refund": _project_refund,
    "load_skill": _project_skill,
    # recall_user_memory：原样（在 digest_tool_result 内特判）
}


# ============================================================
# 通用压缩 / 截断工具（确定性）
# ============================================================
def _generic_compress(payload: dict) -> str:
    """未知工具/未来新工具：仅保留顶层标量（str 截 40 字），嵌套与长文本丢弃。"""
    keep: dict = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str):
            keep[key] = _cut(value, _TEXT_CUT)
        elif isinstance(value, (int, float, bool)):
            keep[key] = value
    return json.dumps(keep, ensure_ascii=False, sort_keys=True)


def _budget(budget: Optional[int]) -> int:
    if budget is None:
        budget = settings.tool_digest_budget_chars
    return max(int(budget), 1)


def _cut(text: str, limit: int) -> str:
    """字段级截断：超限带省略号（limit ≤ 0 返回空——0 档=放弃该字段）。"""
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _prefix(content: str, budget: int) -> str:
    """旧版前缀截断（golden 兼容：开关关闭/第 3 层降级都走它）。"""
    budget = _budget(budget)
    return content if len(content) <= budget else content[:budget] + "..."


def _clamp(text: str, budget: int) -> str:
    """总预算夹逼：投影后仍超预算则截到预算（带 …），token 成本零回归。"""
    return text if len(text) <= budget else text[: budget - 1] + "…"

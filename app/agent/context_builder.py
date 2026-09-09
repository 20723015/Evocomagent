"""ContextBuilder（单 Agent 全量优化计划·阶段A/B）：模型上下文唯一构建点。

职责：
- 系统提示 + Skill 目录；
- 记忆片段（MemoryManager 相关性筛选结果，按 10% 预算裁剪）；
- 交互摘要（history summary）；
- 对话历史（按 50% 水位裁剪；旧版 JSON 助手消息在此折叠为纯文本）。

模型上下文与持久化历史是两种不同的数据形态：消息里的内部
``metadata`` 只供服务端持久化、审计和跨轮状态迁移使用，绝不发送给
Chat Completions。

阶段B约定：
- 每轮历史只保存一个 assistant 消息：content = 用户可见回复，结构化字段在
  消息 metadata（新格式）；构建模型上下文时把旧版「JSON 副本」content 折叠
  为 reply 文本——只折叠模型窗口，不修改历史正本；
- 历史窗口按 token 水位裁剪（不再按消息条数），孤儿 tool 消息不进窗口；
- 上一轮的 pending 写操作（退款确认令牌）以受控系统说明注入，保证跨轮
  两段式确认可用（工具完整结果不再进入后续上下文）。
"""

from __future__ import annotations

import json

from app.agent.token_budget import (
    budget_shares,
    estimate_messages_tokens,
    estimate_tokens,
    trim_messages_to_budget,
)
from app.prompts.customer_service import SYSTEM_PROMPT

# metadata 承载键（新格式 assistant 消息）
METADATA_KEY = "metadata"
_PENDING_WRITE_FIELDS = ("tool", "order_id", "reason")  # 不含凭证/幂等键（Review 修复）


def assistant_metadata(msg: dict) -> dict:
    return msg.get(METADATA_KEY) or {}


def _fold_assistant_content(content: str) -> tuple[str, dict | None]:
    """旧版助手消息折叠：content 为结构化 JSON 副本 → (reply 文本, 元数据)。

    新格式（content 为纯文本）原样返回；非 JSON / 无 reply 键不折叠——
    模型在 ReAct 中间步骤的普通 assistant 文本（思路说明）也原样保留。
    """
    text = (content or "").strip()
    if not text.startswith("{"):
        return content, None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return content, None
    if not isinstance(data, dict) or "reply" not in data:
        return content, None
    reply = str(data.get("reply") or "")
    metadata = {
        k: data[k] for k in ("intent", "confidence", "requires_human",
                             "follow_up_question")
        if k in data
    }
    return reply, (metadata or None)


# 模型窗口必须剔除的 metadata.pending_writes 保留字段（Review 修复：
# token/幂等键只存确认存储，仅供服务端迁移，绝不回注模型上下文）
_PENDING_RESERVED_KEYS = ("confirmation_token", "idempotency_key", "refund_id",
                          "token", "access_token")


def _strip_persistence_metadata(msg: dict) -> dict:
    """返回可发送给模型的消息副本。

    ``metadata`` 是持久化层的内部字段，不是 Chat Completions message
    schema 的字段。这里必须对所有角色统一删除，不能只处理 assistant，
    也不能只删除 token 等敏感子字段；否则 OpenAI 请求会收到未定义字段，
    且内部状态仍可能泄漏到模型上下文。
    """
    return {key: value for key, value in msg.items() if key != METADATA_KEY}


def fold_history(messages: list[dict]) -> list[dict]:
    """构建模型窗口用的折叠视图（不修改传入列表与正本）。

    - 旧版 JSON 助手消息 → 纯文本 reply；
    - 所有消息的持久化 ``metadata`` 整体剥离；
    - metadata 中的工具摘要不展开为消息。

    该函数的返回值会直接进入 OpenAI 请求，因此不能携带持久化层字段。
    ``raw_messages`` 仍保留完整 metadata，供保存、审计和服务端确认逻辑
    使用。
    """
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" not in msg:
            content, _ = _fold_assistant_content(str(msg.get("content") or ""))
            folded = _strip_persistence_metadata(msg)
            folded["content"] = content
            out.append(folded)
            continue
        out.append(_strip_persistence_metadata(msg))
    return out


def pending_write_note(messages: list[dict], decision=None) -> str | None:
    """扫描最近一条助手消息的 metadata.pending_writes → 跨轮确认系统说明。

    Review 修复：说明**不含任何确认凭证/幂等键**（token 只存确认存储，
    由服务端判定 confirm 后经执行器内部通道注入，模型永不接触）；
    说明内容随本轮服务端确认判定变化（confirm/cancel/等待确认）。
    decision: refund_gate.ConfirmationDecision（pipeline 注入；可 None）。
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant" or "tool_calls" in msg:
            continue
        pending = assistant_metadata(msg).get("pending_writes") or []
        # 兼容旧 metadata：剔除 token/幂等键等保留字段（仅供服务端迁移）
        pending = [
            {k: v for k, v in entry.items()
             if k not in _PENDING_RESERVED_KEYS}
            for entry in pending if isinstance(entry, dict)
        ]
        pending = [entry for entry in pending if entry]
        if not pending:
            return None
        action = getattr(decision, "action", "none")
        matched = getattr(decision, "matched_order_id", "")
        if action == "cancel":
            return None  # 已取消：不再注入待确认说明
        lines = []
        for w in pending:
            tool = str(w.get("tool", ""))
            fields = [f"{k}={w[k]}" for k in _PENDING_WRITE_FIELDS
                      if w.get(k) not in (None, "")]
            lines.append(f"- {tool}: {'；'.join(fields)}")
        if action == "confirm" and matched:
            target = next((w for w in pending
                           if w.get("order_id") == matched), pending[0])
            return (
                "用户已明确确认退款（服务端已判定，凭证由系统自动注入）：\n"
                f"- 请立即调用 {target.get('tool', 'apply_refund')}"
                f"（order_id={target.get('order_id', '')}，"
                f"reason={target.get('reason', '')}）完成提交。\n"
                "- 不要向用户索要、复述或编造任何凭证/令牌字段。"
            )
        return (
            "上一轮你已发起需要用户确认的写操作（尚未完成）：\n"
            + "\n".join(lines)
            + "\n等待用户明确确认；用户确认后再次调用对应工具（相同订单号与原因），"
              "系统会自动附加确认凭证完成提交。"
              "不要向用户索要、复述或编造任何凭证/令牌字段。"
        )
    return None


class ContextBuilder:
    """按 token 水位构建模型上下文（无 LLM 调用，纯确定性）。"""

    def __init__(self, memory_manager=None, skill_manager=None,
                 context_window_tokens: int = 32768):
        self._memory = memory_manager
        self._skills = skill_manager
        self._window = max(int(context_window_tokens), 4096)
        self.last_context_tokens = 0  # 最近一次 build 的上下文估算 token

    def build(self, agent, query: str = "") -> list[dict]:
        """组装完整模型消息列表（system → memory → summary → history）。"""
        shares = budget_shares(self._window)
        system_content = SYSTEM_PROMPT
        if self._skills and self._skills.enabled:
            system_content += self._skills.build_catalog_prompt()
        system_content += _final_response_protocol()
        messages: list[dict] = [{"role": "system", "content": system_content}]

        # Memory（≤10% 预算；MemoryManager 已做相关性筛选）
        if self._memory is not None:
            sections = self._memory.build_memory_prompt_sections(query)
            used = 0
            for section in sections:
                cost = estimate_tokens(str(section.get("content", "")))
                if used + cost > shares["memory"]:
                    break
                messages.append(section)
                used += cost

        # 交互摘要
        if agent.summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{agent.summary}",
            })

        # 跨轮 pending 写操作说明（受控字段）
        note = pending_write_note(
            agent.raw_messages, getattr(agent, "current_confirmation", None),
        )
        if note:
            messages.append({"role": "system", "content": note})

        # 对话历史（折叠视图 + 50% 水位裁剪）
        folded = fold_history(agent.raw_messages)
        kept = trim_messages_to_budget(folded, shares["dialog"])
        messages.extend(kept)
        self.last_context_tokens = estimate_messages_tokens(messages)
        return messages

    def history_overflow(self, raw_messages: list[dict]) -> tuple[bool, list[dict]]:
        """判断折叠后历史是否超过水位（压缩触发依据）。

        返回 (是否超限, 折叠视图)。
        """
        folded = fold_history(raw_messages)
        shares = budget_shares(self._window)
        return estimate_messages_tokens(folded) > shares["dialog"], folded


def _final_response_protocol() -> str:
    from app.agent.final_response import FINAL_RESPONSE_INSTRUCTIONS

    return FINAL_RESPONSE_INSTRUCTIONS

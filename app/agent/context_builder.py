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
- 历史窗口按 token 水位裁剪（不再按消息条数），孤儿 tool 消息不进窗口。
"""

from __future__ import annotations

import json

from app.agent.input_policy import emotion_tone_hint
from app.agent.token_budget import (
    budget_shares,
    estimate_messages_tokens,
    estimate_tokens,
    trim_messages_to_budget,
)
from app.prompts.customer_service import SYSTEM_PROMPT

# metadata 承载键（新格式 assistant 消息）
METADATA_KEY = "metadata"


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

    - 旧版 JSON 助手消息 → 纯文本 reply（仅字符串 content；结构化块列表原样透传）；
    - 所有消息的持久化 ``metadata`` 整体剥离；
    - metadata 中的工具摘要不展开为消息。

    该函数的返回值会直接进入 OpenAI 请求，因此不能携带持久化层字段。
    ``raw_messages`` 仍保留完整 metadata，供保存与审计使用。
    """
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" not in msg:
            raw = msg.get("content")
            if isinstance(raw, str):
                content, _ = _fold_assistant_content(raw)
                folded = _strip_persistence_metadata(msg)
                folded["content"] = content
                out.append(folded)
            else:
                # 推理模型适配 T3：结构化 content（thinking 块列表）**原样**透传。
                # 早期实现无条件 str() 会把块列表变成 Python repr——既丢 signature
                # （厂商 400），也让 required_signed 画像的窗口回传断在这里。
                out.append(_strip_persistence_metadata(msg))
            continue
        out.append(_strip_persistence_metadata(msg))
    return out


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

        # Memory（≤10% 预算；MemoryManager 已做相关性筛选；STM 段在前）
        if self._memory is not None:
            # 记忆系统重构·阶段2.5：query 嵌入每轮至多一次（预算内、失败
            # 降级纯词面）；语义开关关闭时 get_memory_embedder 返回 None，
            # 零额外调用
            query_embedding = None
            if query and query.strip():
                from app.llm.embeddings import get_memory_embedder

                embedder = get_memory_embedder()
                if embedder is not None:
                    query_embedding = embedder.encode(query, stage="query")
            sections = self._memory.build_memory_prompt_sections(
                query, query_embedding=query_embedding,
            )
            used = 0
            for section in sections:
                cost = estimate_tokens(str(section.get("content", "")))
                if used + cost > shares["memory"]:
                    # 批次7（Review #8）：超预算 continue 而非 break——
                    # 大的 LTM 段不得吞掉后面更小、更新的 STM 段
                    continue
                messages.append(section)
                used += cost

        # 情绪语气提示（P1-1）：不满级才注入，独立 system 消息放在记忆之后、
        # 摘要之前——语气属"本轮交互策略"，紧跟记忆（个性化依据）之后、早于
        # 历史摘要更符合「先看本轮语气再看历史」的阅读顺序，且不会被摘要/历史
        # 裁剪逻辑改动。提示很短（约 100 字），直接 append：system 段（角色提示 +
        # 技能目录 + 终答协议）本来就不走 token 水位裁剪，单条短提示不改变预算口径。
        tone_hint = emotion_tone_hint(query)
        if tone_hint:
            messages.append({"role": "system", "content": tone_hint})

        # 交互摘要
        if agent.summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{agent.summary}",
            })

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

"""上下文 token 水位（单 Agent 全量优化计划·阶段B）。

历史压缩从「消息条数」改为 token 水位。预算份额（计划原文）：
- 系统与 Skill：20%
- Memory：10%
- 最近对话及工具结果：50%
- 最终输出预留：20%

token 估算为确定性启发式（不依赖 tokenizer）：CJK 字符 ≈ 1 token/字，
ASCII 连续段 ≈ 4 字符/token——只用于水位控制，不用于计费。
"""

from __future__ import annotations

import json
import re

# 预算份额（默认值；settings 可覆盖用于评测）
SHARE_SYSTEM = 0.20
SHARE_MEMORY = 0.10
SHARE_DIALOG = 0.50
SHARE_OUTPUT_RESERVE = 0.20

_ASCII_RUN = re.compile(r"[\x00-\x7f]+")


def estimate_tokens(text: str) -> int:
    """确定性 token 估算：CJK 记 1/字，ASCII 段记 len/4（下限 1）。"""
    if not text:
        return 0
    total = 0
    pos = 0
    for match in _ASCII_RUN.finditer(text):
        total += len(match.group(0)) // 4
        total += len(text[pos:match.start()])
        pos = match.end()
    total += len(text[pos:])
    return max(total, 1 if text.strip() else 0)


def estimate_message_tokens(message: dict) -> int:
    """单条消息的 token 估算（含 tool_calls 参数）。"""
    total = estimate_tokens(str(message.get("content") or ""))
    for tc in message.get("tool_calls") or []:
        func = tc.get("function") or {}
        total += estimate_tokens(str(func.get("arguments") or ""))
        total += estimate_tokens(str(func.get("name") or ""))
    return total + 4  # 角色与分隔开销


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def budget_shares(context_window_tokens: int) -> dict[str, int]:
    """按上下文窗口切预算（含输出预留）。"""
    window = max(int(context_window_tokens), 1024)
    return {
        "system": int(window * SHARE_SYSTEM),
        "memory": int(window * SHARE_MEMORY),
        "dialog": int(window * SHARE_DIALOG),
        "output_reserve": int(window * SHARE_OUTPUT_RESERVE),
    }


def trim_messages_to_budget(messages: list[dict], budget: int) -> list[dict]:
    """从尾部保留消息，直到预算耗尽；绝不从中间截断对话对。

    返回 (保留列表)；被裁掉的部分由调用方交摘要/压缩处理。
    user/assistant/tool 三元组尽量整组保留：遇到 tool 消息起头的窗口边界时
    向前回退到其所属 assistant(tool_calls) 消息，避免孤儿 tool 消息破坏
    OpenAI 对话约束。
    """
    if budget <= 0:
        return []
    kept: list[dict] = []
    used = 0
    for message in reversed(messages):
        cost = estimate_message_tokens(message)
        if kept and used + cost > budget:
            break
        if not kept and cost > budget:
            # 首条即超预算（极端）：仍保留最后一条，保底非空对话
            kept.append(message)
            break
        kept.append(message)
        used += cost
    kept.reverse()
    # 孤儿 tool 消息修复：窗口起头若是 tool，回退吞掉它（其父 assistant 已被裁掉）
    drop = 0
    for message in kept:
        if message.get("role") == "tool":
            drop += 1
        else:
            break
    if drop and drop < len(kept):
        kept = kept[drop:]
    elif drop and drop >= len(kept):
        return []
    return kept


def messages_token_json(messages: list[dict]) -> str:
    return json.dumps([m.get("content", "") for m in messages], ensure_ascii=False)

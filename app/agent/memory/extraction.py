"""LLM 事实提取：从对话中抽取长期记忆事实。

模式与 app/agent/summarizer.py 一致：格式化对话 → 调用 LLM → 解析结果。

记忆系统重构：会话内短期记忆（槽位层）已整体删除——会话内上下文由对话
历史 + rolling 摘要承担；本模块是 LTM 事实的唯一提取入口（memory job
异步巩固），写侧脱敏统一走 models.mask_sensitive。
"""

from __future__ import annotations

import json

from openai import OpenAI

from app.agent.memory.models import MemoryFact, MemoryMutation, mask_sensitive
from app.agent.tools.digest import (
    render_tool_result_line,
    tool_call_name_map,
    visible_assistant_text,
)
from app.evolution.sanitizer import MEMORY_INSTRUCTION_HINTS, has_injection
from app.prompts.memory import LTM_EXTRACTION_PROMPT


def _build_transcript(messages: list[dict]) -> str:
    """将消息列表格式化为文本摘要（复用 summarizer 的逻辑）。

    工具结果行经 digest 投影（按工具字段降维，工具名自 tool_call_id 反查），
    与 summarizer.py 共用单一直现——不再各复制一份前缀截断。assistant 的
    块列表 content（推理画像回传形态）同样走 digest.visible_assistant_text，
    只取 text 块，避免渲染成 Python repr。
    """
    lines = []
    call_map = tool_call_name_map(messages)
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    lines.append(f"客服：[调用工具 {name}]")
            text = visible_assistant_text(msg.get("content"))
            if text:
                lines.append(f"客服：{text}")
        elif role == "tool":
            name = call_map.get(msg.get("tool_call_id"), "?")
            lines.append(render_tool_result_line(name, content))

    return "\n".join(lines)


def extract_long_term_facts(
    client: OpenAI,
    model: str,
    messages: list[dict],
    summary: str | None,
    existing_facts: list,
) -> tuple[list[MemoryMutation], str]:
    """从完整会话中提取长期记忆事实 + 交互摘要。"""
    parts = []
    if summary:
        parts.append(f"【对话摘要】\n{summary}")

    transcript = _build_transcript(messages)
    if transcript:
        parts.append(f"【对话内容】\n{transcript}")

    if not parts:
        return [], ""

    existing_text = _render_existing(existing_facts)
    prompt = LTM_EXTRACTION_PROMPT.format(existing_ltm=existing_text)

    from app.config.settings import settings

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        max_tokens=settings.llm_max_tokens,
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [], ""
    if not isinstance(data, dict):
        return [], ""
    return _parse_mutation_items(
        data.get("mutations", []), _user_evidence(messages), source="ltm",
    ), str(
        data.get("interaction_summary", "") or ""
    ).strip()


def _render_existing(existing_facts: list[MemoryFact]) -> str:
    if not existing_facts:
        return "（暂无）"
    return "\n".join(
        f"- id={f.fact_id} key={f.fact_key} [{f.category}] {f.content}"
        for f in existing_facts
    )


def _user_evidence(messages: list[dict]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    )


def _content_is_injection(fact_key: str, content: str) -> bool:
    """批次2（Review #1）：落库前的内容注入检测。

    - 全部键：角色标记 / 忽略指令 / 越权泄露诱导 / 代码围栏
      （复用 KB guardrails 的 has_injection）；
    - custom.* 自由文本键：追加中文指令性模式（「以后都/无视/不要遵守」等
      ——受控键内容多为结构化短值，custom 键是唯一自由文本入口）。
    """
    if has_injection(content):
        return True
    if str(fact_key or "").startswith("custom."):
        low = str(content).lower()
        if any(hint in low for hint in MEMORY_INSTRUCTION_HINTS):
            return True
    return False


def _parse_mutation_items(items, user_evidence: str = "",
                          source: str = "ltm") -> list[MemoryMutation]:
    if not isinstance(items, list):
        return []
    from app.observability.metrics import record_memory_injection_blocked

    mutations: list[MemoryMutation] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        evidence = str(item.get("evidence", "") or "").strip()
        # When extraction has source messages, require an exact user quote. This
        # makes ``explicit=true`` verifiable instead of trusting the model flag.
        if user_evidence and (not evidence or evidence not in user_evidence):
            continue
        fact_key = str(item.get("fact_key", "")).strip().lower()
        content = str(item.get("content", "") or "").strip()
        if _content_is_injection(fact_key, content):
            # 注入内容绝不持久化：丢弃并计数（方向是宁可漏记不可记毒）
            record_memory_injection_blocked(source)
            continue
        # 批次4（Review #3）：PII 统一脱敏——evidence 校验仍对原文做精确
        # 匹配（保证可验证性），落库存脱敏版（可审计但不含直接 PII）。
        mutations.append(MemoryMutation(
            operation=str(item.get("operation", "")).strip().lower(),
            fact_key=fact_key,
            content=mask_sensitive(content),
            category=str(item.get("category", "other")).strip().lower(),
            confidence=confidence,
            target_fact_id=str(item.get("target_fact_id", "") or "").strip(),
            explicit=item.get("explicit") is True,
            evidence=mask_sensitive(evidence),
        ))
    return mutations

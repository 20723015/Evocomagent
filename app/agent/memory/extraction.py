"""LLM 事实提取：从对话中抽取短期/长期记忆事实。

模式与 app/agent/summarizer.py 一致：格式化对话 → 调用 LLM → 解析结果。
"""

from __future__ import annotations

import json

from openai import OpenAI

from app.agent.memory.models import MemoryFact, MemoryMutation
from app.agent.tools.digest import render_tool_result_line, tool_call_name_map
from app.prompts.memory import LTM_EXTRACTION_PROMPT, STM_EXTRACTION_PROMPT


def _build_transcript(messages: list[dict]) -> str:
    """将消息列表格式化为文本摘要（复用 summarizer 的逻辑）。

    工具结果行经 digest 投影（按工具字段降维，工具名自 tool_call_id 反查），
    与 summarizer.py 共用单一直现——不再各复制一份前缀截断。
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
            if content:
                lines.append(f"客服：{content}")
        elif role == "tool":
            name = call_map.get(msg.get("tool_call_id"), "?")
            lines.append(render_tool_result_line(name, content))

    return "\n".join(lines)


def extract_short_term_facts(
    client: OpenAI,
    model: str,
    recent_messages: list[dict],
    existing_facts: list[MemoryFact],
) -> list[MemoryMutation]:
    """从最近对话提取显式记忆变更；非法输出保持原状态。"""
    transcript = _build_transcript(recent_messages)
    if not transcript.strip():
        return []

    existing_text = _render_existing(existing_facts)
    prompt = STM_EXTRACTION_PROMPT.format(existing_facts=existing_text)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return _parse_mutations(raw, _user_evidence(recent_messages))


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

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [], ""
    if not isinstance(data, dict):
        return [], ""
    return _parse_mutation_items(
        data.get("mutations", []), _user_evidence(messages),
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


def _parse_mutations(raw: str, user_evidence: str = "") -> list[MemoryMutation]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    return _parse_mutation_items(data.get("mutations", []), user_evidence)


def _parse_mutation_items(items, user_evidence: str = "") -> list[MemoryMutation]:
    if not isinstance(items, list):
        return []
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
        mutations.append(MemoryMutation(
            operation=str(item.get("operation", "")).strip().lower(),
            fact_key=str(item.get("fact_key", "")).strip().lower(),
            content=str(item.get("content", "") or "").strip(),
            category=str(item.get("category", "other")).strip().lower(),
            confidence=confidence,
            target_fact_id=str(item.get("target_fact_id", "") or "").strip(),
            explicit=item.get("explicit") is True,
            evidence=evidence,
        ))
    return mutations

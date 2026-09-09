"""确定性 STM 提取（单 Agent 全量优化计划·阶段F）。

短期记忆改为零 LLM 的确定性规则即时提取（少量会话槽位）：
- 显式陈述才入槽（「我叫X」「我喜欢X」「我是X」等固定句式）；
- 每类保留最新一条，覆盖旧值（换地址/换偏好以最新为准）；
- 证据 = 用户原话（与 LTM 一致的可审计口径）；
- 复杂事实与跨会话事实由持久化 memory job 异步处理（LLM 提取）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.agent.memory.models import MemoryFact, MemoryMutation

# 槽位规则：(槽位名, 受控 fact_key, 正则, category)；显式陈述句式，避免过度提取。
# fact_key 必须命中 MEMORY_KEY_SPECS（或 custom.<snake_case>），否则
# apply_memory_mutations 的 key_spec 校验会静默丢弃整条变更；
# operation 一律 upsert（单值键语义：每类保留最新一条，覆盖旧值）。
_SLOT_RULES: tuple[tuple[str, str, re.Pattern, str], ...] = (
    ("name", "identity.name", re.compile(r"(?:我叫|我的名字(?:是|叫))([\u4e00-\u9fa5A-Za-z]{1,12})"), "identity"),
    ("occupation", "identity.occupation", re.compile(r"我(?:是|在)一?[名个位家]([\u4e00-\u9fa5A-Za-z]{2,12})(?:师|员|生|者|工程师|经理)?"), "identity"),
    ("preference", "custom.preference", re.compile(r"我(?:比较|特别|更喜欢|最喜欢)?喜欢([\u4e00-\u9fa5A-Za-z0-9]{1,16})"), "preference"),
    ("address", "identity.address", re.compile(r"(?:收货地址|地址)(?:是|改为?|换成?)([\u4e00-\u9fa5A-Za-z0-9]{4,40})"), "identity"),
    ("size", "preference.size", re.compile(r"我(?:穿|平时穿|的尺码是)([XSsMmLlXx]{1,3}|\d{2,3}码|[大小][号码])"), "preference"),
)
_PHONE_RE = re.compile(r"1[3-9]\d{9}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mask_sensitive(content: str) -> str:
    """敏感级别控制：手机号等 PII 不入记忆（阶段F·敏感级别）。"""
    return _PHONE_RE.sub("[已脱敏号码]", content)


def extract_stm_slots(
    recent_messages: list[dict], existing: list[MemoryFact],
) -> list[MemoryMutation]:
    """从最近消息的**用户原话**中确定性地提取槽位变更。

    返回 MemoryMutation 列表（统一 upsert 语义，由 apply_memory_mutations
    落库：同键旧值 superseded、内容不变时零写入）。
    """
    user_lines: list[str] = []
    for msg in recent_messages:
        if msg.get("role") == "user":
            text = str(msg.get("content") or "").strip()
            if text:
                user_lines.append(text)
    if not user_lines:
        return []

    existing_by_key: dict[str, MemoryFact] = {
        f.fact_key: f for f in existing if f.status == "active" and f.fact_key
    }
    mutations: list[MemoryMutation] = []
    seen_keys: set[str] = set()
    for line in reversed(user_lines):  # 最新消息优先
        for slot, key, pattern, category in _SLOT_RULES:
            if key in seen_keys:
                continue
            match = pattern.search(line)
            if match is None:
                continue
            content = _mask_sensitive(f"{slot}:{match.group(1).strip()}")
            existing_fact = existing_by_key.get(key)
            if existing_fact is not None and existing_fact.content == content:
                seen_keys.add(key)
                continue
            mutations.append(MemoryMutation(
                operation="upsert",
                fact_key=key,
                content=content,
                category=category,
                confidence=0.9,
                explicit=True,
                evidence=line[:80],
            ))
            seen_keys.add(key)
    return mutations

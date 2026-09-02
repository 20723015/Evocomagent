"""短期记忆：会话内事实提取与管理。

从当前对话中提取用户关键信息（身份、偏好、情绪等），
注入 system prompt 增强 Agent 的上下文感知能力。
"""

from __future__ import annotations

from openai import OpenAI

from app.agent.memory.extraction import extract_short_term_facts
from app.agent.memory.models import ACTIVE, MemoryFact, MemoryMutation, apply_memory_mutations


class ShortTermMemory:
    """会话内短期记忆：从对话中提取结构化事实，增强上下文感知。"""

    def __init__(self):
        self.records: list[MemoryFact] = []

    @property
    def facts(self) -> list[str]:
        """兼容旧调用：只暴露当前有效事实的文本。"""
        return [fact.content for fact in self.records if fact.status == ACTIVE]

    @facts.setter
    def facts(self, values) -> None:
        self.records = [
            value if isinstance(value, MemoryFact) else MemoryFact(
                content=str(value), category="other", created_at="",
            )
            for value in (values or [])
        ]

    def update(self, client: OpenAI, model: str, recent_messages: list[dict]) -> None:
        """从最近的对话消息中提取/更新事实。"""
        changes = extract_short_term_facts(
            client, model, recent_messages,
            [fact for fact in self.records if fact.status == ACTIVE],
        )
        if all(isinstance(item, MemoryMutation) for item in changes):
            self.records = apply_memory_mutations(
                self.records, changes, max_active=50,
            )
        else:
            # Compatibility for legacy custom extractors/tests returning strings.
            self.facts = changes

    def build_prompt_section(self) -> str | None:
        """生成注入 system prompt 的短期记忆片段。"""
        if not self.facts:
            return None
        facts_text = "\n".join(f"- {f}" for f in self.facts)
        return f"以下是本次对话中提取的用户关键信息（短期记忆）：\n{facts_text}"

    def reset(self) -> None:
        self.records = []

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "facts": [fact.to_dict() for fact in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ShortTermMemory:
        stm = cls()
        values = data.get("facts", [])
        if values and all(isinstance(value, dict) for value in values):
            stm.records = [MemoryFact.from_dict(value) for value in values]
        else:
            stm.facts = values
        return stm

"""长期记忆：跨会话持久化用户知识。

将用户在多次会话中表现出的偏好、身份、行为模式等提取并持久化为 JSON，
在新会话启动时加载并注入 prompt，让 Agent 具备"记住老客户"的能力。

相关性注入（Agent能力强化计划·改造四，naive 版）：
- 得分 = 词面分（Dice 系数）+ 类别权重 + 新近度衰减；
- 注入严格 ≤8 条，identity/preference 各保底 1 条；
- ES 记忆检索另立项目（facts >500 或 naive 命中率可量化不足时触发）。
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI

from app.agent.memory.extraction import extract_long_term_facts
from app.agent.memory.models import (
    ACTIVE,
    MemoryFact,
    MemoryMutation,
    apply_memory_mutations,
)

# 类别权重（改造四定死）
CATEGORY_WEIGHTS = {
    "identity": 0.15,
    "preference": 0.10,
    "behavior": 0.05,
    "issue": 0.05,
    "other": 0.0,
}
_RECENCY_TAU_DAYS = 180.0
_RECENCY_SCALE = 0.1
_MISSING_CREATED_AT_AGE_DAYS = 90.0

_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _token_set(text: str) -> set:
    """改造四词面集合：汉字 bigram（按连续汉字段分段生成）+ ASCII 整词。

    修复计划：bigram 只在**连续的汉字段**内生成——空格/标点/ASCII 段都是
    分隔符，不得跨它们制造虚假 bigram（如「苹果，香蕉」不能出「果香」）。
    """
    value = unicodedata.normalize("NFKC", text or "")
    tokens: set = set()
    tokens.update(word.casefold() for word in _ASCII_WORD_RE.findall(value))
    run: list[str] = []
    for ch in value:
        if "\u4e00" <= ch <= "\u9fff":
            run.append(ch)
        else:
            if run:
                tokens.update(f"{a}{b}" for a, b in zip(run, run[1:]))
                run = []
    if run:
        tokens.update(f"{a}{b}" for a, b in zip(run, run[1:]))
    return tokens


def dice_score(query_tokens: set, fact_tokens: set) -> float:
    """Dice 系数 = 2|A∩B| / (|A|+|B|)；任一集合为空返回 0（改造四定死）。"""
    if not query_tokens or not fact_tokens:
        return 0.0
    inter = len(query_tokens & fact_tokens)
    if inter == 0:
        return 0.0
    return 2.0 * inter / (len(query_tokens) + len(fact_tokens))


def _created_at_utc(created_at: str, now_utc: datetime) -> datetime:
    """created_at → UTC；缺失按 90 天前；未来时间 clamp 到 now（避免负 age）。"""
    if created_at:
        try:
            parsed = datetime.fromisoformat(str(created_at).replace(" ", "T"))
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)  # naive 统一按 UTC
            parsed = parsed.astimezone(timezone.utc)
            if parsed > now_utc:
                return now_utc  # 未来时间 clamp 到 0 age
            return parsed
    return now_utc - timedelta(days=_MISSING_CREATED_AT_AGE_DAYS)


def score_fact(content: str, category: str, created_at: str,
               query_tokens: set, now_utc: datetime) -> float:
    """得分 = Dice + 类别权重 + 0.1×exp(−age_days/180)（UTC 计算）。"""
    weight = CATEGORY_WEIGHTS.get(category, 0.0)
    created = _created_at_utc(created_at, now_utc)
    age_days = max(0.0, (now_utc - created).total_seconds() / 86400.0)
    recency = _RECENCY_SCALE * math.exp(-age_days / _RECENCY_TAU_DAYS)
    return dice_score(query_tokens, _token_set(content)) + weight + recency


class LongTermMemory:
    """跨会话长期记忆：持久化用户知识。"""

    def __init__(
        self,
        user_id: str = "default",
        memory_dir: str = "app/sessions/memory",
        max_facts: int = 50,
        store=None,  # LTMStore（阶段二 2.3）；None 时维持文件直读（默认）
        source_session: str = "",
    ):
        self.user_id = user_id
        self.memory_dir = Path(memory_dir)
        self.max_facts = max_facts
        self._store = store
        self.source_session = source_session
        self.facts: list[MemoryFact] = []
        self.interaction_summaries: list[dict] = []

    @property
    def memory_path(self) -> Path:
        return self.memory_dir / f"{self.user_id}.json"

    def load(self) -> None:
        """加载用户的长期记忆：store 注入走外置存储，否则读 JSON 文件。"""
        data = None
        if self._store is not None:
            data = self._store.load(self.user_id)
        elif self.memory_path.exists():
            try:
                with self.memory_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                return
        if not isinstance(data, dict):
            return

        self.facts = [
            MemoryFact.from_dict(item)
            for item in data.get("facts", [])
            if isinstance(item, dict) and item.get("content")
        ]
        self.interaction_summaries = data.get("interaction_summaries", [])

    def save(self) -> None:
        """持久化（store 注入走外置存储；否则原子写 JSON 文件）。

        schema v3（2.4）：facts 携带 evidence（用户原话依据）；
        v2 旧文件读取时 evidence 缺省空串，无损升级。
        """
        payload = {
            "schema_version": 3,
            "version": 3,
            "user_id": self.user_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "facts": [f.to_dict() for f in self.facts],
            "interaction_summaries": self.interaction_summaries,
        }

        if self._store is not None:
            self._store.save(self.user_id, payload)
            return

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.memory_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.memory_path)

    def add_facts(self, new_facts: list[MemoryFact]) -> None:
        """兼容旧调用：添加事实并按内容去重，历史版本不计入活跃上限。"""
        existing_contents = {f.content.lower() for f in self.active_facts}
        for fact in new_facts:
            if fact.content.lower() not in existing_contents:
                self.facts.append(fact)
                existing_contents.add(fact.content.lower())

        active = self.active_facts
        if len(active) > self.max_facts:
            overflow_ids = {f.fact_id for f in active[:len(active) - self.max_facts]}
            now = datetime.now().isoformat(timespec="seconds")
            for fact in self.facts:
                if fact.fact_id in overflow_ids:
                    fact.status = "deleted"
                    fact.updated_at = now

    @property
    def active_facts(self) -> list[MemoryFact]:
        return [fact for fact in self.facts if fact.status == ACTIVE]

    def _apply_changes(self, current: list[MemoryFact], changes) -> list[MemoryFact]:
        """Apply v2 mutations; accept MemoryFact lists from legacy tests/callers."""
        changes = list(changes or [])
        if all(isinstance(item, MemoryMutation) for item in changes):
            return apply_memory_mutations(
                current, changes, max_active=self.max_facts,
                source_session=self.source_session,
            )

        merged = list(current)
        existing = {f.content.casefold() for f in merged if f.active}
        for item in changes:
            if isinstance(item, MemoryFact) and item.content.casefold() not in existing:
                merged.append(item)
                existing.add(item.content.casefold())
        active = [f for f in merged if f.active]
        if len(active) > self.max_facts:
            overflow_ids = {f.fact_id for f in active[:len(active) - self.max_facts]}
            now = datetime.now().isoformat(timespec="seconds")
            for fact in merged:
                if fact.fact_id in overflow_ids:
                    fact.status = "deleted"
                    fact.updated_at = now
        return merged

    def add_interaction_summary(self, summary: str) -> None:
        if any(
            item.get("summary") == summary
            and item.get("source_session", "") == self.source_session
            for item in self.interaction_summaries
        ):
            return
        self.interaction_summaries.append({
            "summary": summary,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_session": self.source_session,
        })

    def extract_and_save(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict],
        summary: Optional[str],
    ) -> None:
        """从会话消息中提取长期记忆事实并保存。

        安全修复 P2：store 支持 merge 时走「原子读-改-写」——同用户多会话
        并发巩固不再 last-write-wins 互相覆盖；未实现 merge 的 store（含
        无 store 的文件直读）维持整包写兜底。
        """
        if not messages and not summary:
            return

        mutations, interaction_summary = extract_long_term_facts(
            client, model, messages, summary, self.active_facts,
        )

        merger = getattr(self._store, "merge", None) if self._store else None
        if merger is None:
            if mutations:
                self.facts = self._apply_changes(self.facts, mutations)
            if interaction_summary:
                self.add_interaction_summary(interaction_summary)
            self.save()
            return

        def _apply(current: Optional[dict]) -> dict:
            base = current if isinstance(current, dict) else {}
            facts = [
                MemoryFact.from_dict(f)
                for f in base.get("facts", [])
                if isinstance(f, dict) and f.get("content")
            ]
            summaries = list(base.get("interaction_summaries", []))

            facts = self._apply_changes(facts, mutations)
            if interaction_summary and not any(
                item.get("summary") == interaction_summary
                and item.get("source_session", "") == self.source_session
                for item in summaries
            ):
                summaries.append({
                    "summary": interaction_summary,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source_session": self.source_session,
                })

            return {
                "schema_version": 3,
                "version": int(base.get("version", 1) or 1) + 1,
                "user_id": self.user_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "facts": [f.to_dict() for f in facts],
                "interaction_summaries": summaries,
            }

        merged = merger(self.user_id, _apply)
        # 回写内存态：本进程后续 prompt 注入/去重与正本保持一致
        self.facts = [MemoryFact.from_dict(f) for f in merged.get("facts", [])]
        self.interaction_summaries = list(merged.get("interaction_summaries", []))

    def ranked_facts(self, query: str, limit: int = 8,
                     now_utc: Optional[datetime] = None) -> list[MemoryFact]:
        """按相关性排序取 top-N（改造四）。

        得分 = Dice + 类别权重 + 新近度；同分排序：得分降序 → created_at
        降序 → content 字典序。query 为空时词面分全 0（退化为权重+新近度）。
        """
        now_utc = now_utc or datetime.now(timezone.utc)
        query_tokens = _token_set(query)
        scored = [
            (
                score_fact(f.content, f.category, f.created_at, query_tokens, now_utc),
                _created_at_utc(f.created_at, now_utc),
                f.content,
                f,
            )
            for f in self.active_facts
        ]
        scored.sort(key=lambda t: (-t[0], -t[1].timestamp(), t[2]))
        return [t[3] for t in scored[: max(limit, 0)]]

    def select_facts_for_prompt(self, query: str, max_facts: int = 8,
                                now_utc: Optional[datetime] = None
                                ) -> list[MemoryFact]:
        """最终注入集合：严格 ≤ max_facts（含保底）。

        修复计划（保底算法修正）：只保护**实际作为保底插入**的对象（guaranteed
        集合）；top-N 里自然选中的 identity/preference 不享受保护。插入缺失
        类别时淘汰最低分且非保底的事实——因此即使 top-N 全是 identity，
        preference 保底仍能挤入（类内淘汰同类高分者），两类各至少 1 条
        （该类别存在时），总数恒 ≤ max_facts。
        """
        active_facts = self.active_facts
        if not active_facts:
            return []
        now_utc = now_utc or datetime.now(timezone.utc)
        query_tokens = _token_set(query)
        scored = [
            (
                score_fact(f.content, f.category, f.created_at, query_tokens, now_utc),
                _created_at_utc(f.created_at, now_utc),
                f.content,
                f,
            )
            for f in active_facts
        ]
        scored.sort(key=lambda t: (-t[0], -t[1].timestamp(), t[2]))

        selected = scored[: max(max_facts, 0)]
        guaranteed_ids: set[int] = set()  # 仅记录实际保底插入的对象

        def _best_of(category: str):
            for item in scored:
                if item[3].category == category:
                    return item
            return None

        for category in ("identity", "preference"):
            best = _best_of(category)
            if best is None:
                continue
            if id(best[3]) in {id(t[3]) for t in selected}:
                # 该类最高分已自然入选，无需占用保底名额
                continue
            if len(selected) < max_facts:
                selected.append(best)
                selected_ids = {id(t[3]) for t in selected}
                guaranteed_ids.add(id(best[3]))
                continue
            # 满员：淘汰最低分且非保底的事实（含 top-N 自然选中的 identity/
            # preference——只有已插入的保底对象受保护，保证 preference 能挤入
            # 全 identity 的 top-N）
            evicted = None
            for pos in range(len(selected) - 1, -1, -1):
                if id(selected[pos][3]) not in guaranteed_ids:
                    evicted = selected.pop(pos)
                    break
            if evicted is None:
                continue  # 理论不可达（保底对象 ≤ 2，其余均可淘汰）
            selected.append(best)
            selected_ids = {id(t[3]) for t in selected}
            guaranteed_ids.add(id(best[3]))

        # 恢复得分序输出
        selected.sort(key=lambda t: (-t[0], -t[1].timestamp(), t[2]))
        return [t[3] for t in selected]

    def build_prompt_section(self, query: str = "") -> str | None:
        """生成注入 system prompt 的长期记忆片段（相关性筛选，严格 ≤8 条）。"""
        if not self.active_facts and not self.interaction_summaries:
            return None

        parts = []
        if self.active_facts:
            selected = self.select_facts_for_prompt(query, max_facts=8)
            if selected:
                facts_text = "\n".join(
                    f"- [{f.category}] {f.content}" for f in selected
                )
                header = (
                    "该用户的历史记忆（来自过往会话，已按与当前问题的相关性筛选）："
                    if query else "该用户的历史记忆（来自过往会话）："
                )
                parts.append(f"{header}\n{facts_text}")

        if self.interaction_summaries:
            recent = self.interaction_summaries[-3:]  # 交互摘要保持最近 3 条不变
            summaries_text = "\n".join(f"- {s['summary']}" for s in recent)
            parts.append(f"最近的交互记录：\n{summaries_text}")

        return "\n\n".join(parts) if parts else None

    def reset(self) -> None:
        """清空该用户的长期记忆（store 版置空写回；文件版删除）。"""
        self.facts = []
        self.interaction_summaries = []
        if self._store is not None:
            self.save()
        elif self.memory_path.exists():
            self.memory_path.unlink()

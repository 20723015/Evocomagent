"""批次6（Review #6）：记忆存储 GC——统一修剪函数覆盖两条写路径。

- interaction_summaries 只保留最近 memory_summary_keep 条；
- facts 版本链中 superseded/deleted 按 updated_at 保留
  memory_version_keep_days 天，过期物理删除；
- active 事实永不修剪；updated_at 不可解析 fail-safe 保留。
全程无网络、无 LLM。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agent.memory.long_term import LongTermMemory, prune_memory_payload
from app.agent.memory.models import (
    ACTIVE,
    SUPERSEDED,
    MemoryFact,
    MemoryMutation,
    apply_memory_mutations,
)


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(
        timespec="seconds"
    )


def _fact(content="内容", key="identity.name", status=ACTIVE,
          updated_days_ago=1.0) -> MemoryFact:
    return MemoryFact(
        content=content, category="identity", created_at=_iso(200),
        fact_key=key, status=status, updated_at=_iso(updated_days_ago),
    )


class TestPruneMemoryPayload:
    def test_summaries_kept_latest_n(self, monkeypatch):
        from app.config.settings import settings

        monkeypatch.setattr(settings, "memory_summary_keep", 3)
        summaries = [{"summary": f"s{i}"} for i in range(5)]
        _, out = prune_memory_payload([], summaries)
        assert [item["summary"] for item in out] == ["s2", "s3", "s4"]

    def test_superseded_pruned_after_window_active_never(self, monkeypatch):
        from app.config.settings import settings

        monkeypatch.setattr(settings, "memory_version_keep_days", 90)
        facts = [
            _fact("旧值", status=SUPERSEDED, updated_days_ago=100),
            _fact("新值", status=ACTIVE, updated_days_ago=365),
            _fact("近期版本", status=SUPERSEDED, updated_days_ago=10),
            _fact("删除标记", status="deleted", updated_days_ago=365),
        ]
        kept, _ = prune_memory_payload(facts, [])
        contents = [f.content for f in kept]
        assert "旧值" not in contents  # superseded 超 90 天 → 物理删除
        assert "删除标记" not in contents
        assert "新值" in contents  # active 永不修剪（即使一年未更新）
        assert "近期版本" in contents  # 窗口内的版本保留

    def test_unparsable_updated_at_fail_safe_kept(self, monkeypatch):
        from app.config.settings import settings

        monkeypatch.setattr(settings, "memory_version_keep_days", 90)
        broken = _fact("损坏时间戳", status=SUPERSEDED, updated_days_ago=365)
        broken.updated_at = "not-a-timestamp"
        kept, _ = prune_memory_payload([broken], [])
        assert [f.content for f in kept] == ["损坏时间戳"]


class TestWritePathsUsePrune:
    def _ltm(self, tmp_path) -> LongTermMemory:
        return LongTermMemory(user_id="u-gc", memory_dir=str(tmp_path))

    def test_save_path_prunes(self, tmp_path, monkeypatch):
        from app.config.settings import settings

        monkeypatch.setattr(settings, "memory_summary_keep", 2)
        ltm = self._ltm(tmp_path)
        for i in range(5):
            ltm.source_session = f"s{i}"
            ltm.add_interaction_summary(f"摘要{i}")
        records = apply_memory_mutations([], [
            MemoryMutation("upsert", "identity.name", "旧名", "identity",
                           0.9, explicit=True, evidence="我叫旧名"),
            MemoryMutation("upsert", "identity.name", "新名", "identity",
                           0.9, explicit=True, evidence="我叫新名"),
        ], max_active=50)
        superseded = [f for f in records if f.status == SUPERSEDED]
        assert superseded, "预期产生版本链"
        for f in records:
            if f.status == SUPERSEDED:
                f.updated_at = _iso(365)  # 版本超窗
        ltm.facts = list(records)
        ltm.save()

        reloaded = LongTermMemory(user_id="u-gc", memory_dir=str(tmp_path))
        reloaded.load()
        assert [s["summary"] for s in reloaded.interaction_summaries] == [
            "摘要3", "摘要4",
        ]
        statuses = {f.status for f in reloaded.facts}
        assert statuses == {ACTIVE}  # 过期版本不落盘，active 保留

    def test_merge_path_prunes(self, tmp_path, monkeypatch):
        from app.config.settings import settings

        monkeypatch.setattr(settings, "memory_summary_keep", 1)
        monkeypatch.setattr(settings, "memory_version_keep_days", 90)

        class MergeStore:
            """最小 merge 语义：读-改-写由 apply_fn 完成。"""

            def __init__(self, base):
                self.base = base
                self.saved = None

            def merge(self, user_id, apply_fn):
                self.saved = apply_fn(self.base)
                return self.saved

        stale = _fact("过期版本", status=SUPERSEDED, updated_days_ago=365)
        base = {
            "facts": [stale.to_dict()],
            "interaction_summaries": [
                {"summary": f"s{i}", "source_session": f"s{i}"} for i in range(4)
            ],
        }
        store = MergeStore(base)
        ltm = self._ltm(tmp_path)
        ltm._store = store
        ltm.source_session = "sx"

        def fake_extract(client, model, messages, summary, existing):
            return [], "新会话摘要"

        # extract_and_save 调用模块级函数 → patch long_term 命名空间（零 LLM）
        monkeypatch.setattr(
            "app.agent.memory.long_term.extract_long_term_facts", fake_extract,
        )
        ltm.extract_and_save(None, "fake", [{"role": "user", "content": "x"}], "")
        payload = store.saved
        assert [s["summary"] for s in payload["interaction_summaries"]] == ["新会话摘要"]
        assert all(f["status"] == ACTIVE for f in payload["facts"])
        assert "过期版本" not in [f["content"] for f in payload["facts"]]

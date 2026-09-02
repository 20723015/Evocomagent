"""Structured STM/LTM mutation, versioning, and compatibility tests."""

from __future__ import annotations

import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, inspect

from app.agent.memory.extraction import extract_long_term_facts
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.models import (
    ACTIVE,
    DELETED,
    SUPERSEDED,
    MemoryFact,
    MemoryMutation,
    apply_memory_mutations,
)
from app.agent.memory.short_term import ShortTermMemory
from app.agent.context import ToolContext
from app.agent.tools.memory_tool import recall_user_memory
from app.stores.base import StorageUnavailableError
from app.stores.memory_store import LocalFileLTMStore, RedisLTMStore
from app.stores.sql.engine import _ensure_upgrades
from app.stores.sql.memory_store import SqlLTMStore
from app.stores.sql.schema import metadata
from tests.unit.conftest import FakeChatClient


def _fact(content: str, key: str, *, fact_id: str = "old", category: str = "preference"):
    return MemoryFact(
        content=content, category=category, created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00", fact_id=fact_id, fact_key=key,
    )


def _mutation(operation: str, key: str, content: str = "", **kwargs):
    return MemoryMutation(
        operation=operation, fact_key=key, content=content,
        category=kwargs.pop("category", "preference"), confidence=0.95,
        explicit=True, **kwargs,
    )


def test_single_value_upsert_versions_old_fact_and_prompt_uses_latest_only():
    records = apply_memory_mutations(
        [_fact("用户喜欢红色", "preference.color")],
        [_mutation("upsert", "preference.color", "用户现在喜欢蓝色")],
        max_active=50, now="2026-08-31T12:00:00",
    )

    old = next(f for f in records if f.fact_id == "old")
    current = next(f for f in records if f.status == ACTIVE)
    assert old.status == SUPERSEDED
    assert current.content == "用户现在喜欢蓝色"
    assert current.supersedes_id == "old"

    ltm = LongTermMemory()
    ltm.facts = records
    prompt = ltm.build_prompt_section("想买衣服")
    assert "蓝色" in prompt
    assert "红色" not in prompt


def test_explicit_delete_hides_fact_but_keeps_audit_record():
    records = apply_memory_mutations(
        [_fact("用户喜欢红色", "preference.color")],
        [_mutation("delete", "preference.color")],
        max_active=50,
    )
    assert len(records) == 1
    assert records[0].status == DELETED
    ltm = LongTermMemory()
    ltm.facts = records
    assert ltm.select_facts_for_prompt("红色") == []
    assert ltm.build_prompt_section("红色") is None


def test_set_add_and_remove_only_touch_selected_value():
    nike = _fact("用户喜欢 Nike", "preference.brand", fact_id="nike")
    adidas = _fact("用户喜欢 Adidas", "preference.brand", fact_id="adidas")
    records = apply_memory_mutations(
        [nike, adidas],
        [_mutation("remove", "preference.brand", "用户喜欢 Nike")],
        max_active=50,
    )
    assert {f.fact_id for f in records if f.active} == {"adidas"}
    assert next(f for f in records if f.fact_id == "nike").status == DELETED

    records = apply_memory_mutations(
        records,
        [_mutation("add", "preference.brand", "用户喜欢 Puma")],
        max_active=50,
    )
    assert {f.content for f in records if f.active} == {
        "用户喜欢 Adidas", "用户喜欢 Puma",
    }


def test_invalid_or_low_confidence_mutations_are_fail_closed():
    old = _fact("用户喜欢红色", "preference.color")
    changes = [
        MemoryMutation("upsert", "preference.color", "用户喜欢蓝色", "preference", 0.7, explicit=True),
        MemoryMutation("upsert", "free.form.key", "用户喜欢绿色", "preference", 0.99, explicit=True),
        _mutation("upsert", "preference.color", "用户喜欢黄色", target_fact_id="missing"),
        MemoryMutation("upsert", "preference.color", "用户喜欢黑色", "preference", 0.99, explicit=False),
    ]
    records = apply_memory_mutations([old], changes, max_active=50)
    assert [(f.content, f.status) for f in records] == [("用户喜欢红色", ACTIVE)]


def test_active_limit_does_not_discard_audit_history():
    facts = [
        _fact(
            f"事实{i}", f"custom.fact_{i}", fact_id=f"f{i}", category="other",
        )
        for i in range(3)
    ]
    records = apply_memory_mutations(
        facts,
        [_mutation(
            "upsert", "custom.latest", "最新事实", category="other",
        )],
        max_active=3, now="2026-08-31T12:00:00",
    )
    assert len(records) == 4
    assert len([fact for fact in records if fact.active]) == 3
    assert next(fact for fact in records if fact.fact_id == "f0").status == DELETED


def test_target_fact_id_migrates_legacy_fact_to_controlled_key():
    legacy = MemoryFact(content="用户喜欢红色", category="preference", created_at="")
    records = apply_memory_mutations(
        [legacy],
        [_mutation(
            "upsert", "preference.color", "用户现在喜欢蓝色",
            target_fact_id=legacy.fact_id,
        )],
        max_active=50,
    )
    assert legacy.fact_key.startswith("legacy.")
    assert next(f for f in records if f.fact_id == legacy.fact_id).status == SUPERSEDED
    assert next(f for f in records if f.active).fact_key == "preference.color"


def test_stm_applies_structured_mutation_and_malformed_output_keeps_state():
    client = FakeChatClient().enqueue_chat(json.dumps({
        "mutations": [{
            "operation": "upsert", "fact_key": "preference.color",
            "content": "用户喜欢蓝色", "category": "preference",
            "confidence": 0.95, "target_fact_id": "", "explicit": True,
            "evidence": "我喜欢蓝色",
        }],
    }, ensure_ascii=False)).enqueue_chat("not-json")
    stm = ShortTermMemory()
    stm.update(client, "m", [{"role": "user", "content": "我喜欢蓝色"}])
    assert stm.facts == ["用户喜欢蓝色"]
    stm.update(client, "m", [{"role": "user", "content": "随便聊聊"}])
    assert stm.facts == ["用户喜欢蓝色"]


def test_ltm_invalid_json_keeps_mutations_empty():
    client = FakeChatClient().enqueue_chat("invalid")
    mutations, summary = extract_long_term_facts(
        client, "m", [{"role": "user", "content": "我喜欢蓝色"}], None, [],
    )
    assert mutations == []
    assert summary == ""


def test_extraction_rejects_evidence_not_present_in_user_message():
    client = FakeChatClient().enqueue_chat(json.dumps({
        "mutations": [{
            "operation": "upsert", "fact_key": "preference.color",
            "content": "用户喜欢蓝色", "category": "preference",
            "confidence": 0.99, "explicit": True,
            "evidence": "我喜欢蓝色",
        }],
        "interaction_summary": "普通咨询",
    }, ensure_ascii=False))
    mutations, summary = extract_long_term_facts(
        client, "m", [{"role": "user", "content": "今天天气不错"}], None, [],
    )
    assert mutations == []
    assert summary == "普通咨询"


def test_v1_file_payload_loads_and_is_rewritten_as_v2(tmp_path):
    store = LocalFileLTMStore(tmp_path)
    store.save("u1", {
        "version": 1,
        "facts": [{"content": "用户喜欢红色", "category": "preference",
                   "created_at": "", "source_session": ""}],
        "interaction_summaries": [],
    })
    ltm = LongTermMemory(user_id="u1", store=store)
    ltm.load()
    assert ltm.active_facts[0].fact_key.startswith("legacy.")
    ltm.save()
    payload = store.load("u1")
    assert payload["schema_version"] == 3
    assert payload["facts"][0]["status"] == ACTIVE


def test_sql_store_roundtrips_inactive_history():
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    store = SqlLTMStore(engine)
    records = apply_memory_mutations(
        [_fact("用户喜欢红色", "preference.color")],
        [_mutation("upsert", "preference.color", "用户喜欢蓝色")],
        max_active=50,
    )
    store.save("u1", {
        "schema_version": 2, "facts": [f.to_dict() for f in records],
        "interaction_summaries": [],
    })
    loaded = store.load("u1")
    assert {f["status"] for f in loaded["facts"]} == {ACTIVE, SUPERSEDED}
    assert any(f["supersedes_id"] == "old" for f in loaded["facts"])


def test_recall_returns_active_facts_only_with_additive_metadata():
    ltm = LongTermMemory()
    ltm.facts = apply_memory_mutations(
        [_fact("用户喜欢红色", "preference.color")],
        [_mutation("upsert", "preference.color", "用户喜欢蓝色")],
        max_active=50,
    )
    manager = SimpleNamespace(
        memory_enabled=True, ltm=ltm, stm=SimpleNamespace(facts=[]),
    )
    result = recall_user_memory(ctx=ToolContext(user_id="u1", memory=manager))
    assert [item["content"] for item in result["long_term_facts"]] == ["用户喜欢蓝色"]
    assert result["long_term_facts"][0]["fact_key"] == "preference.color"


def test_sql_legacy_schema_is_upgraded_and_readable(tmp_path):
    db_file = tmp_path / "legacy-memory.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(64) NOT NULL,
            category VARCHAR(32), content TEXT NOT NULL,
            source_session VARCHAR(64), created_at DATETIME
        )
    """)
    conn.execute("""
        CREATE TABLE interaction_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(64) NOT NULL,
            summary TEXT NOT NULL, created_at DATETIME
        )
    """)
    conn.execute(
        "INSERT INTO memory_facts "
        "(user_id, category, content, source_session, created_at) "
        "VALUES ('u1', 'preference', '用户喜欢红色', '', CURRENT_TIMESTAMP)"
    )
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{db_file}")
    _ensure_upgrades(engine)
    columns = {column["name"] for column in inspect(engine).get_columns("memory_facts")}
    assert {"fact_id", "fact_key", "status", "confidence", "supersedes_id", "updated_at"} <= columns
    summary_columns = {
        column["name"] for column in inspect(engine).get_columns("interaction_summaries")
    }
    assert "source_session" in summary_columns

    ltm = LongTermMemory(user_id="u1", store=SqlLTMStore(engine))
    ltm.load()
    assert ltm.active_facts[0].content == "用户喜欢红色"
    assert ltm.active_facts[0].fact_key.startswith("legacy.")


def test_sql_merge_serializes_concurrent_writers(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'memory.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    metadata.create_all(engine)
    stores = [SqlLTMStore(engine), SqlLTMStore(engine)]

    def worker(store, prefix):
        for index in range(5):
            def merger(current, i=index):
                payload = current or {"facts": [], "interaction_summaries": []}
                facts = list(payload.get("facts", []))
                facts.append(_fact(
                    f"{prefix}-{i}", f"custom.{prefix}_{i}", fact_id=f"{prefix}-{i}",
                    category="other",
                ).to_dict())
                return {**payload, "facts": facts}
            store.merge("u1", merger)

    threads = [
        threading.Thread(target=worker, args=(stores[0], "a")),
        threading.Thread(target=worker, args=(stores[1], "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(stores[0].load("u1")["facts"]) == 10


def test_sql_mysql_merge_releases_lock_after_commit_or_rollback():
    """GET_LOCK is connection-scoped: transaction boundary must precede release."""

    events = []

    class _Result:
        def scalar(self):
            return 1

    class _Transaction:
        is_active = True

        def commit(self):
            events.append("commit")
            self.is_active = False

        def rollback(self):
            events.append("rollback")
            self.is_active = False

    class _Connection:
        dialect = SimpleNamespace(name="mysql")

        def begin(self):
            events.append("begin")
            self.transaction = _Transaction()
            return self.transaction

        def commit(self):
            events.append("autobegin_commit")

        def rollback(self):
            events.append("autobegin_rollback")

        def in_transaction(self):
            return False

        def execute(self, statement, params=None):
            sql = str(statement)
            if "GET_LOCK" in sql:
                events.append("get_lock")
            elif "RELEASE_LOCK" in sql:
                events.append("release_lock")
            return _Result()

        def close(self):
            events.append("close")

    class _Engine:
        url = "mysql://unit-test"

        def connect(self):
            return _Connection()

    store = SqlLTMStore(_Engine())
    store._load_via = lambda conn, user_id: {"facts": [], "interaction_summaries": []}
    store._write_via = lambda conn, user_id, payload: events.append("write")
    assert store.merge("u1", lambda current: current) == {
        "facts": [], "interaction_summaries": []
    }
    assert events == [
        "get_lock", "autobegin_commit", "begin", "write", "commit",
        "release_lock", "close",
    ]

    events.clear()
    store._write_via = lambda conn, user_id, payload: (_ for _ in ()).throw(
        RuntimeError("write failed")
    )
    with pytest.raises(StorageUnavailableError, match="SQL 记忆合并失败"):
        store.merge("u1", lambda current: current)
    assert events == [
        "get_lock", "autobegin_commit", "begin", "rollback",
        "release_lock", "close",
    ]


def test_redis_merge_failure_is_fail_closed():
    class BrokenRedis:
        def pipeline(self):
            raise TimeoutError("redis unavailable")

    with pytest.raises(StorageUnavailableError, match="Redis LTM merge"):
        RedisLTMStore(BrokenRedis()).merge("u1", lambda current: current or {})


def test_cross_session_reload_only_exposes_active_fact(tmp_path):
    store = LocalFileLTMStore(tmp_path)
    first = LongTermMemory(user_id="u1", store=store)
    first.facts = apply_memory_mutations(
        [_fact("用户喜欢红色", "preference.color")],
        [_mutation("upsert", "preference.color", "用户喜欢蓝色")],
        max_active=50,
    )
    first.save()

    second = LongTermMemory(user_id="u1", store=store)
    second.load()
    assert [f.content for f in second.active_facts] == ["用户喜欢蓝色"]
    assert len(second.facts) == 2


def test_ltm_extraction_updates_persisted_fact_across_sessions(tmp_path):
    store = LocalFileLTMStore(tmp_path)
    first_client = FakeChatClient().enqueue_chat(json.dumps({
        "mutations": [{
            "operation": "upsert", "fact_key": "preference.color",
            "content": "用户喜欢红色", "category": "preference",
            "confidence": 0.96, "target_fact_id": "", "explicit": True,
            "evidence": "我喜欢红色",
        }],
        "interaction_summary": "用户说明颜色偏好",
    }, ensure_ascii=False))
    first = LongTermMemory(user_id="u1", store=store, source_session="s1")
    first.extract_and_save(
        first_client, "m", [{"role": "user", "content": "我喜欢红色"}], None,
    )
    red_id = first.active_facts[0].fact_id

    second_client = FakeChatClient().enqueue_chat(json.dumps({
        "mutations": [{
            "operation": "upsert", "fact_key": "preference.color",
            "content": "用户现在喜欢蓝色", "category": "preference",
            "confidence": 0.98, "target_fact_id": red_id, "explicit": True,
            "evidence": "我现在喜欢蓝色",
        }],
        "interaction_summary": "用户更新颜色偏好",
    }, ensure_ascii=False))
    second = LongTermMemory(user_id="u1", store=store, source_session="s2")
    second.load()
    second.extract_and_save(
        second_client, "m", [{"role": "user", "content": "我现在喜欢蓝色"}], None,
    )

    reloaded = LongTermMemory(user_id="u1", store=store)
    reloaded.load()
    assert [fact.content for fact in reloaded.active_facts] == ["用户现在喜欢蓝色"]
    old = next(fact for fact in reloaded.facts if fact.fact_id == red_id)
    current = reloaded.active_facts[0]
    assert old.status == SUPERSEDED
    assert current.supersedes_id == red_id
    assert current.source_session == "s2"


def test_replayed_consolidation_does_not_duplicate_fact_or_summary(tmp_path, monkeypatch):
    store = LocalFileLTMStore(tmp_path)
    ltm = LongTermMemory(user_id="u1", store=store, source_session="s1")
    mutation = _mutation("upsert", "preference.color", "用户喜欢蓝色")
    mutation.evidence = "我喜欢蓝色"
    monkeypatch.setattr(
        "app.agent.memory.long_term.extract_long_term_facts",
        lambda *_args, **_kwargs: ([mutation], "用户说明颜色偏好"),
    )
    messages = [{"role": "user", "content": "我喜欢蓝色"}]
    ltm.extract_and_save(None, "m", messages, None)
    ltm.extract_and_save(None, "m", messages, None)

    payload = store.load("u1")
    active = [fact for fact in payload["facts"] if fact["status"] == ACTIVE]
    assert len(active) == 1
    assert len(payload["interaction_summaries"]) == 1


# ============================================================
# 2.4 evidence：持久化 + 版本链 + 三存储 round-trip
# ============================================================
def test_upsert_persists_evidence_and_keeps_old_version_chain():
    """upsert 后：新事实带新 evidence；旧事实保留原内容与原 evidence。"""
    old = _fact("用户喜欢红色", "preference.color")
    old.evidence = "我喜欢红色"
    records = apply_memory_mutations(
        [old],
        [_mutation("upsert", "preference.color", "用户现在喜欢蓝色")],
        max_active=50, now="2026-08-31T12:00:00",
    )
    superseded = next(f for f in records if f.fact_id == "old")
    current = next(f for f in records if f.status == ACTIVE)
    assert superseded.status == SUPERSEDED
    assert superseded.content == "用户喜欢红色"
    assert superseded.evidence == "我喜欢红色"  # 旧版本链不丢 evidence
    assert current.content == "用户现在喜欢蓝色"
    assert current.evidence == ""  # mutation 未带 evidence → 新事实空串
    assert current.supersedes_id == "old"


def test_upsert_with_evidence_on_mutation():
    old = _fact("用户喜欢红色", "preference.color")
    old.evidence = "我喜欢红色"
    mut = _mutation("upsert", "preference.color", "用户现在喜欢蓝色")
    mut.evidence = "我现在喜欢蓝色"
    records = apply_memory_mutations([old], [mut], max_active=50)
    current = next(f for f in records if f.status == ACTIVE)
    assert current.evidence == "我现在喜欢蓝色"


def test_v2_payload_without_evidence_loads_as_empty(tmp_path):
    """v2 老数据（无 evidence 字段）→ 读取为空串，不抛错、不丢内容。"""
    store = LocalFileLTMStore(tmp_path)
    store.save("u1", {
        "schema_version": 2,
        "facts": [{"content": "用户喜欢红色", "category": "preference",
                   "created_at": "2026-01-01T00:00:00", "source_session": "",
                   "fact_id": "f1", "fact_key": "preference.color",
                   "status": ACTIVE, "confidence": 0.95,
                   "supersedes_id": "", "updated_at": "2026-01-01T00:00:00"}],
        "interaction_summaries": [],
    })
    ltm = LongTermMemory(user_id="u1", store=store)
    ltm.load()
    assert ltm.active_facts[0].content == "用户喜欢红色"
    assert ltm.active_facts[0].evidence == ""


def test_file_roundtrip_preserves_evidence(tmp_path):
    store = LocalFileLTMStore(tmp_path)
    ltm = LongTermMemory(user_id="u1", store=store, source_session="s1")
    mut = _mutation("upsert", "preference.color", "用户喜欢蓝色")
    mut.evidence = "我喜欢蓝色"
    ltm.facts = apply_memory_mutations([], [mut], max_active=50)
    ltm.save()

    reloaded = LongTermMemory(user_id="u1", store=store)
    reloaded.load()
    assert reloaded.active_facts[0].evidence == "我喜欢蓝色"
    assert reloaded.active_facts[0].content == "用户喜欢蓝色"


def test_redis_roundtrip_preserves_evidence():
    import fakeredis

    redis = fakeredis.FakeStrictRedis(server=fakeredis.FakeServer())
    store = RedisLTMStore(redis)
    ltm = LongTermMemory(user_id="u1", store=store)
    mut = _mutation("upsert", "preference.color", "用户喜欢蓝色")
    mut.evidence = "我喜欢蓝色"
    ltm.facts = apply_memory_mutations([], [mut], max_active=50)
    ltm.save()

    reloaded = LongTermMemory(user_id="u1", store=store)
    reloaded.load()
    assert reloaded.active_facts[0].evidence == "我喜欢蓝色"
    assert store.load("u1")["schema_version"] == 3


def test_sql_roundtrip_preserves_evidence():
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    store = SqlLTMStore(engine)
    ltm = LongTermMemory(user_id="u1", store=store)
    mut = _mutation("upsert", "preference.color", "用户喜欢蓝色")
    mut.evidence = "我喜欢蓝色"
    ltm.facts = apply_memory_mutations([], [mut], max_active=50)
    ltm.save()

    reloaded = LongTermMemory(user_id="u1", store=store)
    reloaded.load()
    assert reloaded.active_facts[0].evidence == "我喜欢蓝色"
    assert store.load("u1")["schema_version"] == 3


def test_sql_legacy_schema_upgrade_adds_evidence_column(tmp_path):
    """004 迁移语义单测：旧表补 evidence 列后数据可读且 evidence 为空串。"""
    db_file = tmp_path / "legacy-v2.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(64) NOT NULL,
            category VARCHAR(32), content TEXT NOT NULL,
            source_session VARCHAR(64), created_at DATETIME,
            fact_id VARCHAR(64) NOT NULL DEFAULT '',
            fact_key VARCHAR(128) NOT NULL DEFAULT '',
            status VARCHAR(16) NOT NULL DEFAULT 'active',
            confidence DOUBLE NOT NULL DEFAULT 1.0,
            supersedes_id VARCHAR(64) NOT NULL DEFAULT '',
            updated_at DATETIME NULL
        )
    """)
    conn.execute("""
        CREATE TABLE interaction_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(64) NOT NULL,
            summary TEXT NOT NULL, created_at DATETIME
        )
    """)
    conn.execute(
        "INSERT INTO memory_facts "
        "(user_id, category, content, source_session, created_at) "
        "VALUES ('u1', 'preference', '用户喜欢红色', '', CURRENT_TIMESTAMP)"
    )
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{db_file}")
    _ensure_upgrades(engine)
    columns = {column["name"] for column in inspect(engine).get_columns("memory_facts")}
    assert "evidence" in columns

    store = SqlLTMStore(engine)
    ltm = LongTermMemory(user_id="u1", store=store)
    ltm.load()
    assert ltm.active_facts[0].content == "用户喜欢红色"
    assert ltm.active_facts[0].evidence == ""  # 旧数据按空串（无损）

    # 升级后写入新事实带 evidence，再读回
    mut = _mutation("upsert", "preference.color", "用户喜欢蓝色")
    mut.evidence = "我喜欢蓝色"
    ltm.facts = apply_memory_mutations([], [mut], max_active=50)
    ltm.save()
    reloaded = LongTermMemory(user_id="u1", store=store)
    reloaded.load()
    assert reloaded.active_facts[0].evidence == "我喜欢蓝色"

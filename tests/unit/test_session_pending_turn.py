"""记忆系统重构·Step6：掉线半截回复恢复（pending_turn 草稿标记）。

语义：轮次开始（ReAct 之前）随 user 消息同一次 save 落库草稿标记
{turn_id, user_message, started_at}；成功收尾的 save 清除它。
遗留非空 = 上次回复未完成 → API 带回该字段，客户端可提示重发。

覆盖：存储层往返（文件/Redis/SQL）+ 读侧容错、正常收尾清空、收尾中断遗留、
reset 清空、API 字段透出。
全程无网络、无 LLM。
"""

from __future__ import annotations

import json

import fakeredis
import pytest
from sqlalchemy import create_engine

from app.config.settings import settings
from app.stores.base import SessionState, StorageUnavailableError
from app.stores.session_store import LocalFileSessionStore, RedisSessionStore
from app.stores.sql.schema import metadata
from tests.unit.conftest import FakeChatClient

PENDING = {
    "turn_id": "t-1",
    "user_message": "我叫李四",
    "started_at": "2026-09-17T10:00:00+00:00",
}


def _state(**overrides) -> SessionState:
    base = dict(
        session_id="s-1",
        user_id="u1",
        messages=[{"role": "user", "content": "我叫李四"}],
        pending_turn=dict(PENDING),
    )
    base.update(overrides)
    return SessionState(**base)


# ------------------------------------------------------------
# 存储层往返 + 读侧容错
# ------------------------------------------------------------
def test_file_store_pending_turn_roundtrip(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    store.save("u1", "s1", _state())
    loaded = store.load("u1", "s1")
    assert loaded.pending_turn == PENDING

    # 收尾保存显式清空
    store.save("u1", "s1", _state(version=loaded.version, pending_turn=None))
    assert store.load("u1", "s1").pending_turn is None


def test_file_store_tolerates_corrupt_pending_turn(tmp_path):
    """脏字段（异形/非 dict）按「无草稿」处理，不毁整个会话加载。"""
    path = tmp_path / "u1" / "s1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 2, "session_id": "s-1", "user_id": "u1",
        "messages": [{"role": "user", "content": "hi"}],
        "pending_turn": "not-a-dict",
    }, ensure_ascii=False), encoding="utf-8")
    loaded = LocalFileSessionStore(tmp_path).load("u1", "s1")
    assert loaded is not None
    assert loaded.pending_turn is None


def test_redis_store_pending_turn_roundtrip():
    store = RedisSessionStore(fakeredis.FakeRedis(server=fakeredis.FakeServer()))
    store.save("u1", "s1", _state())
    assert store.load("u1", "s1").pending_turn == PENDING


def test_sql_store_pending_turn_roundtrip_and_corrupt_tolerated(tmp_path):
    from app.stores.sql.session_store import SqlSessionStore

    engine = create_engine(f"sqlite:///{tmp_path / 'ecom.sqlite'}")
    metadata.create_all(engine)
    store = SqlSessionStore(engine)
    saved = store.save("u1", "s1", _state())
    assert store.load("u1", "s1").pending_turn == PENDING

    # 清空（收尾路径）
    store.save("u1", "s1", _state(version=saved.version, pending_turn=None))
    assert store.load("u1", "s1").pending_turn is None

    # 损坏的 pending_turn_json：读侧容错为 None（不抛）
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE sessions SET pending_turn_json = '{not json' "
            "WHERE session_key = 'u1/s1'"
        ))
    assert store.load("u1", "s1").pending_turn is None


def test_sql_legacy_schema_upgrade_adds_pending_turn_column(tmp_path):
    """本地 sqlite 无 alembic：_ensure_upgrades 补列后即可读写。"""
    import sqlite3

    from sqlalchemy import inspect

    from app.stores.sql.engine import _ensure_upgrades

    db_file = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE sessions (
            session_key VARCHAR(200) PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            session_uuid VARCHAR(32) DEFAULT '',
            version INTEGER NOT NULL DEFAULT 0,
            summary TEXT NULL,
            consolidated_len INTEGER NOT NULL DEFAULT 0,
            status VARCHAR(12) DEFAULT 'active',
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{db_file}")
    _ensure_upgrades(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("sessions")}
    assert "pending_turn_json" in columns


# ------------------------------------------------------------
# Agent 端到端：正常收尾清空 / 收尾中断遗留 / reset 清空
# ------------------------------------------------------------
def _make_agent(tmp_path, monkeypatch, session_store, client, user_id="u1"):
    from app.agent.chat import EcomAgent

    monkeypatch.setattr(settings, "session_dir", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    monkeypatch.setattr(settings, "evolve_capture_enabled", False)
    agent = EcomAgent(
        user_id=user_id, session_store=session_store,
        client=client, memory_enabled=False, use_mcp=False,
    )
    agent.context_builder._window = 8192  # 不触发历史压缩
    return agent


def test_normal_turn_clears_pending_turn(tmp_path, monkeypatch, reset_settings):
    store = LocalFileSessionStore(tmp_path / "sessions")
    client = FakeChatClient()
    client.enqueue_final_response("好的，李四先生。")
    agent = _make_agent(tmp_path, monkeypatch, store, client)

    agent.chat("我叫李四")

    loaded = store.load("u1", "")
    assert loaded is not None
    assert loaded.pending_turn is None            # 收尾成功 → 草稿清除
    assert loaded.messages[-1]["role"] == "assistant"

    # 新 Agent 续聊：无「上次回复未完成」提示
    agent2 = _make_agent(
        tmp_path, monkeypatch, LocalFileSessionStore(tmp_path / "sessions"),
        FakeChatClient(),
    )
    assert agent2.pending_turn is None


class _InterruptOnCommitStore:
    """收尾保存（第 2 次 save）抛错：模拟进程掉线/存储中断。"""

    def __init__(self, inner, fail_at: int = 2):
        self._inner = inner
        self._fail_at = fail_at
        self._calls = 0

    def load(self, user_id, session_id):
        return self._inner.load(user_id, session_id)

    def delete(self, user_id, session_id):
        return self._inner.delete(user_id, session_id)

    def save(self, user_id, session_id, state, **kwargs):
        self._calls += 1
        if self._calls >= self._fail_at:
            raise StorageUnavailableError("模拟收尾保存中断")
        return self._inner.save(user_id, session_id, state, **kwargs)


def test_interrupted_turn_leaves_pending_turn(tmp_path, monkeypatch, reset_settings):
    """收尾中断 → user 消息已在（轮次开始落库），pending_turn 遗留可提示重发。"""
    store = _InterruptOnCommitStore(LocalFileSessionStore(tmp_path / "sessions"))
    client = FakeChatClient()
    client.enqueue_final_response("这条回复落不了库")
    agent = _make_agent(tmp_path, monkeypatch, store, client)

    with pytest.raises(StorageUnavailableError):
        agent.chat("我叫李四")

    loaded = LocalFileSessionStore(tmp_path / "sessions").load("u1", "")
    assert loaded is not None
    assert loaded.pending_turn is not None
    assert loaded.pending_turn["user_message"] == "我叫李四"
    assert loaded.pending_turn["turn_id"]
    assert loaded.pending_turn["started_at"]
    # user 消息未丢（轮次开始已落库），assistant 未落库 = 半截状态
    assert [m["role"] for m in loaded.messages] == ["user"]

    # 下一次请求（正常存储）能带回草稿标记，并在本轮收尾后清除
    client2 = FakeChatClient()
    client2.enqueue_final_response("这次回复成功")
    agent2 = _make_agent(
        tmp_path, monkeypatch, LocalFileSessionStore(tmp_path / "sessions"),
        client2,
    )
    assert agent2.pending_turn is not None       # API 层据此提示「上次回复未完成」
    assert agent2.pending_turn["user_message"] == "我叫李四"
    agent2.chat("请重新回答")
    assert LocalFileSessionStore(tmp_path / "sessions").load(
        "u1", ""
    ).pending_turn is None


def test_reset_clears_pending_turn(tmp_path, monkeypatch, reset_settings):
    store = LocalFileSessionStore(tmp_path / "sessions")
    store.save("u1", "", _state(session_id="s-1"))
    client = FakeChatClient()
    client.enqueue_final_response("新会话回复")
    agent = _make_agent(tmp_path, monkeypatch, store, client)
    assert agent.pending_turn is not None

    agent.reset()

    assert agent.pending_turn is None
    assert store.load("u1", "") is None  # 旧会话文档已删除（含草稿标记）


def test_sql_mode_two_saves_per_turn_keep_history_ordered(
    tmp_path, monkeypatch, reset_settings,
):
    """SQL 正本：一轮两次 save（begin 落 user + commit 落终答）后，
    模型历史仍是 user→assistant 顺序且跨轮不丢消息（turn_id 分组回归）。"""
    from app.stores.sql.session_store import SqlSessionStore

    engine = create_engine(f"sqlite:///{tmp_path / 'ecom.sqlite'}")
    metadata.create_all(engine)
    store = SqlSessionStore(engine)

    client = FakeChatClient()
    client.enqueue_final_response("第一答")
    agent = _make_agent(tmp_path, monkeypatch, store, client)
    agent.chat("第一问")

    loaded = store.load("u1", "")
    assert [m["role"] for m in loaded.messages] == ["user", "assistant"]
    assert [m["content"] for m in loaded.messages] == ["第一问", "第一答"]
    assert loaded.pending_turn is None

    client2 = FakeChatClient()
    client2.enqueue_final_response("第二答")
    agent2 = _make_agent(tmp_path, monkeypatch, store, client2)
    assert [m["content"] for m in agent2.raw_messages] == ["第一问", "第一答"]
    agent2.chat("第二问")
    assert [m["content"] for m in store.load("u1", "").messages] == [
        "第一问", "第一答", "第二问", "第二答",
    ]

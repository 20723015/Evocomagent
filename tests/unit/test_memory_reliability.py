"""Memory queue/watermark reliability regressions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from fakeredis import FakeRedis, FakeServer
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.agent.memory.jobs import FileMemoryJobStore
from app.stores.base import SessionState
from app.stores.idle_consolidator import find_idle_sessions
from app.stores.sql.schema import metadata
from app.stores.sql.session_store import SqlSessionStore


def _engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return engine


def test_sql_save_returns_and_caches_db_clamped_watermark():
    engine = _engine()
    redis = FakeRedis(server=FakeServer())
    store = SqlSessionStore(engine, redis=redis)
    first = store.save(
        "u1", "s1",
        SessionState(
            session_id="s1", user_id="u1",
            messages=[{"role": "user", "content": "hi"}],
        ),
        new_messages=[{"role": "user", "content": "hi"}],
    )
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE sessions SET consolidated_len=1 WHERE session_key='u1/s1'"
        ))

    saved = store.save(
        "u1", "s1",
        SessionState(
            session_id="s1", user_id="u1", messages=first.messages,
            version=first.version, consolidated_len=0,
        ),
        new_messages=[],
    )
    assert saved.consolidated_len == 1
    assert store.load("u1", "s1").consolidated_len == 1


def test_sql_idle_scanner_is_disabled_when_memory_jobs_are_authoritative():
    engine = _engine()
    store = SqlSessionStore(engine)
    store.save(
        "u1", "s1",
        SessionState(
            session_id="s1", user_id="u1",
            messages=[{"role": "user", "content": "hi"}],
            updated_at=(datetime.now() - timedelta(hours=1)).isoformat(),
        ),
    )
    assert find_idle_sessions(store, idle_minutes=1) == []


def test_file_done_job_cannot_be_enqueued_again(tmp_path):
    store = FileMemoryJobStore(str(tmp_path))
    payload = [{"role": "user", "content": "hi"}]
    store.enqueue("u1/s1", "u1", 1, messages=payload, turn_id="turn-1")
    job = store.claim("worker")[0]
    store.complete(job)
    store.enqueue("u1/s1", "u1", 1, messages=payload, turn_id="turn-1")
    assert store.jobs_snapshot() == []


def test_file_queue_uses_turn_id_and_serializes_concurrent_enqueue(tmp_path):
    def enqueue(turn: int):
        FileMemoryJobStore(str(tmp_path)).enqueue(
            "u1/s1", "u1", 1, messages=[], turn_id=f"turn-{turn}"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(enqueue, range(20)))
    rows = FileMemoryJobStore(str(tmp_path)).jobs_snapshot()
    assert len(rows) == 20
    assert {row["turn_id"] for row in rows} == {f"turn-{i}" for i in range(20)}
    assert {row["id"] for row in rows} == {
        f"u1/s1:turn:turn-{i}" for i in range(20)
    }

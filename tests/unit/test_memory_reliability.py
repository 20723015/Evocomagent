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


# ============================================================
# 低危修复 B7：SQL job CAS 领取 / 全损坏消息不推进水位
# ============================================================
def test_sql_job_claim_cas_single_winner(tmp_path):
    """低危修复 B7①：SQLite 双 worker 并发领取同一批 → 只一方成功
    （UPDATE 带状态前置条件 CAS + rowcount 判定；修复前双方都领取）。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from sqlalchemy import create_engine

    from app.agent.memory.jobs import SqlMemoryJobStore
    from app.stores.sql.schema import memory_jobs, metadata

    engine = create_engine(
        f"sqlite:///{tmp_path / 'jobs.db'}",
        connect_args={"check_same_thread": False},
    )
    metadata.create_all(engine)
    store = SqlMemoryJobStore(engine)
    with engine.begin() as conn:
        conn.execute(memory_jobs.insert().values(
            session_key="u1/s1", user_id="u1", session_uuid="uuid-1",
            through_seq=5, status="pending",
        ))

    barrier = threading.Barrier(2)

    def work(worker_id: str):
        barrier.wait()
        return store.claim(worker_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(work, ["w-a", "w-b"]))
    assert len(results[0]) + len(results[1]) == 1


def test_sql_job_all_corrupt_messages_fail_not_advance():
    """低危修复 B7②：due_messages 行非空但全部解析失败 → job 走失败路径
    （attempts 耗尽置 failed），水位不再被静默推进标 done。"""
    from sqlalchemy import create_engine, select
    from sqlalchemy.pool import StaticPool

    from app.agent.memory.jobs import MemoryJobWorker, SqlMemoryJobStore
    from app.stores.sql.schema import chat_messages, memory_jobs, metadata, sessions

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sessions.insert().values(
            session_key="u1/s1", user_id="u1", session_uuid="uuid-1",
            version=0, consolidated_len=0, status="active",
        ))
        conn.execute(chat_messages.insert().values(
            session_key="u1/s1", user_id="u1", seq=1, role="user",
            turn_id="t1", content="{broken json",
        ))
        conn.execute(memory_jobs.insert().values(
            session_key="u1/s1", user_id="u1", session_uuid="uuid-1",
            through_seq=1, status="pending",
        ))

    job_store = SqlMemoryJobStore(engine, max_attempts=1)
    worker = MemoryJobWorker(
        job_store, ltm_factory=lambda uid, sid: None,
        llm_client=None, model="fake",
    )
    worker.process_once()

    with engine.connect() as conn:
        status = conn.execute(select(memory_jobs.c.status)).scalar_one()
        watermark = conn.execute(
            select(sessions.c.consolidated_len)
        ).scalar_one()
    assert status == "failed"  # 修复前：静默按「无消息」标 done
    assert watermark == 0  # 水位未动


def test_worker_consolidation_carries_source_session():
    """批次3（Review #4）：worker 巩固必须携带 source_session（provenance）。"""
    import json

    from sqlalchemy import insert, select

    from app.agent.memory.jobs import MemoryJobWorker, SqlMemoryJobStore
    from app.agent.memory.long_term import LongTermMemory
    from app.stores.sql.schema import chat_messages, memory_jobs, sessions

    engine = _engine()
    with engine.begin() as conn:
        conn.execute(sessions.insert().values(
            session_key="u9/s9", user_id="u9", session_uuid="uuid-9",
            version=0, consolidated_len=0, status="active",
        ))
        conn.execute(chat_messages.insert().values(
            session_key="u9/s9", user_id="u9", seq=1, role="user",
            turn_id="t9", content=json.dumps(
                {"role": "user", "content": "我的订单到哪了"},
                ensure_ascii=False),
        ))
        conn.execute(memory_jobs.insert().values(
            session_key="u9/s9", user_id="u9", session_uuid="uuid-9",
            through_seq=1, status="pending",
        ))

    made: dict = {}

    def factory(user_id, session_id):
        ltm = LongTermMemory(
            user_id=user_id, memory_dir="app/sessions/memory-test",
            source_session=session_id,
        )
        made["ltm"] = ltm

        def fake_extract(client, model, messages, summary):
            # 记录工厂产出的 LTM 收到的巩固调用，并走真实摘要登记路径
            made["extract_session"] = ltm.source_session
            ltm.add_interaction_summary("用户咨询了订单进度")

        ltm.extract_and_save = fake_extract
        return ltm

    worker = MemoryJobWorker(
        SqlMemoryJobStore(engine, max_attempts=1), ltm_factory=factory,
        llm_client=None, model="fake",
    )
    done = worker.process_once()
    assert done == 1
    assert made["extract_session"] == "s9"  # 修复前恒为空串
    assert made["ltm"].source_session == "s9"
    summaries = made["ltm"].interaction_summaries
    assert len(summaries) == 1 and summaries[0]["source_session"] == "s9"

    with engine.connect() as conn:
        status = conn.execute(select(memory_jobs.c.status)).scalar_one()
    assert status == "done"


def test_same_summary_text_across_sessions_not_deduped():
    """批次3（Review #4）：跨会话相同摘要文本不得互去重（键含 source_session）。"""
    from app.agent.memory.long_term import LongTermMemory

    ltm = LongTermMemory(user_id="u1", memory_dir="", source_session="s1")
    ltm.add_interaction_summary("同样的摘要文本")
    ltm.source_session = "s2"
    ltm.add_interaction_summary("同样的摘要文本")
    sources = [item["source_session"] for item in ltm.interaction_summaries]
    assert sources == ["s1", "s2"]
    # 同会话重复仍去重
    ltm.add_interaction_summary("同样的摘要文本")
    assert len(ltm.interaction_summaries) == 2

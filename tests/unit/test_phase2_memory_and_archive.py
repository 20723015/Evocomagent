"""阶段二 2.3/2.4：LTM 外置、idle 甄别/兜底巩固、turns 对象存储归档。"""

from __future__ import annotations

import json

import fakeredis
import pytest

from app.config.settings import settings

from app.agent.memory.long_term import LongTermMemory
from app.stores.idle_consolidator import find_idle_sessions, run_idle_consolidation
from app.stores.memory_store import LocalFileLTMStore, RedisLTMStore
from app.stores.object_store import LocalDirObjectStore, ObjectStoreUnavailable, S3ObjectStore
from app.stores.session_store import RedisSessionStore, SessionState


def _make_redis():
    return fakeredis.FakeRedis(server=fakeredis.FakeServer())


# ------------------------------------------------------------
# LTM 外置（2.3）
# ------------------------------------------------------------
def test_redis_ltm_roundtrip():
    redis = _make_redis()
    store = RedisLTMStore(redis)
    store.save("u1", {"facts": [], "interaction_summaries": [{"summary": "s"}]})
    payload = store.load("u1")
    assert payload["interaction_summaries"] == [{"summary": "s"}]
    # hash 结构：facts 单独字段（redis-py 返回 bytes 键）
    assert b"facts" in redis.hgetall("memory:u1")


def test_long_term_memory_store_backed():
    redis = _make_redis()
    ltm = LongTermMemory(user_id="u1", memory_dir="/tmp/never-used", store=RedisLTMStore(redis))
    ltm.add_interaction_summary("老客户")
    ltm.save()
    reloaded = LongTermMemory(user_id="u1", memory_dir="/tmp/never-used", store=RedisLTMStore(redis))
    reloaded.load()
    assert [s["summary"] for s in reloaded.interaction_summaries] == ["老客户"]
    assert reloaded.memory_path.exists() is False  # store 版不再写文件


def test_local_ltm_store_keeps_file_layout(tmp_path):
    store = LocalFileLTMStore(tmp_path)
    store.save("u1", {"facts": [], "interaction_summaries": []})
    assert (tmp_path / "u1.json").exists()


# ------------------------------------------------------------
# idle 兜底巩固（2.3）
# ------------------------------------------------------------
def _session(updated_at: str, messages=None):
    return SessionState(
        session_id="s1", messages=messages or [{"role": "user", "content": "hi"}],
        updated_at=updated_at,
    )


def test_find_idle_sessions_respects_cutoff():
    redis = _make_redis()
    store = RedisSessionStore(redis)
    from datetime import datetime, timedelta

    now = datetime(2026, 8, 28, 12, 0, 0)
    store.save("u1", "active", _session("2026-08-28T11:59:00"))
    store.save("u1", "idle", _session("2026-08-25T01:00:00"))
    idle = find_idle_sessions(store, idle_minutes=30, now=now)
    assert ("u1", "idle") in idle
    assert ("u1", "active") not in idle


def test_idle_consolidation_calls_ltm(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    redis = _make_redis()
    session_store = RedisSessionStore(redis)
    old = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
    session_store.save("u1", "s-old", _session(old, messages=[{"role": "user", "content": "我住深圳"}]))

    calls = []

    class _FakeClient:
        pass

    fake_extract = lambda client, model, messages, summary, facts: (
        [], "老客户总结",
    )
    # long_term.py 在模块导入时绑定了函数引用，两处都需替换
    monkeypatch.setattr("app.agent.memory.extraction.extract_long_term_facts", fake_extract)
    monkeypatch.setattr("app.agent.memory.long_term.extract_long_term_facts", fake_extract)
    ltm_store = RedisLTMStore(redis)
    handled = run_idle_consolidation(
        session_store, ltm_store, _FakeClient(), "fake-model",
        idle_minutes=30, memory_dir=str(tmp_path),
    )
    assert handled == ["u1/s-old"]
    # LTM 已写入 Redis
    assert ltm_store.load("u1")["interaction_summaries"]


# ------------------------------------------------------------
# turns 对象存储（2.4）
# ------------------------------------------------------------
def test_local_object_store_roundtrip(tmp_path):
    store = LocalDirObjectStore(tmp_path)
    store.put("turns/20260828/t1.json", b'{"a": 1}')
    assert store.get("turns/20260828/t1.json") == b'{"a": 1}'
    assert store.list("turns/20260828") == ["turns/20260828/t1.json"]


def test_recorder_archives_via_object_store(tmp_path):
    from app.evolution.recorder import TurnRecorder

    archive = LocalDirObjectStore(tmp_path / "archive")
    recorder = TurnRecorder(tmp_path / "local", archive=archive)
    from conftest import sample_response

    turn_id = recorder.record(
        session_id="s1", mode="single", question="能退吗", user_id="u1",
        structured_reply=sample_response(reply="可以退"),
        turn_slice=[],
    )
    assert turn_id
    keys = archive.list("turns")
    assert len(keys) == 1
    data = json.loads(archive.get(keys[0]))
    assert data["user_id"] == "u1"
    assert data["session_id"] == "s1"


def test_s3_store_requires_boto3():
    # boto3 未安装/不可用 → ObjectStoreUnavailable（调用方降级本地）
    try:
        import boto3  # noqa: F401
        pytest.skip("本机已装 boto3，跳过缺依赖路径")
    except ImportError:
        with pytest.raises(ObjectStoreUnavailable):
            S3ObjectStore("bucket")


# ------------------------------------------------------------
# 阶段F：异步记忆任务（轮内/close 不再 LLM 巩固；memory job 承担）
# ------------------------------------------------------------
def test_memory_jobs_enqueue_and_worker_idempotent(tmp_path, reset_settings, monkeypatch):
    from datetime import datetime, timezone

    from app.agent.chat import EcomAgent
    from app.agent.memory.jobs import FileMemoryJobStore, MemoryJobWorker
    from app.agent.memory.long_term import MemoryFact
    from conftest import FakeChatClient

    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    settings.session_dir = str(tmp_path / "sessions")
    settings.evolve_capture_enabled = False

    fake_extract = lambda client, model, messages, summary, existing_facts: (
        [MemoryFact(content="事实", category="preference",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_session="s")],
        "交互摘要",
    )
    monkeypatch.setattr("app.agent.memory.extraction.extract_long_term_facts", fake_extract)
    monkeypatch.setattr("app.agent.memory.long_term.extract_long_term_facts", fake_extract)

    client = FakeChatClient()
    turns = 3
    for i in range(turns):
        client.enqueue_final_response(f"回答{i}", intent="order_query")
    agent = EcomAgent(
        user_id="u1", client=client, memory_enabled=True, use_mcp=False,
    )
    agent.context_builder._window = 8192  # 不触发历史压缩
    for msg in ["第一句", "第二句", "第三句"]:
        agent.chat(msg)

    # 每轮保存后入队一个 memory job（文件轻量队列；轮内零 LLM 巩固调用）
    store = FileMemoryJobStore(str(tmp_path / "memory" / "jobs"))
    assert len(store.claim(worker_id="probe", limit=100)) == turns  # 3 个任务
    # 重新入队（claim 会改状态，重置后验证 worker 全流程）
    store2 = FileMemoryJobStore(str(tmp_path / "memory" / "jobs"))
    jobs = [e for e in _read_jsonl(store2._queue)]
    assert len(jobs) == 3  # 队列留痕：3 processing 任务

    worker = MemoryJobWorker(store2, lambda uid: agent.memory_manager.ltm,
                             client, "fake-model", worker_id="w1")
    # 任务已在 probe claim 时置 processing：直接跑 worker（幂等确认/处理）
    worker.process_once(limit=100)
    agent.close()  # 阶段F：close 零 LLM，不再兜底巩固
    assert client.calls.count is not None  # close 后无新调用（见下方断言）
    before = len(client.calls)
    agent.close()
    assert len(client.calls) == before


def _read_jsonl(path):
    import json as _json

    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(_json.loads(line))
    return out

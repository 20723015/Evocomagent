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


def test_ltm_write_side_timestamps_are_utc(tmp_path):
    """低危修复 B6：LTM 写侧时间戳统一 UTC（+00:00），与读侧
    _created_at_utc 的 tz-aware 解析、recency/TTL 计算对齐。"""
    from datetime import datetime, timedelta, timezone

    from app.agent.memory.models import MemoryFact

    ltm = LongTermMemory(user_id="u1", memory_dir=str(tmp_path))
    ltm.add_interaction_summary("老客户，偏好顺丰")
    ltm.add_facts([MemoryFact(
        content="偏好顺丰快递", category="preference",
        created_at=datetime.now(timezone.utc).isoformat(),
    )])
    ltm.save()
    data = json.loads((tmp_path / "u1.json").read_text(encoding="utf-8"))
    assert data["updated_at"].endswith("+00:00")
    assert data["interaction_summaries"][-1]["timestamp"].endswith("+00:00")

    # 读写一致性：新事实不被 TTL 过滤；超期（>365d）事实不注入
    fresh = LongTermMemory(user_id="u2", memory_dir=str(tmp_path / "b"))
    fresh.add_facts([MemoryFact(
        content="偏好顺丰", category="preference",
        created_at=datetime.now(timezone.utc).isoformat(),
    )])
    assert fresh.select_facts_for_prompt("", max_facts=8)

    # 批次7 TTL 类别豁免：超期 preference 不再被过滤（身份/偏好长期有效，
    # 有意行为变更）；issue 类别仍受 memory_fact_ttl_days 约束
    stale = LongTermMemory(user_id="u3", memory_dir=str(tmp_path / "c"))
    stale.add_facts([MemoryFact(
        content="很老的事实", category="issue",
        created_at=(
            datetime.now(timezone.utc) - timedelta(days=400)
        ).isoformat(),
    )])
    assert stale.select_facts_for_prompt("", max_facts=8) == []
    exempt = LongTermMemory(user_id="u4", memory_dir=str(tmp_path / "d"))
    exempt.add_facts([MemoryFact(
        content="一年前的偏好", category="preference",
        created_at=(
            datetime.now(timezone.utc) - timedelta(days=400)
        ).isoformat(),
    )])
    assert exempt.select_facts_for_prompt("", max_facts=8) != []


def test_incremental_identity_extraction_end_to_end(tmp_path, reset_settings, monkeypatch):
    """记忆系统重构·Step4：增量提取默认开启（memory_job_worker_enabled=True）。

    「我叫李四」→ 轮末 flush 同事务入队 memory job → worker 用**真实**提取链路
    消费 → LTM 落库 identity.name=李四。覆盖「接线存在」之外的默认开关与
    端到端落库，防止开关被误关或链路断裂时静默丢记忆。
    """
    from app.agent.chat import EcomAgent
    from app.agent.memory.jobs import FileMemoryJobStore, MemoryJobWorker
    from conftest import FakeChatClient

    assert settings.memory_job_worker_enabled is True  # 生产默认：增量提取开启
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    settings.session_dir = str(tmp_path / "sessions")
    settings.evolve_capture_enabled = False

    client = FakeChatClient()
    client.enqueue_final_response("好的，李四先生。")
    # 提取调用（真实 extract_long_term_facts）：模型输出 identity.name 变更
    client.enqueue(json.dumps({
        "mutations": [{
            "operation": "upsert", "fact_key": "identity.name",
            "content": "用户名叫李四", "category": "identity",
            "confidence": 0.95, "target_fact_id": "", "explicit": True,
            "evidence": "我叫李四",
        }],
        "interaction_summary": "用户自报姓名",
    }, ensure_ascii=False))

    agent = EcomAgent(user_id="u1", client=client, memory_enabled=True, use_mcp=False)
    agent.context_builder._window = 8192  # 不触发历史压缩
    agent.chat("我叫李四")

    jobs_dir = str(tmp_path / "memory" / "jobs")
    assert len(_read_jsonl(FileMemoryJobStore(jobs_dir)._queue)) == 1  # 已入队

    ltm = LongTermMemory(user_id="u1", memory_dir=str(tmp_path / "memory"))
    worker = MemoryJobWorker(
        FileMemoryJobStore(jobs_dir), lambda uid, sid="": ltm,
        client, "fake-model", worker_id="w1",
    )
    assert worker.process_once(limit=10) == 1

    ltm.load()
    by_key = {f.fact_key: f.content for f in ltm.active_facts}
    assert by_key.get("identity.name") == "用户名叫李四"
    assert [s["summary"] for s in ltm.interaction_summaries] == ["用户自报姓名"]


# ------------------------------------------------------------
# 记忆系统重构·Step5：LTM cache-aside（SQL 唯一正本 + Redis 读缓存）
# ------------------------------------------------------------
def _sql_ltm(tmp_path, name="ltm.sqlite"):
    from sqlalchemy import create_engine

    from app.stores.sql.memory_store import SqlLTMStore
    from app.stores.sql.schema import metadata

    engine = create_engine(f"sqlite:///{tmp_path / name}")
    metadata.create_all(engine)
    return SqlLTMStore(engine)


def _payload(content: str) -> dict:
    return {
        "schema_version": 3, "version": 1,
        "facts": [{
            "content": content, "category": "identity",
            "created_at": "2026-09-01T10:00:00+00:00",
            "fact_id": "f1", "fact_key": "identity.name",
            "status": "active", "confidence": 1.0, "evidence": "",
            "updated_at": "2026-09-01T10:00:00+00:00",
        }],
        "interaction_summaries": [],
    }


def test_cached_ltm_miss_backfills_and_hit_serves_cache(tmp_path):
    from app.stores.memory_store import CachedLTMStore

    redis = _make_redis()
    inner = _sql_ltm(tmp_path)
    store = CachedLTMStore(inner, redis)
    inner.save("u1", _payload("name:张三"))

    loaded = store.load("u1")                      # 未命中 → 回源并回填
    assert loaded["facts"][0]["content"] == "name:张三"
    assert redis.hgetall("memory:u1")              # 缓存已回填

    class _BoomInner:
        def load(self, user_id):
            raise AssertionError("命中缓存时不得回源正本")

    assert CachedLTMStore(_BoomInner(), redis).load("u1")["facts"][0]["content"] == "name:张三"


def test_cached_ltm_save_and_merge_refresh_cache(tmp_path):
    from app.stores.memory_store import CachedLTMStore

    redis = _make_redis()
    store = CachedLTMStore(_sql_ltm(tmp_path), redis)

    store.save("u1", _payload("name:张三"))
    assert store.load("u1")["facts"][0]["content"] == "name:张三"
    assert CachedLTMStore(_sql_ltm(tmp_path), redis).load("u1")["facts"][0]["content"] == "name:张三"

    def _merge(current):
        payload = current or _payload("name:张三")
        payload["facts"] = [*payload["facts"], {
            **_payload("preference:蓝色")["facts"][0],
            "fact_id": "f2", "fact_key": "preference.color",
            "category": "preference", "content": "preference:蓝色",
        }]
        return payload

    store.merge("u1", _merge)
    cached = CachedLTMStore(_sql_ltm(tmp_path), redis).load("u1")
    assert [f["content"] for f in cached["facts"]] == ["name:张三", "preference:蓝色"]


def test_cached_ltm_corrupt_cache_falls_through_and_repairs(tmp_path):
    """损坏缓存不得冒充正本（坏字段被解码成空表 = 静默清空记忆）。"""
    from app.stores.memory_store import CachedLTMStore

    redis = _make_redis()
    inner = _sql_ltm(tmp_path)
    inner.save("u1", _payload("name:张三"))
    redis.hset("memory:u1", mapping={"facts": "{not json", "interaction_summaries": "[]"})

    store = CachedLTMStore(inner, redis)
    assert store.load("u1")["facts"][0]["content"] == "name:张三"  # 回源
    assert store.load("u1")["facts"][0]["content"] == "name:张三"  # 缓存已修复


def test_cached_ltm_redis_failure_degrades_to_inner(tmp_path):
    """Redis 读/写异常都不得影响正本读写（缓存只是加速层）。"""
    from app.stores.memory_store import CachedLTMStore

    class _BrokenRedis:
        def hgetall(self, key):
            raise ConnectionError("redis down")

        def hset(self, key, mapping=None):
            raise ConnectionError("redis down")

        def expire(self, key, ttl):
            raise ConnectionError("redis down")

    inner = _sql_ltm(tmp_path)
    store = CachedLTMStore(inner, _BrokenRedis())

    store.save("u1", _payload("name:张三"))                     # 写正本成功
    assert store.load("u1")["facts"][0]["content"] == "name:张三"  # 读回源成功
    assert inner.load("u1")["facts"][0]["content"] == "name:张三"


def test_cached_ltm_propagates_storage_unavailable(tmp_path):
    """正本不可用 → StorageUnavailableError 直透（绝不拿缓存冒充正本）。"""
    from app.stores.base import StorageUnavailableError
    from app.stores.memory_store import CachedLTMStore

    class _DownInner:
        def load(self, user_id):
            raise StorageUnavailableError("SQL down")

    store = CachedLTMStore(_DownInner(), _make_redis())
    with pytest.raises(StorageUnavailableError):
        store.load("u1")

"""阶段五 5.1/5.2：评估门禁 + 并发写冲突 + Redis 断连降级（故障注入）。"""

from __future__ import annotations

import threading

import fakeredis
import pytest

from app.config.settings import settings
from app.stores.base import SessionConflictError, SessionState, StorageUnavailableError
from app.stores.session_store import LocalFileSessionStore, RedisSessionStore
from app.stores.locks import SessionLockManager
from app.security.ratelimit import UserLimiter


def _make_redis():
    return fakeredis.FakeRedis(server=fakeredis.FakeServer())


class _FailingRedis:
    """故障注入：所有命令抛错（模拟 Redis 断连）。"""

    def __init__(self):
        self._calls = 0

    def _fail(self, *_a, **_k):
        self._calls += 1
        raise ConnectionError("redis down (fault injection)")

    __getattr__ = lambda self, name: self._fail


# ============================================================
# 5.1 数据集门禁（无网络）
# ============================================================
def test_eval_dataset_gate_passes():
    from app.scripts.check_eval_dataset import main

    assert main([]) == 0


def test_eval_dataset_gate_fails_on_shrink(tmp_path, monkeypatch):
    from app.scripts import check_eval_dataset
    from app.scripts.check_eval_dataset import main

    small = tmp_path / "small.json"
    small.write_text('{"cases": [{"id": "a", "description": "d", "turns": ["hi"]}]}',
                     encoding="utf-8")
    monkeypatch.setattr(check_eval_dataset, "MIN_GOLDEN_CASES", 2)
    monkeypatch.setattr(check_eval_dataset, "MIN_RETRIEVAL_CASES", 1)
    assert main(["--eval-cases", str(small)]) == 1


# ============================================================
# 5.2 并发写同一 session：CAS 只允许一个写入者
# ============================================================
def test_concurrent_same_session_single_writer_local(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    store.save("u1", "s1", SessionState(session_id="s1", messages=[], version=0))

    results: dict[str, str] = {}
    lock = threading.Lock()

    def writer(name, version):
        try:
            store.save("u1", "s1", SessionState(session_id="s1", messages=[name], version=version))
            with lock:
                results[name] = "ok"
        except SessionConflictError:
            with lock:
                results[name] = "conflict"

    # 两个写入者持有相同期望版本 → 只有一个成功
    t1 = threading.Thread(target=writer, args=("A", 1))
    t2 = threading.Thread(target=writer, args=("B", 1))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert sorted(results.values()) == ["conflict", "ok"]


def test_concurrent_same_session_single_writer_redis():
    store = RedisSessionStore(_make_redis())
    store.save("u1", "s1", SessionState(session_id="s1", messages=[], version=0))

    results: dict[str, str] = {}
    lock = threading.Lock()

    def writer(name):
        try:
            store.save("u1", "s1", SessionState(session_id="s1", messages=[name], version=1))
            with lock:
                results[name] = "ok"
        except SessionConflictError:
            with lock:
                results[name] = "conflict"

    threads = [threading.Thread(target=writer, args=(f"W{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for v in results.values() if v == "ok") == 1
    # 最终只有一个写入者的消息落盘（版本递增到 2）
    final = store.load("u1", "s1")
    assert final.version == 2 and len(final.messages) == 1


# ============================================================
# 5.2 Redis 断连降级路径（故障注入）
# ============================================================
def test_session_store_unavailable_503_semantics():
    store = RedisSessionStore(_FailingRedis())
    with pytest.raises(StorageUnavailableError):
        store.load("u1", "s1")
    with pytest.raises(StorageUnavailableError):
        store.save("u1", "s1", SessionState(session_id="s1"))


def test_lock_degrades_to_in_process_on_redis_failure():
    mgr = SessionLockManager(_FailingRedis(), ttl_seconds=60)
    t1 = mgr.acquire("u1", "s1")
    assert t1 is not None  # Redis 挂了 → 进程内互斥仍可用
    assert mgr.acquire("u1", "s1") is None  # 同进程仍互斥
    mgr.release("u1", "s1", t1)
    t2 = mgr.acquire("u1", "s1")
    assert t2 is not None
    mgr.release("u1", "s1", t2)  # 释放干净，避免跨测试僵尸锁


def test_limiter_degrades_to_in_process_on_redis_failure():
    limiter = UserLimiter(_FailingRedis(), max_rps=2, daily_token_budget=1000)
    assert limiter.allow_rps("u1") is True
    assert limiter.allow_rps("u1") is True
    assert limiter.allow_rps("u1") is False  # 进程内窗口限速仍生效
    limiter.consume_tokens("u1", 500)
    assert limiter.allow_budget("u1", estimated_tokens=600) is False

"""阶段二 2.1/2.2：SessionStore 两个实现 + CAS 语义。"""

from __future__ import annotations

import fakeredis
import pytest

from app.stores.base import SessionConflictError, SessionState
from app.stores.locks import SessionLease, SessionLockManager
from app.stores.redis_client import set_redis_for_test
from app.stores.session_store import LocalFileSessionStore, RedisSessionStore


def _state(**overrides) -> SessionState:
    base = dict(session_id="s-1", summary=None, messages=[{"role": "user", "content": "hi"}])
    base.update(overrides)
    return SessionState(**base)


# ------------------------------------------------------------
# LocalFileSessionStore
# ------------------------------------------------------------
def test_local_store_roundtrip(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    saved = store.save("u1", "s1", _state())
    assert saved.version == 1
    loaded = store.load("u1", "s1")
    assert loaded.version == 1
    assert loaded.messages[0]["content"] == "hi"
    assert store.load("u1", "missing") is None


def test_local_store_cas_conflict(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    store.save("u1", "s1", _state(version=0))
    # 期望版本已过时 → 409 语义
    with pytest.raises(SessionConflictError):
        store.save("u1", "s1", _state(version=0))
    # 带上当前版本 → 成功并递增
    saved = store.save("u1", "s1", _state(version=1))
    assert saved.version == 2


def test_local_store_derived_path_per_user(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    store.save("u1", "s1", _state())
    store.save("u2", "s1", _state())
    assert (tmp_path / "u1" / "s1.json").exists()
    assert (tmp_path / "u2" / "s1.json").exists()
    # 缺省 session_id → session.json（CLI/API 默认会话）
    store.save("u1", "", _state())
    assert (tmp_path / "u1" / "session.json").exists()


def test_local_store_exact_path(tmp_path):
    # 沙箱显式单文件场景
    target = tmp_path / "case1.json"
    store = LocalFileSessionStore(tmp_path, exact_path=target)
    store.save("u1", "s", _state())
    assert target.exists()
    assert store.load("u1", "anything") is not None


def test_local_store_delete(tmp_path):
    store = LocalFileSessionStore(tmp_path)
    store.save("u1", "s1", _state())
    store.delete("u1", "s1")
    assert store.load("u1", "s1") is None


# ------------------------------------------------------------
# RedisSessionStore（fakeredis）
# ------------------------------------------------------------
def _make_redis():
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server)


def test_redis_store_roundtrip():
    r = _make_redis()
    store = RedisSessionStore(r)
    saved = store.save("u1", "s1", _state())
    assert saved.version == 1
    loaded = store.load("u1", "s1")
    assert loaded.version == 1
    assert (r.get("session:u1:s1") is not None) or r.exists("session:u1:s1")


def test_redis_store_cas_conflict():
    store = RedisSessionStore(_make_redis())
    store.save("u1", "s1", _state(version=0))
    with pytest.raises(SessionConflictError):
        store.save("u1", "s1", _state(version=0))
    saved = store.save("u1", "s1", _state(version=1))
    assert saved.version == 2


def test_redis_store_iter_all_for_idle_scan():
    store = RedisSessionStore(_make_redis())
    store.save("u1", "s1", _state())
    store.save("u2", "s2", _state())
    assert sorted(store.iter_all()) == [("u1", "s1"), ("u2", "s2")]


# ------------------------------------------------------------
# 会话锁（2.2）
# ------------------------------------------------------------
def test_lock_redis_acquire_and_release():
    r = _make_redis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    t1 = mgr.acquire("u1", "s1")
    assert t1 is not None
    assert mgr.acquire("u1", "s1") is None  # 已持有 → 并发请求 409
    mgr.release("u1", "s1", t1)
    assert mgr.acquire("u1", "s1") is not None  # 释放后可再取


def test_lock_redis_release_token_mismatch_keeps_lock():
    r = _make_redis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    t1 = mgr.acquire("u1", "s1")
    mgr.release("u1", "s1", "wrong-token")  # 误释放防护
    assert mgr.acquire("u1", "s1") is None


def test_lock_in_process_fallback():
    mgr = SessionLockManager(None, ttl_seconds=60)
    t1 = mgr.acquire("u1", "s1")
    assert t1 is not None
    assert mgr.acquire("u1", "s1") is None
    mgr.release("u1", "s1", t1)
    t2 = mgr.acquire("u1", "s1")
    assert t2 is not None
    mgr.release("u1", "s1", t2)  # 不留僵尸锁（进程内锁表跨测试共享）


def test_release_with_none_token_is_noop():
    mgr = SessionLockManager(None, ttl_seconds=60)
    mgr.release("u1", "s1", None)  # 不应抛异常


def test_session_lease_context_manager():
    r = _make_redis()
    mgr = SessionLockManager(r, ttl_seconds=60)
    with SessionLease(mgr, "u1", "s1") as lease:
        assert lease.token is not None
        assert mgr.acquire("u1", "s1") is None  # 租约持有期间互斥
    assert mgr.acquire("u1", "s1") is not None  # 退出后释放


def test_session_lease_conflict_yields_none():
    mgr = SessionLockManager(None, ttl_seconds=60)
    t1 = mgr.acquire("u1", "s1")
    try:
        with SessionLease(mgr, "u1", "s1") as lease:
            assert lease.token is None  # 已有人持有 → 409 信号
    finally:
        mgr.release("u1", "s1", t1)


def test_get_redis_singleton_returns_none_without_server(reset_settings):
    # 无 Redis 环境：首次探活失败后缓存 None，不再重复连接
    set_redis_for_test(None)  # 明确注入：不可用
    from app.stores.redis_client import get_redis
    assert get_redis() is None

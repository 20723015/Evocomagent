"""会话栅栏测试（010 D1）：字典序获取 / 部分失败逆序释放 / 心跳校验 / sqlite no-op。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.evolution.fence import (
    ConversationFence,
    FenceLost,
    FenceTimeout,
    conversation_fence_key,
)
from app.stores.sql.schema import metadata


class _RecordingConn:
    """GET_LOCK/RELEASE_LOCK/CONNECTION_ID/IS_USED_LOCK 记录替身（MySQL 语义）。"""

    def __init__(self, locks, *, fail_on=None, drop=None):
        self._locks = locks  # shared dict: name → conn_id
        self._fail_on = set(fail_on or ())
        self._drop = set(drop or ())  # assert_held 时模拟已被他人持有/丢失
        self.conn_id = 4242
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def exec_driver_sql(self, sql, params=None):
        op = sql.split("(")[0].split()[-1]
        name = params[0] if params else ""
        self.calls.append((op, name))
        if op == "GET_LOCK":
            if name in self._fail_on:
                return _Scalar(0)
            self._locks[name] = self.conn_id
            return _Scalar(1)
        if op == "RELEASE_LOCK":
            self._locks.pop(name, None)
            return _Scalar(1)
        if op == "CONNECTION_ID":
            return _Scalar(self.conn_id)
        if op == "IS_USED_LOCK":
            if name in self._drop:
                return _Scalar(9999)  # 被其他连接持有
            owner = self._locks.get(name)
            return _Scalar(owner if owner is not None else 9999)
        raise AssertionError(f"unexpected sql {sql}")

    def close(self):
        self.closed = True


class _Scalar:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class _FakeEngine:
    dialect_name = "mysql"

    @property
    def dialect(self):
        return type("D", (), {"name": self.dialect_name})()

    def connect(self):
        conn = self.handout.pop(0)
        conn._locks = self._locks
        return conn

    def __init__(self):
        self._locks = {}
        self.handout: list[_RecordingConn] = []


def test_conversation_fence_key_shared_across_versions():
    first = conversation_fence_key("cs", "conv-1")
    assert first == conversation_fence_key("cs", "conv-1")
    assert first.startswith("hk:") and len(first.encode("ascii")) == 63


def test_conversation_fence_key_has_no_separator_collision_and_fits_mysql():
    assert conversation_fence_key("a:b", "c") != conversation_fence_key("a", "b:c")
    key = conversation_fence_key("源" * 64, "会话" * 128)
    assert len(key.encode("utf-8")) <= 64


def test_acquire_sorted_and_release_reverse_order():
    engine = _FakeEngine()
    conn = _RecordingConn(engine._locks)
    engine.handout.append(conn)
    fence = ConversationFence(engine, timeout_seconds=1.0)
    fence.acquire(["hk:b:2", "hk:a:1", "hk:b:2"])
    get_calls = [name for op, name in conn.calls if op == "GET_LOCK"]
    assert get_calls == ["hk:a:1", "hk:b:2"]  # 字典序 + 去重
    fence.release()
    release_calls = [name for op, name in conn.calls if op == "RELEASE_LOCK"]
    assert release_calls == ["hk:b:2", "hk:a:1"]  # 逆序释放
    assert conn.closed and fence.held_keys == []


def test_partial_failure_releases_acquired_and_raises_timeout():
    engine = _FakeEngine()
    conn = _RecordingConn(engine._locks, fail_on=("hk:c:3",))
    engine.handout.append(conn)
    fence = ConversationFence(engine, timeout_seconds=1.0)
    with pytest.raises(FenceTimeout):
        fence.acquire(["hk:a:1", "hk:c:3", "hk:b:2"])
    # a/b 已获取后失败 → 逆序全释放，不残留
    release_calls = [name for op, name in conn.calls if op == "RELEASE_LOCK"]
    assert release_calls == ["hk:b:2", "hk:a:1"]
    assert engine._locks == {} and fence.held_keys == []


def test_assert_held_mismatch_raises_lost():
    engine = _FakeEngine()
    conn = _RecordingConn(engine._locks, drop=("hk:a:1",))
    engine.handout.append(conn)
    fence = ConversationFence(engine, timeout_seconds=1.0)
    fence.acquire(["hk:a:1"])
    with pytest.raises(FenceLost):
        fence.assert_held()  # 锁被其他连接持有 → 租约丢失同路径
    # 释放后再校验同样失败
    fence.release()
    with pytest.raises(FenceLost):
        fence.assert_held()


def test_assert_held_ok_when_owned():
    engine = _FakeEngine()
    conn = _RecordingConn(engine._locks)
    engine.handout.append(conn)
    fence = ConversationFence(engine, timeout_seconds=1.0)
    fence.acquire(["hk:a:1"])
    fence.assert_held()  # 不抛
    fence.release()


def test_sqlite_noop():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    metadata.create_all(engine)
    fence = ConversationFence(engine, timeout_seconds=0.1)
    fence.acquire(["hk:a:1", "hk:b:2"])
    fence.assert_held()  # sqlite no-op：恒通过
    assert fence.held_keys == ["hk:a:1", "hk:b:2"]
    fence.release()
    assert fence.held_keys == []

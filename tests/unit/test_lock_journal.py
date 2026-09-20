"""lock：PID 存活/死亡、跨主机、stale 归档接管、force-unlock；journal 两分支恢复素材。"""

from __future__ import annotations

import json

import pytest

from app.evolution.lock import (
    CrossHostLockError,
    Journal,
    LockGuard,
    LockHeldError,
    journal_entry,
)


# ============================================================
# LockGuard
# ============================================================
def test_acquire_and_release(tmp_path):
    lock = LockGuard(tmp_path / "run.lock", stale_seconds=10)
    lock.acquire(phase="run")
    assert lock.path.exists()
    data = json.loads(lock.path.read_text(encoding="utf-8"))
    assert data["hostname"]
    assert data["phase"] == "run"
    lock.release()
    assert not lock.path.exists()


def test_held_by_alive_process(tmp_path, frozen_clock):
    lock1 = LockGuard(tmp_path / "run.lock", stale_seconds=100, clock=frozen_clock)
    lock1.acquire()
    lock2 = LockGuard(tmp_path / "run.lock", stale_seconds=100,
                      clock=frozen_clock, pid_alive_fn=lambda pid: True)
    with pytest.raises(LockHeldError):
        lock2.acquire()
    lock1.release()


def test_dead_pid_not_stale_rejected(tmp_path, frozen_clock):
    lock1 = LockGuard(tmp_path / "run.lock", stale_seconds=100, clock=frozen_clock)
    lock1.acquire()
    lock2 = LockGuard(tmp_path / "run.lock", stale_seconds=100,
                      clock=frozen_clock, pid_alive_fn=lambda pid: False)
    with pytest.raises(LockHeldError):
        lock2.acquire()
    lock1.release()


def test_dead_pid_stale_takeover(tmp_path, frozen_clock):
    lock1 = LockGuard(tmp_path / "run.lock", stale_seconds=10, clock=frozen_clock)
    lock1.acquire()
    frozen_clock.advance(days=1)  # 超过 stale
    lock2 = LockGuard(tmp_path / "run.lock", stale_seconds=10,
                      clock=frozen_clock, pid_alive_fn=lambda pid: False)
    lock2.acquire()  # 接管成功：旧锁归档
    assert lock2.path.exists()
    stale = tmp_path / "run.lock.stale"
    assert stale.exists()
    lock1.release()
    lock2.release()


def test_cross_host_rejected(tmp_path, frozen_clock):
    lock1 = LockGuard(tmp_path / "run.lock", stale_seconds=100,
                      clock=frozen_clock, hostname="host-a")
    lock1.acquire()
    lock2 = LockGuard(tmp_path / "run.lock", stale_seconds=100,
                      clock=frozen_clock, hostname="host-b")
    with pytest.raises(CrossHostLockError):
        lock2.acquire()
    lock1.release()


def test_force_unlock(tmp_path):
    lock1 = LockGuard(tmp_path / "run.lock", stale_seconds=100)
    lock1.acquire()
    lock2 = LockGuard(tmp_path / "run.lock", stale_seconds=100,
                      pid_alive_fn=lambda pid: True)
    with pytest.raises(LockHeldError):
        lock2.acquire()
    lock2.force_unlock()
    lock3 = LockGuard(tmp_path / "run.lock", stale_seconds=100)
    lock3.acquire()  # 释放后可再获取
    lock3.release()


# ============================================================
# Journal
# ============================================================
def test_journal_roundtrip_clear(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    assert journal.read() is None
    entry = journal_entry(
        staging_docs=["evolved/20260828-a.md"],
        backend="numpy",
        index_info=None,
    )
    journal.write(entry)
    assert journal.read()["staging_docs"] == ["evolved/20260828-a.md"]
    journal.clear()
    assert journal.read() is None


def test_journal_archive(tmp_path):
    journal = Journal(tmp_path / "journal.json")
    journal.write({"phase": "publish", "staging_docs": []})
    journal.archive()
    assert journal.read() is None
    assert (tmp_path / "journal.json.archived").exists()


def test_journal_entry_shape():
    from app.evolution.generation import GenerationInfo

    info = GenerationInfo(generation_id="g1", target="/tmp/x.json",
                          embedding_model="m")
    entry = journal_entry(["evolved/a.md"], "numpy", info)
    assert entry["phase"] == "publish"
    assert entry["index"]["generation_id"] == "g1"
    assert entry["backend"] == "numpy"


# ============================================================
# 低危修复 C4：损坏锁 fail-closed / RedisLockGuard.assert_held
# ============================================================
def test_file_lock_corrupt_holder_fails_closed(tmp_path):
    """半截 JSON / 无持有者字段的锁文件 → LockHeldError（提示 --force-unlock），
    不再误报 CrossHostLockError「跨主机持有」。"""
    path = tmp_path / "evolution.lock"
    path.write_text('{"pid": 12', encoding="utf-8")  # 半截 JSON
    with pytest.raises(LockHeldError, match="损坏"):
        LockGuard(path, stale_seconds=1000).acquire(phase="run")

    path.write_text("{}", encoding="utf-8")  # 合法 JSON 但识别不出持有者
    with pytest.raises(LockHeldError, match="损坏"):
        LockGuard(path, stale_seconds=1000).acquire(phase="run")

    # 正常流程不回归（文件不存在 → O_EXCL 创建）
    path.unlink()
    guard = LockGuard(path, stale_seconds=1000)
    guard.acquire(phase="run")
    guard.release()


def test_redis_lock_guard_assert_held():
    """RedisLockGuard 补 assert_held：未持有 / 键值不匹配（TTL 过期或被抢占）
    → LockLostError；持有时通过。"""
    import fakeredis

    from app.evolution.lock import LockLostError, RedisLockGuard

    r = fakeredis.FakeRedis(server=fakeredis.FakeServer())
    guard = RedisLockGuard(r, "evolution", ttl_seconds=300)
    with pytest.raises(LockLostError):
        guard.assert_held()  # 未持有
    guard.acquire(phase="run")
    guard.assert_held()  # 持有 → 通过
    r.set(guard._key, "other-token")  # 模拟被抢占
    with pytest.raises(LockLostError):
        guard.assert_held()
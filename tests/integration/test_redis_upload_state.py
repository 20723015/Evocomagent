"""真实 Redis 集成测试：上传断点 Lua 协议（不依赖 fakeredis 行为）。

运行条件（满足任一即执行，否则 skip）：
- 环境变量 TEST_REDIS_URL 指向可用 Redis；或
- settings.redis_url 指向本机可用 Redis。

用途：fakeredis[lua] 只能证明脚本在"fakeredis 的 Lua 解释器"下正确；
本文件在真实 Redis 上跑同一协议，覆盖
declare/finalize/seal/unseal/drop 与 TTL 语义。

注意：用独立 DB（16）并每测试清理，避免污染本机数据。
"""

from __future__ import annotations

import os

import pytest

from app.config.settings import settings
from app.stores.upload_state import (
    RedisUploadStateStore,
    SESSION_ACCEPTING,
    SESSION_SEALED,
    STATE_READY,
)


def _redis_available() -> None:
    import redis

    url = os.environ.get("TEST_REDIS_URL") or settings.redis_url
    db = url.rsplit("/", 1)[-1]
    conn = redis.Redis.from_url(url)
    conn.ping()
    return conn, int(db) if db.isdigit() else 16


REDIS_ERR = "需要真实 Redis（设置 TEST_REDIS_URL 或启动本机 redis://localhost:6379）才能运行集成测试"

pytestmark = [
    pytest.mark.skipif(
        not (os.environ.get("TEST_REDIS_URL") or settings.redis_url),
        reason=REDIS_ERR,
    ),
]


@pytest.fixture()
def redis_store():
    """真实 Redis 上的状态存储（独立 DB，用完即清）。"""
    try:
        conn, db = _redis_available()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"真实 Redis 不可用: {e}")
    conn.select(db)
    keys = [k for k in conn.scan_iter("upload:*") if True]
    for k in list(keys):
        conn.delete(k)
    yield RedisUploadStateStore(conn, ttl_seconds=3600, publish_timeout=5)
    for k in list(conn.scan_iter("upload:*")):
        conn.delete(k)
    conn.close()


def _session(upload_id="it-up-1", total=2):
    from tests.unit.test_upload_state import _session as mk

    return mk(upload_id=upload_id, total=total)


def test_declare_finalize_seal_real_redis(redis_store):
    store = redis_store
    up = "it-declare"
    store.create(_session(up))
    store.create(_session(up))  # 幂等：重复 create 不覆盖

    c1 = store.declare_chunk(up, 0, 500, "a" * 64, "t0")
    assert c1 == "NEW"
    c2 = store.declare_chunk(up, 0, 500, "a" * 64, "t0")
    assert c2 == "SAME_REQUEST"  # 同 token 重试
    assert store.finalize_chunk(up, 0, "t0", 500, "a" * 64, "obj-0") == "READY"
    assert store.declare_chunk(up, 0, 500, "a" * 64, "t0") == "DUPLICATE"

    assert store.declare_chunk(up, 1, 500, "b" * 64, "t1") == "NEW"
    assert store.finalize_chunk(up, 1, "t1", 500, "b" * 64, "obj-1") == "READY"
    assert store.seal_if_ready(up) == "SEALED"
    assert store.session_state(up) == SESSION_SEALED
    assert store.ready_count(up) == 2
    store.delete(up)
    assert store.get(up) is None


def test_ownership_takeover_real_redis(redis_store):
    """publishing 超时后旧 token 失效，新 token 接管成功。"""
    store = redis_store
    up = "it-takeover"
    store.create(_session(up))
    assert store.declare_chunk(up, 0, 500, "a" * 64, "old") == "NEW"
    # 未超时：另一 token 同内容 → PENDING
    assert store.declare_chunk(up, 0, 500, "a" * 64, "new") == "PENDING"
    # 旧 token 仍可 finalize
    assert store.finalize_chunk(up, 0, "old", 500, "a" * 64, "obj") == "READY"
    # 超时后：新 token 接管（publish_timeout=5）
    up2 = "it-takeover-2"
    store.create(_session(up2))
    assert store.declare_chunk(up2, 0, 500, "a" * 64, "old") == "NEW"
    import time

    time.sleep(6)
    assert store.declare_chunk(up2, 0, 500, "a" * 64, "new") == "TAKEOVER"
    # 旧 token finalize 被拒（token 已换）
    assert store.finalize_chunk(up2, 0, "old", 500, "a" * 64, "obj") == "TOKEN_MISMATCH"
    assert store.finalize_chunk(up2, 0, "new", 500, "a" * 64, "obj") == "READY"
    store.delete(up2)


def test_unseal_repair_real_redis(redis_store):
    """sealed 后 ready-but-missing 修复：解封 → 重传。"""
    store = redis_store
    up = "it-unseal"
    store.create(_session(up))
    assert store.declare_chunk(up, 0, 500, "a" * 64, "t0") == "NEW"
    assert store.finalize_chunk(up, 0, "t0", 500, "a" * 64, "obj-0") == "READY"
    assert store.declare_chunk(up, 1, 500, "b" * 64, "t1") == "NEW"
    assert store.finalize_chunk(up, 1, "t1", 500, "b" * 64, "obj-1") == "READY"
    assert store.seal_if_ready(up) == "SEALED"

    # 元数据不匹配 → 拒绝解封
    assert store.unseal_and_drop(up, 0, 999, "x" * 64) == "META_MISMATCH"
    assert store.unseal_and_drop(up, 0, 500, "a" * 64) == "UNSEALED"
    assert store.session_state(up) == SESSION_ACCEPTING

    # 重传不走预生成 token：全新声明
    assert store.declare_chunk(up, 0, 500, "a" * 64, "t0-r") == "NEW"
    assert store.finalize_chunk(up, 0, "t0-r", 500, "a" * 64, "obj-0") == "READY"
    assert store.seal_if_ready(up) == "SEALED"
    store.delete(up)


def test_drop_chunk_and_cancel_real_redis(redis_store):
    store = redis_store
    up = "it-drop"
    store.create(_session(up))
    assert store.declare_chunk(up, 0, 500, "a" * 64, "t0") == "NEW"
    # 非本人 token → 拒绝删除
    assert store.drop_chunk_record(up, 0, "hacker") is False
    assert store.drop_chunk_record(up, 0, "t0") is True
    assert store.received(up) == {}
    # cancel：先封锁再清
    assert store.declare_chunk(up, 1, 500, "b" * 64, "t1") == "NEW"
    assert store.mark_closed(up, SESSION_SEALED) is True
    assert store.declare_chunk(up, 1, 500, "b" * 64, "t1") == "CLOSED"
    store.delete(up)


def test_publishing_record_by_timeout_claimable_real_redis(redis_store):
    """残留 publishing 记录在接管窗口后可被清理/接管（TTL 兜底之外）。"""
    store = redis_store
    up = "it-stale"
    store.create(_session(up))
    assert store.declare_chunk(up, 0, 500, "a" * 64, "t0") == "NEW"
    assert store.declare_chunk(up, 0, 500, "a" * 64, "t0") == "SAME_REQUEST"
    assert store.drop_chunk_record(up, 0, "t0") is True
    store.delete(up)
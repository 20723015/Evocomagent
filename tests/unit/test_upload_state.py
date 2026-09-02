"""断点状态协议单测（v7 冻结：session 三态 + publishing/ready + Lua 原子语义）。

Redis 版（fakeredis[lua] 支持 eval）与进程内版行为对齐；
语义化异常（UploadConflict/UploadClosed）由调用方在 code 上自行映射。
"""

from __future__ import annotations

import pytest

import fakeredis

from app.stores.upload_state import (
    InProcessUploadStateStore,
    RedisUploadStateStore,
    SESSION_ACCEPTING,
    SESSION_CANCELLED,
    SESSION_SEALED,
    STATE_PUBLISHING,
    STATE_READY,
    UploadSession,
)
from app.config.settings import settings


def _session(upload_id="up-1", total=2) -> UploadSession:
    return UploadSession(
        upload_id=upload_id,
        filename="政策.pdf",
        size_bytes=1000,
        chunk_size=500,
        total_chunks=total,
        uploader="ops-a",
        content_type="application/pdf",
        format="pdf",
        sha256="",
    )


CHUNK1 = (500, "a" * 64)  # (size, sha256)
CHUNK2 = (500, "b" * 64)


def _put(state, upload_id, seq, size, sha256, token="tok"):
    """模拟一次 PUT 全流程：声明 → 发布 → finalize。返回中间 code。"""
    code = state.declare_chunk(upload_id, seq, size, sha256, token)
    if code in ("NEW", "TAKEOVER", "SAME_REQUEST"):
        f = state.finalize_chunk(
            upload_id, seq, token, size, sha256, object_key=f"{seq:05d}-{sha256}",
        )
        assert f in ("READY", "CLOSED"), f
        return f
    return code


@pytest.fixture(params=["redis", "inproc"])
def state(request):
    if request.param == "redis":
        redis = fakeredis.FakeStrictRedis(
            server=fakeredis.FakeServer(), decode_responses=False,
        )
        s = RedisUploadStateStore(
            redis, ttl_seconds=600, publish_timeout=60,
        )
    else:
        s = InProcessUploadStateStore(ttl_seconds=600, publish_timeout=60)
    s.create(_session())
    return s


class TestSession:
    def test_create_idempotent(self, state):
        s2 = _session(upload_id="up-1", total=99)
        state.create(s2)
        got = state.get("up-1")
        assert got.total_chunks == 2  # 原会话保留，不被覆盖

    def test_session_state_tri_state(self, state):
        assert state.session_state("up-1") == SESSION_ACCEPTING
        assert state.mark_closed("up-1", SESSION_SEALED) is True
        assert state.session_state("up-1") == SESSION_SEALED


class TestDeclareFinalize:
    def test_new_then_ready(self, state):
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t1") == "NEW"
        assert state.finalize_chunk("up-1", 0, "t1", 500, "a" * 64, "k1") == "READY"
        meta = state.received("up-1")[0]
        assert meta["state"] == STATE_READY
        assert meta["object_key"] == "k1"

    def test_duplicate_only_when_ready_and_same(self, state):
        _put(state, "up-1", 0, *CHUNK1, token="t1")
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t2") == "DUPLICATE"

    def test_conflict_keeps_original_chunk(self, state):
        _put(state, "up-1", 0, *CHUNK1, token="t1")
        code = state.declare_chunk("up-1", 0, 500, "c" * 64, "t2")
        assert code == "CONFLICT"
        meta = state.received("up-1")[0]
        assert meta["sha256"] == "a" * 64  # 原分片不被覆盖

    def test_same_request_retry(self, state):
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t1") == "NEW"
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t1") == "SAME_REQUEST"
        assert state.finalize_chunk("up-1", 0, "t1", 500, "a" * 64, "k1") == "READY"

    def test_pending_then_takeover_after_timeout(self, state):
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t1") == "NEW"
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t2") == "PENDING"
        # 越过接管超时：Redis 版 publish_timeout=0 的 store 直接接管；进程内版改时间戳
        if isinstance(state, RedisUploadStateStore):
            state._publish_timeout = 0
        else:
            with state._lock:
                state._chunks["up-1"][0]["publishing_at"] -= 100
        code = state.declare_chunk("up-1", 0, 500, "a" * 64, "t2")
        assert code == "TAKEOVER"
        # 旧 token 不能 finalize（token 已换）
        assert state.finalize_chunk("up-1", 0, "t1", 500, "a" * 64, "k1") == "TOKEN_MISMATCH"
        assert state.finalize_chunk("up-1", 0, "t2", 500, "a" * 64, "k2") == "READY"

    def test_seal_blocks_new_puts(self, state):
        _put(state, "up-1", 0, *CHUNK1, token="t1")
        _put(state, "up-1", 1, *CHUNK2, token="t2")
        assert state.seal_if_ready("up-1") == "SEALED"
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t3") == "CLOSED"
        assert state.finalize_chunk("up-1", 0, "t3", 500, "a" * 64, "k3") == "CLOSED"

    def test_seal_incomplete_and_publishing(self, state):
        assert state.seal_if_ready("up-1") == "INCOMPLETE"
        state.declare_chunk("up-1", 0, 500, "a" * 64, "t1")  # publishing 未 finalize
        assert state.seal_if_ready("up-1") == "HAS_PUBLISHING"
        state.finalize_chunk("up-1", 0, "t1", 500, "a" * 64, "k1")
        assert state.seal_if_ready("up-1") == "INCOMPLETE"  # 仍缺 1 片

    def test_unseal_and_drop_ready_missing(self, state):
        _put(state, "up-1", 0, *CHUNK1, token="t1")
        _put(state, "up-1", 1, *CHUNK2, token="t2")
        assert state.seal_if_ready("up-1") == "SEALED"
        # 模拟 complete 发现 seq=0 对象缺失 → 解封（不预生成 token）
        assert state.unseal_and_drop("up-1", 0, 500, "a" * 64) == "UNSEALED"
        assert state.session_state("up-1") == SESSION_ACCEPTING
        assert state.received("up-1") == {1: state.received("up-1")[1]}  # seq0 已删除
        # 重传走全新声明（NEW 而非 duplicate）
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t-new") == "NEW"
        assert state.finalize_chunk("up-1", 0, "t-new", 500, "a" * 64, "k-new") == "READY"

    def test_unseal_precondition_guards(self, state):
        _put(state, "up-1", 0, *CHUNK1, token="t1")
        assert state.unseal_and_drop("up-1", 0, 500, "a" * 64) == "NOT_SEALED"  # accepting 不解封
        assert state.seal_if_ready("up-1") == "INCOMPLETE"

    def test_drop_chunk_record_same_token_only(self, state):
        state.declare_chunk("up-1", 0, 500, "a" * 64, "t1")
        assert state.drop_chunk_record("up-1", 0, "t1") is True
        assert state.received("up-1") == {}  # 声明被撤回
        state.declare_chunk("up-1", 0, 500, "a" * 64, "t2")
        assert state.drop_chunk_record("up-1", 0, "t-other") is False

    def test_ready_count(self, state):
        _put(state, "up-1", 0, *CHUNK1, token="t1")
        assert state.ready_count("up-1") == 1
        state.declare_chunk("up-1", 1, 500, "b" * 64, "t2")  # publishing 不计数
        assert state.ready_count("up-1") == 1

    def test_cancel_closes_puts(self, state):
        state.mark_closed("up-1", SESSION_CANCELLED)
        assert state.session_state("up-1") == SESSION_CANCELLED
        assert state.declare_chunk("up-1", 0, 500, "a" * 64, "t1") == "CLOSED"

    def test_delete_removes_session(self, state):
        state.delete("up-1")
        assert state.get("up-1") is None

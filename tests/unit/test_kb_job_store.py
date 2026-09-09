"""KbIndexJobStore 单测（多实例异步建库改造）：事务入队、唯一键幂等、
SKIP LOCKED 领取不重复、租约接管、heartbeat、所有权检查、失败分流、
取消/人工重试/保留期清理。全程 sqlite + StaticPool（进程内并发安全）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.agent.rag.job_store import (
    JOB_BLOCKED,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RETRY_WAIT,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JobConflictError,
    JobLeaseLost,
    KbIndexJobStore,
)
from app.stores.base import StorageUnavailableError
from app.stores.sql.document_store import SqlDocumentStore
from app.stores.sql.schema import metadata


def _utcnow_naive():
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.fixture()
def env():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return engine, KbIndexJobStore(engine, backoff_base=0), SqlDocumentStore(engine)


def _make_doc(doc_store, upload_id="up-1"):
    from app.stores.sql.document_store import DocumentRecord

    return doc_store.create(DocumentRecord(
        doc_id=f"doc-{upload_id}", upload_id=upload_id, storage_key=f"{upload_id}.md",
        filename="a.md", format="md", size_bytes=10, sha256="",
        uploader="ops-a",
    ))


class TestEnqueue:
    def test_enqueue_atomic_with_doc_cas(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        job, created = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a",
                                            rec.version)
        assert created is True
        assert job["status"] == JOB_QUEUED
        assert doc_store.get(rec.doc_id).status == "queued"

    def test_duplicate_request_returns_same_job(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        job1, c1 = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job2, c2 = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        assert (c1, c2) == (True, False)
        assert job1["job_id"] == job2["job_id"]

    def test_version_conflict_leaves_no_job(self, env):
        """文档 CAS 失败（并发已变化）→ 任务不入队（同一事务回滚语义）。"""
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        with pytest.raises(JobConflictError):
            store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a",
                                 rec.version + 99)
        assert store.find("upload", doc_id=rec.doc_id) is None

    def test_cas_conflict_rereads_winner_task(self, env, monkeypatch):
        """并发入队输家在 CAS 冲突后返回赢家任务，而不是 409/500。"""
        engine, store, doc_store = env
        rec = _make_doc(doc_store)
        winner = {
            "job_id": "winner-job", "operation": "upload", "doc_id": rec.doc_id,
            "upload_id": rec.upload_id, "status": JOB_QUEUED,
        }

        class _Result:
            rowcount = 0

            def mappings(self):
                return self

            def first(self):
                return None

        class _Conn:
            def execute(self, *_args, **_kwargs):
                return _Result()

        class _Begin:
            def __enter__(self):
                return _Conn()

            def __exit__(self, *_args):
                return False

        class _Engine:
            dialect = engine.dialect

            def begin(self):
                return _Begin()

        store._engine = _Engine()
        monkeypatch.setattr(
            store, "find",
            lambda _operation, *, doc_id="", upload_id="": winner,
        )
        job, created = store.enqueue_upload(
            rec.doc_id, rec.upload_id, "ops-a", rec.version,
        )
        assert (job, created) == (winner, False)

    def test_sql_failure_then_retry_can_enqueue(self, env):
        """入队 SQL 失败（连接异常）→ 文档仍在 uploading，重复 complete 可补入队。"""
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)

        class _Boom:
            def __enter__(self):
                raise RuntimeError("连接闪断（注入）")

            def __exit__(self, *a):
                return False

        orig = store._engine
        store._engine = type("P", (), {
            "begin": lambda self: _Boom(),
            "connect": lambda self: _Boom(),
            "dialect": orig.dialect,
        })()
        with pytest.raises(StorageUnavailableError):
            store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        store._engine = orig
        assert doc_store.get(rec.doc_id).status == "uploading"  # CAS 已回滚
        _job, created = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a",
                                            doc_store.get(rec.doc_id).version)
        assert created is True

    def test_delete_enqueue_cas(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        doc_store.update_status(rec.doc_id, "uploading", "validating", rec.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "validating", "indexing", cur.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "indexing", "indexed", cur.version)
        cur = doc_store.get(rec.doc_id)
        with pytest.raises(JobConflictError):
            store.enqueue_delete(rec.doc_id, "ops-a", rec.version)  # 版本不匹配
        job, created = store.enqueue_delete(rec.doc_id, "ops-a", cur.version)
        assert created and job["operation"] == "delete"
        assert doc_store.get(rec.doc_id).status == "delete_queued"

    def test_cancelled_upload_session_cannot_reenqueue(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        job1, _ = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        assert store.cancel(job1["job_id"]) is not None
        with pytest.raises(JobConflictError):
            store.enqueue_upload(
                rec.doc_id, rec.upload_id, "ops-a",
                doc_store.get(rec.doc_id).version,
            )
        assert store.get(job1["job_id"])["status"] == JOB_CANCELLED


class TestClaimLease:
    def test_claim_no_duplicates_across_workers(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        first = store.claim("w1", limit=1)
        second = store.claim("w2", limit=1)
        assert len(first) == 1
        assert second == []  # 已被 w1 领取（running），不重复发放
        assert first[0]["lease_owner"] == "w1"
        assert first[0]["lease_token"]
        assert first[0]["attempts"] == 1

    def test_lease_expiry_takeover(self, env):
        engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        first = store.claim("w1")[0]
        # 模拟 w1 崩溃：租约到期
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET lease_until = :ts WHERE job_id = :jid"
            ), {"ts": _utcnow_naive() - timedelta(seconds=1),
                "jid": first["job_id"]})
        taken = store.claim("w2")
        assert len(taken) == 1
        assert taken[0]["lease_token"] != first["lease_token"]  # 新 token → 旧持有者失主
        assert taken[0]["attempts"] == 2

    def test_heartbeat_and_ownership(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        assert store.heartbeat(job["job_id"], job["lease_token"]) is True
        assert store.heartbeat(job["job_id"], "wrong-token") is False
        assert store.check_owner(job["job_id"], job["lease_token"]) is True
        assert store.check_owner(job["job_id"], "wrong-token") is False

    def test_expired_lease_cannot_renew_or_write(self, env):
        """token 未被接管前也不能跨过 lease_until fencing。"""
        engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET lease_until = :ts WHERE job_id = :jid"
            ), {"ts": _utcnow_naive() - timedelta(seconds=1),
                "jid": job["job_id"]})

        assert store.check_owner(job["job_id"], job["lease_token"]) is False
        assert store.heartbeat(job["job_id"], job["lease_token"]) is False
        store.release_lease(job["job_id"], job["lease_token"])
        assert store.get(job["job_id"])["status"] == JOB_RUNNING
        with pytest.raises(JobLeaseLost):
            store.set_stage(job["job_id"], job["lease_token"], "embedding")
        with pytest.raises(JobLeaseLost):
            store.succeed(job["job_id"], job["lease_token"])
        with pytest.raises(JobLeaseLost):
            store.fail(job, RuntimeError("stale"))
        with pytest.raises(JobLeaseLost):
            store.mark_blocked(job["job_id"], job["lease_token"], "stale")

    def test_set_stage_ownership_enforced(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        store.set_stage(job["job_id"], job["lease_token"], "parsing", progress=10)
        cur = store.get(job["job_id"])
        assert cur["stage"] == "parsing" and cur["progress"] == 10
        with pytest.raises(JobLeaseLost):
            store.set_stage(job["job_id"], "wrong-token", "embedding", progress=30)

    def test_stale_token_cannot_succeed(self, env):
        """旧 lease token 无法完成任务（提交前所有权检查的存储层保证）。"""
        engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET lease_until = :ts WHERE job_id = :jid"
            ), {"ts": _utcnow_naive() - timedelta(seconds=1),
                "jid": job["job_id"]})
        taken = store.claim("w2")[0]
        with pytest.raises(JobLeaseLost):
            store.succeed(job["job_id"], job["lease_token"])  # 旧 Worker 提交被拒
        store.succeed(taken["job_id"], taken["lease_token"])  # 新持有者可提交
        assert store.get(taken["job_id"])["status"] == JOB_SUCCEEDED

    def test_release_lease_requeues(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        store.release_lease(job["job_id"], job["lease_token"])
        row = store.get(job["job_id"])
        assert row["status"] == JOB_QUEUED
        assert row["lease_token"] == ""
        assert row["attempts"] == 0  # 让出不计尝试


class TestFailClassification:
    def test_lock_wait_not_counted_as_failure(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]  # attempts = 1
        status = store.fail(job, RuntimeError("锁被持有"), lock_wait=True)
        row = store.get(job["job_id"])
        assert (status, row["status"]) == (JOB_QUEUED, JOB_QUEUED)
        assert row["attempts"] == 0  # 锁等待不计失败次数
        assert row["next_run_at"] is not None

    def test_transient_retry_then_dead_letter(self, env):
        store = KbIndexJobStore(env[0], max_attempts=2, backoff_base=0)
        doc_store = env[2]
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        assert store.fail(job, RuntimeError("ES 5xx")) == JOB_RETRY_WAIT
        job2 = store.claim("w1")[0]  # backoff=0 立即可领取
        assert job2["attempts"] == 2
        assert store.fail(job2, RuntimeError("ES 5xx")) == JOB_FAILED
        row = store.get(job["job_id"])
        assert row["status"] == JOB_FAILED and row["retryable"] == 1
        # 人工重试：重置本轮 attempts，审计数只增
        retried = store.retry(job["job_id"])
        assert retried["status"] == JOB_QUEUED and retried["attempts"] == 0
        assert retried["manual_retry_count"] == 1

    def test_permanent_error_not_retryable(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        assert store.fail(job, ValueError("解析失败"), retryable=False) == JOB_FAILED
        row = store.get(job["job_id"])
        assert row["retryable"] == 0
        assert store.retry(job["job_id"]) is None  # 永久错误不可人工重试

    def test_post_commit_never_dead_letter(self, env):
        """提交点后可确定故障：持续重试，绝不进入普通死信。"""
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        for attempt in range(1, 8):  # 远超 max_attempts
            job = store.claim("w1")[0]
            assert store.fail(job, RuntimeError("指针写失败"), post_commit=True) \
                == JOB_RETRY_WAIT
        assert store.get(job["job_id"])["status"] == JOB_RETRY_WAIT


class TestCancelAndCleanup:
    def test_cancel_queued_cancels_upload_doc_atomically(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        job, _ = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        cancelled = store.cancel(job["job_id"])
        assert cancelled["status"] == JOB_CANCELLED
        assert doc_store.get(rec.doc_id).status == "cancelled"

    def test_cancel_running_returns_none(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        job, _ = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        store.claim("w1")
        assert store.cancel(job["job_id"]) is None  # 运行中不可取消（409 语义）
        assert store.get(job["job_id"])["status"] == JOB_RUNNING

    def test_cancel_post_commit_retry_wait_returns_none(self, env):
        """提交点后 retry_wait 不能取消，避免中断 journal 前进恢复。"""
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        job, _ = store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "queued", "validating", cur.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "validating", "indexing", cur.version)
        claimed = store.claim("w1")[0]
        assert store.fail(claimed, RuntimeError("pointer"), post_commit=True) \
            == JOB_RETRY_WAIT

        assert store.cancel(job["job_id"]) is None
        assert store.get(job["job_id"])["status"] == JOB_RETRY_WAIT
        assert doc_store.get(rec.doc_id).status == "indexing"

    def test_cancel_delete_post_commit_retry_wait_returns_none(self, env):
        """下架越过提交点后也只能前进恢复，不能被取消回 indexed。"""
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        doc_store.update_status(rec.doc_id, "uploading", "validating", rec.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "validating", "indexing", cur.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "indexing", "indexed", cur.version)
        cur = doc_store.get(rec.doc_id)
        job, _ = store.enqueue_delete(rec.doc_id, "ops-a", cur.version)
        claimed = store.claim("w1")[0]
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(
            rec.doc_id, "delete_queued", "deleting", cur.version,
            operation_id=job["job_id"],
        )
        assert store.fail(
            claimed, RuntimeError("pointer"), post_commit=True,
        ) == JOB_RETRY_WAIT

        assert store.cancel(job["job_id"]) is None
        assert store.get(job["job_id"])["status"] == JOB_RETRY_WAIT
        assert doc_store.get(rec.doc_id).status == "deleting"

    def test_cancel_delete_queued_restores_indexed(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        doc_store.update_status(rec.doc_id, "uploading", "validating", rec.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "validating", "indexing", cur.version)
        cur = doc_store.get(rec.doc_id)
        doc_store.update_status(rec.doc_id, "indexing", "indexed", cur.version)
        cur = doc_store.get(rec.doc_id)
        job, _ = store.enqueue_delete(rec.doc_id, "ops-a", cur.version)
        store.cancel(job["job_id"])
        assert doc_store.get(rec.doc_id).status == "indexed"

    def test_cleanup_retention(self, env):
        engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        store.succeed(job["job_id"], job["lease_token"])
        rec2 = _make_doc(doc_store, "up-2")
        job2, _ = store.enqueue_upload(rec2.doc_id, rec2.upload_id, "ops-a",
                                       rec2.version)
        claimed2 = store.claim("w1")[0]
        store.fail(claimed2, ValueError("解析失败"), retryable=False)  # 直接终态
        # 手工把终态时间回拨（91 天 / 181 天）
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET finished_at = :ts WHERE job_id = :jid"),
                {"ts": _utcnow_naive() - timedelta(days=91),
                 "jid": job["job_id"]})
        assert store.cleanup(done_days=90, failed_days=180) == 1  # 只清 succeeded
        assert store.get(job["job_id"]) is None
        assert store.get(job2["job_id"]) is not None
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET finished_at = :ts WHERE job_id = :jid"),
                {"ts": _utcnow_naive() - timedelta(days=181),
                 "jid": job2["job_id"]})
        assert store.cleanup(done_days=90, failed_days=180) == 1
        assert store.get(job2["job_id"]) is None

    def test_stats_backlog_and_oldest(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        stats = store.stats()
        assert stats["backlog"]["upload"] == 1
        assert stats["oldest_age_seconds"]["upload"] >= 0
        assert stats["blocked"] == 0

    def test_mark_blocked(self, env):
        _engine, store, doc_store = env
        rec = _make_doc(doc_store)
        store.enqueue_upload(rec.doc_id, rec.upload_id, "ops-a", rec.version)
        job = store.claim("w1")[0]
        store.mark_blocked(job["job_id"], job["lease_token"], "alias 指向未知代")
        row = store.get(job["job_id"])
        assert row["status"] == JOB_BLOCKED
        assert store.retry(job["job_id"])["status"] == JOB_QUEUED  # 人工可重试

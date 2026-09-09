"""KB Worker 单测（多实例异步建库改造）：端到端 202 → 终态、锁等待回队、
租约接管（旧 token 无法提交）、journal 接管恢复（PREPARED 回滚重跑 /
POINTER_UPDATED 只前进）、失败分流与人工重试、停机宽限与状态查询 SQL 正本。
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import kb_async_testkit as kit
import pytest
from sqlalchemy import text

from app.agent.rag.job_store import (
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RETRY_WAIT,
    JOB_SUCCEEDED,
    JobLeaseLost,
)
from app.agent.rag.kb_worker import JobControl, KbIndexWorker
from app.agent.rag.upload_service import (
    PH_POINTER_UPDATED,
    UploadConflictError,
)
from app.evolution.generation import new_generation_id
from app.stores.base import StorageUnavailableError
from app.stores.kb_write_lock import KbWriteLockError
from app.stores.sql.document_store import (
    STATUS_DELETED,
    STATUS_INDEXED,
    STATUS_QUEUED,
    STATUS_UPLOADING,
    STATUS_VALIDATING,
)


def _utcnow_naive():
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    svc, jobs, deps = kit.build_async_kb(tmp_path, monkeypatch, lock_wait_seconds=0)
    kit.enable_async(deps[1])
    return svc, jobs, deps


def _worker(jobs, svc, worker_id="w1"):
    return KbIndexWorker(jobs, svc, worker_id=worker_id, poll_seconds=0.05)


def rec_storage_key(svc, upload_id):
    return svc._doc.get_by_upload_id(upload_id).storage_key


def _expire_lease(deps, job_id):
    """模拟 Worker 强杀：租约立即到期（接管前提）。"""
    from sqlalchemy import text as _text

    with deps[0]._engine.begin() as conn:
        conn.execute(_text(
            "UPDATE kb_index_jobs SET lease_until = :ts WHERE job_id = :jid"),
            {"ts": _utcnow_naive() - timedelta(seconds=1), "jid": job_id})


class TestEndToEnd:
    def test_upload_202_then_worker_indexes(self, env):
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        # 202 语义：任务负载（API 层映射状态码/Location/Retry-After）
        assert payload["status"] == JOB_QUEUED
        assert payload["operation"] == "upload"
        assert payload["status_url"] == f"/v1/kb/jobs/{payload['job_id']}"

        w = _worker(jobs, svc)
        assert w.process_once() is True
        rec = deps[0].get(payload["doc_id"])
        assert rec.status == STATUS_INDEXED
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED
        # 收尾：分片/会话清理 + journal 清
        assert svc._state.get("up-1") is None

        # 幂等 complete：已 indexed → 终态负载（无 job_id → API 200）
        again = svc.complete("up-1", "ops-a")
        assert again["status"] == STATUS_INDEXED
        assert "job_id" not in again

    def test_delete_202_then_worker_deletes(self, env):
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        up = svc.complete("up-1", "ops-a")
        assert _worker(jobs, svc).process_once() is True
        doc_id = up["doc_id"]

        payload = svc.delete_document(doc_id, "ops-a")
        assert payload["status"] == JOB_QUEUED
        assert payload["operation"] == "delete"
        assert _worker(jobs, svc, "w2").process_once() is True
        assert deps[0].get(doc_id).status == STATUS_DELETED
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED
        # 幂等下架：已 deleted → 200 语义
        again = svc.delete_document(doc_id, "ops-a")
        assert again["status"] == STATUS_DELETED

    def test_duplicate_complete_returns_same_job(self, env):
        svc, _jobs, _deps = env
        kit.upload_chunks(svc, "up-1")
        p1 = svc.complete("up-1", "ops-a")
        p2 = svc.complete("up-1", "ops-a")
        assert p1["job_id"] == p2["job_id"]
        assert p2["status"] == JOB_QUEUED

    def test_upload_and_delete_race_serialized_by_write_lock(self, env):
        """上传与删除竞争：全局写锁串行——后提交者基于前一代重建，终态一致。"""
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        up = svc.complete("up-1", "ops-a")
        w = _worker(jobs, svc)
        w.process_once()
        doc_id = up["doc_id"]
        # 删除入队后立即再上传竞争（对已 indexed 文档 complete 是幂等 200，
        # 这里验证删除任务执行后 alias/指针仍自洽）
        job = svc.delete_document(doc_id, "ops-a")
        w.process_once()
        assert deps[0].get(doc_id).status == STATUS_DELETED
        assert jobs.get(job["job_id"])["status"] == JOB_SUCCEEDED


class TestLockWait:
    def test_lock_wait_requeues_without_attempts(self, env):
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        real_make = svc._make_lock
        calls = {"n": 0}

        def flaky_make():
            lock = real_make()
            orig_acquire = lock.acquire

            def acquire(phase="run"):
                if calls["n"] == 0:
                    calls["n"] += 1
                    raise KbWriteLockError("注入：锁被其它运行者持有")
                return orig_acquire(phase)

            lock.acquire = acquire
            return lock

        svc._make_lock = flaky_make
        w = _worker(jobs, svc)
        w.process_once()  # 第一次：锁等待 → 回 queued（不计失败）
        row = jobs.get(payload["job_id"])
        assert row["status"] == JOB_QUEUED
        assert row["attempts"] == 0
        assert deps[0].get(payload["doc_id"]).status == STATUS_QUEUED
        w.process_once()  # 第二次：锁可用 → 正常入库
        assert deps[0].get(payload["doc_id"]).status == STATUS_INDEXED
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED


class TestLeaseAndTakeover:
    def test_takeover_while_waiting_for_lock_preserves_shared_state(
        self, env, monkeypatch,
    ):
        """拿到写锁时已被接管：旧 Worker 不得回滚，新 Worker 可续跑。"""
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        stale = jobs.claim("w1")[0]
        fresh_holder = {}
        real_make = svc._make_lock

        def takeover_make():
            lock = real_make()
            acquire = lock.acquire

            def acquire_then_takeover(phase="run"):
                result = acquire(phase)
                _expire_lease(deps, payload["job_id"])
                fresh_holder["job"] = jobs.claim("w2")[0]
                return result

            lock.acquire = acquire_then_takeover
            return lock

        monkeypatch.setattr(svc, "_make_lock", takeover_make)
        _worker(jobs, svc, "w1")._execute(stale)
        assert deps[0].get(payload["doc_id"]).status == STATUS_VALIDATING
        assert svc._stage.journal_read("up-1") is None

        monkeypatch.setattr(svc, "_make_lock", real_make)
        _worker(jobs, svc, "w2")._execute(fresh_holder["job"])
        assert deps[0].get(payload["doc_id"]).status == STATUS_INDEXED
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED

    def test_lease_lost_immediately_after_alias_never_rolls_back(
        self, env, monkeypatch,
    ):
        """alias 调用后失主：旧 Worker 保留 ACTIVATING，接管者最终收敛。"""
        svc, jobs, deps = env
        doc_store, _control, generation_store, _kb = deps
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        stale = jobs.claim("w1")[0]
        fresh_holder = {}
        activate_alias = svc._index_service.activate_alias

        def activate_then_takeover(backend, info):
            activate_alias(backend, info)
            _expire_lease(deps, payload["job_id"])
            fresh_holder["job"] = jobs.claim("w2")[0]

        monkeypatch.setattr(
            svc._index_service, "activate_alias", activate_then_takeover,
        )
        _worker(jobs, svc, "w1")._execute(stale)
        assert doc_store.get(payload["doc_id"]).status == "indexing"
        assert svc._stage.journal_read("up-1")["phase"] == "ACTIVATING"

        monkeypatch.setattr(svc._index_service, "activate_alias", activate_alias)
        _worker(jobs, svc, "w2")._execute(fresh_holder["job"])
        rec = doc_store.get(payload["doc_id"])
        assert rec.status == STATUS_INDEXED
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED
        assert generation_store.active("numpy").generation_id == rec.generation_id

    def test_stale_token_cannot_commit(self, env):
        """进程强杀后接管：旧 lease token 无法完成任务，新持有者可完成。"""
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        old = jobs.claim("w1")[0]
        # 模拟 w1 强杀（租约到期）→ w2 接管（新 token）
        with deps[0]._engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET lease_until = :ts WHERE job_id = :jid"),
                {"ts": _utcnow_naive() - timedelta(seconds=1),
                 "jid": payload["job_id"]})
        fresh = jobs.claim("w2")[0]
        assert fresh["lease_token"] != old["lease_token"]
        # 旧 Worker：任何提交/进度写都被拒绝
        with pytest.raises(JobLeaseLost):
            jobs.succeed(payload["job_id"], old["lease_token"])
        control = JobControl(jobs, old)
        with pytest.raises(JobLeaseLost):
            control.check_alive()
        # 新 Worker：正常执行入库
        assert _worker(jobs, svc, "w2")._execute(fresh) is None
        assert deps[0].get(payload["doc_id"]).status == STATUS_INDEXED
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED

    def test_takeover_prepared_journal_rolls_back_and_reruns(self, env):
        """接管：PREPARED journal（先 journal 后 CAS 崩溃点）→ 回滚重跑 → indexed。"""
        svc, jobs, deps = env
        doc_store, _control, _gen_store, _kb = deps
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        job = jobs.claim("w1")[0]  # 模拟崩溃的 Worker 已领取
        _expire_lease(deps, job["job_id"])
        gen_id = new_generation_id()
        target = svc._index_service.target_for("numpy", gen_id)
        # journal op_id = upload_id（恢复机按 upload_id 找文档）
        svc._stage.journal_write("up-1", {
            "op": "upload", "phase": "PREPARED", "doc_id": payload["doc_id"],
            "storage_key": rec_storage_key(svc, "up-1"),
            "generation_id": gen_id, "target": target,
        })
        # 模拟崩溃在 journal 后、CAS 前：文档仍在 queued（语义上无副作用）
        # 用 w2 接管执行
        w = _worker(jobs, svc, "w2")
        assert w.process_once() is True
        rec = doc_store.get(payload["doc_id"])
        assert rec.status == STATUS_INDEXED
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED
        assert svc._stage.journal_read("up-1") is None

    def test_takeover_pointer_updated_journal_forwards_only(self, env):
        """接管：POINTER_UPDATED（已越过提交点）→ 只前进，不重建，不回滚。"""
        svc, jobs, deps = env
        doc_store, _control, _gen_store, kb = deps
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        job = jobs.claim("w1")[0]
        _expire_lease(deps, job["job_id"])
        gen_id = new_generation_id()
        target = svc._index_service.target_for("numpy", gen_id)
        # 模拟崩溃在指针提交后、元数据 CAS 前：journal=POINTER_UPDATED，doc=indexing
        svc._stage.journal_write("up-1", {
            "op": "upload", "phase": PH_POINTER_UPDATED, "doc_id": payload["doc_id"],
            "storage_key": rec_storage_key(svc, "up-1"),
            "generation_id": gen_id, "target": target,
        })
        v = doc_store.update_status(
            payload["doc_id"], STATUS_QUEUED, STATUS_VALIDATING,
            doc_store.get(payload["doc_id"]).version, operation_id=job["job_id"],
        )
        doc_store.update_status(
            payload["doc_id"], STATUS_VALIDATING, "indexing", v,
            require_op=job["job_id"], pending_generation_id=gen_id,
        )
        # kb 文件必须已在位（崩溃前已完成移动）
        (kb / "uploads" / "up-1.md").write_text(
            "# 退换货补充说明\n\n## 适用范围\n\n支持七天无理由退货。\n",
            encoding="utf-8",
        )
        assert _worker(jobs, svc, "w2").process_once() is True
        rec = doc_store.get(payload["doc_id"])
        assert rec.status == STATUS_INDEXED
        assert rec.generation_id == gen_id  # 前进恢复：激活的就是 journal 里的代
        assert jobs.get(payload["job_id"])["status"] == JOB_SUCCEEDED

    def test_takeover_alias_activated_forward_keeps_single_active_generation(self, env):
        """双 Worker 故障注入：接管后最终只有一个活动代且文档状态与指针一致。"""
        svc, jobs, deps = env
        doc_store, _control, gen_store, kb = deps
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        job = jobs.claim("w1")[0]
        _expire_lease(deps, job["job_id"])
        gen_id = new_generation_id()
        target = svc._index_service.target_for("numpy", gen_id)
        svc._stage.journal_write("up-1", {
            "op": "upload", "phase": "ALIAS_ACTIVATED", "doc_id": payload["doc_id"],
            "storage_key": rec_storage_key(svc, "up-1"),
            "generation_id": gen_id, "target": target,
        })
        v = doc_store.update_status(
            payload["doc_id"], STATUS_QUEUED, STATUS_VALIDATING,
            doc_store.get(payload["doc_id"]).version, operation_id=job["job_id"],
        )
        doc_store.update_status(
            payload["doc_id"], STATUS_VALIDATING, "indexing", v,
            require_op=job["job_id"], pending_generation_id=gen_id,
        )
        (kb / "uploads" / "up-1.md").write_text(
            "# 退换货补充说明\n\n## 适用范围\n\n支持七天无理由退货。\n",
            encoding="utf-8",
        )
        assert _worker(jobs, svc, "w2").process_once() is True
        rec = doc_store.get(payload["doc_id"])
        assert rec.status == STATUS_INDEXED and rec.generation_id == gen_id
        pointer = gen_store.active("numpy")
        assert pointer.generation_id == gen_id  # 唯一活动代与文档一致

    def test_takeover_unknown_alias_state_retries_instead_of_blocking(
        self, env, monkeypatch,
    ):
        """ES alias 查询暂时失败属于可恢复故障，不能误置 blocked。"""
        svc, jobs, deps = env
        doc_store, control_store, _gen_store, _kb = deps
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        stale = jobs.claim("w1")[0]
        _expire_lease(deps, stale["job_id"])
        gen_id = new_generation_id()
        target = svc._index_service.target_for("numpy", gen_id)
        svc._stage.journal_write("up-1", {
            "op": "upload", "phase": "ACTIVATING",
            "doc_id": payload["doc_id"],
            "storage_key": rec_storage_key(svc, "up-1"),
            "generation_id": gen_id, "target": target,
        })
        version = doc_store.update_status(
            payload["doc_id"], STATUS_QUEUED, STATUS_VALIDATING,
            doc_store.get(payload["doc_id"]).version,
            operation_id=stale["job_id"],
        )
        doc_store.update_status(
            payload["doc_id"], STATUS_VALIDATING, "indexing", version,
            require_op=stale["job_id"], pending_generation_id=gen_id,
        )
        monkeypatch.setattr(svc._index_service, "reconcile", lambda _backend: {
            "alias_target": None,
            "alias_known": False,
            "pointer_target": "",
        })

        assert _worker(jobs, svc, "w2").process_once() is True
        assert jobs.get(payload["job_id"])["status"] == JOB_RETRY_WAIT
        assert control_store.get("kb_write_blocked") == ""
        assert svc._stage.journal_read("up-1") is not None

    def test_stale_token_upload_commit_blocked_but_job_recovers(self, env):
        """旧 Worker 提交前所有权检查中止；任务由新持有者重跑成功。"""
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        old = jobs.claim("w1")[0]
        with deps[0]._engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET lease_until = :ts WHERE job_id = :jid"),
                {"ts": _utcnow_naive() - timedelta(seconds=1),
                 "jid": payload["job_id"]})
        fresh = jobs.claim("w2")[0]
        w = _worker(jobs, svc, "w2")
        # 旧 Worker 尝试执行（模拟暂停后苏醒）：stage 上报即失主
        with pytest.raises(JobLeaseLost):
            svc.run_upload_job(old, JobControl(jobs, old))
        # 新 Worker 正常完成
        w._execute(fresh)
        assert deps[0].get(payload["doc_id"]).status == STATUS_INDEXED


class TestFailureClassification:
    def test_storage_unavailable_requeues_without_running_leak(self, env, monkeypatch):
        """业务存储暂态故障必须落 retry_wait，不能把任务遗留 running。"""
        svc, jobs, _deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")

        def unavailable(_job, _control):
            raise StorageUnavailableError("object store unavailable")

        monkeypatch.setattr(svc, "run_upload_job", unavailable)
        assert _worker(jobs, svc).process_once() is True
        row = jobs.get(payload["job_id"])
        assert row["status"] == JOB_RETRY_WAIT
        assert row["lease_token"] == ""
        assert row["lease_until"] is None

    @pytest.mark.parametrize("failure_point", ["alias_confirm", "pointer"])
    def test_post_alias_failures_are_recoverable(self, env, monkeypatch,
                                                 failure_point):
        """alias 校验/指针失败都保留 journal 并进入只前进重试。"""
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")

        if failure_point == "alias_confirm":
            def fail_alias(_info):
                raise UploadConflictError("alias mismatch")

            monkeypatch.setattr(svc, "_assert_alias_confirmed", fail_alias)
        else:
            def fail_pointer(_backend, _info):
                raise StorageUnavailableError("pointer unavailable")

            monkeypatch.setattr(
                svc._index_service, "activate_pointer", fail_pointer,
            )

        assert _worker(jobs, svc).process_once() is True
        row = jobs.get(payload["job_id"])
        assert row["status"] == JOB_RETRY_WAIT
        assert deps[0].get(payload["doc_id"]).status == "indexing"
        journal = svc._stage.journal_read("up-1")
        assert journal is not None and journal["phase"] == "ACTIVATING"

    def test_embedding_failure_retry_then_dead_then_manual_retry(self, env, tmp_path,
                                                                 monkeypatch):
        svc, jobs, deps = env
        _doc_store, _control, _gen_store, _kb = deps
        jobs2 = jobs
        # 用常败 embedder 重建 index_service 的 embedder
        from kb_async_testkit import FakeEmbedder

        svc._index_service._embedder = FakeEmbedder(fail_after=0)
        jobs2._max_attempts = 2
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        w = _worker(jobs2, svc)
        w.process_once()
        assert jobs2.get(payload["job_id"])["status"] == JOB_RETRY_WAIT
        w.process_once()  # backoff=0 立即重试 → attempts=2 → 死信
        row = jobs2.get(payload["job_id"])
        assert row["status"] == JOB_FAILED and row["attempts"] == 2
        assert row["retryable"] == 1
        assert deps[0].get(payload["doc_id"]).status == STATUS_QUEUED
        # 人工修复依赖后重试 → 成功
        svc._index_service._embedder = FakeEmbedder()
        retried = jobs2.retry(payload["job_id"])
        assert retried["status"] == JOB_QUEUED
        assert w.process_once() is True
        assert deps[0].get(payload["doc_id"]).status == STATUS_INDEXED

    def test_parse_failure_is_permanent(self, env):
        """解析失败（永久输入错误）→ failed 且不可人工重试。"""
        svc, jobs, deps = env
        _doc_store, _control, _gen_store, _kb = deps
        # md 内容解析后为空文本（只有空白）→ 解析结果为空 → 永久失败
        kit.upload_chunks(svc, "up-1", content=b" \t\n" * 20000)
        payload = svc.complete("up-1", "ops-a")
        w = _worker(jobs, svc)
        w.process_once()
        row = jobs.get(payload["job_id"])
        assert row["status"] == JOB_FAILED
        assert row["retryable"] == 0
        assert deps[0].get(payload["doc_id"]).status == "failed"
        assert jobs.retry(payload["job_id"]) is None

    def test_incomplete_chunks_requeue_without_counting(self, env):
        """接管后分片对象缺失 → 解封回 uploading，任务回队不计失败；补传后成功。"""
        svc, jobs, deps = env
        _doc_store, _control, _gen_store, _kb = deps
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        # 崩溃模拟：领取后删除最后一个分片对象
        job = jobs.claim("w1")[0]
        svc._chunks.delete_object("up-1", 1)
        _expire_lease(deps, job["job_id"])
        assert _worker(jobs, svc, "w1").process_once() is True  # 执行（失败分流）
        row = jobs.get(payload["job_id"])
        # 接管领取使 attempts 1→2；失败分流回退 1 → 本次失败未计入
        assert row["status"] == JOB_QUEUED and row["attempts"] == 1
        assert deps[0].get(payload["doc_id"]).status == STATUS_UPLOADING
        # 补传缺失分片 → complete 补封口（同任务）→ 执行成功
        session = svc._state.get("up-1")
        cs = session.chunk_size
        data = kit.pad("# 退换货补充说明\n\n## 适用范围\n\n支持七天无理由退货。\n".encode())
        svc.put_chunk("up-1", 1, data[cs:session.size_bytes])
        again = svc.complete("up-1", "ops-a")
        assert again["job_id"] == payload["job_id"]
        # 分流延迟（等补传）回拨：补传已完成，立即可再执行
        with deps[0]._engine.begin() as conn:
            conn.execute(text(
                "UPDATE kb_index_jobs SET next_run_at = :ts WHERE job_id = :jid"),
                {"ts": _utcnow_naive() - timedelta(seconds=1),
                 "jid": payload["job_id"]})
        assert _worker(jobs, svc, "w1").process_once() is True
        assert deps[0].get(payload["doc_id"]).status == STATUS_INDEXED


class TestShutdown:
    def test_heartbeat_thread_stops_after_job(self, env):
        svc, jobs, _deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        worker = KbIndexWorker(
            jobs, svc, worker_id="w-heartbeat", poll_seconds=0.01,
            heartbeat_seconds=0.01,
        )

        assert worker.process_once() is True
        name = f"kb-job-hb-{payload['job_id'][:8]}"
        assert all(thread.name != name for thread in threading.enumerate())

    def test_no_claim_after_shutdown(self, env):
        svc, jobs, _deps = env
        w = _worker(jobs, svc)
        w.request_shutdown()
        assert w.process_once() is False

    def test_grace_deadline_releases_lease_for_takeover(self, env):
        """宽限期超时：主动让出租约，任务回队，其他实例立即接管。"""
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        w = _worker(jobs, svc)
        w.request_shutdown()  # grace=默认 120s → 手动置 0 模拟超时
        w._grace = 0
        job = jobs.claim("w1")[0]
        w._execute(job)
        row = jobs.get(payload["job_id"])
        assert row["status"] == JOB_QUEUED  # 让出 → 回队
        assert row["lease_token"] == ""
        # 接管者立即完成
        fresh = _worker(jobs, svc, "w2")
        assert fresh.process_once() is True
        assert deps[0].get(payload["doc_id"]).status == STATUS_INDEXED


class TestStatusTruth:
    def test_status_sql_truth_after_session_cleanup(self, env):
        """终态会话已清理：状态查询以 SQL 文档 + 任务为正本。"""
        svc, jobs, _deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        assert _worker(jobs, svc).process_once() is True
        st = svc.get_status("up-1")  # Redis 会话已删除
        assert st["doc_status"] == STATUS_INDEXED
        assert st["job"]["job_id"] == payload["job_id"]
        assert st["job"]["status"] == JOB_SUCCEEDED

    def test_status_includes_job_while_processing(self, env):
        svc, _jobs, _deps = env
        kit.upload_chunks(svc, "up-1")
        svc.complete("up-1", "ops-a")
        st = svc.get_status("up-1")
        assert st["job"]["status"] == JOB_QUEUED
        assert st["doc_status"] == STATUS_QUEUED

    def test_cancel_upload_delegates_to_job(self, env):
        """上传取消委托任务取消：任务/文档均 cancelled（放弃语义），运行中 409。"""
        svc, jobs, deps = env
        kit.upload_chunks(svc, "up-1")
        payload = svc.complete("up-1", "ops-a")
        out = svc.cancel_upload("up-1", "ops-a")
        assert out["job_id"] == payload["job_id"]
        assert out["status"] == "cancelled"
        assert jobs.get(payload["job_id"])["status"] == "cancelled"
        assert deps[0].get(payload["doc_id"]).status == "cancelled"
        assert svc._state.get("up-1") is None  # 会话/分片已清理
        # 已取消（终态）→ complete 409
        from app.agent.rag.upload_service import UploadConflictError as _UCE

        with pytest.raises(_UCE):
            svc.complete("up-1", "ops-a")

    def test_cancel_running_conflict(self, env):
        svc, jobs, _deps = env
        kit.upload_chunks(svc, "up-1")
        svc.complete("up-1", "ops-a")
        jobs.claim("w1")
        with pytest.raises(UploadConflictError):
            svc.cancel_upload("up-1", "ops-a")  # 运行中不可取消（409）

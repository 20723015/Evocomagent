"""KB Index Worker：kb_index_jobs 的独立执行进程（多实例异步建库改造）。

模型（每实例并发 1；多实例并行解析，全局单写者由 kb_write 锁保证）：
- 领取：claim(1)（SKIP LOCKED + lease token），单任务串行；
- 执行：run_upload_job / run_delete_job（复用 DocumentUploadService 的
  验证区 + 锁内提交序列），JobControl 注入阶段/进度上报与「副作用前
  写锁 + lease_token 所有权双重检查」——失去所有权的旧 Worker 不得提交；
- 心跳：独立线程每 heartbeat_seconds 续租（慢请求与等待写锁期间持续）；
  续租失败 → lost，下一次所有权检查立即中止；
- 错误分流（见 job_store.fail）：锁等待/分片未就绪 → 回队不计失败；
  输入错误 → 永久失败；提交点后 → 只前进持续重试；blocked → 人工 reconcile；
- 优雅停机：SIGTERM → 停止领取；当前任务在宽限期内收尾，超时主动让出
  租约（接管立即发生）；期间心跳照常，保证宽限期内任务不被误判接管。

用法（Worker Deployment 专用；Web Pod 不启动本进程）：
    KB_WORKER_ENABLED=1 python -m app.agent.rag.kb_worker
"""

from __future__ import annotations

import signal
import socket
import threading
import time

from app.agent.rag.job_store import JobLeaseLost
from app.config.settings import settings
from app.observability.logging import get_logger
from app.stores.base import StorageUnavailableError

log = get_logger("app.agent.rag.kb_worker")

# embedding 进度映射（chunking 结束 20% → writing_index 70%）
_EMBED_PROGRESS_START = 20
_EMBED_PROGRESS_END = 70
_EMBED_BATCH = 64


def embedding_progress_proxy(inner, control, *, batch_size: int = _EMBED_BATCH,
                             start: int = _EMBED_PROGRESS_START,
                             end: int = _EMBED_PROGRESS_END):
    """embedder 代理：encode 按批切分并上报 embedding 阶段进度（每批续租）。"""
    if control is None:
        return inner

    class _ProgressEmbedder:
        model = getattr(inner, "model", "")

        def _encode_batch(self, texts, timeout):
            if timeout is not None:
                return inner.encode(texts, timeout=timeout)
            return inner.encode(texts)

        def encode(self, texts, timeout=None):
            texts = list(texts)
            if not texts:
                return []
            control.stage("embedding", progress=start)
            vectors: list = []
            total = len(texts)
            for i in range(0, total, max(batch_size, 1)):
                batch = texts[i:i + max(batch_size, 1)]
                vectors.extend(self._encode_batch(batch, timeout))
                done = min(i + len(batch), total)
                progress = start + int((end - start) * done / total)
                control.stage("embedding", progress=progress)  # 每批续租
            control.stage("writing_index", progress=end)
            return vectors

        def encode_one(self, text, timeout=None):
            if timeout is not None:
                return inner.encode_one(text, timeout=timeout)
            return inner.encode_one(text)

        def __getattr__(self, name):
            return getattr(inner, name)

    return _ProgressEmbedder()


class JobControl:
    """异步执行控制柄：副作用前所有权检查 + 阶段/进度上报（顺带续租）。

    check_alive 同时检查三件事（任一失败 → JobLeaseLost，立即停止提交）：
    1. 心跳线程标记的「续租已失败」；
    2. 停机宽限期是否已过（超时主动让出租约等待接管）；
    3. store 侧 lease_token 所有权（任务已被接管/取消）。
    """

    def __init__(self, store, job: dict, *,
                 lost: threading.Event | None = None,
                 deadline_fn=None):
        self._store = store
        self._job_id = job["job_id"]
        self._token = job.get("lease_token", "")
        self._operation = job.get("operation", "")
        self._lost = lost if lost is not None else threading.Event()
        self._deadline_fn = deadline_fn  # () -> float | None：停机宽限截止（monotonic）
        self._stage = ""
        self._stage_started = time.monotonic()

    @property
    def lost(self) -> threading.Event:
        return self._lost

    def check_alive(self) -> None:
        if self._lost.is_set():
            from app.observability.metrics import record_kb_job_lease_lost

            record_kb_job_lease_lost("heartbeat")
            raise JobLeaseLost("租约续期失败（heartbeat），停止提交")
        deadline = self._deadline_fn() if self._deadline_fn is not None else None
        if deadline is not None and time.monotonic() >= deadline:
            # 优雅停机超时：主动让出（清租约）→ 其他实例立即接管
            self._store.release_lease(self._job_id, self._token)
            raise JobLeaseLost("停机宽限期已过，任务让出等待接管")
        if not self._store.check_owner(self._job_id, self._token):
            from app.observability.metrics import record_kb_job_lease_lost

            record_kb_job_lease_lost("ownership")
            raise JobLeaseLost("任务所有权已丢失（lease_token 不匹配），停止提交")

    def stage(self, stage: str, progress: int | None = None) -> None:
        # 阶段上报本身会续租并写任务状态；在写入前同时检查停机宽限和
        # store 侧所有权，避免 stale worker 仅靠 set_stage 继续执行。
        self.check_alive()
        from app.observability.metrics import record_kb_job_stage

        prev, started = self._stage, self._stage_started
        self._store.set_stage(self._job_id, self._token, stage, progress)
        if prev and prev != stage:
            record_kb_job_stage(prev, time.monotonic() - started)
        self._stage = stage
        self._stage_started = time.monotonic()


class KbIndexWorker:
    """kb_index_jobs 后台执行器：领取 → 心跳续租 → 执行 → 终态分流。"""

    def __init__(self, job_store, upload_service, *, worker_id: str = "",
                 poll_seconds: float | None = None,
                 heartbeat_seconds: float | None = None,
                 grace_seconds: float | None = None):
        self._jobs = job_store
        self._svc = upload_service
        self._worker_id = worker_id or (
            f"kbworker-{socket.gethostname()}-{threading.get_ident() % 100000}"
        )
        self._poll = float(poll_seconds if poll_seconds is not None
                           else settings.kb_job_poll_seconds)
        self._heartbeat = float(heartbeat_seconds if heartbeat_seconds is not None
                                else settings.kb_job_heartbeat_seconds)
        self._grace = float(grace_seconds if grace_seconds is not None
                            else settings.kb_worker_grace_seconds)
        self._stop = threading.Event()
        self._shutdown_at: float | None = None

    # ---------- 生命周期 ----------
    def request_shutdown(self) -> None:
        """SIGTERM：停止领取新任务；当前任务在宽限期内收尾。"""
        if not self._stop.is_set():
            self._stop.set()
            self._shutdown_at = time.monotonic()
            log.info("kb_worker shutdown_requested grace=%.0fs", self._grace)

    def run_forever(self) -> None:
        log.info("kb_worker started worker=%s poll=%.1fs heartbeat=%.0fs",
                 self._worker_id, self._poll, self._heartbeat)
        last_cleanup = 0.0
        while not self._stop.is_set():
            try:
                processed = self.process_once()
            except Exception as e:  # noqa: BLE001
                log.warning("kb_worker loop error: %s", type(e).__name__)
                processed = False
            # 保留期清理（succeeded/cancelled 90 天；failed/blocked 180 天）：
            # 跟随 GC 间隔低频执行；删除幂等，多实例并发安全
            if time.monotonic() - last_cleanup >= settings.kb_gc_interval_seconds:
                last_cleanup = time.monotonic()
                try:
                    removed = self._jobs.cleanup()
                    if removed:
                        log.info("kb_job_cleanup removed=%s", removed)
                except Exception as e:  # noqa: BLE001
                    log.warning("kb_job_cleanup failed: %s", type(e).__name__)
            if not processed and not self._stop.is_set():
                self._stop.wait(self._poll)
        log.info("kb_worker stopped")

    # ---------- 领取与执行 ----------
    def process_once(self) -> bool:
        """领取并执行一个任务；返回是否领取到（供循环/测试断言）。

        已请求停机 → 不再领取新任务（当前任务由 run_forever 中的执行收尾）。
        """
        if self._stop.is_set():
            return False
        claimed = self._jobs.claim(self._worker_id, limit=1)
        if not claimed:
            self._publish_stats()
            return False
        self._execute(claimed[0])
        self._publish_stats()
        return True

    def _publish_stats(self) -> None:
        """积压/最老任务年龄/blocked 指标 + 告警日志。"""
        try:
            from app.observability.metrics import set_kb_job_stats

            stats = self._jobs.stats()
            set_kb_job_stats(stats)
            oldest = stats.get("oldest_age_seconds") or {}
            threshold = settings.kb_job_backlog_alert_seconds
            for op, age in oldest.items():
                if age >= threshold:
                    log.warning(
                        "kb_job_backlog_alert operation=%s oldest_age=%.0fs "
                        "backlog=%s", op, age, stats.get("backlog"),
                    )
            if stats.get("blocked"):
                log.warning("kb_job_blocked_alert blocked=%s（等待人工 reconcile）",
                            stats["blocked"])
        except Exception:  # noqa: BLE001, S110
            pass

    def _execute(self, job: dict) -> None:
        lost = threading.Event()
        heartbeat_stop = threading.Event()
        # 停机宽限截止动态计算：SIGTERM 可能在任务执行中途到达
        def deadline_fn():
            return (
                None if self._shutdown_at is None
                else self._shutdown_at + self._grace
            )

        control = JobControl(self._jobs, job, lost=lost, deadline_fn=deadline_fn)
        hb = self._start_heartbeat(job, lost, heartbeat_stop)
        job_id = job["job_id"]
        op = job.get("operation", "")
        try:
            if op == "delete":
                self._svc.run_delete_job(job, control)
            else:
                self._svc.run_upload_job(job, control)
            # 业务编排可能在最后一次阶段上报后仍执行了很久；提交终态前
            # 再做一次 fencing，避免 heartbeat 已丢失时误把任务标成成功。
            control.check_alive()
            self._jobs.succeed(job_id, job.get("lease_token", ""))
            log.info("kb_job_succeeded job=%s operation=%s attempts=%s",
                     job_id, op, job.get("attempts"))
        except JobLeaseLost as e:
            # 所有权已丢失：任务行归新持有者，本 Worker 不得再改写
            log.warning("kb_job_lease_lost job=%s err=%s", job_id, e)
        except StorageUnavailableError as e:
            # _fail 的默认分支已将存储故障归为可重试；不要传入不存在的
            # retryable 关键字，否则这里会二次抛 TypeError，任务卡在 running。
            self._fail(job, e)
        except Exception as e:  # noqa: BLE001
            self._fail(job, e)
        finally:
            self._stop_heartbeat(hb, heartbeat_stop)

    # ---------- 心跳 ----------
    def _start_heartbeat(self, job: dict, lost: threading.Event,
                         stop_event: threading.Event | None = None):
        if self._heartbeat <= 0:
            return None
        stop_event = stop_event or threading.Event()

        def _loop():
            while (
                not lost.is_set()
                and not stop_event.is_set()
                and not stop_event.wait(self._heartbeat)
            ):
                try:
                    ok = self._jobs.heartbeat(job["job_id"], job.get("lease_token", ""))
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    lost.set()
                    return

        t = threading.Thread(target=_loop, name=f"kb-job-hb-{job['job_id'][:8]}",
                             daemon=True)
        # 保留在 thread 上供旧测试/调用方只传 thread 的停止路径使用；
        # 正式执行路径显式持有独立 stop event，绝不复用 lost event。
        t._kb_stop_event = stop_event
        t.start()
        return t

    @staticmethod
    def _stop_heartbeat(thread, stop_event: threading.Event | None = None) -> None:
        if stop_event is None and thread is not None:
            stop_event = getattr(thread, "_kb_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=2)

    # ---------- 失败分流 ----------
    def _fail(self, job: dict, exc: Exception) -> None:
        """异常 → 任务终态/重试语义（见 job_store.fail 的分流表）。"""
        from app.agent.rag.upload_service import (
            UploadBlocked,
            UploadIncomplete,
            UploadLockWait,
            UploadRecovering,
        )

        job_id = job["job_id"]
        log.warning("kb_job_failed job=%s operation=%s err=%s",
                    job_id, job.get("operation"), type(exc).__name__)
        try:
            if isinstance(exc, UploadLockWait):
                # 锁等待不计失败次数：回 queued 短延迟重试
                from app.observability.metrics import record_kb_job_lock_wait

                record_kb_job_lock_wait(job.get("operation", ""))
                self._jobs.fail(job, exc, lock_wait=True)
            elif isinstance(exc, UploadIncomplete):
                # 分片未就绪（接管后发现对象缺失等）：等补传，不计失败
                self._jobs.fail(job, exc, lock_wait=True, delay_seconds=30)
            elif isinstance(exc, UploadBlocked):
                self._jobs.mark_blocked(
                    job_id, job.get("lease_token", ""), str(exc),
                )
                log.error(
                    "kb_job_blocked job=%s（alias 指向未知代，已维持全局写阻塞，"
                    "等待人工 reconcile）", job_id,
                )
            elif isinstance(exc, UploadRecovering):
                # 已越过提交点的可确定故障：持续重试（不进入普通死信）
                self._jobs.fail(job, exc, post_commit=True)
            else:
                from app.agent.rag.upload_service import UploadError

                if isinstance(exc, UploadError):
                    # 永久输入错误（解析/消毒/超限/会话取消）：不可自动重试
                    self._jobs.fail(job, exc, retryable=False)
                else:
                    self._jobs.fail(job, exc, retryable=True)
        except Exception as e:  # noqa: BLE001
            log.error("kb_job_fail_record_failed job=%s err=%s", job_id, e)


# ============================================================
# 进程装配（独立 Worker Deployment；Web Pod 不启动）
# ============================================================
def build_kb_worker(worker_id: str = "") -> KbIndexWorker | None:
    """复用 Web 侧装配（MySQL/ES/S3/RWX/模型凭据一致），构造 Worker。"""
    from app.server.deps import _build_upload_service
    from app.stores.redis_client import get_redis
    from app.stores.sql.engine import get_engine

    engine = get_engine()
    if engine is None:
        log.error("kb_worker 需要 MySQL（DB 未配置），拒绝启动")
        return None
    redis = get_redis()
    object_store = None
    if settings.s3_bucket and settings.s3_endpoint_url:
        from app.stores.object_store import ObjectStoreUnavailable, S3ObjectStore

        try:
            object_store = S3ObjectStore(
                settings.s3_bucket,
                endpoint_url=settings.s3_endpoint_url,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
            )
        except ObjectStoreUnavailable as e:
            log.warning("S3 初始化失败（%s），分片/原件降级本地", e)
    from app.agent.rag.job_store import KbIndexJobStore

    job_store = KbIndexJobStore(engine)
    svc = _build_upload_service(engine, redis, object_store, job_store=job_store)
    if svc is None:
        log.error("kb_worker 装配 upload_service 失败，拒绝启动")
        return None
    return KbIndexWorker(job_store, svc, worker_id=worker_id)


def main() -> int:
    if not settings.kb_worker_enabled:
        log.error("KB_WORKER_ENABLED 未开启：本进程不应在 Web Pod 启动")
        return 2
    worker = build_kb_worker()
    if worker is None:
        return 1

    def _sig(_sig_num, _frame):
        worker.request_shutdown()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig_obj = getattr(signal, sig_name, None)
        if sig_obj is not None:
            try:
                signal.signal(sig_obj, _sig)
            except (ValueError, OSError):
                pass
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

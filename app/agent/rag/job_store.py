"""kb_index_jobs 任务队列（多实例异步知识库建库改造）。

模型（复用 memory_jobs 的 SKIP LOCKED + lease，增强提交安全）：
- 入队：INSERT kb_index_jobs 与文档状态 CAS（uploading→queued /
  indexed→delete_queued）在同一事务完成——「队列行」与「文档状态」互为正本，
  SQL 失败整体回滚，重复 complete 可补入队；唯一键 (operation, doc_id)
  保证重复请求返回同一任务。
- 领取：UPDATE … WHERE queued/退避到期/租约过期，FOR UPDATE SKIP LOCKED；
  单次领取 1 个任务，领取即生成新 lease_token——旧持有者立即失去所有权。
- 租约：MySQL 服务端时间计算（NOW() + INTERVAL）；heartbeat 续租；
  提交前所有权检查（lease_token 匹配）——失去所有权的旧 Worker 不得继续提交。
- 重试：attempts 在领取时 +1（=执行次数）；失败时按错误类别分流：
  锁等待 → 回 queued 且 attempts 回退（不计失败）；提交点后可确定故障 →
  retry_wait 持续重试（不进入死信）；暂态故障 → 退避重试，超上限 failed；
  永久输入错误 → failed 且 retryable=0。
- 时间口径：MySQL 用服务端时间（NOW()），sqlite（测试）用进程时间——
  同一方言内读写一致，跨实例比较无时钟漂移。
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, delete, func, or_, select, text, update

from app.config.settings import settings
from app.observability.logging import get_logger
from app.stores.base import StorageUnavailableError
from app.stores.sql.document_store import (
    STATUS_CANCELLED,
    STATUS_DELETE_QUEUED,
    STATUS_INDEXED,
    STATUS_QUEUED,
    STATUS_UPLOADING,
)
from app.stores.sql.schema import kb_documents, kb_index_jobs

log = get_logger("app.agent.rag.job_store")

# ---- 任务状态 ----
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_RETRY_WAIT = "retry_wait"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
JOB_BLOCKED = "blocked"

JOB_ACTIVE = (JOB_QUEUED, JOB_RUNNING, JOB_RETRY_WAIT, JOB_BLOCKED)
JOB_TERMINAL = (JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED)

OP_UPLOAD = "upload"
OP_DELETE = "delete"

_MAX_ERROR_CHARS = 300


def _utcnow_naive() -> datetime:
    """SQL DateTime 列使用无时区 UTC，避免本机时区影响测试/退避。"""
    return datetime.now(UTC).replace(tzinfo=None)


def _sanitize_error(exc: Exception) -> str:
    """脱敏错误摘要：只留异常类型与首行，不落用户内容/SQL 细节。"""
    return f"{type(exc).__name__}: {exc}".splitlines()[0][:_MAX_ERROR_CHARS]


class JobLeaseLost(RuntimeError):
    """任务所有权已丢失（lease_token 不匹配/已被接管/取消）——必须立即停止提交。"""


class JobConflictError(ValueError):
    """任务与文档状态冲突（映射 409）。"""


class KbIndexJobStore:
    """SQLAlchemy 实现（生产 mysql+pymysql；本地/测试 sqlite 同表）。"""

    def __init__(self, engine, *, lease_seconds: int | None = None,
                 max_attempts: int | None = None, lock_wait_seconds: int | None = None,
                 backoff_base: int | None = None, backoff_max: int | None = None):
        self._engine = engine
        self._lease = int(lease_seconds if lease_seconds is not None
                          else settings.kb_job_lease_seconds)
        self._max_attempts = int(max_attempts if max_attempts is not None
                                 else settings.kb_job_max_attempts)
        self._lock_wait = int(lock_wait_seconds if lock_wait_seconds is not None
                              else settings.kb_job_lock_wait_seconds)
        self._backoff_base = int(backoff_base if backoff_base is not None
                                 else settings.kb_job_backoff_base_seconds)
        self._backoff_max = int(backoff_max if backoff_max is not None
                                else settings.kb_job_backoff_max_seconds)
        self._server_time = engine.dialect.name == "mysql"

    # ---------- 时间口径（MySQL 服务端时间；sqlite 进程时间） ----------
    def _now(self):
        return func.now() if self._server_time else _utcnow_naive()

    def _now_plus(self, seconds: int):
        if self._server_time:
            return text(f"NOW() + INTERVAL {int(max(seconds, 0))} SECOND")
        return _utcnow_naive() + timedelta(seconds=max(seconds, 0))

    def _claimable(self):
        """领取候选：排队中（到点）或退避到期；running 且租约过期 = 接管崩溃任务。

        时间比较与本模块写入同一时钟口径（MySQL 服务端时间 / sqlite 进程时间）。
        """
        now = self._now()
        return or_(
            and_(
                kb_index_jobs.c.status.in_((JOB_QUEUED,)),
                or_(
                    kb_index_jobs.c.next_run_at.is_(None),
                    kb_index_jobs.c.next_run_at <= now,
                ),
            ),
            and_(
                kb_index_jobs.c.status == JOB_RETRY_WAIT,
                kb_index_jobs.c.next_run_at.isnot(None),
                kb_index_jobs.c.next_run_at <= now,
            ),
            and_(
                kb_index_jobs.c.status == JOB_RUNNING,
                kb_index_jobs.c.lease_until.isnot(None),
                kb_index_jobs.c.lease_until <= now,
            ),
        )

    def _backoff_seconds(self, attempts: int) -> int:
        """指数退避 + 随机抖动，上限 backoff_max。"""
        base = min(self._backoff_max, self._backoff_base * (2 ** max(attempts, 1)))
        return int(min(self._backoff_max, base * random.uniform(0.7, 1.3)))

    def _clamp_at_zero(self, expr):
        """attempts 回退表达式：MySQL greatest(x,0) / sqlite max(x,0)（标量）。"""
        if self._server_time:
            return func.greatest(expr, 0)
        return func.max(expr, 0)

    def _owner_where(self, job_id: str, lease_token: str):
        """所有者写操作的 fencing 条件。

        token 只保证「最近一次领取者」身份；租约过期后，在接管者尚未
        领取的窗口内 token 仍可能相同，因此每个写操作还必须确认租约
        尚未过期。时间比较使用与领取/续租相同的时钟口径。
        """
        return (
            kb_index_jobs.c.job_id == job_id,
            kb_index_jobs.c.lease_token == lease_token,
            kb_index_jobs.c.status == JOB_RUNNING,
            kb_index_jobs.c.lease_until.isnot(None),
            kb_index_jobs.c.lease_until > self._now(),
        )

    # ---------- 入队（与文档状态 CAS 同事务） ----------
    def _enqueue(self, operation: str, doc_id: str, upload_id: str,
                 requested_by: str, src_status: str, queued_status: str,
                 expected_doc_version: int) -> tuple[dict, bool]:
        """同一事务：文档 CAS → queued + INSERT 任务行。返回 (job, created)。

        任务已存在 → 幂等返回现有任务（created=False，不动文档）；
        文档状态/版本不匹配 → JobConflictError（调用方 409）。
        """
        try:
            with self._engine.begin() as conn:
                row = conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.operation == operation,
                           kb_index_jobs.c.doc_id == doc_id)
                ).mappings().first()
                if row is not None and row["status"] != JOB_CANCELLED:
                    return dict(row), False
                if row is not None:
                    # 已取消 → 换新任务重入队（保唯一键：同事务删旧插新；
                    # 文档状态已被 cancel 恢复到 src，CAS 照常执行）
                    conn.execute(
                        delete(kb_index_jobs)
                        .where(kb_index_jobs.c.job_id == row["job_id"])
                    )
                cas = conn.execute(
                    update(kb_documents)
                    .where(
                        kb_documents.c.doc_id == doc_id,
                        kb_documents.c.status == src_status,
                        kb_documents.c.version == expected_doc_version,
                    )
                    .values(
                        status=queued_status,
                        version=expected_doc_version + 1,
                        operation_id="",
                        error="",
                        status_changed_at=self._now(),
                    )
                )
                if cas.rowcount != 1:
                    raise JobConflictError(
                        f"文档 {doc_id} 状态已变化，无法入队（期望 {src_status})"
                    )
                job_id = uuid.uuid4().hex
                conn.execute(kb_index_jobs.insert().values(
                    job_id=job_id, operation=operation, doc_id=doc_id,
                    upload_id=upload_id, requested_by=requested_by,
                    status=JOB_QUEUED,
                ))
                created = conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                ).mappings().first()
                return dict(created), True
        except JobConflictError:
            # 两个实例可能都在初始 SELECT 中看不到任务；赢家提交文档
            # CAS+INSERT 后，输家会在自己的 CAS 上得到 rowcount=0。事务
            # 已回滚，必须在新连接重读赢家任务，保持 complete/delete 幂等。
            existing = self.find(operation, doc_id=doc_id)
            if existing is not None and existing["status"] != JOB_CANCELLED:
                return existing, False
            raise
        except Exception as e:
            existing = self.find(operation, doc_id=doc_id)
            if existing is not None:
                return existing, False
            raise StorageUnavailableError(f"KB 任务入队失败: {e}") from e

    def enqueue_upload(self, doc_id: str, upload_id: str, requested_by: str,
                       expected_doc_version: int) -> tuple[dict, bool]:
        return self._enqueue(OP_UPLOAD, doc_id, upload_id, requested_by,
                             STATUS_UPLOADING, STATUS_QUEUED, expected_doc_version)

    def enqueue_delete(self, doc_id: str, requested_by: str,
                       expected_doc_version: int) -> tuple[dict, bool]:
        return self._enqueue(OP_DELETE, doc_id, "", requested_by,
                             STATUS_INDEXED, STATUS_DELETE_QUEUED, expected_doc_version)

    # ---------- 读 ----------
    def get(self, job_id: str) -> dict | None:
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(kb_index_jobs).where(kb_index_jobs.c.job_id == job_id)
                ).mappings().first()
        except Exception as e:
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e
        return dict(row) if row is not None else None

    def find(self, operation: str, *, doc_id: str = "",
             upload_id: str = "") -> dict | None:
        """按唯一键 (operation, doc_id) 或 upload_id 查任务（幂等锚点）。"""
        try:
            with self._engine.connect() as conn:
                if doc_id:
                    cond = (kb_index_jobs.c.operation == operation,
                            kb_index_jobs.c.doc_id == doc_id)
                elif upload_id:
                    cond = (kb_index_jobs.c.operation == operation,
                            kb_index_jobs.c.upload_id == upload_id)
                else:
                    return None
                row = conn.execute(
                    select(kb_index_jobs).where(*cond)
                ).mappings().first()
        except Exception as e:
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e
        return dict(row) if row is not None else None

    def latest_for_doc(self, doc_id: str) -> dict | None:
        """文档当前/最近一次任务（上传状态查询的 SQL 正本）。"""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.doc_id == doc_id)
                    .order_by(kb_index_jobs.c.created_at.desc(),
                              kb_index_jobs.c.job_id.desc())
                    .limit(1)
                ).mappings().first()
        except Exception as e:
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e
        return dict(row) if row is not None else None

    def list(self, status: str = "", operation: str = "",
             limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """分页查询（按状态/操作类型过滤；created_at 倒序）。"""
        limit = min(max(int(limit), 1), 200)
        offset = max(int(offset), 0)
        conds = []
        if status:
            conds.append(kb_index_jobs.c.status == status)
        if operation:
            conds.append(kb_index_jobs.c.operation == operation)
        try:
            with self._engine.connect() as conn:
                total = int(conn.execute(
                    select(func.count()).select_from(kb_index_jobs).where(*conds)
                ).scalar() or 0)
                rows = conn.execute(
                    select(kb_index_jobs).where(*conds)
                    .order_by(kb_index_jobs.c.created_at.desc(),
                              kb_index_jobs.c.job_id.desc())
                    .limit(limit).offset(offset)
                ).mappings().all()
        except Exception as e:
            raise StorageUnavailableError(f"SQL 读取失败: {e}") from e
        return [dict(r) for r in rows], total

    # ---------- 领取 / 租约 ----------
    def claim(self, worker_id: str, limit: int = 1) -> list[dict]:
        """原子领取（queued/退避到期/租约过期接管）；领取即生成新 lease_token。"""
        claimed: list[dict] = []
        try:
            with self._engine.begin() as conn:
                rows = conn.execute(
                    select(kb_index_jobs)
                    .where(self._claimable())
                    .order_by(kb_index_jobs.c.created_at, kb_index_jobs.c.job_id)
                    .limit(max(int(limit), 1))
                    .with_for_update(skip_locked=True)
                ).mappings().all()
                for row in rows:
                    token = uuid.uuid4().hex
                    conn.execute(
                        update(kb_index_jobs)
                        .where(kb_index_jobs.c.job_id == row["job_id"])
                        .values(
                            status=JOB_RUNNING,
                            lease_owner=worker_id,
                            lease_token=token,
                            lease_until=self._now_plus(self._lease),
                            attempts=row["attempts"] + 1,
                            started_at=func.coalesce(
                                kb_index_jobs.c.started_at, self._now(),
                            ),
                            updated_at=self._now(),
                        )
                    )
                    job = dict(row)
                    job["status"] = JOB_RUNNING
                    job["lease_owner"] = worker_id
                    job["lease_token"] = token
                    job["attempts"] = int(row["attempts"] or 0) + 1
                    if row["status"] == JOB_RUNNING:  # 租约过期接管（原持有者崩溃）
                        from app.observability.metrics import record_kb_job_takeover

                        record_kb_job_takeover(job["operation"])
                    claimed.append(job)
            return claimed
        except Exception as e:  # noqa: BLE001
            log.warning("kb_job.claim_failed err=%s", type(e).__name__)
            return []

    def heartbeat(self, job_id: str, lease_token: str) -> bool:
        """续租（独立线程调用）；False = 所有权已丢失。"""
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(kb_index_jobs)
                    .where(*self._owner_where(job_id, lease_token))
                    .values(lease_until=self._now_plus(self._lease),
                            updated_at=self._now())
                )
        except Exception:  # noqa: BLE001
            return False
        return updated.rowcount == 1

    def check_owner(self, job_id: str, lease_token: str) -> bool:
        """提交前所有权检查（轻量只读）：token 匹配且仍 running。"""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(kb_index_jobs.c.job_id)
                    .where(*self._owner_where(job_id, lease_token))
                ).first()
        except Exception:  # noqa: BLE001
            return False
        return row is not None

    def set_stage(self, job_id: str, lease_token: str, stage: str,
                  progress: int | None = None) -> None:
        """阶段/进度更新（每次都顺带续租到完整租约期）；失主 → JobLeaseLost。"""
        values: dict = {"stage": stage, "updated_at": self._now(),
                        "lease_until": self._now_plus(self._lease)}
        if progress is not None:
            values["progress"] = max(0, min(int(progress), 100))
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(kb_index_jobs)
                    .where(*self._owner_where(job_id, lease_token))
                    .values(**values)
                )
        except Exception as e:
            raise JobLeaseLost(f"阶段更新失败（视为失主）: {e}") from e
        if updated.rowcount != 1:
            raise JobLeaseLost(f"任务所有权已丢失: {job_id}")

    def release_lease(self, job_id: str, lease_token: str) -> None:
        """主动让出（优雅停机超时）：清租约使接管立即发生；任务回到排队。"""
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    update(kb_index_jobs)
                    .where(*self._owner_where(job_id, lease_token))
                    .values(
                        status=JOB_QUEUED, lease_owner="", lease_token="",
                        lease_until=None, updated_at=self._now(),
                        attempts=self._clamp_at_zero(kb_index_jobs.c.attempts - 1),
                    )
                )
        except Exception:  # noqa: BLE001, S110
            pass

    # ---------- 终态 ----------
    def succeed(self, job_id: str, lease_token: str) -> None:
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(kb_index_jobs)
                    .where(*self._owner_where(job_id, lease_token))
                    .values(
                        status=JOB_SUCCEEDED, progress=100, error="",
                        lease_owner="", lease_token="", lease_until=None,
                        finished_at=self._now(), updated_at=self._now(),
                    )
                )
        except Exception as e:
            raise StorageUnavailableError(f"KB 任务完成写库失败: {e}") from e
        if updated.rowcount != 1:
            raise JobLeaseLost(f"任务所有权已丢失: {job_id}")
        self._record_e2e(job_id)

    def fail(self, job: dict, exc: Exception, *, retryable: bool = True,
             lock_wait: bool = False, post_commit: bool = False,
             delay_seconds: int | None = None) -> str:
        """失败分流（WHERE lease_token——失主的旧 Worker 不得改写状态）。

        - lock_wait：锁等待不计失败次数 → 回 queued（attempts 回退）+ 短延迟重试
          （delay_seconds 可覆盖默认等待，如分片未就绪等待补传）；
        - post_commit：已越过提交点的可确定故障 → retry_wait 持续重试（不进死信）；
        - 永久输入错误（retryable=False）→ failed（不可人工重试）；
        - 暂态故障超上限 → failed（可人工重试，死信）。
        返回落库后的任务状态。
        """
        job_id = job["job_id"]
        token = job.get("lease_token", "")
        attempts = int(job.get("attempts", 0) or 0)
        now_expr = self._now()
        values: dict = {"updated_at": now_expr,
                        "error": _sanitize_error(exc),
                        "lease_owner": "", "lease_token": "", "lease_until": None}
        if lock_wait:
            # 锁等待不计失败：attempts 回退 + 短延迟重试（等待全局写锁持有者收尾）
            values.update({
                "status": JOB_QUEUED,
                "attempts": self._clamp_at_zero(kb_index_jobs.c.attempts - 1),
                "next_run_at": self._now_plus(
                    self._lock_wait if delay_seconds is None else int(delay_seconds),
                ),
            })
            new_status = JOB_QUEUED
        elif post_commit:
            # 提交点后只前进：持续重试，绝不进入普通死信
            values.update({
                "status": JOB_RETRY_WAIT, "retryable": 1,
                "next_run_at": self._now_plus(self._backoff_seconds(attempts)),
            })
            new_status = JOB_RETRY_WAIT
        elif not retryable:
            values.update({
                "status": JOB_FAILED, "retryable": 0,
                "finished_at": now_expr,
            })
            new_status = JOB_FAILED
        elif attempts >= self._max_attempts:
            values.update({
                "status": JOB_FAILED, "retryable": 1,
                "finished_at": now_expr,
            })
            new_status = JOB_FAILED
        else:
            values.update({
                "status": JOB_RETRY_WAIT,
                "next_run_at": self._now_plus(self._backoff_seconds(attempts)),
            })
            new_status = JOB_RETRY_WAIT
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(kb_index_jobs)
                    .where(*self._owner_where(job_id, token))
                    .values(**values)
                )
        except Exception as e:
            raise StorageUnavailableError(f"KB 任务失败落库失败: {e}") from e
        if updated.rowcount != 1:
            raise JobLeaseLost(f"任务所有权已丢失: {job_id}")
        self._record_fail_metrics(job, new_status, lock_wait=lock_wait,
                                  post_commit=post_commit)
        return new_status

    def _record_e2e(self, job_id: str) -> None:
        try:
            job = self.get(job_id) or {}
            created = job.get("created_at")
            finished = job.get("finished_at")
            if created is not None and finished is not None:
                from app.observability.metrics import record_kb_job_e2e

                record_kb_job_e2e(
                    job.get("operation", ""),
                    (finished - created).total_seconds(),
                )
        except Exception:  # noqa: BLE001, S110
            pass

    def _record_fail_metrics(self, job: dict, new_status: str, *,
                             lock_wait: bool, post_commit: bool) -> None:
        try:
            from app.observability.metrics import (
                record_kb_job_dead,
                record_kb_job_retry,
            )

            if new_status == JOB_FAILED:
                record_kb_job_dead(job.get("operation", ""))
            elif not lock_wait and not post_commit:
                record_kb_job_retry(job.get("operation", ""))
        except Exception:  # noqa: BLE001, S110
            pass

    # ---------- 管理（取消 / 重试 / 清理） ----------
    def cancel(self, job_id: str) -> dict | None:
        """取消 queued/retry_wait 任务，并同事务更新文档状态。

        upload→cancelled，delete→indexed；提交点后的 retry_wait 因文档已处
        indexing/deleting 而拒绝取消。running/终态同样返回 None（409 语义）。
        """
        try:
            with self._engine.begin() as conn:
                row = conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                    .with_for_update()
                ).mappings().first()
                if row is None or row["status"] not in (JOB_QUEUED, JOB_RETRY_WAIT):
                    return None

                # retry_wait 既可能是提交点前的暂态失败，也可能是提交点
                # 后的恢复任务。后者的文档已在 indexing/deleting，不能被
                # 取消，否则 journal 永远没有机会前进恢复。锁住文档并在
                # 同一事务内判定，避免任务状态与文档状态分裂。
                doc = conn.execute(
                    select(kb_documents)
                    .where(kb_documents.c.doc_id == row["doc_id"])
                    .with_for_update()
                ).mappings().first()
                expected_doc_status = (
                    STATUS_QUEUED if row["operation"] == OP_UPLOAD
                    else STATUS_DELETE_QUEUED
                )
                if doc is None or doc["status"] != expected_doc_status:
                    return None

                # 上传取消就是放弃本次会话，文档必须在同一事务进入终态
                # cancelled；若先恢复 uploading 再由 service 二次更新，另一
                # 实例可能在事务窗口内重新 complete 并创建新任务。删除取消
                # 则仅撤销下架请求，恢复为 indexed。
                restored_doc_status = (
                    STATUS_CANCELLED if row["operation"] == OP_UPLOAD
                    else STATUS_INDEXED
                )
                doc_updated = conn.execute(
                    update(kb_documents)
                    .where(
                        kb_documents.c.doc_id == row["doc_id"],
                        kb_documents.c.status == expected_doc_status,
                        kb_documents.c.version == doc["version"],
                    )
                    .values(status=restored_doc_status,
                            version=doc["version"] + 1,
                            status_changed_at=self._now())
                )
                if doc_updated.rowcount != 1:
                    return None
                conn.execute(
                    update(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                    .values(
                        status=JOB_CANCELLED, error="已取消",
                        lease_owner="", lease_token="", lease_until=None,
                        next_run_at=None, finished_at=self._now(),
                        updated_at=self._now(),
                    )
                )
                return dict(conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                ).mappings().first())
        except StorageUnavailableError:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"KB 任务取消失败: {e}") from e

    def cancel_upload_atomic(self, job_id: str) -> dict | None:
        """原子取消上传任务：任务与文档同事务进入终态 cancelled。

        该显式接口供上传编排层使用；只允许 upload 任务且文档仍为 queued，
        提交点后的 retry_wait（indexing）会返回 None，确保 journal 恢复链
        不被截断。
        """
        try:
            with self._engine.begin() as conn:
                row = conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                    .with_for_update()
                ).mappings().first()
                if row is None or row["operation"] != OP_UPLOAD:
                    return None
                if row["status"] not in (JOB_QUEUED, JOB_RETRY_WAIT):
                    return None
                doc = conn.execute(
                    select(kb_documents)
                    .where(kb_documents.c.doc_id == row["doc_id"])
                    .with_for_update()
                ).mappings().first()
                if doc is None or doc["status"] != STATUS_QUEUED:
                    return None
                doc_updated = conn.execute(
                    update(kb_documents)
                    .where(
                        kb_documents.c.doc_id == row["doc_id"],
                        kb_documents.c.status == STATUS_QUEUED,
                        kb_documents.c.version == doc["version"],
                    )
                    .values(
                        status=STATUS_CANCELLED,
                        version=doc["version"] + 1,
                        operation_id="",
                        pending_generation_id="",
                        error="已取消",
                        status_changed_at=self._now(),
                    )
                )
                if doc_updated.rowcount != 1:
                    return None
                conn.execute(
                    update(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                    .values(
                        status=JOB_CANCELLED, error="已取消",
                        lease_owner="", lease_token="", lease_until=None,
                        next_run_at=None, finished_at=self._now(),
                        updated_at=self._now(),
                    )
                )
                return dict(conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                ).mappings().first())
        except StorageUnavailableError:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"KB 上传任务原子取消失败: {e}") from e

    def retry(self, job_id: str) -> dict | None:
        """人工重试：仅可重试的 failed/blocked；重置本轮 attempts（审计数只增）。"""
        try:
            with self._engine.begin() as conn:
                row = conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                    .with_for_update()
                ).mappings().first()
                if row is None or row["status"] not in (JOB_FAILED, JOB_BLOCKED):
                    return None
                if not int(row["retryable"] or 0):
                    return None
                conn.execute(
                    update(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                    .values(
                        status=JOB_QUEUED, attempts=0, next_run_at=None,
                        error="", stage="", progress=0,
                        manual_retry_count=row["manual_retry_count"] + 1,
                        lease_owner="", lease_token="", lease_until=None,
                        finished_at=None, updated_at=self._now(),
                    )
                )
                return dict(conn.execute(
                    select(kb_index_jobs)
                    .where(kb_index_jobs.c.job_id == job_id)
                ).mappings().first())
        except Exception as e:
            raise StorageUnavailableError(f"KB 任务重试失败: {e}") from e

    def mark_blocked(self, job_id: str, lease_token: str, reason: str) -> None:
        """alias 指向未知代：任务置 blocked，等待人工 reconcile。"""
        try:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(kb_index_jobs)
                    .where(*self._owner_where(job_id, lease_token))
                    .values(
                        status=JOB_BLOCKED, error=reason[:_MAX_ERROR_CHARS],
                        lease_owner="", lease_token="", lease_until=None,
                        finished_at=self._now(), updated_at=self._now(),
                    )
                )
                if updated.rowcount != 1:
                    raise JobLeaseLost(f"任务所有权已丢失: {job_id}")
        except JobLeaseLost:
            raise
        except Exception as e:
            raise StorageUnavailableError(f"KB 任务标记 blocked 失败: {e}") from e

    def cleanup(self, *, done_days: int | None = None,
                failed_days: int | None = None, limit: int = 200) -> int:
        """保留期清理：succeeded/cancelled 90 天；failed/blocked 180 天。"""
        done_days = done_days if done_days is not None else settings.kb_job_retention_done_days
        failed_days = failed_days if failed_days is not None else settings.kb_job_retention_failed_days
        removed = 0
        try:
            with self._engine.begin() as conn:
                for statuses, days in (
                    ((JOB_SUCCEEDED, JOB_CANCELLED), done_days),
                    ((JOB_FAILED, JOB_BLOCKED), failed_days),
                ):
                    if days <= 0:
                        continue
                    cutoff = (
                        text(f"NOW() - INTERVAL {int(days * 86400)} SECOND")
                        if self._server_time
                        else _utcnow_naive() - timedelta(seconds=days * 86400)
                    )
                    cond = (
                        kb_index_jobs.c.status.in_(statuses),
                        kb_index_jobs.c.finished_at.isnot(None),
                        kb_index_jobs.c.finished_at < cutoff,
                    )
                    ids = [
                        r[0] for r in conn.execute(
                            select(kb_index_jobs.c.job_id).where(*cond).limit(limit)
                        ).fetchall()
                    ]
                    if ids:
                        conn.execute(
                            delete(kb_index_jobs)
                            .where(kb_index_jobs.c.job_id.in_(ids))
                        )
                        removed += len(ids)
        except Exception as e:
            raise StorageUnavailableError(f"KB 任务清理失败: {e}") from e
        return removed

    # ---------- 观测 ----------
    def stats(self) -> dict:
        """积压/最老任务年龄/blocked 计数（指标与告警同源）。"""
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(
                        kb_index_jobs.c.status,
                        kb_index_jobs.c.operation,
                        func.count().label("n"),
                        func.min(kb_index_jobs.c.created_at).label("oldest"),
                    ).group_by(kb_index_jobs.c.status, kb_index_jobs.c.operation)
                ).mappings().all()
                now = conn.execute(select(func.now())).scalar() or _utcnow_naive()
        except Exception:  # noqa: BLE001
            return {}
        # created_at 由库端 CURRENT_TIMESTAMP 写入（MySQL=服务端时间，
        # sqlite=UTC）——age 计算必须用同一时钟族，否则 sqlite 下差出时区偏移
        by_status: dict[str, int] = {}
        backlog: dict[str, int] = {"upload": 0, "delete": 0}
        oldest: dict[str, float] = {}
        blocked = 0
        for r in rows:
            status, op, n = r["status"], r["operation"], int(r["n"])
            by_status[status] = by_status.get(status, 0) + n
            if status in (JOB_QUEUED, JOB_RETRY_WAIT, JOB_RUNNING, JOB_BLOCKED):
                backlog[op] = backlog.get(op, 0) + n
                if r["oldest"] is not None:
                    age = max((now - r["oldest"]).total_seconds(), 0.0)
                    oldest[op] = max(oldest.get(op, 0.0), age)
            if status == JOB_BLOCKED:
                blocked += n
        return {"by_status": by_status, "backlog": backlog,
                "oldest_age_seconds": oldest, "blocked": blocked}

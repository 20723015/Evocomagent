"""KB 文档上传编排服务（v7 冻结 + 多实例异步改造）：Redis 断点续传 + MySQL 元数据 + ES 版本化重建。

生命周期（先 journal 后 CAS；提交点 = ES alias 切换）：

    uploading ──①CAS──→ validating ──(seal/合并/逐片验证/解析/消毒，无锁)──
        通过 → ②锁 → ③写 PREPARED journal → ④CAS validating→indexing
        → 原件保存 → 移入 knowledge/uploads → build(strict) → INDEX_BUILT
        → ACTIVATING(fsync) → activate_alias → assert_held → 查真实 alias
        → ALIAS_ACTIVATED → activate_pointer → POINTER_UPDATED → CAS indexed
        → journal 清 → 最后清分片/会话
    输入错误（hash/magic/解析/消毒/超限）→ failed（终态，重新上传）
    分片未就绪/锁冲突 → 回退 uploading（分片/staging 保留，幂等重试）
    提交点后失败 → 只前进（journal 保留，恢复机凑齐指针与元数据）

下架：CAS indexed→deleting → 锁 → 先 journal(PREPARED) → 移 trash →
    build(strict, 不含本文档) → … 提交点前失败移回（deleting→indexed），
    提交点后只前进（deleting→deleted）。

恢复：持有统一写锁后、接受任何新写前——先恢复全部残留 journal（相位判定：
    < ALIAS_ACTIVATED 回滚 / == ACTIVATING 查真实 alias 三分支 / ≥ 只前进），
    再检查 kb_write_blocked（alias 指向其它代 → 阻塞全部后续写，人工 reconcile）。

异步模式（kb_control 共享开关 kb_async_enabled 启用后）：
    complete/delete 只做「幂等封口 + 事务入队」（文档 CAS 与 kb_index_jobs
    INSERT 同事务）即返回 202；执行由独立 KB Worker 领取：
    run_upload_job / run_delete_job 复用同一提交序列，经 JobControl 注入
    阶段/进度上报与「副作用前 lease_token 所有权检查」；锁等待回 queued
    （不计失败次数）；提交点后故障只前进持续重试。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from app.agent.rag.job_store import JobLeaseLost
from app.config.settings import settings
from app.evolution.generation import GenerationInfo, GenerationStore, new_generation_id
from app.evolution.index_service import IndexBuildService
from app.evolution.lock import LockHeldError
from app.stores.base import StorageUnavailableError
from app.stores.kb_write_lock import (
    KbWriteLockBackendError,
    KbWriteLockError,
    get_kb_write_lock,
)
from app.stores.sql.document_store import (
    STATUS_CANCELLED,
    STATUS_DELETE_QUEUED,
    STATUS_DELETED,
    STATUS_DELETING,
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_INDEXING,
    STATUS_QUEUED,
    STATUS_UPLOADING,
    STATUS_VALIDATING,
    DocumentRecord,
    KbControlStore,
    SqlDocumentStore,
)
from app.stores.upload_state import (
    SESSION_CANCELLED,
    SESSION_SEALED,
    STATE_READY,
    UploadSession,
    build_upload_state_store,
)
from app.stores.upload_storage import (
    OriginalStore,
    StageDir,
    build_chunk_storage,
)

# 相位常量（journal.phase）
PH_PREPARED = "PREPARED"
PH_INDEX_BUILT = "INDEX_BUILT"
PH_ACTIVATING = "ACTIVATING"
PH_ALIAS_ACTIVATED = "ALIAS_ACTIVATED"
PH_POINTER_UPDATED = "POINTER_UPDATED"

_ALIAS_PHASES = (PH_ACTIVATING, PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED)

BLOCKED_KEY = "kb_write_blocked"
BLOCKED_VALUE = "reconcile-needed"

_UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ============================================================
# 编排错误（端点层映射 HTTP 状态）
# ============================================================
class UploadError(ValueError):
    status_code = 400


class UploadNotFound(UploadError):
    status_code = 404


class UploadForbidden(UploadError):
    status_code = 403


class UploadConflictError(UploadError):
    status_code = 409


class UploadParamsConflict(UploadConflictError):
    """幂等 create 参数不一致（同 upload_id 不同参数）。"""


class UploadClosedError(UploadConflictError):
    """会话已封口/取消（PUT/complete 时序错误）。"""


class UploadIncomplete(UploadConflictError):
    """分片不齐/有 publishing（可继续上传后重试 complete）。"""


class UploadTooLarge(UploadError):
    status_code = 413


class UploadParseFailed(UploadError):
    status_code = 400


class UploadSanitized(UploadError):
    status_code = 400


class UploadBlocked(UploadConflictError):
    """知识库被全局阻塞（alias 指向其它代，待人工 reconcile）。"""


class UploadLockWait(UploadConflictError):
    """异步模式：全局写锁被占用——任务回 queued 等待（不计失败次数）。"""


# 异步执行期的 control 协议（KbWorker 注入）：副作用前所有权检查 + 阶段上报
class JobControlLike:  # pragma: no cover —— 仅类型说明；运行时见 kb_worker.JobControl
    def check_alive(self) -> None:
        """写锁 + lease_token 所有权检查；失主立即抛错中止提交。"""

    def stage(self, stage: str, progress: int | None = None) -> None:
        """阶段/进度上报（每次顺带续租）。"""


# ============================================================
# 服务
# ============================================================
class DocumentUploadService:
    """文档上传编排（依赖注入：测试可全 fake；生产从 deps 构造）。"""

    def __init__(
        self,
        *,
        doc_store: SqlDocumentStore,
        control_store: KbControlStore,
        generation_store: GenerationStore,
        index_service: IndexBuildService,
        state_store=None,
        chunk_storage=None,
        stage: StageDir | None = None,
        originals: OriginalStore | None = None,
        engine=None,
        redis=None,
        kb_root=None,
        job_store=None,
    ):
        self._doc = doc_store
        self._control = control_store
        self._gen = generation_store
        self._index_service = index_service
        self._state = state_store or build_upload_state_store(redis)
        self._chunks = chunk_storage or build_chunk_storage()
        self._stage = stage or StageDir()
        self._originals = originals or OriginalStore()
        self._engine = engine
        self._redis = redis
        self._kb_root = Path(kb_root or settings.kb_dir)
        self._jobs = job_store  # KbIndexJobStore（异步模式；None = 仅同步语义）

    # ---------- 基础 ----------
    def _make_lock(self):
        return get_kb_write_lock(engine=self._engine, redis=self._redis)

    @staticmethod
    def _assert_write_authority(lock, control=None) -> None:
        """在共享写副作用前同时确认写锁与异步任务所有权。

        ``control`` 为空时用于同步恢复路径，仍必须确认锁未丢失；异步
        Worker 则额外校验 lease token/heartbeat/停机宽限。所有恢复函数
        都通过此闸门进入 alias、pointer、文档 CAS 和 journal 清理。
        """
        lock.assert_held()
        if control is not None:
            control.check_alive()

    def _kb_uploads_dir(self) -> Path:
        return self._kb_root / "uploads"

    def _backend(self) -> str:
        return (settings.rag_backend or "numpy").lower()

    def _new_operation(self) -> str:
        return uuid.uuid4().hex

    # ---------- 异步模式（kb_control 共享开关；默认同步语义） ----------
    def _async_enabled(self) -> bool:
        """两阶段上线开关：kb_control.kb_async_enabled=1 启用异步。

        共享开关未设置时回退 settings.kb_async_api（默认 False，保持同步语义）；
        读取失败向上抛（503 fail-closed，不猜测语义）。
        """
        if self._jobs is None:
            return False
        raw = (self._control.get(settings.kb_async_control_key) or "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        return bool(settings.kb_async_api)

    def _find_job(self, operation: str, rec: DocumentRecord) -> dict | None:
        if self._jobs is None or not rec.doc_id:
            return None
        return self._jobs.find(operation, doc_id=rec.doc_id)

    def _job_payload(self, job: dict) -> dict:
        """任务负载（API 层映射 202/200/409 与 Location/Retry-After 头）。"""
        return {
            "job_id": job["job_id"],
            "operation": job["operation"],
            "doc_id": job["doc_id"],
            "upload_id": job.get("upload_id", "") or "",
            "status": job["status"],
            "stage": job.get("stage", "") or "",
            "progress": int(job.get("progress") or 0),
            "attempts": int(job.get("attempts") or 0),
            "error": job.get("error", "") or "",
            "retryable": bool(int(job.get("retryable") or 0)),
            "status_url": f"/v1/kb/jobs/{job['job_id']}",
        }

    def _seal_idempotent(self, rec: DocumentRecord) -> None:
        """入队前幂等封口：SQL 失败后重复 complete 可补入队（封口可重放）。"""
        session = self._state.get(rec.upload_id)
        if session is None:
            raise UploadParseFailed("断点会话已过期，请重新上传")
        if session.session_state == SESSION_CANCELLED:
            raise UploadClosedError("会话已取消")
        if session.session_state != SESSION_SEALED:
            out = self._state.seal_if_ready(rec.upload_id)
            if out != "SEALED":
                raise UploadIncomplete(f"分片未就绪（{out}），请补传后重试")

    # ---------- create ----------
    def create_upload(self, uploader: str, filename: str, size_bytes: int,
                      content_type: str = "", chunk_size: int = 0,
                      sha256: str = "", upload_id: str = "") -> dict:
        fmt = _format_of(filename)
        size_bytes = int(size_bytes)
        if size_bytes <= 0:
            raise UploadError("size_bytes 必须 > 0")
        if size_bytes > settings.kb_upload_max_bytes:
            raise UploadError(f"文件超过大小上限 {settings.kb_upload_max_bytes} 字节")
        if not upload_id:
            upload_id = uuid.uuid4().hex
        upload_id = _validate_upload_id(upload_id)
        chunk_size = _clamp_chunk_size(chunk_size or settings.kb_upload_chunk_size)
        total_chunks = max((size_bytes + chunk_size - 1) // chunk_size, 1)
        if total_chunks > settings.kb_upload_max_chunks:
            raise UploadError(f"分片数 {total_chunks} 超过上限 {settings.kb_upload_max_chunks}")

        existing = self._doc.get_by_upload_id(upload_id)
        if existing is not None:
            self._assert_create_params(existing, uploader, filename, size_bytes,
                                       chunk_size, sha256, content_type)
            return self._status_payload(existing)

        record = DocumentRecord(
            doc_id=uuid.uuid4().hex,
            upload_id=upload_id,
            uploader=uploader,
            storage_key=self._storage_key(filename, upload_id),
            filename=filename,
            content_type=content_type,
            format=fmt,
            chunk_size=chunk_size,
            size_bytes=size_bytes,
            sha256=sha256 or "",
            upload_chunk_count=total_chunks,
            provenance=f"upload:{upload_id}",
            expires_at=(datetime.now() + timedelta(seconds=settings.kb_upload_session_ttl))  # noqa: DTZ005
            .isoformat(timespec="seconds"),
        )
        record = self._doc.create(record)
        self._state.create(UploadSession(
            upload_id=upload_id, filename=filename, size_bytes=size_bytes,
            chunk_size=chunk_size, total_chunks=total_chunks,
            uploader=uploader, content_type=content_type, format=fmt,
            sha256=sha256 or "",
        ))
        return self._status_payload(record)

    def _assert_create_params(self, existing: DocumentRecord, uploader: str,
                              filename: str, size_bytes: int, chunk_size: int,
                              sha256: str, content_type: str) -> None:
        mismatch = (
            existing.uploader != uploader
            or existing.filename != filename
            or int(existing.size_bytes) != int(size_bytes)
            or int(existing.chunk_size) != int(chunk_size)
            or (existing.sha256 or "") != (sha256 or "")
            or (existing.content_type or "") != (content_type or "")
        )
        if mismatch:
            raise UploadParamsConflict(
                f"upload_id {existing.upload_id} 已存在且参数不一致"
                "（uploader/filename/size/chunk_size/sha256/content_type）"
            )

    def _storage_key(self, filename: str, upload_id: str) -> str:
        return f"{_slug_from(filename)}-{upload_id[:8]}.md"

    # ---------- PUT 分片 ----------
    def put_chunk(self, upload_id: str, seq: int, body: bytes) -> dict:
        session = self._state.get(upload_id)
        if session is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        seq = int(seq)
        if not (0 <= seq < session.total_chunks):
            raise UploadError(f"分片序号越界: {seq}（0..{session.total_chunks - 1}）")

        expected = _expected_chunk_size(session, seq)
        if len(body) != expected:
            if len(body) > expected:
                raise UploadTooLarge("分片超过预期大小（按实际读取字节数限流）")
            raise UploadError(f"分片大小不符: 期望 {expected}，实际 {len(body)}")

        token = uuid.uuid4().hex
        self._chunks.write_temp(upload_id, seq, token, body)
        sha256 = hashlib.sha256(body).hexdigest()
        code = self._state.declare_chunk(upload_id, seq, len(body), sha256, token)
        try:
            if code in ("NEW", "TAKEOVER", "SAME_REQUEST"):
                object_key = self._chunks.finalize_object(upload_id, seq, token, sha256)
                out = self._state.finalize_chunk(
                    upload_id, seq, token, len(body), sha256, object_key,
                )
                if out not in ("READY", "DUPLICATE"):
                    self._state.drop_chunk_record(upload_id, seq, token)
                    raise UploadClosedError(f"分片发布确认失败（{out}）")
            elif code in ("DUPLICATE", "CONFLICT", "PENDING", "CLOSED"):
                # 只删本次临时对象——正式对象（原有分片）绝不因重传被破坏
                self._chunks.delete_temp(upload_id, seq, token)
                if code == "DUPLICATE":
                    pass  # 幂等成功
                elif code == "CONFLICT":
                    raise UploadParamsConflict(f"分片 {seq} 已存在且内容不一致")
                elif code == "PENDING":
                    raise UploadConflictError(f"分片 {seq} 正在处理中，请稍后重试")
                else:
                    raise UploadClosedError("会话已封口/取消，拒绝新分片")
            else:  # NOSESSION
                self._chunks.delete_temp(upload_id, seq, token)
                raise UploadNotFound(f"上传会话不存在: {upload_id}")
        except UploadError:
            raise
        except Exception as e:
            self._state.drop_chunk_record(upload_id, seq, token)
            raise UploadError(f"分片发布失败: {e}") from e
        return {"upload_id": upload_id, "seq": seq,
                "received": self._state.ready_count(upload_id),
                "total_chunks": session.total_chunks}

    # ---------- 状态 ----------
    def get_status(self, upload_id: str) -> dict:
        session = self._state.get(upload_id)
        rec = self._doc.get_by_upload_id(upload_id)
        if session is None and rec is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        job = self._jobs.latest_for_doc(rec.doc_id) if (self._jobs and rec) else None
        if session is None:
            # Redis 会话/分片已清理：以 SQL 文档 + 任务为正本返回（终态可见）
            return {
                "upload_id": upload_id,
                "doc_id": rec.doc_id if rec else "",
                "session_state": "",
                "filename": rec.filename if rec else "",
                "size_bytes": int(rec.size_bytes) if rec else 0,
                "chunk_size": int(rec.chunk_size) if rec else 0,
                "total_chunks": int(rec.upload_chunk_count) if rec else 0,
                "received": [],
                "ready_count": 0,
                "doc_status": rec.status if rec else "",
                "operation_id": rec.operation_id if rec else "",
                "job": self._job_payload(job) if job else None,
            }
        received = self._state.received(upload_id)
        return {
            "upload_id": upload_id,
            "session_state": session.session_state,
            "filename": session.filename,
            "size_bytes": session.size_bytes,
            "chunk_size": session.chunk_size,
            "total_chunks": session.total_chunks,
            "received": sorted(k for k, m in received.items()
                               if m.get("state") == STATE_READY),
            "ready_count": self._state.ready_count(upload_id),
            "doc_status": rec.status if rec else "",
            "operation_id": rec.operation_id if rec else "",
            "job": self._job_payload(job) if job else None,
        }

    def _status_payload(self, rec: DocumentRecord) -> dict:
        received = self._state.received(rec.upload_id)
        return {
            "upload_id": rec.upload_id,
            "doc_id": rec.doc_id,
            "filename": rec.filename,
            "size_bytes": rec.size_bytes,
            "chunk_size": rec.chunk_size,
            "total_chunks": rec.upload_chunk_count,
            "received": sorted(k for k, m in received.items()
                               if m.get("state") == STATE_READY),
            "status": rec.status,
            "operation_id": rec.operation_id,
        }

    # ---------- cancel ----------
    def cancel_upload(self, upload_id: str, uploader: str) -> dict:
        rec = self._doc.get_by_upload_id(upload_id)
        if rec is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        _require_owner(rec, uploader)
        if self._async_enabled():
            return self._cancel_async(rec, upload_id, uploader)
        return self._cancel_sync(rec, upload_id)

    def _cancel_async(self, rec: DocumentRecord, upload_id: str,
                      uploader: str) -> dict:
        """委托任务取消：仅 queued/retry_wait 可取消；运行中 409。

        取消 = 放弃上传（与同步语义一致）：任务 cancelled + 文档 cancelled +
        清分片/会话。若用户想换文件，重新 create_upload 即可。
        """
        from app.agent.rag.job_store import JOB_QUEUED, JOB_RETRY_WAIT

        job = self._find_job("upload", rec)
        if job is not None:
            if job["status"] not in (JOB_QUEUED, JOB_RETRY_WAIT):
                raise UploadConflictError(
                    f"任务已开始执行，无法取消（status={job['status']}，"
                    f"查询 {self._job_payload(job)['status_url']}）"
                )
            cancelled = self.cancel_job(job["job_id"])
            if cancelled is None:  # 竞态：任务刚被领取
                raise UploadConflictError("任务已开始执行，无法取消")
            return {"upload_id": upload_id, "status": STATUS_CANCELLED,
                    "job_id": job["job_id"]}
        if rec.status not in (STATUS_UPLOADING, STATUS_QUEUED):
            raise UploadConflictError(f"当前状态 {rec.status} 不可取消")
        # 无任务（尚未 complete）：仅清会话与分片，文档置 cancelled
        updated = self._doc.update_status(
            rec.doc_id, rec.status, STATUS_CANCELLED, rec.version,
        )
        if updated < 0:
            raise UploadConflictError("上传状态已变化，无法取消")
        self._cleanup_upload_site(upload_id)
        return {"upload_id": upload_id, "status": STATUS_CANCELLED}

    def cancel_job(self, job_id: str) -> dict | None:
        """取消排队任务；SQL 事务提交后再清理上传会话和分片。

        JobStore 在一个事务中完成 job cancelled 与文档状态转换；这里只
        编排不可事务化的对象存储/Redis 清理，避免多实例重新入队窗口。
        """
        if self._jobs is None:
            return None
        operation = str((self._jobs.get(job_id) or {}).get("operation") or "")
        if operation == "upload" and hasattr(self._jobs, "cancel_upload_atomic"):
            cancelled = self._jobs.cancel_upload_atomic(job_id)
        else:
            cancelled = self._jobs.cancel(job_id)
        if (cancelled is not None
                and cancelled.get("operation") == "upload"
                and cancelled.get("upload_id")):
            self._cleanup_upload_site(str(cancelled["upload_id"]))
        return cancelled

    def _cancel_sync(self, rec: DocumentRecord, upload_id: str) -> dict:
        self._state.mark_closed(upload_id, SESSION_CANCELLED)
        if rec.status in (STATUS_UPLOADING, STATUS_VALIDATING):
            self._doc.update_status(rec.doc_id, rec.status, STATUS_CANCELLED, rec.version)
        self._cleanup_upload_site(upload_id)
        return {"upload_id": upload_id, "status": STATUS_CANCELLED}

    def _cleanup_upload_site(self, upload_id: str) -> None:
        """收尾清理：分片对象 + Redis 会话（尽力而为，失败不影响结果）。"""
        try:
            self._chunks.delete_chunks(upload_id)
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self._state.delete(upload_id)
        except Exception:  # noqa: BLE001, S110
            pass

    # ---------- complete ----------
    def complete(self, upload_id: str, uploader: str) -> dict:
        rec = self._doc.get_by_upload_id(upload_id)
        if rec is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        _require_owner(rec, uploader)

        if rec.status == STATUS_INDEXED:
            return self._indexed_payload(rec)
        if self._async_enabled():
            return self._complete_async(rec, uploader)
        return self._complete_sync(rec, upload_id, uploader)

    def _complete_async(self, rec: DocumentRecord, uploader: str) -> dict:
        """异步入队：幂等封口 + 事务（文档 CAS + 任务 INSERT），立即返回任务。"""
        job = self._find_job("upload", rec)
        if job is not None:
            from app.agent.rag.job_store import JOB_ACTIVE

            if job["status"] in JOB_ACTIVE:
                return self._job_payload(job)  # 202：重复请求返回同一任务
            if job["status"] == "succeeded":
                return self._indexed_payload(self._doc.get(rec.doc_id) or rec)
            if job["status"] == "failed":
                # 永久失败/重试耗尽：返回任务负载（API 映射 409 + 任务地址，
                # 人工 POST .../retry 可重排——仅限 retryable 任务）
                return self._job_payload(job)
            # cancelled → 继续走重新入队（换新任务）
        if rec.status == STATUS_FAILED:
            raise UploadConflictError("文档为永久失败状态，请修正后重新上传")
        if rec.status in (STATUS_VALIDATING, STATUS_INDEXING, STATUS_DELETING):
            # 无任务的中间态：同步路径遗留（回滚中/接管前），拒绝并发入队
            raise UploadRecovering(f"文档正在处理（{rec.status}），请稍后重试")
        if rec.status not in (STATUS_UPLOADING, STATUS_QUEUED):
            raise UploadConflictError(f"当前状态 {rec.status} 不允许 complete")
        self._seal_idempotent(rec)
        job, _created = self._jobs.enqueue_upload(
            rec.doc_id, rec.upload_id, uploader, rec.version,
        )
        return self._job_payload(job)

    def _complete_sync(self, rec: DocumentRecord, upload_id: str,
                       uploader: str) -> dict:
        """同步兼容路径（kb_async 开关关闭）：保持 v7 冻结语义不变。"""
        if rec.status in (STATUS_VALIDATING, STATUS_INDEXING, STATUS_DELETING):
            return self._resume_after_recover(rec, upload_id)

        # ---- ① CAS 抢占 validating（合并前）----
        op_id = self._new_operation()
        v = self._doc.update_status(rec.doc_id, STATUS_UPLOADING, STATUS_VALIDATING,
                                    rec.version, operation_id=op_id)
        if v < 0:
            raise UploadConflictError("并发处理中（另一控制者已抢占），请稍后重试")

        # ---- ② 无锁校验区 ----
        try:
            merged, text = self._validate_and_prepare(rec)
        except UploadIncomplete as e:
            self._doc.update_status(rec.doc_id, STATUS_VALIDATING, STATUS_UPLOADING,
                                    v, operation_id="", error=str(e)[:200])
            raise
        except (UploadParseFailed, UploadSanitized, UploadTooLarge) as e:
            self._doc.update_status(rec.doc_id, STATUS_VALIDATING, STATUS_FAILED,
                                    v, operation_id="", error=str(e)[:200])
            raise

        # ---- ③ 锁 + 提交点序列 ----
        return self._commit_locked(rec, v, op_id, merged, text)

    def _validate_and_prepare(self, rec: DocumentRecord):
        """封口 → 合并（ready 校验/缺失解封）→ 魔数/zip/页数限 → 子进程解析 → 消毒。"""
        session = self._state.get(rec.upload_id)
        if session is None:
            raise UploadParseFailed("断点会话已过期，请重新上传")
        if session.session_state == SESSION_CANCELLED:
            raise UploadClosedError("会话已取消")
        if session.session_state != SESSION_SEALED:
            out = self._state.seal_if_ready(rec.upload_id)
            if out != "SEALED":
                raise UploadIncomplete(f"分片未就绪（{out}），请补传后重试")
        merged = self._merge_ready(session, rec)
        self._stage.write_merged(rec.upload_id, merged, rec.format)
        text = self._parse_final(merged, rec, session)
        return merged, text

    def _merge_ready(self, session: UploadSession, rec: DocumentRecord) -> bytes:
        received = self._state.received(session.upload_id)
        ready = sorted((k, m) for k, m in received.items() if m.get("state") == STATE_READY)
        if len(ready) != session.total_chunks:
            raise UploadIncomplete("分片未就绪（数量不足）")
        parts: list[bytes] = []
        for seq, meta in ready:
            data = self._chunks.get_chunk(session.upload_id, seq)
            if data is None or hashlib.sha256(data).hexdigest() != meta.get("sha256"):
                out = self._state.unseal_and_drop(
                    session.upload_id, seq,
                    int(meta.get("size", 0) or 0), meta.get("sha256", ""),
                )
                if out == "UNSEALED":
                    raise UploadIncomplete(
                        f"分片 {seq} 对象缺失，已解封等待重传"
                    )
                raise UploadConflictError(f"分片 {seq} 校验失败且无法解封（{out}）")
            parts.append(data)
        merged = b"".join(parts)
        if len(merged) != session.size_bytes:
            raise UploadIncomplete(f"合并大小不符: {len(merged)} vs {session.size_bytes}")
        declared = rec.sha256 or session.sha256
        if declared and declared != hashlib.sha256(merged).hexdigest():
            raise UploadParseFailed("sha256 与客户端声明不符")
        return merged

    def _parse_final(self, merged: bytes, rec: DocumentRecord,
                     session: UploadSession) -> str:
        fmt = rec.format
        _magic_check(merged, fmt)
        _zip_limits(merged, fmt)
        _page_limits(merged, fmt)

        from app.agent.rag.parse_guard import ParseTimeout, parse_with_timeout

        try:
            text = parse_with_timeout(
                self._stage.merged_path(session.upload_id, rec.format),
            )
        except (ParseTimeout, ValueError) as e:
            raise UploadParseFailed(f"解析失败: {e}") from e
        if not text.strip():
            raise UploadParseFailed("解析结果为空（扫描件请走 OCR）")
        if len(text) > settings.kb_upload_max_text_chars:
            raise UploadParseFailed("解析文本超过长度上限")

        from app.evolution.sanitizer import has_injection, has_pii

        title = _slug_from(rec.filename)
        frontmatter = (
            "---\n"
            f"provenance: upload:{rec.upload_id}\n"
            "owner: ops\n"
            f"title: {title}\n"
            f"source_format: {fmt}\n"
            "---\n"
        )
        normalized = f"{frontmatter}# {title}\n\n{text}\n"
        if has_pii(normalized) or has_injection(normalized):
            raise UploadSanitized("文档包含 PII/注入内容，已拒绝（与沉淀发布同标准）")
        self._stage.write_normalized(session.upload_id, normalized)
        return normalized

    # ---------- 锁内提交（upload 语义；下架复用同相位逻辑） ----------
    def _commit_locked(self, rec: DocumentRecord, validating_version: int,
                       op_id: str, merged: bytes, text: str,
                       control=None) -> dict:
        """锁内提交序列（提交点 = alias 切换）。

        control=None：同步语义（锁冲突回 uploading 抛 409；失败回滚）。
        control=JobControl：异步语义——阶段/进度上报 + 每个副作用前所有权检查；
        锁等待回 queued（UploadLockWait，不计失败）；提交点后失败只前进
        （UploadRecovering，Worker 持续重试）。
        """
        backend = self._backend()
        gen_id = new_generation_id()
        info = GenerationInfo(
            generation_id=gen_id,
            target=self._index_service.target_for(backend, gen_id),
            embedding_model=settings.embedding_model,
        )
        storage_key = rec.storage_key
        phase = PH_PREPARED
        lock = self._make_lock()
        try:
            if control is not None:
                control.stage("waiting_for_lock", progress=10)
            lock.acquire(phase="upload")
        except (KbWriteLockError, KbWriteLockBackendError, LockHeldError) as e:
            if control is not None:
                # 等锁期间可能已被另一实例接管；失主不得把文档状态回写成
                # queued（新 Worker 可能已经推进到提交点）。
                control.check_alive()
                # 异步：锁等待不计失败——文档回 queued，任务短延迟重试
                self._doc.update_status(rec.doc_id, STATUS_VALIDATING, STATUS_QUEUED,
                                        validating_version, operation_id="",
                                        error=f"锁被持有: {type(e).__name__}")
                raise UploadLockWait(f"知识库构建中: {e}") from e
            self._doc.update_status(rec.doc_id, STATUS_VALIDATING, STATUS_UPLOADING,
                                    validating_version, operation_id="",
                                    error=f"锁被持有: {type(e).__name__}")
            raise UploadConflictError(f"知识库构建中: {e}") from e
        try:
            # 持锁后、接受任何新写前：先恢复历史事务，再检查全局阻塞
            self._assert_write_authority(lock, control)
            self._recover_all_journals(lock, control=control)
            self._assert_write_authority(lock, control)
            self._assert_not_blocked()

            # ③ 先 journal 后 CAS（消除「indexing 无 journal」窗口）
            self._assert_write_authority(lock, control)
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_PREPARED,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_PREPARED
            self._assert_write_authority(lock, control)
            ok = self._doc.update_status(rec.doc_id, STATUS_VALIDATING, STATUS_INDEXING,
                                         validating_version, require_op=op_id,
                                         pending_generation_id=gen_id, error="")
            if ok < 0:
                self._assert_write_authority(lock, control)
                self._stage.journal_clear(rec.upload_id)
                raise UploadBusyState("状态已变化，无法进入 indexing")

            # ④ 原件（提交点前）→ 移入知识库（同卷 rename）
            self._assert_write_authority(lock, control)
            self._originals.put(rec.doc_id, rec.format, merged)
            self._assert_write_authority(lock, control)
            self._stage.move_into_kb(rec.upload_id, self._kb_uploads_dir(), storage_key)

            # ⑤ 构建（strict：任何源文件解析失败即中止）
            self._assert_write_authority(lock, control)
            if control is not None:
                control.stage("chunking", progress=15)
            built = self._build_with_progress(backend, gen_id, control)
            self._assert_write_authority(lock, control)
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_INDEX_BUILT,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_INDEX_BUILT

            # ⑥ 提交点：ACTIVATING → alias → assert → 查真实 alias
            self._assert_write_authority(lock, control)
            if control is not None:
                control.stage("activating", progress=80)
            self._assert_write_authority(lock, control)
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_ACTIVATING,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_ACTIVATING
            # journal 写入与 alias 切换之间也可能跨过租约/写锁边界，
            # 必须在真正提交点前重新 fencing。
            self._assert_write_authority(lock, control)
            self._index_service.activate_alias(backend, built)
            self._assert_write_authority(lock, control)  # alias 后、指针前
            self._assert_alias_confirmed(built)

            # ⑦ 指针 → 元数据提交
            self._assert_write_authority(lock, control)
            self._index_service.activate_pointer(backend, built)
            self._assert_write_authority(lock, control)
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_POINTER_UPDATED,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_POINTER_UPDATED
            if control is not None:
                control.stage("finalizing", progress=90)
            self._assert_write_authority(lock, control)
            ok = self._doc.update_status(
                rec.doc_id, STATUS_INDEXING, STATUS_INDEXED, validating_version + 1,
                require_op=op_id, generation_id=gen_id,
                indexed_chunk_count=self._index_service.last_built_size,
                pending_generation_id="", error="",
            )
            if ok < 0:
                raise UploadRecovering("元数据提交失败，journal 保留等待恢复")

            # ⑧ 成功收尾：清 journal + 分片/会话（最后清理）
            self._assert_write_authority(lock, control)
            self._stage.journal_clear(rec.upload_id)
            try:
                self._chunks.delete_chunks(rec.upload_id)
                self._state.delete(rec.upload_id)
            except Exception:  # noqa: BLE001, S110
                pass
            return self._indexed_payload(self._doc.get(rec.doc_id) or rec)
        except JobLeaseLost:
            # 旧 Worker 失主后绝不能进入 rollback：共享卷、journal 和
            # 文档状态必须留给新 lease 持有者恢复。
            raise
        except UploadError as e:
            # alias 已切换后的确认/其它编排错误属于提交点后故障，必须
            # 保留 journal 并走只前进恢复；只有明确 blocked 继续原语义。
            if phase in _ALIAS_PHASES and not isinstance(e, UploadBlocked):
                raise UploadRecovering(
                    f"提交点后异常待恢复（{rec.upload_id}）: {type(e).__name__}"
                ) from e
            raise
        except KbWriteLockError as e:
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(f"处理中断待恢复（{rec.upload_id}）: {e}") from e
            if control is not None:
                self._rollback_upload(rec, storage_key, phase, dst=STATUS_QUEUED,
                                      lock=lock, control=control)
                raise UploadLockWait(f"锁已丢失，已回退 queued: {e}") from e
            self._rollback_upload(rec, storage_key, phase, lock=lock)
            raise UploadConflictError(f"锁已丢失，已回滚可重试: {e}") from e
        except (StorageUnavailableError, Exception) as e:
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(
                    f"处理中断待恢复（{rec.upload_id}）: {type(e).__name__}") from e
            if control is not None:
                # 异步：提交点前失败回 queued（暂态重试 / 输入错误由 Worker 分流）
                self._rollback_upload(rec, storage_key, phase, dst=STATUS_QUEUED,
                                      lock=lock, control=control)
                raise
            self._rollback_upload(rec, storage_key, phase, lock=lock)
            raise UploadConflictError(
                f"构建失败已回滚，可重试（{type(e).__name__}: {e}）") from e
        finally:
            try:
                lock.release()
            except Exception:  # noqa: BLE001, S110
                pass

    def _build_with_progress(self, backend: str, gen_id: str, control,
                             *, allow_empty: bool = False) -> GenerationInfo:
        """构建 + embedding 批级进度上报（control=None 时原样调用）。

        allow_empty：下架流程传 True——删掉最后一篇文档后 KB 为空是合法终态；
        上传路径保持默认 strict（防误清空保护不回归）。
        """
        if control is None:
            return self._index_service.build(
                backend, generation_id=gen_id, allow_empty=allow_empty,
            )
        from app.agent.rag.kb_worker import embedding_progress_proxy

        original = self._index_service._embedder
        self._index_service._embedder = embedding_progress_proxy(original, control)
        try:
            return self._index_service.build(
                backend, generation_id=gen_id, allow_empty=allow_empty,
            )
        finally:
            self._index_service._embedder = original

    # ---------- Worker 执行入口（异步模式；kb_worker 调用） ----------
    def run_upload_job(self, job: dict, control) -> None:
        """上传任务执行：接管恢复 → CAS validating → 验证区 → 锁内提交。

        异常分类由 Worker 处理（输入错误=永久 / 锁等待=回队不计失败 /
        提交点后=只前进持续重试 / blocked=人工 reconcile）。
        """
        rec = self._doc.get(job["doc_id"])
        if rec is None or rec.upload_id != (job.get("upload_id") or rec.upload_id):
            raise UploadParseFailed("任务与文档记录不一致（疑似数据异常）")
        # 接管恢复：journal 残留 → 相位表判定（回滚重跑 / 前进 / blocked）
        rec = self._recover_takeover(rec, rec.upload_id, control)
        if rec.status == STATUS_INDEXED:
            return  # 前进恢复已入库
        control.check_alive()
        validating_version = self._enter_validating(rec, job["job_id"])
        control.stage("validating", progress=5)
        try:
            merged, text = self._validate_and_prepare(rec)
        except UploadIncomplete:
            control.check_alive()
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status == STATUS_VALIDATING:
                control.check_alive()
                self._doc.update_status(
                    cur.doc_id, STATUS_VALIDATING, STATUS_UPLOADING, cur.version,
                    operation_id="", error="分片未就绪（等待补传）",
                )
            raise
        except (UploadParseFailed, UploadSanitized, UploadTooLarge) as e:
            control.check_alive()
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status == STATUS_VALIDATING:
                control.check_alive()
                self._doc.update_status(
                    cur.doc_id, STATUS_VALIDATING, STATUS_FAILED, cur.version,
                    operation_id="", error=str(e)[:200],
                )
            raise
        return self._commit_locked(rec, validating_version, job["job_id"],
                                   merged, text, control=control)

    def run_delete_job(self, job: dict, control) -> None:
        """下架任务执行：接管恢复 → CAS deleting → 锁内提交。"""
        rec = self._doc.get(job["doc_id"])
        if rec is None:
            raise UploadParseFailed("文档不存在（job 与文档不一致）")
        rec = self._recover_takeover(rec, f"del-{rec.doc_id}", control)
        if rec.status == STATUS_DELETED:
            return  # 前进恢复已完成下架
        control.check_alive()
        deleting_version = self._enter_deleting(rec, job["job_id"])
        control.stage("validating", progress=5)
        return self._commit_locked_delete(rec, job["job_id"],
                                          control=control,
                                          deleting_version=deleting_version)

    def _recover_takeover(self, rec: DocumentRecord, op_id: str,
                          control=None) -> DocumentRecord:
        """Worker 接管：journal 残留 → 持锁按相位表恢复。

        - 无 journal：原样返回（validating 无 journal → 从解析阶段安全重跑）；
        - PREPARED/INDEX_BUILT：回滚（uploading/indexed）后由调用方重跑；
        - ACTIVATING/ALIAS_ACTIVATED/POINTER_UPDATED：只前进；
        - alias 指向未知代：抛 UploadBlocked（Worker 置 blocked + 全局写阻塞）。
        """
        payload = self._stage.journal_read(op_id)
        if not payload:
            return rec
        lock = self._make_lock()
        try:
            if control is not None:
                control.stage("waiting_for_lock", progress=10)
            lock.acquire(phase="recover")
        except (KbWriteLockError, KbWriteLockBackendError, LockHeldError) as e:
            raise UploadLockWait(f"知识库构建中: {e}") from e
        try:
            # 领取恢复任务后，锁已拿到但租约可能在等待锁时过期；恢复的每个
            # 共享副作用仍须由同一个 lease + write lock 闸门保护。
            self._assert_write_authority(lock, control)
            self._recover_one(op_id, lock=lock, control=control)
        finally:
            try:
                lock.release()
            except Exception:  # noqa: BLE001, S110
                pass
        fresh = self._doc.get(rec.doc_id)
        return fresh or rec

    def _enter_validating(self, rec: DocumentRecord, op_id: str) -> int:
        """queued/uploading/validating → validating（CAS；返回 validating 版本号）。"""
        if rec.status == STATUS_VALIDATING:
            return rec.version  # 接管续跑（同一 job_id/operation_id）
        if rec.status not in (STATUS_QUEUED, STATUS_UPLOADING):
            raise UploadConflictError(f"文档状态 {rec.status} 不允许执行上传任务")
        v = self._doc.update_status(rec.doc_id, rec.status, STATUS_VALIDATING,
                                    rec.version, operation_id=op_id)
        if v < 0:
            raise UploadConflictError("文档状态已变化（可能已被取消）")
        return v

    def _enter_deleting(self, rec: DocumentRecord, op_id: str) -> int:
        """delete_queued/deleting/indexed → deleting（CAS；返回 deleting 版本号）。"""
        if rec.status == STATUS_DELETING:
            return rec.version  # 接管续跑
        if rec.status == STATUS_DELETE_QUEUED:
            v = self._doc.update_status(rec.doc_id, STATUS_DELETE_QUEUED,
                                        STATUS_DELETING, rec.version,
                                        operation_id=op_id)
        elif rec.status == STATUS_INDEXED:
            # 恢复机把提交点前失败回滚到 indexed（异步接管场景）→ 重新进入 deleting
            v = self._doc.update_status(rec.doc_id, STATUS_INDEXED, STATUS_DELETING,
                                        rec.version, operation_id=op_id)
        else:
            raise UploadConflictError(f"文档状态 {rec.status} 不允许执行下架任务")
        if v < 0:
            raise UploadConflictError("文档状态已变化（可能已被取消）")
        return v

    # ---------- 下架 ----------
    def delete_document(self, doc_id: str, uploader: str) -> dict:
        rec = self._doc.get(doc_id)
        if rec is None:
            raise UploadNotFound(f"文档不存在: {doc_id}")
        _require_owner(rec, uploader)
        if rec.status == STATUS_DELETED:
            return {"doc_id": doc_id, "status": STATUS_DELETED}  # 200 幂等
        if self._async_enabled():
            return self._delete_async(rec, doc_id, uploader)
        return self._delete_sync(rec, doc_id, uploader)

    def _delete_async(self, rec: DocumentRecord, doc_id: str,
                      uploader: str) -> dict:
        """异步下架入队：CAS indexed→delete_queued + 任务 INSERT（同事务）。"""
        job = self._find_job("delete", rec)
        if job is not None:
            from app.agent.rag.job_store import JOB_ACTIVE

            if job["status"] in JOB_ACTIVE:
                return self._job_payload(job)  # 202：重复请求返回同一任务
            if job["status"] == "succeeded":
                return {"doc_id": doc_id, "status": STATUS_DELETED}
            # failed / cancelled → 重入队（cancelled 换新任务；failed 幂等返回）
        if rec.status in (STATUS_DELETE_QUEUED, STATUS_DELETING):
            raise UploadRecovering(f"文档正在下架（{rec.status}），请稍后查询")
        if rec.status != STATUS_INDEXED:
            raise UploadConflictError(f"仅已入库文档可下架（当前 {rec.status}）")
        job, _created = self._jobs.enqueue_delete(doc_id, uploader, rec.version)
        return self._job_payload(job)

    def _delete_sync(self, rec: DocumentRecord, doc_id: str, uploader: str) -> dict:
        """同步兼容路径（kb_async 开关关闭）：保持 v7 冻结语义不变。"""
        if rec.status != STATUS_INDEXED:
            raise UploadConflictError(f"仅已入库文档可下架（当前 {rec.status}）")

        op_id = self._new_operation()
        ok = self._doc.update_status(rec.doc_id, STATUS_INDEXED, STATUS_DELETING,
                                     rec.version, operation_id=op_id)
        if ok < 0:
            raise UploadConflictError("文档状态已变化")
        return self._commit_locked_delete(rec, op_id)

    def _commit_locked_delete(self, rec: DocumentRecord, op_id: str,
                              control=None,
                              deleting_version: int | None = None) -> dict:
        """下架锁内提交序列（control 语义同 _commit_locked；同步默认路径）。

        deleting_version：deleting 状态的版本号（异步接管续跑时显式传入；
        同步路径缺省 = rec.version + 1，即刚 CAS 进入 deleting 的版本）。
        """
        doc_id = rec.doc_id
        del_ver = deleting_version if deleting_version is not None else rec.version + 1
        backend = self._backend()
        gen_id = new_generation_id()
        info = GenerationInfo(
            generation_id=gen_id,
            target=self._index_service.target_for(backend, gen_id),
            embedding_model=settings.embedding_model,
        )
        journal_op = f"del-{doc_id}"
        phase = PH_PREPARED
        lock = self._make_lock()
        try:
            if control is not None:
                control.stage("waiting_for_lock", progress=10)
            lock.acquire(phase="delete")
        except (KbWriteLockError, KbWriteLockBackendError, LockHeldError) as e:
            if control is not None:
                control.check_alive()
                self._doc.update_status(rec.doc_id, STATUS_DELETING,
                                        STATUS_DELETE_QUEUED, del_ver,
                                        require_op=op_id,
                                        error=f"锁被持有: {type(e).__name__}")
                raise UploadLockWait(f"知识库构建中: {e}") from e
            self._doc.update_status(rec.doc_id, STATUS_DELETING, STATUS_INDEXED,
                                    del_ver, require_op=op_id,
                                    error=f"锁被持有: {type(e).__name__}")
            raise UploadConflictError(f"知识库构建中: {e}") from e
        try:
            self._assert_write_authority(lock, control)
            self._recover_all_journals(lock, control=control)
            self._assert_write_authority(lock, control)
            self._assert_not_blocked()
            if control is not None:
                control.check_alive()
            # 先 journal 再移动（下架可恢复）
            self._assert_write_authority(lock, control)
            self._journal(journal_op, {
                "op": "delete", "phase": PH_PREPARED,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_PREPARED
            self._assert_write_authority(lock, control)
            if control is not None:
                control.stage("chunking", progress=20)
            self._assert_write_authority(lock, control)
            self._stage.move_to_trash(self._kb_uploads_dir(), rec.storage_key)
            self._assert_write_authority(lock, control)
            # 下架后 KB 为空是合法终态 → allow_empty=True（删最后一篇不再失败）
            built = self._build_with_progress(backend, gen_id, control,
                                              allow_empty=True)
            self._assert_write_authority(lock, control)
            self._journal(journal_op, {
                "op": "delete", "phase": PH_INDEX_BUILT,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_INDEX_BUILT
            self._assert_write_authority(lock, control)
            if control is not None:
                control.stage("activating", progress=80)
            self._assert_write_authority(lock, control)
            self._journal(journal_op, {
                "op": "delete", "phase": PH_ACTIVATING,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_ACTIVATING
            self._assert_write_authority(lock, control)
            self._index_service.activate_alias(backend, built)
            self._assert_write_authority(lock, control)
            self._assert_alias_confirmed(built)
            self._assert_write_authority(lock, control)
            self._index_service.activate_pointer(backend, built)
            self._assert_write_authority(lock, control)
            self._journal(journal_op, {
                "op": "delete", "phase": PH_POINTER_UPDATED,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_POINTER_UPDATED
            if control is not None:
                control.stage("finalizing", progress=90)
            self._assert_write_authority(lock, control)
            ok = self._doc.update_status(rec.doc_id, STATUS_DELETING, STATUS_DELETED,
                                         del_ver, require_op=op_id,
                                         generation_id=gen_id, error="")
            if ok < 0:
                raise UploadRecovering("下架元数据提交失败，journal 保留等待恢复")
            # 成功收尾：清 journal（残留会被恢复机误判为 pending 代并重放旧代激活）
            self._assert_write_authority(lock, control)
            self._stage.journal_clear(journal_op)
        except JobLeaseLost:
            # 旧 Worker 失主后不回滚、不清理共享 journal，也不改文档；新
            # lease 持有者会按相位继续恢复。
            raise
        except UploadError as e:
            if phase in _ALIAS_PHASES and not isinstance(e, UploadBlocked):
                raise UploadRecovering(
                    f"下架提交点后异常待恢复（{doc_id}）: {type(e).__name__}"
                ) from e
            raise
        except KbWriteLockError as e:
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(f"下架中断待恢复（{doc_id}）: {e}") from e
            if control is not None:
                self._rollback_delete(rec, op_id, dst=STATUS_DELETE_QUEUED,
                                      lock=lock, control=control)
                raise UploadLockWait(f"锁丢失已回退（{doc_id}）: {e}") from e
            self._rollback_delete(rec, op_id, lock=lock)
            raise UploadConflictError(f"锁丢失已回滚（{doc_id}）: {e}") from e
        except (StorageUnavailableError, Exception) as e:
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(f"下架中断待恢复（{doc_id}）: {type(e).__name__}") from e
            if control is not None:
                self._rollback_delete(rec, op_id, dst=STATUS_DELETE_QUEUED,
                                      lock=lock, control=control)
                raise
            self._rollback_delete(rec, op_id, lock=lock)
            raise UploadConflictError(f"下架失败已回滚（{doc_id}）: {type(e).__name__}") from e
        finally:
            try:
                lock.release()
            except Exception:  # noqa: BLE001, S110
                pass
        return {"doc_id": doc_id, "status": STATUS_DELETED, "generation_id": gen_id}

    # ---------- 回滚 ----------
    def _rollback_upload(self, rec: DocumentRecord, storage_key: str, phase: str,
                         dst: str = STATUS_UPLOADING, *, lock=None,
                         control=None) -> None:
        """提交点前回滚：文件移出知识目录 + 清 journal + 回退（dst=上传中/排队）。"""
        try:
            if lock is not None:
                self._assert_write_authority(lock, control)
            kb_file = self._kb_uploads_dir() / storage_key
            if kb_file.exists():
                if lock is not None:
                    self._assert_write_authority(lock, control)
                self._stage.move_out_of_kb(self._kb_uploads_dir(), storage_key, rec.doc_id)
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status in (STATUS_INDEXING, STATUS_VALIDATING):
                if lock is not None:
                    self._assert_write_authority(lock, control)
                updated = self._doc.update_status(
                    cur.doc_id, cur.status, dst, cur.version,
                    operation_id="", pending_generation_id="",
                    error=f"回滚（phase={phase}）",
                )
                if updated < 0:
                    return
            # journal 必须最后清：若文档 CAS 失败或进程在 CAS 前崩溃，
            # 接管者仍能看到恢复依据，不能留下无 journal 的中间态。
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(rec.upload_id)
        except JobLeaseLost:
            raise
        except KbWriteLockError:
            # 锁已丢失：保留 journal/中间态，交给接管者前进或回滚。
            return
        except Exception:  # noqa: BLE001
            return

    def _rollback_delete(self, rec: DocumentRecord, op_id: str,
                         dst: str = STATUS_INDEXED, *, lock=None,
                         control=None) -> None:
        try:
            if lock is not None:
                self._assert_write_authority(lock, control)
            if self._stage.trash_exists(rec.storage_key):
                if lock is not None:
                    self._assert_write_authority(lock, control)
                self._stage.restore_from_trash(self._kb_uploads_dir(), rec.storage_key)
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status in (STATUS_DELETING,):
                if lock is not None:
                    self._assert_write_authority(lock, control)
                updated = self._doc.update_status(
                    cur.doc_id, STATUS_DELETING, dst,
                    cur.version, require_op=op_id, error="下架回滚",
                )
                if updated < 0:
                    return
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(f"del-{rec.doc_id}")
        except JobLeaseLost:
            raise
        except KbWriteLockError:
            return
        except Exception:  # noqa: BLE001
            return

    # ---------- 恢复机 ----------
    def _resume_after_recover(self, rec: DocumentRecord, upload_id: str) -> dict:
        lock = self._make_lock()
        try:
            lock.acquire(phase="recover")
        except (KbWriteLockError, KbWriteLockBackendError, LockHeldError) as e:
            raise UploadConflictError(f"知识库构建中: {e}") from e
        try:
            self._recover_all_journals(lock)
        finally:
            try:
                lock.release()
            except Exception:  # noqa: BLE001, S110
                pass
        rec2 = self._doc.get_by_upload_id(upload_id)
        if rec2 is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        if rec2.status == STATUS_INDEXED:
            return self._indexed_payload(rec2)
        if rec2.status == STATUS_UPLOADING:
            raise UploadIncomplete("已回退可重试（请重新 complete）")
        raise UploadRecovering(f"仍在处理（{rec2.status}），请稍后再试")

    def _recover_all_journals(self, lock, control=None) -> None:
        self._assert_write_authority(lock, control)
        for op_id in self._stage.journal_list():
            self._assert_write_authority(lock, control)
            self._recover_one(op_id, lock=lock, control=control)

    def _recover_one(self, op_id: str, *, lock=None, control=None) -> None:
        if op_id.startswith("del-"):
            self._recover_delete(op_id, lock=lock, control=control)
        else:
            self._recover_upload(op_id, lock=lock, control=control)

    def _recover_upload(self, op_id: str, *, lock=None, control=None) -> None:
        payload = self._stage.journal_read(op_id)
        rec = self._doc.get_by_upload_id(op_id)
        phase = (payload or {}).get("phase", "")
        if rec is None:
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(op_id)
            return
        target = (payload or {}).get("target") or ""
        gen_id = (payload or {}).get("generation_id") or ""
        if phase in ("", PH_PREPARED):
            # journal 在但记录未到 indexing（先 journal 后 CAS 崩溃点）→ 清 journal 回退
            kb_file = self._kb_uploads_dir() / rec.storage_key
            if kb_file.exists():
                if lock is not None:
                    self._assert_write_authority(lock, control)
                self._stage.move_out_of_kb(self._kb_uploads_dir(), rec.storage_key,
                                            rec.doc_id)
            if rec.status in (STATUS_VALIDATING, STATUS_INDEXING):
                if lock is not None:
                    self._assert_write_authority(lock, control)
                updated = self._doc.update_status(
                    rec.doc_id, rec.status, STATUS_UPLOADING, rec.version,
                    operation_id="", pending_generation_id="", error="恢复回退",
                )
                if updated < 0:
                    return
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(op_id)
            return
        if phase == PH_INDEX_BUILT:
            # alias 未动：可回滚
            kb_file = self._kb_uploads_dir() / rec.storage_key
            if kb_file.exists():
                if lock is not None:
                    self._assert_write_authority(lock, control)
                self._stage.move_out_of_kb(self._kb_uploads_dir(), rec.storage_key,
                                            rec.doc_id)
            if rec.status == STATUS_INDEXING:
                if lock is not None:
                    self._assert_write_authority(lock, control)
                updated = self._doc.update_status(
                    rec.doc_id, STATUS_INDEXING, STATUS_UPLOADING, rec.version,
                    operation_id="", pending_generation_id="", error="恢复回滚",
                )
                if updated < 0:
                    return
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(op_id)
            return
        if phase == PH_ACTIVATING:
            if lock is not None:
                self._assert_write_authority(lock, control)
            st = self._index_service.reconcile(self._backend())
            if not st.get("alias_known", st.get("alias_target") is not None):
                raise UploadRecovering("alias 状态暂不可确认，保留 journal 等待恢复")
            if st["alias_target"] == target:
                self._forward_upload(op_id, rec, target, gen_id,
                                     lock=lock, control=control)
                return
            if st["alias_target"] == "" or st["alias_target"] == st["pointer_target"]:
                # 回滚：先把文件移出知识库目录（与 PREPARED/INDEX_BUILT 回滚
                # 分支对称）——文件在步骤④已 move_into_kb，只回滚状态不移
                # 文件会留下孤儿，下次重建索引会把未提交文档扫进活动索引
                kb_file = self._kb_uploads_dir() / rec.storage_key
                if kb_file.exists():
                    if lock is not None:
                        self._assert_write_authority(lock, control)
                    self._stage.move_out_of_kb(self._kb_uploads_dir(), rec.storage_key,
                                                rec.doc_id)
                if rec.status == STATUS_INDEXING:
                    if lock is not None:
                        self._assert_write_authority(lock, control)
                    updated = self._doc.update_status(
                        rec.doc_id, STATUS_INDEXING, STATUS_UPLOADING, rec.version,
                        operation_id="", pending_generation_id="", error="恢复回滚(ACTIVATING)",
                    )
                    if updated < 0:
                        return
                if lock is not None:
                    self._assert_write_authority(lock, control)
                self._stage.journal_clear(op_id)
                return
            self._block_for_reconcile(lock=lock, control=control)
            return
        if phase in (PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED):
            self._forward_upload(op_id, rec, target, gen_id,
                                 lock=lock, control=control)
            return

    def _forward_upload(self, op_id: str, rec: DocumentRecord, target: str,
                        gen_id: str, *, lock=None, control=None) -> None:
        """提交点后恢复；所有非 fencing 异常都保持可恢复语义。"""
        try:
            self._forward_upload_impl(op_id, rec, target, gen_id,
                                      lock=lock, control=control)
        except (JobLeaseLost, UploadBlocked, UploadRecovering):
            raise
        except Exception as e:
            raise UploadRecovering(
                f"上传提交点后恢复失败（{rec.upload_id}）: {type(e).__name__}"
            ) from e

    def _forward_upload_impl(self, op_id: str, rec: DocumentRecord, target: str,
                             gen_id: str, *, lock=None, control=None) -> None:
        """提交点后只前进：activate（幂等）→ CAS indexed。"""
        backend = self._backend()
        info = GenerationInfo(
            generation_id=gen_id, target=target,
            embedding_model=settings.embedding_model,
        )
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._index_service.activate_alias(backend, info)
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._assert_alias_confirmed(info)
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._index_service.activate_pointer(backend, info)
        if lock is not None:
            self._assert_write_authority(lock, control)
        cur = self._doc.get(rec.doc_id)
        if cur is not None and cur.status == STATUS_INDEXING:
            if lock is not None:
                self._assert_write_authority(lock, control)
            ok = self._doc.update_status(cur.doc_id, STATUS_INDEXING, STATUS_INDEXED,
                                         cur.version, generation_id=gen_id,
                                         pending_generation_id="", error="")
            if ok < 0:
                raise UploadRecovering("元数据提交待恢复（人工）")
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._stage.journal_clear(op_id)

    def _recover_delete(self, op_id: str, *, lock=None, control=None) -> None:
        payload = self._stage.journal_read(op_id)
        doc_id = op_id[len("del-"):]
        rec = self._doc.get(doc_id)
        phase = (payload or {}).get("phase", "")
        if rec is None or phase == "":
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(op_id)
            return
        target = (payload or {}).get("target") or ""
        gen_id = (payload or {}).get("generation_id") or ""
        if phase in (PH_PREPARED, PH_INDEX_BUILT):
            if rec.status == STATUS_DELETING:
                try:
                    if lock is not None:
                        self._assert_write_authority(lock, control)
                    if self._stage.trash_exists(rec.storage_key):
                        if lock is not None:
                            self._assert_write_authority(lock, control)
                        self._stage.restore_from_trash(self._kb_uploads_dir(), rec.storage_key)
                    if lock is not None:
                        self._assert_write_authority(lock, control)
                    updated = self._doc.update_status(
                        rec.doc_id, STATUS_DELETING, STATUS_INDEXED,
                        rec.version, error=f"恢复回滚（{phase}）",
                    )
                    if updated < 0:
                        return
                except JobLeaseLost:
                    raise
                except Exception:  # noqa: BLE001
                    return
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(op_id)
            return
        if phase == PH_ACTIVATING:
            if lock is not None:
                self._assert_write_authority(lock, control)
            st = self._index_service.reconcile(self._backend())
            if not st.get("alias_known", st.get("alias_target") is not None):
                raise UploadRecovering("alias 状态暂不可确认，保留 journal 等待恢复")
            if st["alias_target"] == target:
                self._forward_delete(op_id, rec, target, gen_id,
                                     lock=lock, control=control)
                return
            if st["alias_target"] == "" or st["alias_target"] == st["pointer_target"]:
                self._rollback_delete_static(rec, op_id, lock=lock, control=control)
                return
            self._block_for_reconcile(lock=lock, control=control)
            return
        if phase in (PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED):
            self._forward_delete(op_id, rec, target, gen_id,
                                 lock=lock, control=control)
            return

    def _forward_delete(self, op_id: str, rec: DocumentRecord, target: str,
                        gen_id: str, *, lock=None, control=None) -> None:
        """下架提交点后恢复；失败必须保留 journal 供下次继续。"""
        try:
            self._forward_delete_impl(op_id, rec, target, gen_id,
                                      lock=lock, control=control)
        except (JobLeaseLost, UploadBlocked, UploadRecovering):
            raise
        except Exception as e:
            raise UploadRecovering(
                f"下架提交点后恢复失败（{rec.doc_id}）: {type(e).__name__}"
            ) from e

    def _forward_delete_impl(self, op_id: str, rec: DocumentRecord, target: str,
                             gen_id: str, *, lock=None, control=None) -> None:
        backend = self._backend()
        info = GenerationInfo(
            generation_id=gen_id, target=target,
            embedding_model=settings.embedding_model,
        )
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._index_service.activate_alias(backend, info)
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._assert_alias_confirmed(info)
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._index_service.activate_pointer(backend, info)
        if lock is not None:
            self._assert_write_authority(lock, control)
        cur = self._doc.get(rec.doc_id)
        if cur is not None and cur.status == STATUS_DELETING:
            if lock is not None:
                self._assert_write_authority(lock, control)
            ok = self._doc.update_status(cur.doc_id, STATUS_DELETING, STATUS_DELETED,
                                         cur.version, generation_id=gen_id, error="")
            if ok < 0:
                raise UploadRecovering("下架元数据提交待恢复（人工）")
        if lock is not None:
            self._assert_write_authority(lock, control)
        self._stage.journal_clear(op_id)

    def _rollback_delete_static(self, rec: DocumentRecord, op_id: str,
                                *, lock=None, control=None) -> None:
        try:
            if lock is not None:
                self._assert_write_authority(lock, control)
            if self._stage.trash_exists(rec.storage_key):
                if lock is not None:
                    self._assert_write_authority(lock, control)
                self._stage.restore_from_trash(self._kb_uploads_dir(), rec.storage_key)
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status == STATUS_DELETING:
                if lock is not None:
                    self._assert_write_authority(lock, control)
                updated = self._doc.update_status(
                    cur.doc_id, STATUS_DELETING, STATUS_INDEXED,
                    cur.version, error="恢复回滚(ACTIVATING)",
                )
                if updated < 0:
                    return
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._stage.journal_clear(op_id)
        except JobLeaseLost:
            raise
        except Exception:  # noqa: BLE001
            return

    def _block_for_reconcile(self, *, lock=None, control=None) -> None:
        try:
            if lock is not None:
                self._assert_write_authority(lock, control)
            self._control.set(BLOCKED_KEY, BLOCKED_VALUE)
        except StorageUnavailableError:
            pass  # 下次恢复仍会重判
        raise UploadBlocked("alias 指向未知代，已阻塞知识库写入，请人工 reconcile")

    # ---------- 辅助 ----------
    def _assert_not_blocked(self) -> None:
        if self._control.get(BLOCKED_KEY):
            raise UploadBlocked("知识库被全局阻塞（alias 不一致），请先人工 reconcile")

    def _assert_alias_confirmed(self, info: GenerationInfo) -> None:
        """提交点核验：仅 ES 后端有 alias 概念（numpy/chroma 的 activate 是 no-op）。"""
        if self._backend() != "es":
            return
        st = self._index_service.reconcile("es")
        if st["alias_target"] != info.target:
            raise UploadConflictError(
                f"alias 确认失败（目标 {info.target}，实际 {st['alias_target']}）"
            )

    def _journal(self, op_id: str, payload: dict) -> None:
        self._stage.journal_write(op_id, payload)

    def _indexed_payload(self, rec: DocumentRecord) -> dict:
        return {
            "doc_id": rec.doc_id, "upload_id": rec.upload_id, "status": rec.status,
            "generation_id": rec.generation_id,
            "indexed_chunk_count": rec.indexed_chunk_count,
        }

    # ---------- GC（须持 kb_write 锁调用；上限处理） ----------
    def gc(self, limit: int = 100) -> int:
        now = datetime.now()  # noqa: DTZ005
        handled = 0

        for rec in self._doc.list_before_status_changed(
            (STATUS_DELETED,),
            now - timedelta(days=settings.kb_upload_retention_trash_days),
        ):
            if handled >= limit:
                return handled
            self._stage.trash_delete(rec.storage_key)
            handled += 1
        for rec in self._doc.list_before_status_changed(
            (STATUS_DELETED,),
            now - timedelta(days=settings.kb_upload_retention_original_days),
        ):
            if handled >= limit:
                return handled
            self._originals.delete(rec.doc_id)
            handled += 1
        for rec in self._doc.list_before_status_changed(
            (STATUS_FAILED,),
            now - timedelta(days=settings.kb_upload_retention_failed_days),
        ):
            if handled >= limit:
                return handled
            self._stage.cleanup_failed(rec.doc_id)
            self._stage.cleanup_staging(rec.upload_id)
            self._originals.delete(rec.doc_id)
            handled += 1
        for rec in self._doc.list_before_status_changed(
            (STATUS_CANCELLED,), now - timedelta(minutes=1),
        ):
            if handled >= limit:
                return handled
            self._originals.delete(rec.doc_id)
            handled += 1
        return handled


# ============================================================
# 模块级辅助
# ============================================================
class UploadBusyState(UploadConflictError):
    pass


class UploadRecovering(UploadConflictError):
    """处理中断，journal 已保留（提交点后）——客户端稍后重试 complete。"""


def _require_owner(rec: DocumentRecord, uploader: str) -> None:
    if rec.uploader != uploader:
        raise UploadForbidden("该操作仅限上传者本人（upload_id 不会跨操作者接管）")


def _slug_from(filename: str) -> str:
    from app.evolution.sanitizer import make_slug

    stem = Path(filename).stem
    return make_slug(stem or "doc", 40) or "doc"


def _format_of(filename: str) -> str:
    from app.agent.rag.parsers import SUPPORTED_SUFFIXES

    allowed = tuple(sorted({s.lstrip(".") for s in SUPPORTED_SUFFIXES}))
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext == "doc":
        raise UploadError(
            "不支持的文件格式: doc（旧版 Word 格式，请先转换为 .docx 后上传）"
        )
    if ext not in allowed:
        raise UploadError(f"不支持的文件格式: {ext or '(无后缀)'}（支持 {', '.join(allowed)}）")
    return ext


def _validate_upload_id(upload_id: str) -> str:
    if not _UPLOAD_ID_RE.match(upload_id or ""):
        raise UploadError("upload_id 只允许字母/数字开头，[A-Za-z0-9._-]，≤64 字符")
    return upload_id


def _clamp_chunk_size(chunk_size: int) -> int:
    lo = settings.kb_upload_min_chunk_size
    hi = settings.kb_upload_max_chunk_size
    if not (lo <= chunk_size <= hi):
        raise UploadError(f"chunk_size 必须在 [{lo}, {hi}] 字节之间")
    return chunk_size


def _expected_chunk_size(session: UploadSession, seq: int) -> int:
    if seq < session.total_chunks - 1:
        return session.chunk_size
    return session.size_bytes - (session.total_chunks - 1) * session.chunk_size


def _magic_check(data: bytes, fmt: str) -> None:
    if fmt == "pdf" and not data.startswith(b"%PDF-"):
        raise UploadParseFailed("PDF 魔数校验失败（非 PDF 文件）")
    if fmt == "docx" and not data.startswith(b"PK\x03\x04"):
        raise UploadParseFailed("DOCX 魔数校验失败（非 zip 文档）")


def _zip_limits(data: bytes, fmt: str) -> None:
    if fmt != "docx":
        return
    import zipfile
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            infos = zf.infolist()
            if len(infos) > settings.kb_upload_max_zip_entries:
                raise UploadParseFailed(f"zip 条目 {len(infos)} 超过上限")
            total = sum(i.file_size for i in infos)
            packed = sum(i.compress_size for i in infos) or 1
            if total > settings.kb_upload_max_zip_bytes:
                raise UploadParseFailed("zip 累计解压字节超过上限（zip bomb 防护）")
            if total / packed > settings.kb_upload_max_zip_ratio:
                raise UploadParseFailed("zip 压缩比超过上限（zip bomb 防护）")
    except UploadParseFailed:
        raise
    except Exception as e:
        raise UploadParseFailed(f"zip 结构检查失败: {e}") from e


def _page_limits(data: bytes, fmt: str) -> None:
    if fmt != "pdf" or not data.startswith(b"%PDF-"):
        return
    from io import BytesIO

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        n = len(reader.pages)
        if n > settings.kb_upload_max_pages:
            raise UploadParseFailed(f"PDF 页数 {n} 超过上限 {settings.kb_upload_max_pages}")
    except UploadParseFailed:
        raise
    except Exception:  # noqa: BLE001
        return

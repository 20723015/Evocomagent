"""KB 文档上传编排服务（v7 冻结）：Redis 断点续传 + MySQL 元数据 + ES 版本化重建。

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
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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
    DocumentRecord,
    KbControlStore,
    SqlDocumentStore,
    STATUS_CANCELLED,
    STATUS_DELETED,
    STATUS_DELETING,
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_INDEXING,
    STATUS_UPLOADING,
    STATUS_VALIDATING,
)
from app.stores.upload_state import (
    SESSION_CANCELLED,
    SESSION_SEALED,
    UploadSession,
    STATE_READY,
    build_upload_state_store,
)
from app.stores.upload_storage import (
    OriginalStore,
    StageDir,
    UploadStorageError,
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
        stage: Optional[StageDir] = None,
        originals: Optional[OriginalStore] = None,
        engine=None,
        redis=None,
        kb_root=None,
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

    # ---------- 基础 ----------
    def _make_lock(self):
        return get_kb_write_lock(engine=self._engine, redis=self._redis)

    def _kb_uploads_dir(self) -> Path:
        return self._kb_root / "uploads"

    def _backend(self) -> str:
        return (settings.rag_backend or "numpy").lower()

    def _new_operation(self) -> str:
        return uuid.uuid4().hex

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
            expires_at=(datetime.now() + timedelta(seconds=settings.kb_upload_session_ttl))
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
        except Exception as e:  # noqa: BLE001 —— 发布失败：撤回声明（同 token only）
            self._state.drop_chunk_record(upload_id, seq, token)
            raise UploadError(f"分片发布失败: {e}") from e
        return {"upload_id": upload_id, "seq": seq,
                "received": self._state.ready_count(upload_id),
                "total_chunks": session.total_chunks}

    # ---------- 状态 ----------
    def get_status(self, upload_id: str) -> dict:
        session = self._state.get(upload_id)
        if session is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        rec = self._doc.get_by_upload_id(upload_id)
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
        self._state.mark_closed(upload_id, SESSION_CANCELLED)
        if rec.status in (STATUS_UPLOADING, STATUS_VALIDATING):
            self._doc.update_status(rec.doc_id, rec.status, STATUS_CANCELLED, rec.version)
        self._chunks.delete_chunks(upload_id)
        self._state.delete(upload_id)
        return {"upload_id": upload_id, "status": STATUS_CANCELLED}

    # ---------- complete ----------
    def complete(self, upload_id: str, uploader: str) -> dict:
        rec = self._doc.get_by_upload_id(upload_id)
        if rec is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        _require_owner(rec, uploader)

        if rec.status == STATUS_INDEXED:
            return self._indexed_payload(rec)
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
                       op_id: str, merged: bytes, text: str) -> dict:
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
            lock.acquire(phase="upload")
        except (KbWriteLockError, KbWriteLockBackendError, LockHeldError) as e:
            self._doc.update_status(rec.doc_id, STATUS_VALIDATING, STATUS_UPLOADING,
                                    validating_version, operation_id="",
                                    error=f"锁被持有: {type(e).__name__}")
            raise UploadConflictError(f"知识库构建中: {e}") from e
        try:
            # 持锁后、接受任何新写前：先恢复历史事务，再检查全局阻塞
            self._recover_all_journals(lock)
            self._assert_not_blocked()

            # ③ 先 journal 后 CAS（消除「indexing 无 journal」窗口）
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_PREPARED,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_PREPARED
            ok = self._doc.update_status(rec.doc_id, STATUS_VALIDATING, STATUS_INDEXING,
                                         validating_version, require_op=op_id,
                                         pending_generation_id=gen_id, error="")
            if ok < 0:
                self._stage.journal_clear(rec.upload_id)
                raise UploadBusyState("状态已变化，无法进入 indexing")

            # ④ 原件（提交点前）→ 移入知识库（同卷 rename）
            self._originals.put(rec.doc_id, rec.format, merged)
            lock.assert_held()
            self._stage.move_into_kb(rec.upload_id, self._kb_uploads_dir(), storage_key)

            # ⑤ 构建（strict：任何源文件解析失败即中止）
            lock.assert_held()
            built = self._index_service.build(backend, generation_id=gen_id)
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_INDEX_BUILT,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_INDEX_BUILT

            # ⑥ 提交点：ACTIVATING → alias → assert → 查真实 alias
            lock.assert_held()
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_ACTIVATING,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_ACTIVATING
            self._index_service.activate_alias(backend, built)
            lock.assert_held()  # 紧接 alias 操作（失锁 → 保持 ACTIVATING 等恢复）
            self._assert_alias_confirmed(built)

            # ⑦ 指针 → 元数据提交
            self._index_service.activate_pointer(backend, built)
            self._journal(rec.upload_id, {
                "op": "upload", "phase": PH_POINTER_UPDATED,
                "doc_id": rec.doc_id, "storage_key": storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_POINTER_UPDATED
            ok = self._doc.update_status(
                rec.doc_id, STATUS_INDEXING, STATUS_INDEXED, validating_version + 1,
                require_op=op_id, generation_id=gen_id,
                indexed_chunk_count=self._index_service.last_built_size,
                pending_generation_id="", error="",
            )
            if ok < 0:
                raise UploadRecovering("元数据提交失败，journal 保留等待恢复")

            # ⑧ 成功收尾：清 journal + 分片/会话（最后清理）
            self._stage.journal_clear(rec.upload_id)
            try:
                self._chunks.delete_chunks(rec.upload_id)
                self._state.delete(rec.upload_id)
            except Exception:  # noqa: BLE001 —— 清理失败不影响结果
                pass
            return self._indexed_payload(self._doc.get(rec.doc_id) or rec)
        except UploadError:
            raise
        except (KbWriteLockError,) as e:
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(f"处理中断待恢复（{rec.upload_id}）: {e}") from e
            self._rollback_upload(rec, storage_key, phase)
            raise UploadConflictError(f"锁已丢失，已回滚可重试: {e}") from e
        except (StorageUnavailableError, Exception) as e:  # noqa: BLE001
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(
                    f"处理中断待恢复（{rec.upload_id}）: {type(e).__name__}") from e
            self._rollback_upload(rec, storage_key, phase)
            raise UploadConflictError(
                f"构建失败已回滚，可重试（{type(e).__name__}: {e}）") from e
        finally:
            try:
                lock.release()
            except Exception:  # noqa: BLE001
                pass

    # ---------- 下架 ----------
    def delete_document(self, doc_id: str, uploader: str) -> dict:
        rec = self._doc.get(doc_id)
        if rec is None:
            raise UploadNotFound(f"文档不存在: {doc_id}")
        _require_owner(rec, uploader)
        if rec.status != STATUS_INDEXED:
            raise UploadConflictError(f"仅已入库文档可下架（当前 {rec.status}）")

        op_id = self._new_operation()
        ok = self._doc.update_status(rec.doc_id, STATUS_INDEXED, STATUS_DELETING,
                                     rec.version, operation_id=op_id)
        if ok < 0:
            raise UploadConflictError("文档状态已变化")
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
            lock.acquire(phase="delete")
        except (KbWriteLockError, KbWriteLockBackendError, LockHeldError) as e:
            self._doc.update_status(rec.doc_id, STATUS_DELETING, STATUS_INDEXED,
                                    rec.version + 1, require_op=op_id,
                                    error=f"锁被持有: {type(e).__name__}")
            raise UploadConflictError(f"知识库构建中: {e}") from e
        try:
            self._recover_all_journals(lock)
            self._assert_not_blocked()
            # 先 journal 再移动（下架可恢复）
            self._journal(journal_op, {
                "op": "delete", "phase": PH_PREPARED,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_PREPARED
            lock.assert_held()
            self._stage.move_to_trash(self._kb_uploads_dir(), rec.storage_key)
            lock.assert_held()
            built = self._index_service.build(backend, generation_id=gen_id)
            self._journal(journal_op, {
                "op": "delete", "phase": PH_INDEX_BUILT,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_INDEX_BUILT
            lock.assert_held()
            self._journal(journal_op, {
                "op": "delete", "phase": PH_ACTIVATING,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_ACTIVATING
            self._index_service.activate_alias(backend, built)
            lock.assert_held()
            self._assert_alias_confirmed(built)
            self._index_service.activate_pointer(backend, built)
            self._journal(journal_op, {
                "op": "delete", "phase": PH_POINTER_UPDATED,
                "doc_id": doc_id, "storage_key": rec.storage_key,
                "generation_id": gen_id, "target": info.target,
            })
            phase = PH_POINTER_UPDATED
            ok = self._doc.update_status(rec.doc_id, STATUS_DELETING, STATUS_DELETED,
                                         rec.version + 1, require_op=op_id,
                                         generation_id=gen_id, error="")
            if ok < 0:
                raise UploadRecovering("下架元数据提交失败，journal 保留等待恢复")
            # 成功收尾：清 journal（残留会被恢复机误判为 pending 代并重放旧代激活）
            self._stage.journal_clear(journal_op)
        except UploadError:
            raise
        except (KbWriteLockError,) as e:
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(f"下架中断待恢复（{doc_id}）: {e}") from e
            self._rollback_delete(rec, op_id)
            raise UploadConflictError(f"锁丢失已回滚（{doc_id}）: {e}") from e
        except (StorageUnavailableError, Exception) as e:  # noqa: BLE001
            if phase in _ALIAS_PHASES:
                raise UploadRecovering(f"下架中断待恢复（{doc_id}）: {type(e).__name__}") from e
            self._rollback_delete(rec, op_id)
            raise UploadConflictError(f"下架失败已回滚（{doc_id}）: {type(e).__name__}") from e
        finally:
            try:
                lock.release()
            except Exception:  # noqa: BLE001
                pass
        return {"doc_id": doc_id, "status": STATUS_DELETED, "generation_id": gen_id}

    # ---------- 回滚 ----------
    def _rollback_upload(self, rec: DocumentRecord, storage_key: str, phase: str) -> None:
        """提交点前回滚：文件移出知识目录 + 清 journal + 回退 uploading。"""
        try:
            kb_file = self._kb_uploads_dir() / storage_key
            if kb_file.exists():
                self._stage.move_out_of_kb(self._kb_uploads_dir(), storage_key, rec.doc_id)
            self._stage.journal_clear(rec.upload_id)
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status in (STATUS_INDEXING, STATUS_VALIDATING):
                self._doc.update_status(
                    cur.doc_id, cur.status, STATUS_UPLOADING, cur.version,
                    operation_id="", pending_generation_id="",
                    error=f"回滚（phase={phase}）",
                )
        except Exception:  # noqa: BLE001 —— 回滚尽力而为；残留由恢复机接管
            return

    def _rollback_delete(self, rec: DocumentRecord, op_id: str) -> None:
        try:
            if self._stage.trash_exists(rec.storage_key):
                self._stage.restore_from_trash(self._kb_uploads_dir(), rec.storage_key)
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status == STATUS_DELETING:
                self._doc.update_status(cur.doc_id, STATUS_DELETING, STATUS_INDEXED,
                                        cur.version, require_op=op_id, error="下架回滚")
            self._stage.journal_clear(f"del-{rec.doc_id}")
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
            except Exception:  # noqa: BLE001
                pass
        rec2 = self._doc.get_by_upload_id(upload_id)
        if rec2 is None:
            raise UploadNotFound(f"上传会话不存在: {upload_id}")
        if rec2.status == STATUS_INDEXED:
            return self._indexed_payload(rec2)
        if rec2.status == STATUS_UPLOADING:
            raise UploadIncomplete("已回退可重试（请重新 complete）")
        raise UploadRecovering(f"仍在处理（{rec2.status}），请稍后再试")

    def _recover_all_journals(self, lock) -> None:
        for op_id in self._stage.journal_list():
            self._recover_one(op_id)

    def _recover_one(self, op_id: str) -> None:
        if op_id.startswith("del-"):
            self._recover_delete(op_id)
        else:
            self._recover_upload(op_id)

    def _recover_upload(self, op_id: str) -> None:
        payload = self._stage.journal_read(op_id)
        rec = self._doc.get_by_upload_id(op_id)
        phase = (payload or {}).get("phase", "")
        if rec is None:
            self._stage.journal_clear(op_id)
            return
        target = (payload or {}).get("target") or ""
        gen_id = (payload or {}).get("generation_id") or ""
        if phase in ("", PH_PREPARED):
            # journal 在但记录未到 indexing（先 journal 后 CAS 崩溃点）→ 清 journal 回退
            self._stage.journal_clear(op_id)
            if rec.status in (STATUS_VALIDATING, STATUS_INDEXING):
                self._doc.update_status(
                    rec.doc_id, rec.status, STATUS_UPLOADING, rec.version,
                    operation_id="", pending_generation_id="", error="恢复回退",
                )
            return
        if phase == PH_INDEX_BUILT:
            # alias 未动：可回滚
            self._stage.journal_clear(op_id)
            if rec.status == STATUS_INDEXING:
                self._doc.update_status(
                    rec.doc_id, STATUS_INDEXING, STATUS_UPLOADING, rec.version,
                    operation_id="", pending_generation_id="", error="恢复回滚",
                )
            return
        if phase == PH_ACTIVATING:
            st = self._index_service.reconcile(self._backend())
            if st["alias_target"] == target:
                self._forward_upload(op_id, rec, target, gen_id)
                return
            if st["alias_target"] == "" or st["alias_target"] == st["pointer_target"]:
                self._stage.journal_clear(op_id)
                if rec.status == STATUS_INDEXING:
                    self._doc.update_status(
                        rec.doc_id, STATUS_INDEXING, STATUS_UPLOADING, rec.version,
                        operation_id="", pending_generation_id="", error="恢复回滚(ACTIVATING)",
                    )
                return
            self._block_for_reconcile()
            return
        if phase in (PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED):
            self._forward_upload(op_id, rec, target, gen_id)
            return

    def _forward_upload(self, op_id: str, rec: DocumentRecord, target: str,
                        gen_id: str) -> None:
        """提交点后只前进：activate（幂等）→ CAS indexed。"""
        backend = self._backend()
        info = GenerationInfo(
            generation_id=gen_id, target=target,
            embedding_model=settings.embedding_model,
        )
        self._index_service.activate_alias(backend, info)
        self._index_service.activate_pointer(backend, info)
        cur = self._doc.get(rec.doc_id)
        if cur is not None and cur.status == STATUS_INDEXING:
            ok = self._doc.update_status(cur.doc_id, STATUS_INDEXING, STATUS_INDEXED,
                                         cur.version, generation_id=gen_id,
                                         pending_generation_id="", error="")
            if ok < 0:
                raise UploadRecovering("元数据提交待恢复（人工）")
        self._stage.journal_clear(op_id)

    def _recover_delete(self, op_id: str) -> None:
        payload = self._stage.journal_read(op_id)
        doc_id = op_id[len("del-"):]
        rec = self._doc.get(doc_id)
        phase = (payload or {}).get("phase", "")
        if rec is None or phase == "":
            self._stage.journal_clear(op_id)
            return
        target = (payload or {}).get("target") or ""
        gen_id = (payload or {}).get("generation_id") or ""
        if phase in (PH_PREPARED, PH_INDEX_BUILT):
            if rec.status == STATUS_DELETING:
                try:
                    if self._stage.trash_exists(rec.storage_key):
                        self._stage.restore_from_trash(self._kb_uploads_dir(), rec.storage_key)
                    self._doc.update_status(rec.doc_id, STATUS_DELETING, STATUS_INDEXED,
                                            rec.version, error=f"恢复回滚（{phase}）")
                except Exception:  # noqa: BLE001 —— 尽力
                    pass
            self._stage.journal_clear(op_id)
            return
        if phase == PH_ACTIVATING:
            st = self._index_service.reconcile(self._backend())
            if st["alias_target"] == target:
                self._forward_delete(op_id, rec, target, gen_id)
                return
            if st["alias_target"] == "" or st["alias_target"] == st["pointer_target"]:
                self._rollback_delete_static(rec, op_id)
                return
            self._block_for_reconcile()
            return
        if phase in (PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED):
            self._forward_delete(op_id, rec, target, gen_id)
            return

    def _forward_delete(self, op_id: str, rec: DocumentRecord, target: str,
                        gen_id: str) -> None:
        backend = self._backend()
        info = GenerationInfo(
            generation_id=gen_id, target=target,
            embedding_model=settings.embedding_model,
        )
        self._index_service.activate_alias(backend, info)
        self._index_service.activate_pointer(backend, info)
        cur = self._doc.get(rec.doc_id)
        if cur is not None and cur.status == STATUS_DELETING:
            ok = self._doc.update_status(cur.doc_id, STATUS_DELETING, STATUS_DELETED,
                                         cur.version, generation_id=gen_id, error="")
            if ok < 0:
                raise UploadRecovering("下架元数据提交待恢复（人工）")
        self._stage.journal_clear(op_id)

    def _rollback_delete_static(self, rec: DocumentRecord, op_id: str) -> None:
        try:
            if self._stage.trash_exists(rec.storage_key):
                self._stage.restore_from_trash(self._kb_uploads_dir(), rec.storage_key)
            cur = self._doc.get(rec.doc_id)
            if cur is not None and cur.status == STATUS_DELETING:
                self._doc.update_status(cur.doc_id, STATUS_DELETING, STATUS_INDEXED,
                                        cur.version, error="恢复回滚(ACTIVATING)")
            self._stage.journal_clear(op_id)
        except Exception:  # noqa: BLE001
            return

    def _block_for_reconcile(self) -> None:
        try:
            self._control.set(BLOCKED_KEY, BLOCKED_VALUE)
        except StorageUnavailableError:  # noqa: BLE001 —— blocked 记录失败：journal 仍在，
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
        now = datetime.now()
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
    if fmt not in ("docx", "doc"):
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
    except Exception as e:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001 —— 页数检查失败放行给子进程解析（那里 fail-fast）
        return

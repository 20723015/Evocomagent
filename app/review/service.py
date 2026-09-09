"""ReviewService：知识审核的统一入口（编辑/拒绝/批准）。

修复的三个一致性问题：
1. **Ledger 实例缓存漂移**：审核后台/导入命令各自构建 Ledger，直接改内存态
   会覆盖其他实例的落盘更新——所有写操作先 acquire 单写锁、再 ``ledger.reload()``
   读到磁盘正本，改完落盘、最后释放锁。
2. **审核并发覆盖**：编辑/批准携带 ``expected_revision``（pending 条目的
   乐观锁版本，初始 0，每次编辑 +1）；不匹配 → ``revision_conflict``（409），
   杜绝两个审核员互相覆盖。
3. **发布结果靠 CLI 返回码猜测**：批准在锁内直接调用
   ``pipeline.publish_approved(..., lock_held=True)``，按 ledger 终态返回
   结构化状态（published / duplicate_rejected / not_found /
   revision_conflict / publish_failed），绝不虚报 published=true。

门禁：``SELF_EVOLVE_ENABLED=false`` 时编辑/拒绝可用（候选可继续沉淀与整理），
批准发布明确返回 ``publish_forbidden``（409）——采集与发布解耦。
"""

from __future__ import annotations

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.review.service")

# 审核拒绝的持久化原因（ledger.rejected_details）
REASON_MANUAL_REJECT = "manual_review_reject"
REASON_DUPLICATE_ON_APPROVE = "duplicate_on_approve"


class ReviewLockHeld(RuntimeError):
    """单写锁被其他发布者持有（映射 503，稍后重试）。"""


class ReviewService:
    """依赖注入：ledger/lock/pipeline 来自同一次 _make_services 装配。"""

    def __init__(self, *, ledger, lock, pipeline):
        self._ledger = ledger
        self._lock = lock
        self._pipeline = pipeline

    # ---------- 内部 ----------
    def _acquire(self, phase: str) -> None:
        try:
            self._lock.acquire(phase=phase)
        except Exception as e:

            raise ReviewLockHeld(f"审核锁被占用（{type(e).__name__}），请稍后重试") from e

    @staticmethod
    def _entry_revision(entry: dict | None) -> int:
        return int((entry or {}).get("revision", 0) or 0)

    def _terminal_result(self, candidate_id: str) -> dict | None:
        """按磁盘 ledger 正本读取批准后的终态。

        发布事务可能在抛出异常前已经由 journal 恢复完成（例如 alias/
        pointer 已切换后 ledger 补记成功）。此时 pipeline 的异常不能覆盖
        ledger 的事实；reload 后有明确终态就直接返回给审核端。
        """
        self._ledger.reload()
        published = self._ledger.published()
        if candidate_id in published:
            return {"status": "published", "filename": published[candidate_id]}
        if self._ledger.rejected_reason(candidate_id) == \
                REASON_DUPLICATE_ON_APPROVE:
            return {
                "status": "duplicate_rejected",
                "detail": "审核时发现重复，候选已终结（rejected）",
            }
        return None

    # ---------- 编辑 ----------
    def edit(self, candidate_id: str, question: str, answer: str,
             expected_revision: int) -> dict:
        """编辑规范问题/标准答案；复扫校验先行，锁内 reload + 乐观锁落盘。"""
        from app.evolution.sanitizer import (
            has_injection,
            has_pii,
            normalize_answer,
            normalize_question,
        )

        q = normalize_question(question)
        a = normalize_answer(answer)
        if not q or not a:
            return {"status": "invalid_content",
                    "detail": "编辑后问题或答案不合法/过短（长度闸门）"}
        if has_injection(q + "\n" + a) or has_pii(q + "\n" + a):
            return {"status": "sensitive_content",
                    "detail": "编辑内容命中 PII/注入检测，已拒绝保存"}
        self._acquire("review-edit")
        try:
            self._ledger.reload()
            entry = self._ledger.pending_entry(candidate_id)
            if entry is None:
                return {"status": "not_found"}
            updated, revision = self._ledger.update_pending_edit(
                candidate_id, q, a, expected_revision=expected_revision,
            )
            if updated is None:
                from app.observability.metrics import record_review_conflict

                record_review_conflict()
                return {"status": "revision_conflict",
                        "current_revision": revision}
            return {"status": "edited", "revision": revision}
        finally:
            self._lock.release()

    # ---------- 拒绝 ----------
    def reject(self, candidate_id: str,
               expected_revision: int | None = None) -> dict:
        """手工拒绝：出 pending + ``manual_review_reject`` 原因持久化。"""
        self._acquire("review-reject")
        try:
            self._ledger.reload()
            entry = self._ledger.pending_entry(candidate_id)
            if entry is None:
                return {"status": "not_found"}
            if expected_revision is not None and int(expected_revision) != \
                    self._entry_revision(entry):
                from app.observability.metrics import record_review_conflict

                record_review_conflict()
                return {"status": "revision_conflict",
                        "current_revision": self._entry_revision(entry)}
            self._ledger.mark_rejected(candidate_id, reason=REASON_MANUAL_REJECT)
            return {"status": "rejected"}
        finally:
            self._lock.release()

    # ---------- 批准 ----------
    def approve(self, candidate_id: str, expected_revision: int) -> dict:
        """批准发布：锁内读取最终版本 → 去重 → 发布 → 清账，结构化状态返回。

        重复候选由 publish_approved 记 ``duplicate_on_approve`` 出 pending；
        本方法按 ledger 终态判定，绝不把重复/失败虚报为 published。
        """
        if not settings.self_evolve_enabled:
            return {"status": "publish_forbidden",
                    "detail": "最终发布被安全开关禁止（SELF_EVOLVE_ENABLED=false）；"
                              "候选保留在 pending，可继续编辑"}
        self._acquire("review-approve")
        try:
            self._ledger.reload()
            entry = self._ledger.pending_entry(candidate_id)
            if entry is None:
                return {"status": "not_found"}
            current = self._entry_revision(entry)
            if int(expected_revision) != current:
                from app.observability.metrics import record_review_conflict

                record_review_conflict()
                return {"status": "revision_conflict", "current_revision": current}
            candidate = self._ledger.approve(candidate_id)
            if candidate is None:  # reload 与 approve 之间的竞态（理论上不可达）
                return {"status": "not_found"}
            try:
                report = self._pipeline.publish_approved([candidate],
                                                         lock_held=True)
            except Exception as e:  # noqa: BLE001 - publisher boundary
                # publish_approved 的异常收尾可能已完成 forward/ledger-only
                # 恢复；先看 ledger 正本，避免把已完成的发布虚报为失败。
                try:
                    terminal = self._terminal_result(candidate_id)
                except Exception:  # noqa: BLE001 - ledger read is best effort
                    terminal = None
                if terminal is not None:
                    return terminal
                log.warning("review approve publish_failed cid=%s err=%s",
                            candidate_id, type(e).__name__)
                return {"status": "publish_failed",
                        "detail": f"发布事务失败（{type(e).__name__}），候选保留在 pending"}
            _ = report
            terminal = self._terminal_result(candidate_id)
            if terminal is not None:
                return terminal
            return {"status": "publish_failed",
                    "detail": "候选未发布且未终结（如容量截断），保留在 pending"}
        finally:
            self._lock.release()

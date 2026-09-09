"""人工知识批量发布 Worker：租约 fencing + 全局写锁 + 会话栅栏 + 可前进恢复。

并发正确性（010）：
- 构造期强制注入 SemanticDedupService——发布路径不可能再跳过去重；
- 发布内容取自审批快照（items 不可变列），批准什么就发什么；
- 动手前取会话栅栏（GET_LOCK，字典序），激活前逐项条件复核（候选/快照/
  来源版本），结算走 CAS 条件更新——superseded 在任何阶段都不被覆盖；
- CAS miss（激活后漂移）与更高来源版本 → 补偿下架入队；
- build 后存在性探针（逐文档自检索），失败 → INDEX_BUILT 回滚重建重试。
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.evolution.authority import AUTHORITATIVE, authority_kind
from app.evolution.fence import FenceLost
from app.evolution.generation import GenerationInfo, new_generation_id
from app.evolution.human_store import (
    BATCH_OP_PUBLISH,
    BATCH_OP_RETIRE,
    CAND_PUBLISH_QUEUED,
    CAND_PUBLISHED,
    CAND_REJECTED,
    CAND_SUPERSEDED,
    CLASS_UPDATE,
    ITEM_FAILED,
    ITEM_PUBLISHED,
    HumanLeaseLost,
)
from app.evolution.lock import LockHeldError
from app.evolution.models import CandidateQA
from app.observability.logging import get_logger

log = get_logger("app.evolution.human_publish")
PH_PREPARED = "PREPARED"
PH_INDEX_BUILT = "INDEX_BUILT"
PH_ACTIVATING = "ACTIVATING"
PH_ALIAS_ACTIVATED = "ALIAS_ACTIVATED"
PH_POINTER_UPDATED = "POINTER_UPDATED"
PH_LEDGER_COMMITTED = "LEDGER_COMMITTED"
JOURNAL_KIND = "human_publish"
BLOCKED_KEY = "kb_write_blocked"
BLOCKED_VALUE = "reconcile-needed"


class BatchStale(RuntimeError):
    """激活前复核发现漂移：fail_batch(retry_wait)，journal 走回滚后剔除重发。"""


class _DedupOutcome:
    """_final_dedup 的结果：detail 非空 = 拒绝；replaces 非空 = 替换语义。"""

    __slots__ = ("detail", "replaces")

    def __init__(self, detail: str = "", replaces: str = ""):
        self.detail = detail
        self.replaces = replaces


class HumanBatchPublisher:
    """多实例 Worker：每个共享副作用前同时检查写锁、有效任务租约与会话栅栏。"""

    def __init__(
        self,
        store,
        *,
        index_service,
        generation_store,
        publisher,
        journal,
        lock,
        control_store,
        kb_dir,
        worker_id: str,
        dedup_service,
        enabled: bool | None = None,
        fence=None,
        lifecycle=None,
    ):
        self._store = store
        self._index = index_service
        self._gen_store = generation_store
        self._publisher = publisher
        self._journal = journal
        self._lock = lock
        self._control = control_store
        self._kb_dir = kb_dir
        self._worker_id = worker_id
        self._dedup = dedup_service
        self._enabled = enabled
        self._fence = fence
        self._lifecycle = lifecycle

    # ============================================================
    # 主循环
    # ============================================================
    def process_once(self) -> bool:
        from app.config.settings import settings

        enabled = (
            settings.self_evolve_enabled if self._enabled is None else self._enabled
        )
        entry = self._journal.read()
        recovery_batch_id = None
        if entry:
            if entry.get("kind") != JOURNAL_KIND:
                # 共享 journal 属于机器人发布；不要领取任务，也绝不能覆盖。
                return False
            recovery_batch_id = int(entry.get("batch_id", 0) or 0) or None
            existing = (
                self._store.get_batch(recovery_batch_id)
                if recovery_batch_id is not None
                else None
            )
            if existing is None:
                # journal 本身也是共享状态；必须拿到全局写锁并复读后才能
                # 宣告损坏，避免与另一发布实例的写入窗口竞态。
                self._lock.acquire(phase="human-publish-orphan-check")
                try:
                    current = self._journal.read()
                    if current == entry:
                        try:
                            self._block()
                        except _Blocked:
                            return True
                finally:
                    self._lock.release()
                return True
            if existing is not None and existing.get("status") in (
                "completed",
                "published",
            ):
                # MySQL 已提交而 post-settle/清 journal 前强杀：仍需在 KB
                # 写锁内幂等补偿下架与旧 Ledger 清账，再清现场。
                # （published 为 010 重命名前遗留值，兼容读。）
                self._lock.acquire(phase="human-publish-recovery")
                try:
                    current = self._journal.read()
                    if current == entry:
                        self._recover_post_settle(current, existing)
                        self._journal.clear()
                finally:
                    self._lock.release()
                entry = None
                recovery_batch_id = None
        # 开关关闭时不启动新发布，但已越过提交点的现场仍允许补账。
        if not enabled and entry is None:
            return False
        batch = self._store.claim_publish_batch(
            self._worker_id,
            batch_id=recovery_batch_id,
        )
        if batch is None:
            return False

        lost = threading.Event()
        heartbeat_stop = threading.Event()
        heartbeat = self._start_heartbeat(batch, lost, heartbeat_stop)
        try:
            self._lock.acquire(phase="human-publish")
            try:
                self._assert_owned(batch, lost)
                locked_entry = self._journal.read()
                if locked_entry and locked_entry.get("kind") != JOURNAL_KIND:
                    # 初次无 journal、等锁期间机器人发布留下了恢复现场：
                    # 让出本批次即可，不能把合法的异类 journal 标成损坏。
                    self._store.fail_batch(
                        batch,
                        RuntimeError("另一条知识发布事务正在恢复"),
                    )
                    return True
                had_journal = locked_entry is not None
                if self._recover_journal(batch, lost):
                    return True
                if had_journal and not enabled:
                    self._store.fail_batch(
                        batch, RuntimeError("SELF_EVOLVE_ENABLED=false")
                    )
                    return True
                self._publish_locked(batch, lost)
            finally:
                self._release_fence()
                try:
                    self._lock.release()
                except Exception as exc:  # noqa: BLE001 - release is best effort
                    log.warning(
                        "human_publish.lock_release_failed err=%s", type(exc).__name__
                    )
        except _Blocked:
            return True
        except (HumanLeaseLost, FenceLost) as exc:
            log.warning("human_publish.lease_lost batch=%s err=%s", batch["id"], exc)
        except LockHeldError as exc:
            log.info("human_publish.lock_unavailable batch=%s", batch["id"])
            self._fail_if_owned(batch, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "human_publish.failed batch=%s err=%s", batch["id"], type(exc).__name__
            )
            self._fail_if_owned(batch, exc)
        finally:
            heartbeat_stop.set()
            if heartbeat is not None:
                heartbeat.join(timeout=2)
        return True

    def _release_fence(self) -> None:
        if self._fence is None:
            return
        try:
            self._fence.release()
        except Exception as exc:  # noqa: BLE001 - release is best effort
            log.warning("human_publish.fence_release_failed err=%s", type(exc).__name__)

    def _start_heartbeat(
        self, batch: dict, lost: threading.Event, stop_event: threading.Event
    ):
        interval = max(0.05, float(getattr(self._store, "_lease", 30)) / 3.0)

        def _loop():
            while not stop_event.wait(interval):
                try:
                    ok = self._store.heartbeat_batch(batch["id"], batch["lease_token"])
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    lost.set()
                    return

        thread = threading.Thread(
            target=_loop, name=f"human-publish-hb-{batch['id']}", daemon=True
        )
        thread.start()
        return thread

    def _assert_owned(self, batch: dict, lost: threading.Event) -> None:
        self._lock.assert_held()
        if lost.is_set() or not self._store.check_batch_owner(
            batch["id"],
            batch["lease_token"],
        ):
            lost.set()
            raise HumanLeaseLost(f"发布批次 {batch['id']} 租约已失效")
        # ConversationFence 的持锁连接只能由创建它的发布主线程访问；
        # 不在 heartbeat 线程并发使用 PyMySQL connection。
        if self._fence is not None and self._fence.held_keys:
            self._fence.assert_held()

    def _fail_if_owned(self, batch: dict, exc: Exception) -> None:
        entry = self._journal.read()
        post_commit = bool(
            entry
            and entry.get("kind") == JOURNAL_KIND
            and int(entry.get("batch_id", -1)) == int(batch["id"])
            and entry.get("stage")
            in (PH_ACTIVATING, PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED)
        )
        try:
            self._store.fail_batch(batch, exc, post_commit=post_commit)
        except HumanLeaseLost:
            log.warning("human_publish.fail_write_lost batch=%s", batch["id"])

    # ============================================================
    # 发布 / 下架分发
    # ============================================================
    def _publish_locked(self, batch: dict, lost: threading.Event) -> None:
        self._assert_owned(batch, lost)
        self._assert_not_blocked()
        self._assert_alias_known()
        operation = batch.get("operation") or BATCH_OP_PUBLISH
        if operation == BATCH_OP_RETIRE:
            self._retire_locked(batch, lost)
        else:
            self._publish_items_locked(batch, lost)

    def _acquire_fence(self, batch: dict, lost: threading.Event) -> None:
        """批内全部去重后的会话 key（字典序）取栅栏；超时 → 批次 retry_wait。"""
        if self._fence is None:
            return
        keys = self._store.fence_keys_for_candidates(
            [item["candidate_id"] for item in batch["items"]]
        )
        if not keys:
            return
        self._assert_owned(batch, lost)
        self._fence.acquire(keys)  # FenceTimeout → 外层 fail_batch(retry_wait)

    def _publish_items_locked(self, batch: dict, lost: threading.Event) -> None:
        backend = self._index_backend()
        results: list[dict] = []
        valid: list[tuple[dict, dict]] = []  # (item, fresh candidate)

        # ---- 会话栅栏（收敛竞态窗口；正确性由结算 CAS 保证）----
        self._acquire_fence(batch, lost)

        # ---- 逐项校验（审批快照复核）----
        for item in batch["items"]:
            self._assert_owned(batch, lost)
            cid = int(item["candidate_id"])
            cand = self._store.get_candidate(cid)
            if cand is None:
                results.append(
                    {
                        "candidate_id": cid,
                        "status": ITEM_FAILED,
                        "detail": "candidate_missing",
                    }
                )
                continue
            if cand["status"] != CAND_PUBLISH_QUEUED:
                results.append(
                    {
                        "candidate_id": cid,
                        "status": CAND_REJECTED,
                        "candidate_status": cand["status"],
                        "detail": f"candidate_{cand['status']}",
                    }
                )
                continue
            if int(cand["revision"] or 0) != _int_or(item.get("candidate_revision"), -1):
                self._record_stale("revision_drift")
                results.append(
                    {
                        "candidate_id": cid,
                        "status": CAND_REJECTED,
                        "candidate_status": CAND_REJECTED,
                        "detail": "revision_drift",
                    }
                )
                continue
            if self._approval_digest(item) != (item.get("approval_digest") or ""):
                self._record_stale("digest_mismatch")
                results.append(
                    {
                        "candidate_id": cid,
                        "status": CAND_REJECTED,
                        "candidate_status": CAND_REJECTED,
                        "detail": "digest_mismatch",
                    }
                )
                continue
            valid.append((item, cand))

        # ---- 批内 pairwise（高分胜出，同分小 ID 胜出）----
        if len(valid) > 1:
            from app.evolution.semantic_dedup import drop_in_run_duplicates

            metas = [
                {
                    "candidate_id": int(cand["id"]),
                    "value_score": float(cand.get("value_score") or 0.0),
                }
                for _item, cand in valid
            ]
            vectors = self._dedup.embed([cand["question"] for _i, cand in valid])
            dropped = set(
                drop_in_run_duplicates(
                    vectors, metas, threshold=self._dedup.q_threshold
                )
            )
            for idx in sorted(dropped):
                item, cand = valid[idx]
                results.append(
                    {
                        "candidate_id": int(cand["id"]),
                        "status": CAND_REJECTED,
                        "candidate_status": CAND_REJECTED,
                        "detail": "in_run_duplicate",
                    }
                )
            valid = [pair for i, pair in enumerate(valid) if i not in dropped]

        # ---- 最终去重（替换 final_dedup；规则矩阵）----
        docs: list[tuple[dict, dict, str, str]] = []
        # (item, candidate, filename, replaced_filename)。替换关系必须与成功
        # staging 的文档绑在同一结构里，不能用过滤前后的列表下标回连。
        replaced: list[str] = []
        old_candidate_ids: dict[str, int | None] = {}  # old filename → human id
        for item, cand in valid:
            self._assert_owned(batch, lost)
            outcome = self._final_dedup(item, cand)
            if outcome.detail:  # 拒绝（duplicate / target_changed / answer_side）
                results.append(
                    {
                        "candidate_id": int(cand["id"]),
                        "status": CAND_REJECTED,
                        "candidate_status": CAND_REJECTED,
                        "detail": outcome.detail,
                    }
                )
                continue
            filename = self._publisher.write_staging(
                self._candidate_qa_from_item(item, cand)
            )
            if filename is None:
                results.append(
                    {
                        "candidate_id": int(cand["id"]),
                        "status": CAND_REJECTED,
                        "candidate_status": CAND_REJECTED,
                        "detail": "sensitive_on_publish",
                    }
                )
                continue
            replacement = outcome.replaces
            if replacement:
                replaced.append(replacement)
                old_candidate_ids[replacement] = self._old_candidate_id(
                    f"evolved/{replacement}"
                )
            docs.append((item, cand, filename, replacement))

        if not docs:
            self._assert_owned(batch, lost)
            self._store.settle_batch(batch, generation_id="", results=results)
            return

        # ---- staging generation + journal（扩展字段：operation/digests/old_gen）----
        old_generation = self._gen_store.active(backend)
        generation_id = new_generation_id()
        info = GenerationInfo(
            generation_id=generation_id,
            target=self._index.target_for(backend, generation_id),
            embedding_model=self._index._embedder.model,
        )
        published_results = []
        for item, cand, filename, old_file in docs:
            entry = {
                "candidate_id": int(cand["id"]),
                "status": ITEM_PUBLISHED,
                "filename": filename,
                "candidate_revision": int(item.get("candidate_revision") or 0),
                "lifecycle_revision": int(cand.get("lifecycle_revision") or 0),
            }
            if old_file and isinstance(
                old_candidate_ids.get(old_file), int
            ):
                entry["replaced_candidate_id"] = int(old_candidate_ids[old_file])
                entry["replaced_filename"] = old_file
            published_results.append(entry)
        all_results = list(results) + published_results
        journal_body = {
            "kind": JOURNAL_KIND,
            "phase": "publish",
            "operation": BATCH_OP_PUBLISH,
            "stage": PH_PREPARED,
            "batch_id": batch["id"],
            "staging_docs": [
                {"candidate_id": int(cand["id"]), "filename": filename}
                for _item, cand, filename, _old in docs
            ],
            "replaced_docs": list(dict.fromkeys(replaced)),
            "approval_digests": [
                item.get("approval_digest") or ""
                for item, _cand, _fn, _old in docs
            ],
            "old_generation_id": (
                old_generation.generation_id if old_generation is not None else ""
            ),
            "backend": backend,
            "index": info.to_dict(),
            "results": all_results,
        }
        self._assert_owned(batch, lost)
        self._journal.write(journal_body)
        for _item, _cand, filename, _old in docs:
            self._assert_owned(batch, lost)
            self._publisher.move_into_kb(filename)
        trash_dir = self._journal.path.parent / "trash"
        for old in dict.fromkeys(replaced):
            self._assert_owned(batch, lost)
            self._publisher.unpublish(old, trash_dir)
        self._assert_owned(batch, lost)
        self._index.build(backend, generation_id=generation_id)
        self._assert_owned(batch, lost)
        self._journal.write({**journal_body, "stage": PH_INDEX_BUILT})

        # ---- 存在性探针（确定性完整性检查；失败 → INDEX_BUILT 回滚重试）----
        self._assert_owned(batch, lost)
        try:
            self._index.verify_documents(
                backend,
                info,
                [f"evolved/{filename}" for _i, _c, filename, _old in docs],
            )
        except Exception:
            from app.observability.metrics import record_human_publish_probe_fail

            record_human_publish_probe_fail()
            raise

        # ---- 激活前复核（任何漂移 → retry_wait，回滚后剔除重发）----
        self._assert_owned(batch, lost)
        drift = self._store.revalidate_batch_items(batch["id"])
        if drift:
            raise BatchStale(
                f"激活前复核漂移 {len(drift)} 项: "
                + ",".join(f"{d['candidate_id']}:{d['reason']}" for d in drift[:5])
            )

        # ---- 两阶段激活（阶段机不动）----
        self._assert_owned(batch, lost)
        self._journal.write({**journal_body, "stage": PH_ACTIVATING})
        self._assert_owned(batch, lost)
        self._index.activate_alias(backend, info)
        self._assert_owned(batch, lost)
        if backend == "es":
            state = self._index.reconcile(backend)
            if not state.get("alias_known") or state.get("alias_target") != info.target:
                raise HumanPublishRecovering("alias 激活结果未确认")
        self._journal.write({**journal_body, "stage": PH_ALIAS_ACTIVATED})
        self._assert_owned(batch, lost)
        self._index.activate_pointer(backend, info)
        self._assert_owned(batch, lost)
        self._journal.write({**journal_body, "stage": PH_POINTER_UPDATED})
        self._assert_owned(batch, lost)
        cas_misses = self._store.settle_batch(
            batch, generation_id=generation_id, results=all_results
        )
        committed = {
            **journal_body,
            "stage": PH_LEDGER_COMMITTED,
            "cas_misses": cas_misses,
        }
        self._journal.write(committed)
        # 补偿下架/旧 Ledger 清账也是事务完成的一部分。Journal 保留到这些
        # 幂等动作完成，避免 settle 后强杀造成永久漏补账。
        self._post_settle(
            batch, all_results, cas_misses, replaced, check_versions=True
        )
        self._journal.clear()
        log.info(
            "human_publish.done batch=%s generation=%s docs=%s",
            batch["id"],
            generation_id,
            len(docs),
        )

    # ============================================================
    # 最终去重决策
    # ============================================================
    def _final_dedup(self, item: dict, cand: dict):
        """发布端规则矩阵。返回 _DedupOutcome。

        - 无命中/低于阈值 → 发布（replaces 为被替换旧文件名，替换语义见下）；
        - 命中 authoritative（任一侧）→ duplicate_on_publish:{path}；
        - 仅答案侧命中 managed → 拒绝（答案侧永不触发替换）；
        - 问题侧命中 managed：classification=update 且目标未变化且
          value_score ≥ 旧文档 quality_score（frontmatter 解析）→ 替换；
          目标变化 → dedup_target_changed；分类不符/质量不足 → duplicate。
        """
        from app.observability.metrics import (
            record_human_dedup_decision,
            record_human_replace_cross_caliber,
        )

        result = self._dedup.score(cand["question"], cand["answer"])
        q_hit, a_hit = result.question, result.answer
        q_on = self._dedup.at_threshold(q_hit, side="question")
        a_on = self._dedup.at_threshold(a_hit, side="answer")

        def _auth(hit) -> bool:
            return (
                hit is not None
                and authority_kind(hit.path, kb_dir=self._kb_dir) == AUTHORITATIVE
            )

        if (q_on and _auth(q_hit)) or (a_on and _auth(a_hit)):
            record_human_dedup_decision("duplicate")
            hit = q_hit if q_on else a_hit
            return _DedupOutcome(detail=f"duplicate_on_publish:{hit.path}")
        if q_on:
            if cand.get("classification") != CLASS_UPDATE:
                record_human_dedup_decision("duplicate")
                return _DedupOutcome(detail=f"duplicate_on_publish:{q_hit.path}")
            if q_hit.path != (item.get("dedup_target_path") or ""):
                record_human_dedup_decision("target_changed")
                return _DedupOutcome(detail="dedup_target_changed")
            old_quality, old_kind = self._old_doc_meta(q_hit.path)
            if float(cand.get("value_score") or 0.0) < old_quality:
                record_human_dedup_decision("duplicate")
                return _DedupOutcome(detail=f"duplicate_on_publish:{q_hit.path}")
            if old_kind not in ("human_conversation", "human_handoff"):
                # 跨口径判据（value_score vs 旧 quality_score）：观测待校准
                record_human_replace_cross_caliber()
            record_human_dedup_decision("replaced")
            return _DedupOutcome(replaces=q_hit.path[len("evolved/") :])
        if a_on:
            record_human_dedup_decision("answer_side")
            return _DedupOutcome(detail=f"duplicate_on_publish:{a_hit.path}")
        record_human_dedup_decision("kept")
        return _DedupOutcome()

    def _old_doc_meta(self, source_path: str) -> tuple[float, str]:
        """旧文档 frontmatter 的 (quality_score, source_kind)；不可读 → (0.0, "")。"""
        from app.agent.rag.loader import parse_frontmatter

        try:
            text = (Path(self._kb_dir) / source_path).read_text(encoding="utf-8")
            meta, _body = parse_frontmatter(text)
        except (OSError, ValueError):
            return 0.0, ""
        try:
            quality = float(meta.get("quality_score") or 0.0)
        except (TypeError, ValueError):
            quality = 0.0
        return quality, str(meta.get("source_kind") or "")

    def _old_candidate_id(self, source_path: str) -> int | None:
        """被替换文档 frontmatter 的 candidate_id（human-{id}）；非 human → None。"""
        from app.agent.rag.loader import parse_frontmatter

        try:
            text = (Path(self._kb_dir) / source_path).read_text(encoding="utf-8")
            meta, _body = parse_frontmatter(text)
        except (OSError, ValueError):
            return None
        raw = str(meta.get("candidate_id") or "")
        if raw.startswith("human-") and raw[len("human-") :].isdigit():
            return int(raw[len("human-") :])
        return None

    @staticmethod
    def _record_stale(reason: str) -> None:
        from app.observability.metrics import record_human_approval_stale

        record_human_approval_stale(reason)

    @staticmethod
    def _approval_digest(item: dict) -> str:
        from app.evolution.human_store import approval_digest

        return approval_digest(
            {
                "candidate_id": int(item["candidate_id"]),
                "candidate_revision": item.get("candidate_revision"),
                "source_version": item.get("source_version"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "value_score": item.get("value_score"),
                "classification": item.get("classification"),
                "dedup_target_path": item.get("dedup_target_path") or "",
                "approved_by": item.get("approved_by") or "",
                "approved_at": _iso_seconds(item.get("approved_at")),
            }
        )

    def _candidate_qa_from_item(self, item: dict, cand: dict) -> CandidateQA:
        """发布内容取自审批快照（非候选现行值）；provenance 取会话归属。"""
        return CandidateQA(
            candidate_id=f"human-{item['candidate_id']}",
            turn_id=f"human-conv:{cand['conversation_id']}",
            question=item.get("question") or cand["question"],
            answer=item.get("answer") or cand["answer"],
            quality_score=float(item.get("value_score") or 0.0),
            source_kind="human_conversation",
            submitted_by=item.get("approved_by") or "",
            evidence_paths=[],
        )

    # ============================================================
    # 下架（人工 + 补偿）
    # ============================================================
    def _retire_locked(self, batch: dict, lost: threading.Event) -> None:
        from app.observability.metrics import record_human_lifecycle_inconsistency

        backend = self._index_backend()
        results: list[dict] = []
        docs: list[tuple[dict, dict, str, int]] = []  # (item, candidate, filename, lrev)
        for item in batch["items"]:
            self._assert_owned(batch, lost)
            cid = int(item["candidate_id"])
            cand = self._store.get_candidate(cid)
            if cand is None:
                results.append(
                    {
                        "candidate_id": cid,
                        "status": ITEM_FAILED,
                        "detail": "candidate_missing",
                    }
                )
                continue
            compensation = item.get("approval_digest") == "compensation"
            if compensation:
                # 补偿项（发布激活后漂移）：候选可能已被并发 superseded——
                # 状态校验放宽为 published|superseded，文档移除照常执行
                # （发布内容归我们批次所有；行不被覆盖由结算 CAS 兜底）。
                if cand["status"] not in (CAND_PUBLISHED, CAND_SUPERSEDED):
                    results.append(
                        {
                            "candidate_id": cid,
                            "status": CAND_REJECTED,
                            "detail": f"candidate_{cand['status']}",
                        }
                    )
                    continue
            elif cand["status"] != CAND_PUBLISHED:
                results.append(
                    {
                        "candidate_id": cid,
                        "status": CAND_REJECTED,
                        "detail": f"candidate_{cand['status']}",
                    }
                )
                continue
            filename = item.get("filename") or cand["published_filename"]
            if not filename:
                record_human_lifecycle_inconsistency("document_missing")
                results.append(
                    {
                        "candidate_id": cid,
                        "status": ITEM_FAILED,
                        "detail": "document_missing",
                    }
                )
                continue
            docs.append(
                (item, cand, filename, int(cand.get("lifecycle_revision") or 0))
            )

        if not docs:
            self._assert_owned(batch, lost)
            self._store.settle_batch(batch, generation_id="", results=results)
            return

        self._acquire_fence(batch, lost)

        generation_id = new_generation_id()
        info = GenerationInfo(
            generation_id=generation_id,
            target=self._index.target_for(backend, generation_id),
            embedding_model=self._index._embedder.model,
        )
        retire_results = [
            {
                "candidate_id": int(item["candidate_id"]),
                "status": "retired",
                "filename": filename,
                "candidate_revision": int(item.get("candidate_revision") or 0),
                "lifecycle_revision": lrev,
            }
            for item, _cand, filename, lrev in docs
        ]
        all_results = list(results) + retire_results
        journal_body = {
            "kind": JOURNAL_KIND,
            "phase": "retire",
            "operation": BATCH_OP_RETIRE,
            "stage": PH_PREPARED,
            "batch_id": batch["id"],
            "retire_docs": [filename for _i, _c, filename, _l in docs],
            "backend": backend,
            "index": info.to_dict(),
            "results": all_results,
        }
        self._assert_owned(batch, lost)
        self._journal.write(journal_body)
        trash_dir = self._journal.path.parent / "trash"
        for _item, _cand, filename, _lrev in docs:
            self._assert_owned(batch, lost)
            if self._publisher.unpublish(filename, trash_dir) is None:
                record_human_lifecycle_inconsistency("document_missing")
        self._assert_owned(batch, lost)
        self._index.build(backend, generation_id=generation_id)
        self._assert_owned(batch, lost)
        self._journal.write({**journal_body, "stage": PH_INDEX_BUILT})

        # 激活前复核：人工项要求仍 published 且 lifecycle_revision 未变；
        # 补偿项只要求候选仍在预期状态集（superseded 是补偿的预期现状，
        # 行不被覆盖由结算 CAS 兜底——文档移除不可因此空转）。
        self._assert_owned(batch, lost)
        for item, _cand, _filename, lrev in docs:
            fresh = self._store.get_candidate(int(item["candidate_id"]))
            compensation = item.get("approval_digest") == "compensation"
            if compensation:
                ok = fresh is not None and fresh["status"] in (
                    CAND_PUBLISHED,
                    CAND_SUPERSEDED,
                )
            else:
                ok = (
                    fresh is not None
                    and fresh["status"] == CAND_PUBLISHED
                    and int(fresh.get("lifecycle_revision") or 0) == lrev
                )
            if not ok:
                raise BatchStale(
                    f"下架激活前复核漂移: 候选 {item['candidate_id']}"
                )

        self._assert_owned(batch, lost)
        self._journal.write({**journal_body, "stage": PH_ACTIVATING})
        self._assert_owned(batch, lost)
        self._index.activate_alias(backend, info)
        self._assert_owned(batch, lost)
        if backend == "es":
            state = self._index.reconcile(backend)
            if not state.get("alias_known") or state.get("alias_target") != info.target:
                raise HumanPublishRecovering("alias 激活结果未确认")
        self._journal.write({**journal_body, "stage": PH_ALIAS_ACTIVATED})
        self._assert_owned(batch, lost)
        self._index.activate_pointer(backend, info)
        self._assert_owned(batch, lost)
        self._journal.write({**journal_body, "stage": PH_POINTER_UPDATED})
        self._assert_owned(batch, lost)
        cas_misses = self._store.settle_batch(
            batch, generation_id=generation_id, results=all_results
        )
        self._journal.write({**journal_body, "stage": PH_LEDGER_COMMITTED})
        self._journal.clear()
        for miss in cas_misses:
            # 双下架兜底：不触发再补偿（会成环），只留观测
            log.info(
                "human_publish.retire_cas_miss batch=%s candidate=%s",
                batch["id"],
                miss.get("candidate_id"),
            )
        log.info(
            "human_publish.retired batch=%s generation=%s docs=%s",
            batch["id"],
            generation_id,
            len(docs),
        )

    # ============================================================
    # 结算后补偿（激活后漂移 / 更高来源版本）
    # ============================================================
    def _post_settle(
        self,
        batch: dict,
        results: list[dict],
        cas_misses: list[dict],
        replaced: list[str],
        *,
        check_versions: bool,
    ) -> None:
        from app.observability.metrics import record_human_compensation_retire

        candidates: dict[int, str] = {}
        if batch.get("operation", BATCH_OP_PUBLISH) == BATCH_OP_PUBLISH:
            for miss in cas_misses:
                # 只对「发布本身 CAS miss」补偿；replacement/retire miss 不入队
                # （旧文档已被移除，再下架新文档会成环）。
                if miss.get("kind") == "publish" and miss.get("filename"):
                    candidates[int(miss["candidate_id"])] = miss["filename"]
        if check_versions:
            for r in results:
                if r.get("status") != ITEM_PUBLISHED:
                    continue
                cid = int(r["candidate_id"])
                if cid in candidates:
                    continue
                cand = self._store.get_candidate(cid)
                if cand is None:
                    continue
                conv = (
                    self._store.get_conversation(int(cand["conversation_id"] or 0))
                    if cand.get("conversation_id")
                    else None
                )
                if conv is None:
                    continue
                if self._store.conversation_has_higher_version(
                    conv["source"], conv["external_conversation_id"],
                    int(conv["source_version"] or 0),
                ):
                    candidates[cid] = r.get("filename", "")
        if candidates:
            items = [
                {"candidate_id": cid, "filename": filename}
                for cid, filename in candidates.items()
            ]
            batch_id = self._store.enqueue_retire_batch(
                items,
                reason="superseded_during_publish",
                requested_by="system:compensation",
            )
            if batch_id is not None:
                record_human_compensation_retire()
                log.warning(
                    "human_publish.compensation_retire_enqueued batch=%s new_batch=%s n=%s",
                    batch["id"],
                    batch_id,
                    len(items),
                )
        # 自动沉淀旧文档（无 human candidate_id）→ ledger 清账（人工候选已在
        # 结算事务内回写 superseded）
        if self._lifecycle is not None:
            for old in replaced:
                try:
                    if self._old_candidate_id(f"evolved/{old}") is None:
                        self._lifecycle.settle_replacement(old, None)
                except Exception as exc:  # noqa: BLE001 - 清账失败不回滚发布
                    log.warning(
                        "human_publish.ledger_cleanup_failed file_prefix=%s err=%s",
                        old[:8],
                        type(exc).__name__,
                    )

    def _recover_post_settle(self, entry: dict, batch: dict) -> None:
        """重放 settle 后的幂等副作用；允许处理旧 POINTER_UPDATED journal。

        若强杀发生在 MySQL settle 与 PH_LEDGER_COMMITTED journal 写入之间，
        entry 尚无 cas_misses。此时从候选当前状态反推未成功发布的 item，生成
        补偿下架；更高来源版本仍由 check_versions 二次兜底。
        """
        if entry.get("operation", BATCH_OP_PUBLISH) != BATCH_OP_PUBLISH:
            return
        results = list(entry.get("results") or [])
        misses = list(entry.get("cas_misses") or [])
        known = {
            int(m.get("candidate_id", 0) or 0)
            for m in misses
            if m.get("kind") == "publish"
        }
        for result in results:
            if result.get("status") != ITEM_PUBLISHED:
                continue
            cid = int(result.get("candidate_id", 0) or 0)
            filename = str(result.get("filename") or "")
            if not cid or not filename or cid in known:
                continue
            cand = self._store.get_candidate(cid)
            committed = (
                cand is not None
                and cand.get("status") == CAND_PUBLISHED
                and cand.get("published_filename") == filename
            )
            if not committed:
                misses.append(
                    {
                        "candidate_id": cid,
                        "kind": "publish",
                        "filename": filename,
                    }
                )
        self._post_settle(
            batch,
            results,
            misses,
            list(entry.get("replaced_docs", [])),
            check_versions=True,
        )

    # ============================================================
    # journal 恢复（回滚 / 前进）
    # ============================================================
    def _index_backend(self) -> str:
        from app.config.settings import settings

        return settings.rag_backend.lower()

    def _assert_not_blocked(self) -> None:
        if self._control.get(BLOCKED_KEY):
            from app.evolution.pipeline import EvolutionBlockedError

            raise EvolutionBlockedError("KB 写入被全局阻塞，等待人工 reconcile")

    def _assert_alias_known(self) -> None:
        if not self._index.reconcile(self._index_backend()).get("alias_known", True):
            from app.evolution.pipeline import EvolutionBlockedError

            raise EvolutionBlockedError("alias 状态未知，fail-closed")

    def _recover_journal(self, batch: dict, lost: threading.Event) -> bool:
        entry = self._journal.read()
        if not entry:
            return False
        if entry.get("kind") != JOURNAL_KIND or int(entry.get("batch_id", -1)) != int(
            batch["id"]
        ):
            self._block(batch, lost)
        return self._recover_entry(entry, batch, lost)

    def _recover_entry(self, entry: dict, batch: dict, lost: threading.Event) -> bool:
        self._assert_owned(batch, lost)
        stage = entry.get("stage", PH_PREPARED)
        operation = entry.get("operation") or BATCH_OP_PUBLISH  # 旧 journal 兼容
        if operation == BATCH_OP_RETIRE:
            docs = [
                {"candidate_id": 0, "filename": f}
                for f in entry.get("retire_docs", [])
            ]
            replaced = []
        else:
            docs = entry.get("staging_docs", [])
            replaced = entry.get("replaced_docs", [])
        meta = entry.get("index", {})
        info = GenerationInfo(
            generation_id=meta.get("generation_id", ""),
            target=meta.get("target", ""),
            embedding_model=meta.get("embedding_model", ""),
        )
        backend = entry.get("backend", "numpy")
        if stage in (PH_PREPARED, PH_INDEX_BUILT):
            self._rollback(entry, docs, replaced, batch, lost)
            return False
        if stage == PH_ACTIVATING:
            if backend != "es":
                self._forward(entry, docs, info, backend, batch, lost)
                return True
            state = self._index.reconcile(backend)
            if not state.get("alias_known", False):
                self._block(batch, lost)
            if state.get("alias_target") == info.target:
                self._forward(entry, docs, info, backend, batch, lost)
                return True
            if state.get("alias_target") in ("", state.get("pointer_target")):
                self._rollback(entry, docs, replaced, batch, lost)
                return False
            self._block(batch, lost)
        if stage in (PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED):
            self._forward(entry, docs, info, backend, batch, lost)
            return True
        if stage == PH_LEDGER_COMMITTED:
            self._assert_owned(batch, lost)
            self._recover_post_settle(entry, batch)
            self._journal.clear()
            return True
        self._block(batch, lost)

    def _rollback(
        self,
        entry: dict,
        docs: list[dict],
        replaced: list[str],
        batch: dict,
        lost: threading.Event,
    ) -> None:
        self._delete_candidate_index(entry, batch, lost)
        trash_dir = self._journal.path.parent / "trash"
        if entry.get("operation") == BATCH_OP_RETIRE:
            # 下架回滚：trash → evolved/ 还原
            for doc in docs:
                self._assert_owned(batch, lost)
                self._publisher.restore(doc["filename"], trash_dir)
        else:
            for doc in docs:
                self._assert_owned(batch, lost)
                self._publisher.remove(doc["filename"])
            for old in replaced:
                self._assert_owned(batch, lost)
                self._publisher.restore(old, trash_dir)
        self._assert_owned(batch, lost)
        self._journal.clear()
        log.info("human_publish.recovered_rollback stage=%s", entry.get("stage"))

    def _delete_candidate_index(
        self, entry: dict, batch: dict, lost: threading.Event
    ) -> None:
        """Delete only the exact, inactive generation recorded by this journal."""
        meta = entry.get("index") or {}
        backend = str(entry.get("backend") or "").lower()
        generation_id = str(meta.get("generation_id") or "")
        target = str(meta.get("target") or "")
        if not generation_id or not target:
            return
        self._assert_owned(batch, lost)
        if target != self._index.target_for(backend, generation_id):
            self._block(batch, lost)
        active = self._gen_store.active(backend)
        if active is not None and active.target == target:
            self._block(batch, lost)
        if backend == "numpy":
            expected_dir = Path(self._index.target_for(backend, "probe")).resolve().parent
            if Path(target).resolve().parent != expected_dir:
                self._block(batch, lost)
            self._assert_owned(batch, lost)
            Path(target).unlink(missing_ok=True)
            return
        if backend == "es":
            self._assert_owned(batch, lost)
            if not self._index.delete_candidate(backend, target):
                raise HumanPublishRecovering("候选索引状态无法确认，保留 journal")
            return
        if backend == "chroma":
            try:
                import chromadb

                client = chromadb.PersistentClient(
                    path=str(self._index._chroma_persist_dir())
                )
                self._assert_owned(batch, lost)
                client.delete_collection(target)
            except Exception as exc:
                if "not found" not in str(exc).lower():
                    raise HumanPublishRecovering(
                        "候选 collection 清理未确认，保留 journal"
                    ) from exc
            return
        self._block(batch, lost)

    def _forward(
        self,
        entry: dict,
        docs: list[dict],
        info: GenerationInfo,
        backend: str,
        batch: dict,
        lost: threading.Event,
    ) -> None:
        self._assert_owned(batch, lost)
        self._index.activate_alias(backend, info)
        self._assert_owned(batch, lost)
        if backend == "es":
            state = self._index.reconcile(backend)
            if not state.get("alias_known") or state.get("alias_target") != info.target:
                self._block(batch, lost)
        self._assert_owned(batch, lost)
        self._index.activate_pointer(backend, info)
        self._assert_owned(batch, lost)
        self._journal.write({**entry, "stage": PH_POINTER_UPDATED})
        results = entry.get("results") or [
            {
                "candidate_id": doc["candidate_id"],
                "status": ITEM_PUBLISHED,
                "filename": doc["filename"],
            }
            for doc in docs
            if doc.get("candidate_id")
        ]
        self._assert_owned(batch, lost)
        cas_misses = self._store.settle_batch(
            batch, generation_id=info.generation_id, results=results
        )
        committed = {
            **entry,
            "stage": PH_LEDGER_COMMITTED,
            "cas_misses": cas_misses,
        }
        self._journal.write(committed)
        self._post_settle(
            batch,
            results,
            cas_misses,
            list(entry.get("replaced_docs", [])),
            check_versions=True,
        )
        self._journal.clear()
        log.info("human_publish.recovered_forward stage=%s", entry.get("stage"))

    def _block(
        self,
        batch: dict | None = None,
        lost: threading.Event | None = None,
    ) -> None:
        self._lock.assert_held()
        if batch is not None and lost is not None:
            self._assert_owned(batch, lost)
        try:
            self._control.set(BLOCKED_KEY, BLOCKED_VALUE)
        except Exception as exc:  # noqa: BLE001 - original blocked state remains
            log.error("human_publish.block_flag_failed err=%s", type(exc).__name__)
        raise _Blocked()


class _Blocked(RuntimeError):
    """alias 或 journal 所属不明确：全局阻塞，禁止猜测恢复。"""


class HumanPublishRecovering(RuntimeError):
    """保留 journal，由下一有效租约前进或回滚。"""


def _iso_seconds(value) -> str:
    if value is None:
        return ""
    if getattr(value, "isoformat", None):
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)


def _int_or(value, default: int) -> int:
    """None → default；0/正数原样（0 是合法 revision，不能走 or 兜底）。"""
    return default if value is None else int(value)

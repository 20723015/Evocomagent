"""pipeline.py：EvolutionPipeline —— 12 步编排（第10期 QA 自动沉淀）。

正式运行流程（dry-run 零写入，不取锁）：
1  锁获取（LockGuard，与 build_kb_index / 其他运行互斥）
2  journal 恢复（上次中断：generation 未切换 → 删孤立文档与 staging 索引；
   已切换 → ledger 补记）
3  [with_eval] 活动索引 baseline 评测
4  读未处理 turn（turns 目录 + legacy session 状态机）
5  规则过滤 + 精确去重
6  预去重（0.95，原始脱敏问题）
7  排序取 ≤ max_judge_per_run
8  价值 Judge（≤50）
9  接地 Judge（证据集 = 非 evolved 来源）
10 最终去重（0.9）+ 本轮互查
11 选 ≤ max_per_run → 渲染 + 复扫 → journal → 移入 evolved/ → staging 索引构建验证
12 [with_eval] staging 评测 + 候选探针 → 通过才激活 generation；
    任一阻断 → 候选进 pending（eval_blocked，供人工审核）、清 staging、不切换（CLI 退出码 2）
13 全通过 → 切换 generation → ledger 批量补记（published/processed/pending）→ 报告落盘

事务性：任何位于 journal 之后的失败都由「下次运行的 journal 恢复」兜底，
任一阶段失败旧索引始终可用（不激活就不切换）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from app.observability.logging import get_logger

log = get_logger("app.evolution.pipeline")

from app.agent.rag.loader import parse_frontmatter
from app.config.settings import settings
from app.evolution import dedup
from app.evolution.generation import GenerationInfo, GenerationStore, new_generation_id
from app.evolution.judges import ValueJudge
from app.evolution.ledger import Ledger
from app.evolution.lock import Journal, LockGuard
from app.evolution.miner import build_candidate, mine_turns, scan_legacy_session
from app.evolution.models import CandidateQA, EvolutionReport, TurnRecord
from app.evolution.publish_state import (
    PH_ALIAS_ACTIVATED,
    PH_INDEX_BUILT,
    PH_LEDGER_COMMITTED,
    PH_POINTER_UPDATED,
    PH_PREPARED,
    REC_BLOCKED,
    REC_FORWARD,
    REC_LEDGER_ONLY,
    recover_decide,
)
from app.evolution.publisher import Publisher

EVAL_SCORE_DROP = 0.02  # avg_result_score 允许的下降容差


class EvolutionBlockedError(RuntimeError):
    """知识库被持久化的 reconcile 阻塞标记保护，禁止继续写入。"""


class EvolutionRecoveryPendingError(RuntimeError):
    """恢复状态尚未得到明确成功确认，当前 run/approve 必须终止。"""


def _resource_not_found(exc: BaseException, resource: str) -> bool:
    """只把明确的资源不存在映射为幂等清理成功。"""
    name = type(exc).__name__.lower().replace("_", "")
    if "notfound" in name:
        return True
    for attr in ("status_code", "status"):
        try:
            if int(getattr(exc, attr)) == 404:
                return True
        except (AttributeError, TypeError, ValueError):
            pass
    message = str(exc).lower()
    resource = resource.lower()
    return resource in message and (
        "not found" in message or "does not exist" in message
    )


class EvolutionPipeline:
    """QA 自动沉淀管线：依赖全部注入，便于 CI 用 Fake services 跑 dry-run。"""

    def __init__(
        self,
        *,
        value_judge: ValueJudge,
        grounding_judge,
        embedder,
        retriever_factory,  # () -> 检索器（复用 knowledge 单例，含 generation 热刷新）
        index_service,
        generation_store: GenerationStore,
        ledger: Ledger,
        lock: LockGuard,
        journal: Journal,
        publisher: Publisher,
        turns_dir,
        kb_dir,
        state_dir,
        output_dir,
        session_paths: list[Path] | None = None,  # legacy v1 session 文件
        clock=None,
        evaluator_factory=None,  # () -> Evaluator（with_eval 用）
        eval_cases=None,  # list[EvalCase]
        lifecycle=None,  # KnowledgeLifecycleCoordinator（人工正本路由）
    ):
        self._value_judge = value_judge
        self._grounding_judge = grounding_judge
        self._embedder = embedder
        self._retriever_factory = retriever_factory
        self._index_service = index_service
        self._generation_store = generation_store
        self._ledger = ledger
        self._lock = lock
        self._journal = journal
        self._publisher = publisher
        self._turns_dir = Path(turns_dir)
        self._kb_dir = Path(kb_dir)
        self._state_dir = Path(state_dir)
        self._output_dir = Path(output_dir)
        self._session_paths = session_paths or []
        self._clock = clock
        self._evaluator_factory = evaluator_factory
        self._eval_cases = eval_cases or []
        self._lifecycle = lifecycle
        self.eval_detail: dict | None = None  # with_eval 的 before/after/阻断详情

    def _now(self) -> datetime:
        return self._clock.now() if self._clock else datetime.now()  # noqa: DTZ005

    # ============================================================
    # 入口
    # ============================================================
    def run(self, dry_run: bool = False, with_eval: bool = False) -> EvolutionReport:
        if dry_run:
            return self._dry_run()
        if not settings.self_evolve_enabled:
            raise RuntimeError(
                "SELF_EVOLVE_ENABLED=false 拒绝正式运行（写操作安全开关）。"
                "如需预览请用 --dry-run，或设置环境变量后再运行。"
            )
        if with_eval and self._evaluator_factory is None:
            # fail-closed：请求评测但评测能力缺失时拒绝发布，不静默跳过闸门
            raise RuntimeError(
                "with-eval 需要 evaluator_factory（评测能力缺失，按 fail-closed 拒绝发布）"
            )

        self.eval_detail = None
        report = EvolutionReport()
        # 只有重接地无需执行或当前 generation 已全部核对，才允许在 finally 推进
        # last_human_generation。异常/分页未完成必须保留触发条件供下次续跑。
        generation_ready = False
        self._lock.acquire(phase="run")
        try:
            # blocked 是跨 Pod/重启持久化的全局写保护；必须在任何挖掘、评测、
            # 发布前检查。SQL 是正本，SQL 不可用时才读 state 文件降级标记。
            self._assert_kb_writes_allowed()
            self._recover_journal(report)
            generation_ready = self._maybe_revalidate(report)
            if not generation_ready:
                self._write_report(report, suffix="-revalidate-pending")
                return report

            before_detail = None
            if with_eval:
                before_detail = self._run_eval()
                if before_detail is None:
                    report.failures += 1

            turns = self._mine(report)
            candidates = self._filter(turns, report)
            candidates = self._pre_dedup(candidates, report, abort_on_error=True)
            survivors = self._judge_round(candidates, report)
            survivors = self._final_dedup(survivors, report, abort_on_error=True)

            selected, report = self._cap(survivors, report)
            self._publish(selected, turns, report, before_detail, with_eval)
            return report
        except (EvolutionBlockedError, EvolutionRecoveryPendingError):
            # 当前事务已明确阻塞/待恢复时不能再猜测 rollback；journal 现场
            # 必须保留。普通异常仍走统一恢复决策。
            raise
        except Exception:
            # journal 已写但未切换 → 兜底清理，保证旧索引可用
            # （revalidate 隔离事务例外：走前进式恢复，见 _rollback_publish）
            self._rollback_publish()
            raise
        finally:
            if generation_ready:
                try:
                    self.record_current_generation()
                except Exception:  # noqa: BLE001 - generation bookkeeping is best effort
                    log.info("⚠️  记录 last_human_generation 失败（不影响本次运行）")
            self._lock.release()

    # ============================================================
    # 1-3：锁 + journal 恢复 + baseline 评测
    # ============================================================
    def _recover_journal(self, report: EvolutionReport) -> None:
        entry = self._journal.read()
        # 孤儿 staging 清扫（run/approve 开头，锁内）：未被当前 journal 引用的
        # *.md / *.tmp 一律删除（崩溃在 journal 写之前产生的残留）
        keep = set()
        if entry:
            for doc in entry.get("staging_docs", []):
                fname = doc.get("filename") if isinstance(doc, dict) else doc
                if fname:
                    keep.add(fname)
        self._publisher.clean_staging(keep)
        if not entry:
            return
        if entry.get("kind") == "human_publish":
            raise EvolutionRecoveryPendingError(
                "人工知识发布事务尚未恢复，机器人自进化本轮让路"
            )
        if entry.get("phase") == "revalidate":
            # 隔离事务：前进式恢复（完成隔离 + 重建），不走 publish 回滚
            result = self._recover_revalidate(entry)
            if not result.get("success", False):
                raise EvolutionRecoveryPendingError(
                    result.get("reason") or "revalidate 恢复未明确成功，journal 已保留"
                )
            return
        # 2.7：统一恢复决策（读真实 alias + pointer + journal 三源）
        action = self._recover_publish_transaction(entry, report)
        if action == REC_BLOCKED:
            raise EvolutionBlockedError(
                "知识库恢复状态不可确认，已写入 kb_write_blocked；请人工 reconcile"
            )

    def _read_alias_target(self, backend: str) -> str | None:
        """读取真实 ES alias 指向（空=明确不存在，None=读取失败/未知）。"""
        if backend != "es":
            return ""
        try:
            st = self._index_service.reconcile(backend)
            target = st.get("alias_target")
            if target is None:
                return None
            return str(target or "")
        except Exception as e:  # noqa: BLE001 - alias read must fail closed
            log.warning("读取 ES alias 失败，恢复状态置为 unknown: %s", e)
            return None

    def _recover_publish_transaction(
        self, entry: dict | None, report: EvolutionReport | None = None
    ) -> str | None:
        """2.7 统一恢复：journal 阶段 + 真实 alias + pointer 三源决策。

        恢复表（publish_state.recover_decide）：
        - ledger_only：pointer 已切 → 补齐 ledger，清 journal（绝不回滚已生效知识）；
        - forward：ES alias 已切但 pointer 未切 → 只前进（补 pointer + ledger）；
        - blocked：alias 指向非旧代/非候选代，或 Alias 读取失败/多目标
          （状态未知）→ 写 kb_write_blocked，保留现场；
        - rollback：alias/pointer 均未指向候选 → 删候选索引与新文档、恢复旧文档。
        """
        if not entry:
            return None  # 无 journal：无事务可恢复
        backend = entry.get("backend", settings.rag_backend.lower())
        index_info = entry.get("index") or {}
        candidate_target = str(index_info.get("target", ""))
        previous_target = str(entry.get("previous_target", ""))
        stage = str(entry.get("stage", ""))
        active = self._generation_store.active(backend)
        pointer_target = active.target if active is not None else ""
        alias_target = self._read_alias_target(backend)

        action = recover_decide(
            backend,
            stage,
            pointer_target,
            alias_target,
            candidate_target,
            previous_target,
        )
        from app.observability.metrics import record_evolution_recovery

        record_evolution_recovery(action)
        log.info(
            "🔁 journal 恢复决策: %s（stage=%s alias=%r pointer=%r candidate=%r）",
            action,
            stage,
            alias_target,
            pointer_target,
            candidate_target,
        )

        if action == REC_LEDGER_ONLY:
            self._complete_ledger(entry)
            self._journal.clear()
            return action
        if action == REC_FORWARD:
            # alias 已生效：只前进——补 pointer、ledger、报告，绝不回滚文件
            if not active or active.target != candidate_target:
                self._index_service.activate_pointer(
                    backend,
                    GenerationInfo.from_dict(index_info),
                )
            self._complete_ledger(entry)
            self._journal.clear()
            return action
        if action == REC_BLOCKED:
            self._block_kb_writes(entry)
            return action  # 保留现场（journal 不归档）等待人工 reconcile

        # rollback：删候选索引 + 新文档，恢复被替换文档
        self._rollback_transaction(entry, backend, index_info)
        return action

    def _complete_ledger(self, entry: dict) -> None:
        """ledger 补齐（幂等）：published 补记并移除 pending、replaced 清理。

        ``forward``/``ledger_only`` 恢复都可能发生在发布进程已经把文档
        写入知识库、但尚未完成 ledger 提交的窗口。恢复重放必须和正常
        ``_publish`` 使用同一个原子账本操作；只调用 ``mark_published`` 会
        留下 ``published`` 与 ``pending`` 双重终态，下一次精确去重仍会把
        已发布候选当作待审项。
        """
        publish_entries = []
        for doc in entry.get("staging_docs", []):
            cid = doc.get("candidate_id") if isinstance(doc, dict) else None
            fname = doc.get("filename") if isinstance(doc, dict) else doc
            if cid and fname:
                publish_entries.append((cid, fname))
        if publish_entries:
            # commit_publish 本身是幂等的，并在同一次落盘中 drop pending。
            commit_publish = getattr(self._ledger, "commit_publish", None)
            if callable(commit_publish):
                commit_publish(publish_entries)
            else:
                # 兼容极简依赖注入 fake；真实 Ledger 始终走上面的原子路径。
                self._ledger.mark_published_many(publish_entries)
                drop_pending_many = getattr(self._ledger, "drop_pending_many", None)
                if callable(drop_pending_many):
                    drop_pending_many([cid for cid, _ in publish_entries])
                else:
                    for cid, _ in publish_entries:
                        self._ledger.drop_pending(cid)
        olds = []
        for old in entry.get("replaced_docs", []):
            # 与 _publish 阶段 5/5 同语义：替换结算按旧文档正本路由
            # （journal 不含被替换文档 frontmatter，从 trash 副本读取；
            #  未命中时回退 ledger published 反查，保持幂等）
            if self._lifecycle is not None and self._lifecycle.settle_replacement(
                old, None, metadata=self._replaced_doc_meta(old),
            ):
                continue
            cid = self._cid_for_filename(old)
            if cid is not None:
                olds.append((old, cid))
        if olds:
            self._ledger.batch_cleanup_published(olds)

    def _rollback_transaction(
        self, entry: dict, backend: str, index_info: dict
    ) -> None:
        """回滚：删候选索引与新文档，恢复被替换文档，成功后归档 journal。

        ES candidate 只有在再次确认 alias 可读且未指向 candidate 后才允许删除；
        清理未完成时保留 journal，避免恢复证据丢失。
        """
        # 先做 ES candidate 的二次安全确认/删除；若 alias 读取失败或目标
        # 不安全，任何源文档都不能先被删/还原，否则后续发现 alias 已切时会
        # 形成「线上索引已生效、源文件却已回滚」的不一致。
        if not self._delete_staging_index(backend, index_info):
            log.warning("candidate 清理未获明确成功，保留 journal 等待下次恢复")
            return
        for doc in entry.get("staging_docs", []):
            fname = doc.get("filename") if isinstance(doc, dict) else doc
            self._publisher.remove(fname)
        for old in entry.get("replaced_docs", []):
            self._publisher.restore(old, self._state_dir / "trash")
        self._journal.archive()
        log.info("🔁 journal 恢复：generation 未切换，已清理孤立 staging 文档与索引")

    def _block_kb_writes(self, entry: dict) -> None:
        """alias 指向未知代：写 kb_write_blocked（SQL 优先），保留现场。"""
        from app.stores.sql.document_store import KbControlStore
        from app.stores.sql.engine import get_engine

        reason = (
            f"alias 指向非旧代/非候选代（journal {entry.get('stage', '?')}），"
            "需人工 reconcile"
        )
        engine = get_engine()
        blocked_written = False
        if engine is not None:
            try:
                KbControlStore(engine).set("kb_write_blocked", reason)
                log.error("🚫 KB 写入已全局阻塞: %s", reason)
                blocked_written = True
            except Exception as e:  # noqa: BLE001 - SQL fallback is fail closed
                log.warning("kb_write_blocked 写 SQL 失败，降级文件: %s", e)
        if not blocked_written:
            marker = self._state_dir / "kb_write_blocked"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(reason, encoding="utf-8")
        from app.observability.metrics import set_kb_write_blocked

        set_kb_write_blocked(True)

    def _read_kb_write_blocked(self) -> str:
        """读取全局写保护（SQL 优先，SQL 不可用时文件降级）。"""
        from app.stores.sql.document_store import KbControlStore
        from app.stores.sql.engine import get_engine

        sql_value = ""
        try:
            engine = get_engine()
        except Exception as e:  # noqa: BLE001 - SQL fallback is fail closed
            engine = None
            log.warning("读取 kb_write_blocked SQL 引擎失败，降级文件: %s", e)
        if engine is not None:
            try:
                sql_value = str(KbControlStore(engine).get("kb_write_blocked") or "")
            except Exception as e:  # noqa: BLE001 - SQL fallback is fail closed
                log.warning("读取 kb_write_blocked SQL 失败，降级文件: %s", e)

        # SQL 中有值时优先返回；文件标记是 SQL 不可用时的持久化降级，
        # 即使 SQL 返回空也保守检查它，防止阻塞现场被重启后的新库覆盖。
        if sql_value:
            return sql_value
        marker = self._state_dir / "kb_write_blocked"
        try:
            file_value = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            file_value = ""
        if file_value:
            return file_value
        return ""

    def _assert_kb_writes_allowed(self) -> None:
        reason = self._read_kb_write_blocked()
        if reason:
            raise EvolutionBlockedError(
                f"知识库写入已阻塞：{reason}；请先人工 reconcile 并清除阻塞标记"
            )

    def _maybe_revalidate(self, report: EvolutionReport) -> bool:
        """人工知识变更（上传/下架/手工 CLI 切了 generation）→ 触发存量沉淀重接地。

        state/last_human_generation.json 记录上次运行结束的活动 generation_id；
        当前活动代不同（含首次运行无记录）且 EVOLVE_REVALIDATE_ENABLED 才执行。
        记录在 run() 末尾由 _record_current_generation 统一刷新为最终活动代，
        避免自身的发布/重接地重建造成自我触发。
        """
        if not settings.evolve_revalidate_enabled:
            return True
        backend = settings.rag_backend.lower()
        active = self._generation_store.active(backend)
        if active is None:
            return True  # 知识库尚无索引：无从核对
        progress = self._read_revalidate_progress(active.generation_id)
        if not progress and self._read_last_human_generation() == active.generation_id:
            return True
        from app.evolution.revalidate import revalidate

        result = revalidate(
            kb_dir=self._kb_dir,
            trash_dir=self._state_dir / "trash",
            ledger=self._ledger,
            publisher=self._publisher,
            index_service=self._index_service,
            retriever=self._retriever_factory(),
            grounding_judge=self._grounding_judge,
            journal=self._journal,
            exclude_docs=set(progress),
            backend=backend,
            lifecycle=self._lifecycle,
        )
        report.revalidated_checked = result["checked"]
        report.revalidated_passed = result["passed"]
        report.revalidated_failed = result["failed"]
        report.revalidated_remaining = result["remaining"]
        processed = set(progress)
        processed.update(result["processed_docs"])
        if result["has_more"]:
            # 隔离失败文档可能在本批次内切出一个新 generation；该切代属于当前
            # 重接地事务，后续批次应沿用进度，而不是误判成新一轮人工变更。
            current = self._generation_store.active(backend)
            progress_generation = (
                current.generation_id if current is not None else active.generation_id
            )
            self._write_revalidate_progress(progress_generation, processed)
        log.info(
            f"🔁 重接地（人工知识变更触发）: 核对 {result['checked']} / "
            f"通过 {result['passed']} / 隔离 {result['failed']} / "
            f"剩余 {result['remaining']}"
        )
        if result["has_more"]:
            return False
        self.record_current_generation()
        return True

    def _read_revalidate_progress(self, generation_id: str) -> list[str]:
        """读取同一 generation 的分页进度；代际不匹配时从头开始。"""
        path = self._state_dir / "revalidate_progress.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if str(data.get("generation_id", "")) != generation_id:
                return []
            return [str(name) for name in data.get("processed_docs", []) if name]
        except (json.JSONDecodeError, OSError, TypeError, AttributeError):
            return []

    def _write_revalidate_progress(self, generation_id: str, processed_docs) -> None:
        """原子保存 generation 分页进度。"""
        path = self._state_dir / "revalidate_progress.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "generation_id": generation_id,
                    "processed_docs": sorted(set(processed_docs)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _clear_revalidate_progress(self) -> None:
        (self._state_dir / "revalidate_progress.json").unlink(missing_ok=True)

    def _read_last_human_generation(self) -> str:
        path = self._state_dir / "last_human_generation.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("generation_id", ""))
        except (json.JSONDecodeError, OSError, TypeError, AttributeError):
            return ""

    def record_current_generation(self) -> None:
        """把运行结束时的活动 generation 落盘（重接地/发布的触发基准；公共入口，
        run/publish_approved/`--revalidate` CLI 共用；重接地未完成或失败时不得调用）。"""
        backend = settings.rag_backend.lower()
        active = self._generation_store.active(backend)
        if active is None:
            return
        path = self._state_dir / "last_human_generation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {"backend": backend, "generation_id": active.generation_id},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        self._clear_revalidate_progress()

    def _recover_revalidate(self, entry: dict) -> dict:
        """revalidate 隔离事务的前进式恢复（完成隔离 + 一次重建）。

        只有 ``recover_revalidate`` 明确返回 success 才归档 journal；pointer
        补写失败、alias 未知/blocked 或重建失败均保留现场并由调用方终止本轮。
        """
        from app.evolution.revalidate import recover_revalidate

        result = recover_revalidate(
            entry,
            kb_dir=self._kb_dir,
            trash_dir=self._state_dir / "trash",
            ledger=self._ledger,
            publisher=self._publisher,
            index_service=self._index_service,
            generation_store=self._generation_store,
            backend=entry.get("backend") or settings.rag_backend.lower(),
            lifecycle=self._lifecycle,
        )
        if not result.get("success", False):
            if result.get("blocked"):
                self._block_kb_writes(entry)
            return result
        self._journal.archive()
        if result.get("already_done"):
            log.info("🔁 journal 恢复：revalidate 重建此前已生效，ledger 无需补齐")
        else:
            log.info(
                "🔁 journal 恢复：revalidate 隔离已前进式补齐"
                f"（{len(result.get('retired', []))} 篇 + 索引重建）"
            )
        return result

    def _delete_staging_index(self, backend: str, index_info) -> bool:
        if not index_info:
            return True
        if backend == "numpy" and index_info.get("target"):
            Path(index_info["target"]).unlink(missing_ok=True)
            return True
        if backend == "chroma" and index_info.get("target"):
            try:
                import chromadb

                client = chromadb.PersistentClient(
                    path=str(Path(settings.chroma_persist_dir))
                )
                client.delete_collection(index_info["target"])
                return True
            except Exception as e:  # noqa: BLE001 - resource cleanup is retried by journal
                if _resource_not_found(e, "collection"):
                    # 首次清理成功、随后文件回滚失败时，第二次恢复会收到
                    # NotFound；collection 已不存在即是幂等成功。
                    return True
                return _resource_not_found(e, "collection")
        if backend == "es" and index_info.get("target"):
            delete = getattr(self._index_service, "delete_candidate", None)
            if not callable(delete):
                return False
            return bool(delete("es", index_info["target"]))
        # 没有候选 target 或未知后端时没有可清理对象；未知 target 不宣称成功。
        return not bool(index_info.get("target"))

    def _run_eval(self) -> dict | None:
        """跑一遍沙箱评测（活动索引）；返回 evaluator.run_all 完整报告；无 factory 返回 None。"""
        if self._evaluator_factory is None:
            return None
        evaluator = self._evaluator_factory()
        return evaluator.run_all(self._eval_cases)

    # ============================================================
    # 4：挖掘
    # ============================================================
    def _mine(self, report: EvolutionReport) -> list[TurnRecord]:
        turns = mine_turns(self._turns_dir, self._ledger.processed_set())
        for session_path in self._session_paths:
            p = Path(session_path)
            if p.exists():
                turns.extend(scan_legacy_session(p))
        report.mined = len(turns)
        return turns

    # ============================================================
    # 5：规则过滤 + 精确去重
    # ============================================================
    def _filter(
        self, turns: list[TurnRecord], report: EvolutionReport
    ) -> list[CandidateQA]:
        candidates: list[CandidateQA] = []
        for turn in turns:
            candidate, reason = build_candidate(turn)
            if candidate is None:
                report.skipped[reason] = report.skipped.get(reason, 0) + 1
                continue
            if dedup.exact(candidate.candidate_id, self._ledger):
                report.skipped["duplicate"] = report.skipped.get("duplicate", 0) + 1
                continue
            candidates.append(candidate)
        report.pre_deduped = len(candidates)
        return candidates

    # ============================================================
    # 6：预去重（0.95，原始脱敏问题）
    # ============================================================
    def _pre_dedup(self, candidates, report, abort_on_error: bool) -> list[CandidateQA]:
        if not candidates:
            return candidates
        try:
            retriever = self._retriever_factory()
            kept = []
            for c in candidates:
                question = c.raw_question or c.question
                if dedup.pre_dedup(
                    question, retriever, settings.evolve_pre_dedup_threshold
                ):
                    report.skipped["duplicate"] = report.skipped.get("duplicate", 0) + 1
                else:
                    kept.append(c)
            return kept
        except Exception as e:
            report.failures += 1
            if abort_on_error:
                raise RuntimeError(
                    f"embedding/检索不可用（{type(e).__name__}: {e}），"
                    f"本次运行中止（未写任何文件）"
                ) from e
            log.info(f"⚠️  dedup_unavailable（dry-run 继续）: {e}")
            return candidates

    # ============================================================
    # 7-9：排序 + 价值 Judge + 接地 Judge
    # ============================================================
    def _judge_round(self, candidates, report: EvolutionReport) -> list[CandidateQA]:
        # 预排序：来源分 + 置信度（稳定排序保持时间序），前 N 个进 Judge
        candidates.sort(
            key=lambda c: (
                max((s.score for s in c.sources), default=0.0),
                c.confidence,
            ),
            reverse=True,
        )
        candidates = candidates[: settings.evolve_max_judge_per_run]
        report.judged = min(len(candidates), settings.evolve_max_judge_per_run)

        survivors: list[CandidateQA] = []
        quality: dict[str, float] = {}
        for c in candidates:
            decision = self._value_judge.judge(c, c.sources)
            report.api_calls += 1
            if decision.failed:
                self._ledger.add_pending(c, reason="judge_failed")
                report.pending += 1
                self._per_candidate(report, c, "pending", "judge_failed")
                continue
            if not decision.worth_saving:
                report.skipped["judge_rejected"] = (
                    report.skipped.get("judge_rejected", 0) + 1
                )
                self._per_candidate(
                    report, c, "skipped", decision.reason or "worth_saving=false"
                )
                continue
            if decision.quality_score < settings.evolve_min_quality:
                # 独立的质量分闸门：有价值但质量不足 → 不自动发布
                report.skipped["judge_rejected"] = (
                    report.skipped.get("judge_rejected", 0) + 1
                )
                self._per_candidate(
                    report,
                    c,
                    "skipped",
                    f"quality={decision.quality_score:.2f}<{settings.evolve_min_quality}",
                )
                continue

            c.question = decision.question
            c.answer = decision.answer
            c.quality_score = decision.quality_score
            quality[c.candidate_id] = decision.quality_score

            grounding = self._grounding_judge.judge(c.answer, c.sources)
            report.api_calls += 1
            if not grounding["grounded"]:
                # 值得保存但缺证据（含 no_human_sources / 断言无支撑 / judge 失败）：
                # 这正是 pending 人工审核通道的核心用途，不能直接丢弃
                self._ledger.add_pending(
                    c,
                    reason=f"ungrounded:{grounding.get('reason', 'grounding_failed')}",
                )
                report.pending += 1
                self._per_candidate(
                    report, c, "pending", grounding.get("reason", "grounding_failed")
                )
                continue
            survivors.append(c)
        self._quality = quality
        return survivors

    @staticmethod
    def _per_candidate(
        report: EvolutionReport, c: CandidateQA, status: str, detail: str
    ) -> None:
        report.per_candidate.append(
            {
                "candidate_id": c.candidate_id,
                "turn_id": c.turn_id,
                "status": status,
                "detail": detail,
            }
        )

    # ============================================================
    # 10：最终去重 + 本轮互查
    # ============================================================
    def _final_dedup(
        self, survivors, report, abort_on_error: bool
    ) -> list[CandidateQA]:
        if not survivors:
            return survivors
        try:
            retriever = self._retriever_factory()
            kept = []
            for c in survivors:
                hit, side = dedup.final_dedup(
                    c.question, c.answer, retriever, settings.evolve_dedup_threshold
                )
                if hit is not None:
                    replaces = self._replacement_for(c, hit, side)
                    if replaces is None:
                        report.skipped["duplicate"] = (
                            report.skipped.get("duplicate", 0) + 1
                        )
                        continue
                    c.replaces = replaces
                kept.append(c)
            if not kept:
                return kept
            vectors = self._embedder.encode([f"{c.question}\n{c.answer}" for c in kept])
            metas = [self._meta(c) for c in kept]
            dropped = dedup.in_run_pairwise(
                vectors, metas, threshold=settings.evolve_dedup_threshold
            )
            report.skipped["duplicate"] = report.skipped.get("duplicate", 0) + len(
                dropped
            )
            return [c for i, c in enumerate(kept) if i not in dropped]
        except Exception as e:
            report.failures += 1
            if abort_on_error:
                raise RuntimeError(
                    f"最终去重不可用（{type(e).__name__}: {e}），本次运行中止"
                ) from e
            return survivors

    def _meta(self, c: CandidateQA) -> dict:
        quality = getattr(self, "_quality", {}).get(c.candidate_id, 0.0)
        source_score = max((s.score for s in c.sources), default=0.0)
        return {
            "quality_score": quality,
            "source_score": source_score,
            "confidence": c.confidence,
        }

    def _replacement_for(self, c: CandidateQA, hit, side: str) -> str | None:
        """近重复命中的替换判定（P2-1）。

        仅**问题侧**命中可替换——答案侧命中只说明回答模板化（固定话术），
        问题可能完全不同，替换会误删回答另一个问题的旧沉淀。
        命中 evolved/ 旧沉淀且新候选质量分 ≥ 旧文档 frontmatter 的 quality_score
        （旧文档缺字段视为可替换）→ 返回旧文件名（发布时替换）；
        命中人工文档（根目录 / uploads/）或旧分更高 → None（仍按重复丢弃）。
        """
        if side != "question":
            return None
        path = getattr(getattr(hit, "chunk", None), "source_path", "") or ""
        if not path.startswith("evolved/"):
            return None
        filename = path[len("evolved/") :]
        if c.quality_score < self._evolved_quality(filename):
            return None
        return filename

    def _evolved_quality(self, filename: str) -> float:
        """旧 evolved 文档 frontmatter 的 quality_score；缺失/损坏 → -1.0（视为可替换）。"""
        from app.agent.rag.loader import parse_frontmatter as _parse_fm

        path = self._kb_dir / "evolved" / filename
        try:
            meta, _ = _parse_fm(path.read_text(encoding="utf-8"))
            return float(meta.get("quality_score", ""))
        except (OSError, ValueError, TypeError):
            return -1.0

    # ============================================================
    # 11：容量截断
    # ============================================================
    def _cap(self, survivors, report: EvolutionReport):
        limit = settings.evolve_max_per_run
        selected = survivors[:limit]
        for c in survivors[limit:]:
            self._ledger.add_pending(c, reason="capacity")
            report.pending += 1
            self._per_candidate(report, c, "pending", "capacity")
        return selected, report

    # ============================================================
    # 12-13：渲染复扫 → journal → 移入 → staging 索引 → 评测探针 → 激活
    # ============================================================
    def _publish(
        self,
        selected: list[CandidateQA],
        turns: list[TurnRecord],
        report: EvolutionReport,
        before_detail,
        with_eval: bool,
    ) -> None:
        if not selected:
            self._finalize(report, turns)
            return

        entries: list[tuple[CandidateQA, str]] = []
        for c in selected:
            filename = self._publisher.write_staging(c)
            if filename is None:
                report.skipped["sensitive"] = report.skipped.get("sensitive", 0) + 1
                self._per_candidate(report, c, "skipped", "sensitive_in_rescan")
                continue
            entries.append((c, filename))
        if not entries:
            self._finalize(report, turns)
            return
        for c, _ in entries:
            self._per_candidate(report, c, "staged", "")

        # P2-1：近重复新答案替换旧 evolved 沉淀（去重阶段已标记 c.replaces）
        replaced_docs: list[str] = list(
            dict.fromkeys(c.replaces for c, _ in entries if c.replaces)
        )

        backend = settings.rag_backend.lower()
        generation_id = new_generation_id(self._clock)
        index_info = GenerationInfo(
            generation_id=generation_id,
            target=self._index_service.target_for(backend, generation_id),
            embedding_model=self._embedder.model,
        )
        # 2.7：记录发布前的活动代（恢复判定「非旧代」用）
        previous = self._generation_store.active(backend)
        previous_target = previous.target if previous is not None else ""

        # 阶段 1/5：文档移动前写 PREPARED journal
        journal_body = {
            "phase": "publish",
            "stage": PH_PREPARED,
            "staging_docs": [
                {"candidate_id": c.candidate_id, "filename": f} for c, f in entries
            ],
            "replaced_docs": replaced_docs,
            "backend": backend,
            "index": index_info.to_dict(),
            "previous_target": previous_target,
        }
        self._journal.write(journal_body)
        self._lock.assert_held()
        from app.observability.metrics import set_evolution_phase

        set_evolution_phase(PH_PREPARED)

        for _, filename in entries:
            self._publisher.move_into_kb(filename)
        # 替换：旧文档移 trash（在索引 build 之前 → 新代只含新文档）
        trash_dir = self._state_dir / "trash"
        for old in replaced_docs:
            self._publisher.unpublish(old, trash_dir)

        # 阶段 2/5：构建并验证候选索引（不激活）
        self._lock.assert_held()
        info = self._index_service.build(backend, generation_id=generation_id)
        self._journal.write({**journal_body, "stage": PH_INDEX_BUILT})
        set_evolution_phase(PH_INDEX_BUILT)

        if with_eval and before_detail is not None:
            detail = self._eval_after_and_probe(entries, info, before_detail)
            self.eval_detail = detail
            if detail["blocked"]:
                self._mark_blocked_pending(entries, report, detail)
                self._recover_publish_transaction(self._journal.read())
                self._write_report(report, suffix="-blocked")
                return  # 不 mark processed：下次运行靠 pending 精确去重零成本跳过

        # 阶段 3/5：ES 先切 alias，读真实 alias 确认后写 ALIAS_ACTIVATED
        self._lock.assert_held()
        self._index_service.activate_alias(backend, info)
        self._lock.assert_held()
        if backend == "es":
            actual = self._read_alias_target(backend)
            if actual != info.target:
                raise RuntimeError(
                    f"alias 确认失败：期望 {info.target}，实际 {actual or '(空)'}；"
                    "按恢复表处理（journal 保留）"
                )
        self._journal.write({**journal_body, "stage": PH_ALIAS_ACTIVATED})
        set_evolution_phase(PH_ALIAS_ACTIVATED)

        # 阶段 4/5：写 generation pointer
        self._lock.assert_held()
        self._index_service.activate_pointer(backend, info)
        self._journal.write({**journal_body, "stage": PH_POINTER_UPDATED})
        set_evolution_phase(PH_POINTER_UPDATED)

        # 阶段 5/5：替换结算 + 提交 ledger（幂等批量）
        retired: list[tuple[str, str]] = []
        for old in replaced_docs:
            # 结算按旧文档正本路由（lifecycle）：human 正本 → MySQL
            # published→superseded；ledger 正本 → batch_cleanup_published。
            # 旧文档在阶段 1/5 已移 trash，显式带 frontmatter 防止误路由。
            if self._lifecycle is not None and self._lifecycle.settle_replacement(
                old, self._new_numeric_cid(entries, old),
                metadata=self._replaced_doc_meta(old),
            ):
                continue
            # 回退：lifecycle 未注入（极简测试依赖）或未命中正本
            cid = self._cid_for_filename(old)
            if cid:
                retired.append((old, cid))
        if retired:
            self._ledger.batch_cleanup_published(retired)
        self._ledger.commit_publish(
            [(c.candidate_id, filename) for c, filename in entries]
        )
        self._journal.write({**journal_body, "stage": PH_LEDGER_COMMITTED})
        set_evolution_phase(PH_LEDGER_COMMITTED)

        for c, filename in entries:
            report.per_candidate = [
                p
                for p in report.per_candidate
                if not (p["candidate_id"] == c.candidate_id and p["status"] == "staged")
            ]
            detail = filename + (f"·replaced:{c.replaces}" if c.replaces else "")
            self._per_candidate(report, c, "published", detail)
        report.sedimented = len(entries)
        report.replaced = len(replaced_docs)
        # 完成报告后清 journal（事务终态）；阶段指标归零
        self._journal.clear()
        set_evolution_phase("")
        self._finalize(report, turns)

    def _cid_for_filename(self, filename: str) -> str | None:
        """ledger.published 反查 candidate_id；不存在返回 None。"""
        return next(
            (cid for cid, f in self._ledger.published().items() if f == filename),
            None,
        )

    @staticmethod
    def _new_numeric_cid(entries: list[tuple[CandidateQA, str]], old: str) -> int | None:
        """被替换旧文档对应新候选的数值 id（MySQL superseded 指向用）。

        机器人候选 id 非数值（跨 LLM 稳定哈希）→ None。
        """
        for c, _fname in entries:
            if c.replaces == old:
                try:
                    return int(str(c.candidate_id))
                except (TypeError, ValueError):
                    return None
        return None

    def _replaced_doc_meta(self, old: str) -> dict | None:
        """被替换旧文档的 frontmatter（evolved/ 已移 trash 时读 trash 副本）。

        旧文档在发布阶段 1/5 就已 unpublish 进 trash，resolve 只看
        evolved/ 现存文件读不到 frontmatter，会把人工文档误路由到 Ledger；
        metadata 显式携带 frontmatter 保证正本路由正确（journal 恢复路径
        同样依赖 trash 副本）。
        """
        for base in (self._kb_dir / "evolved", self._state_dir / "trash"):
            try:
                meta, _body = parse_frontmatter(
                    (base / old).read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if meta:
                return meta
        return None

    def _mark_blocked_pending(
        self, entries, report: EvolutionReport, detail: dict
    ) -> None:
        """with-eval 阻断 → 候选进 pending（reason=eval_blocked:<主要阻断原因>）。

        不 mark processed 的代价是下轮重挖重审（temperature=0 决策确定，结果相同）
        而持续烧 API；进 pending 后 dedup.exact 命中 → 下轮零成本跳过，
        人工凭 -blocked 报告的 before/after 在审核后台 approve/reject。
        """
        reasons = detail.get("reasons") or []
        probes = detail.get("probe_failures") or []
        primary = (
            reasons[0] if reasons else (f"probe:{probes[0]}" if probes else "blocked")
        )
        for c, _ in entries:
            self._ledger.add_pending(c, reason=f"eval_blocked:{primary}")
            report.pending += 1
            self._per_candidate(report, c, "pending", f"eval_blocked:{primary}")

    def _finalize(self, report: EvolutionReport, turns: list[TurnRecord]) -> None:
        self._ledger.mark_processed([t.turn_id for t in turns])
        self._write_report(report)

    def _write_report(self, report: EvolutionReport, suffix: str = "") -> None:
        run_id = self._now().strftime("%Y%m%d%H%M%S") + suffix
        self._ledger.write_report(self._output_dir, report, run_id)

    # ============================================================
    # with_eval：staging 评测 + 候选探针
    # ============================================================
    def _eval_after_and_probe(
        self, entries, info: GenerationInfo, before_detail: dict
    ) -> dict:
        """push staging override → after 评测 + 候选探针 → pop。返回阻断详情。

        2.6：候选检索器统一经 index_service.open_retriever 装配——
        ES 直接以 index_name=info.target 查询（不经过活动 alias），
        hybrid/reranker/recall_k 与线上一致（不再是 `else -> chroma` 兜底）。
        探针记录：原问题排名、规范化问题排名、rerank 分数、命中 source_path。
        """
        staged_retriever = self._index_service.open_retriever(info)
        staged_retriever.load()

        from app.agent.tools import knowledge as knowledge_tool

        knowledge_tool.push_retriever_override(staged_retriever)
        try:
            after_detail = self._run_eval()
        finally:
            knowledge_tool.pop_retriever_override()
        if after_detail is None:
            return {
                "blocked": False,
                "before": before_detail,
                "after": None,
                "reasons": ["eval_unavailable"],
                "probe_failures": [],
                "probe_records": [],
            }

        reasons: list[str] = []
        before_by_case = {c["case_id"]: c for c in before_detail["cases"]}
        for case in after_detail["cases"]:
            b = before_by_case.get(case["case_id"])
            if b and b.get("passed") and not case.get("passed"):
                reasons.append(f"regression:{case['case_id']}")
        errors_before = sum(1 for c in before_detail.get("cases", []) if c.get("error"))
        errors_after = sum(1 for c in after_detail.get("cases", []) if c.get("error"))
        if errors_after > errors_before:
            reasons.append(f"errors_up:{errors_before}->{errors_after}")
        rate_before = before_detail["summary"].get("pass_rate", 0.0)
        rate_after = after_detail["summary"].get("pass_rate", 0.0)
        if rate_after < rate_before:
            reasons.append(f"pass_rate_down:{rate_before:.3f}->{rate_after:.3f}")
        score_before = before_detail["summary"].get("avg_result_score") or 0.0
        score_after = after_detail["summary"].get("avg_result_score") or 0.0
        if score_after < score_before - EVAL_SCORE_DROP:
            reasons.append(f"score_drop:{score_before:.3f}->{score_after:.3f}")

        # 候选探针：每条候选的两路问题都经统一最终口径（final_search：门控 +
        # 父块去重 + Top-K）在 Top-K 内命中目标文档才放行。
        # 记录排名与精排分数（rerank 后 hit.score），供阻断报告与人工复核。
        from app.agent.rag.retriever_factory import final_search

        probe_failures: list[str] = []
        probe_records: list[dict] = []
        probe_top_k = 5
        for c, filename in entries:
            target_path = f"evolved/{filename}"
            for label, q in (
                ("original", c.raw_question or c.question),
                ("normalized", c.question),
            ):
                outcome = final_search(
                    staged_retriever,
                    q,
                    probe_top_k,
                    min_score=settings.rag_min_relevance_score,
                )
                rank = 0
                score = None
                for i, h in enumerate(outcome.hits, start=1):
                    if h.chunk.source_path == target_path:
                        rank, score = i, float(h.score)
                        break
                probe_records.append(
                    {
                        "candidate_id": c.candidate_id[:12],
                        "filename": filename,
                        "label": label,
                        "rank": rank,
                        "rerank_score": round(score, 4) if score is not None else None,
                        "hit_source_path": target_path if rank else "",
                    }
                )
                if rank == 0:
                    probe_failures.append(f"{c.candidate_id[:12]}:{label}:not_in_top")

        blocked = bool(reasons or probe_failures)
        return {
            "blocked": blocked,
            "before": before_detail,
            "after": after_detail,
            "reasons": reasons,
            "probe_failures": probe_failures,
            "probe_records": probe_records,
        }

    # ============================================================
    # 回滚
    # ============================================================
    def _rollback_publish(self, report: EvolutionReport | None = None) -> None:
        """异常收尾：2.7 起调用统一恢复决策，不再自行假设「pointer 未切即回滚」。

        - revalidate 隔离事务不做回滚（判定已做出，前进式补齐）；
        - publish 事务按 recover_decide 三源判定：alias 已切只前进、未知代
          阻塞、未切才回滚。
        """
        entry = self._journal.read()
        if not entry:
            return
        if entry.get("phase") == "revalidate":
            # 隔离事务不做回滚：判定已做出，前进式恢复保住已完成的淘汰
            result = self._recover_revalidate(entry)
            if not result.get("success", False):
                raise EvolutionRecoveryPendingError(
                    result.get("reason") or "revalidate 恢复未明确成功，journal 已保留"
                )
            return
        self._recover_publish_transaction(entry, report)

    # ============================================================
    # 人工 approve：走同一发布链路（跳过 Judge，计入 max_per_run）
    # ============================================================
    def publish_approved(
        self, candidates: list[CandidateQA], *, lock_held: bool = False
    ) -> EvolutionReport:
        """审核通过的候选直接发布：最终去重 → 渲染复扫 → journal → 索引 → 激活。

        lock_held=True：调用方（ReviewService）已持同一写锁，并在锁内完成
        「读取最终版本 → 去重 → 发布 → 清账」——本方法跳过 acquire/release
        （file 锁不可重入；mysql/redis 同实例可重入，统一走显式参数）。
        """
        if not candidates:
            raise ValueError("没有要发布的候选")
        report = EvolutionReport()
        if not lock_held:
            self._lock.acquire(phase="approve")
        try:
            self._assert_kb_writes_allowed()
            self._recover_journal(report)
            kept: list[CandidateQA] = []
            try:
                retriever = self._retriever_factory()
                for c in candidates:
                    hit, side = dedup.final_dedup(
                        c.question, c.answer, retriever, settings.evolve_dedup_threshold
                    )
                    if hit is not None:
                        replaces = self._replacement_for(c, hit, side)
                        if replaces is None:
                            report.skipped["duplicate"] = (
                                report.skipped.get("duplicate", 0) + 1
                            )
                            self._per_candidate(
                                report, c, "skipped", "duplicate_on_approve"
                            )
                            # 终结出 pending：审核时发现重复 → rejected（原因持久化）
                            self._ledger.mark_rejected(
                                c.candidate_id,
                                reason="duplicate_on_approve",
                            )
                            continue
                        c.replaces = replaces
                    kept.append(c)
            except Exception as e:
                raise RuntimeError(
                    f"approve 发布失败：检索不可用（{type(e).__name__}: {e}）"
                ) from e
            if not kept:
                self._write_report(report, suffix="-approve")
                return report
            limited = kept[: settings.evolve_max_per_run]
            overflow = kept[settings.evolve_max_per_run :]
            for c in overflow:
                report.pending += 1
                self._per_candidate(report, c, "pending", "capacity_on_approve")
            self._publish(
                limited, turns=[], report=report, before_detail=None, with_eval=False
            )
            return report
        except (EvolutionBlockedError, EvolutionRecoveryPendingError):
            raise
        except Exception:
            self._rollback_publish()
            raise
        finally:
            # approve 切代后同样记录，避免下一次 run 误判为「人工知识变更」
            try:
                self.record_current_generation()
            except Exception:  # noqa: BLE001 - generation bookkeeping is best effort
                log.info("⚠️  记录 last_human_generation 失败（不影响本次 approve）")
            if not lock_held:
                self._lock.release()

    # ============================================================
    # dry-run：零写入、不取锁、完整报告 + 预计 API 调用数
    # ============================================================
    def _dry_run(self) -> EvolutionReport:
        report = EvolutionReport()
        turns = self._mine(report)
        candidates = self._filter(turns, report)
        candidates = self._pre_dedup(candidates, report, abort_on_error=False)
        candidates.sort(
            key=lambda c: (
                max((s.score for s in c.sources), default=0.0),
                c.confidence,
            ),
            reverse=True,
        )
        candidates = candidates[: settings.evolve_max_judge_per_run]
        report.judged = len(candidates)
        report.api_calls = len(candidates) * 2  # 价值 Judge + 接地 Judge 的预计调用数
        report.sedimented = min(len(candidates), settings.evolve_max_per_run)
        return report

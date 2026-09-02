"""第10期 QA 自动沉淀 CLI。

用法：
  # 预览（知识库/索引/ledger 零写入、不取锁；S3 后端会做 turns 幂等缓存下载）
  python app/scripts/run_evolution.py --dry-run

  # 正式运行（需 SELF_EVOLVE_ENABLED=true；带评测则退出码 2 表示阻断）
  # S3 后端（TURNS_ARCHIVE_BACKEND=s3）运行前先从归档同步 turns（幂等）
  python app/scripts/run_evolution.py
  python app/scripts/run_evolution.py --with-eval

  # pending 审核与运维
  python app/scripts/run_evolution.py --list-pending
  python app/scripts/run_evolution.py --approve <cid> [cid...]
  python app/scripts/run_evolution.py --reject <cid> [cid...]
  python app/scripts/run_evolution.py --unpublish <cid>
  python app/scripts/run_evolution.py --revalidate
  python app/scripts/run_evolution.py --prune-turns --older-than 90
  python app/scripts/run_evolution.py --prune-pending
  python app/scripts/run_evolution.py --force-unlock

main(argv, services=None) 支持依赖注入（CI 用 Fake services 跑 dry-run）。
退出码：0 成功 / 1 失败 / 2 with-eval 阻断。
"""

import argparse
import sys
from pathlib import Path
from app.observability.logging import get_logger
log = get_logger("app.scripts.run_evolution")


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.evolution.lock import CrossHostLockError, LockHeldError  # noqa: E402


# ============================================================
# 默认服务装配（真实依赖）
# ============================================================
def _build_services() -> dict:
    from openai import OpenAI

    from app.agent.rag.embedder import create_embedder
    from app.agent.rag.parsers import chunk_kb_dir
    from app.evolution.generation import GenerationStore
    from app.evolution.index_service import IndexBuildService
    from app.evolution.judges import GroundingJudge, ValueJudge
    from app.evolution.ledger import Ledger
    from app.evolution.lock import Journal
    from app.evolution.pipeline import EvolutionPipeline
    from app.evolution.publisher import Publisher

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    embedder = create_embedder()
    # 阶段二 2.4 / 阶段六 6.2：Redis 可用 → 指针共享 + 锁换 Redis（跨 pod）；
    # 无 Redis（开发/单机）保持文件实现
    from app.stores.redis_client import get_redis

    redis = get_redis()
    # v7 统一写锁 + 共享指针 fail-closed（生产路径 strict_shared=True）
    from app.stores.kb_write_lock import get_kb_write_lock
    from app.stores.sql.engine import get_engine

    engine = get_engine()
    strict_shared = redis is not None
    generation_store = GenerationStore(
        ROOT / settings.kb_generation_path, redis_client=redis,
        strict_shared=strict_shared,
    )
    index_service = IndexBuildService(
        embedder=embedder,
        kb_dir=ROOT / settings.kb_dir,
        generation_store=generation_store,
        # v7：自进化重建与上传/手工构建同一约束——多格式扫描 + strict
        # （否则重建会静默丢掉 uploads/ 里的 pdf/docx，或跳过坏文件）
        chunker=chunk_kb_dir,
        strict_build=True,
    )
    ledger = Ledger(ROOT / settings.evolve_state_dir)
    lock = get_kb_write_lock(engine=engine, redis=redis, ttl_seconds=None)
    journal = Journal(ROOT / settings.evolve_state_dir / "journal.json")
    publisher = Publisher(
        kb_dir=ROOT / settings.kb_dir,
        staging_dir=ROOT / settings.evolve_state_dir / "staging",
    )

    def retriever_factory():
        from app.agent.tools import knowledge as knowledge_tool

        return knowledge_tool._get_retriever()

    def evaluator_factory():
        from app.evaluation.evaluator import Evaluator
        from app.evaluation.sandbox import Sandbox

        sandbox = Sandbox(mode="multi" if settings.multi_agent_enabled else "single")
        return Evaluator(
            sandbox=sandbox,
            client=OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            ),
            model=settings.model_name,
            use_judge=True,
            pass_threshold=settings.eval_pass_threshold,
        )

    def eval_cases():
        from app.evaluation.dataset import load_dataset

        dataset_path = ROOT / settings.eval_dataset_path
        if not dataset_path.exists():
            return []
        return load_dataset(dataset_path)

    pipeline = EvolutionPipeline(
        value_judge=ValueJudge(client, settings.model_name),
        grounding_judge=GroundingJudge(client, settings.model_name),
        embedder=embedder,
        retriever_factory=retriever_factory,
        index_service=index_service,
        generation_store=generation_store,
        ledger=ledger,
        lock=lock,
        journal=journal,
        publisher=publisher,
        turns_dir=ROOT / settings.evolve_turns_dir,
        kb_dir=ROOT / settings.kb_dir,
        state_dir=ROOT / settings.evolve_state_dir,
        output_dir=ROOT / settings.evolve_output_dir,
        session_paths=[ROOT / settings.session_path],
        evaluator_factory=evaluator_factory,
        eval_cases=eval_cases(),
    )
    return {
        "pipeline": pipeline,
        "ledger": ledger,
        "lock": lock,
        "journal": journal,
        "publisher": publisher,
        "index_service": index_service,
        "generation_store": generation_store,
        "retriever_factory": retriever_factory,
        "grounding_judge": GroundingJudge(client, settings.model_name),
        "kb_dir": ROOT / settings.kb_dir,
        "turns_dir": ROOT / settings.evolve_turns_dir,
        "state_dir": ROOT / settings.evolve_state_dir,
    }


# ============================================================
# 报告打印
# ============================================================
def _print_report(report, prefix: str = "  ") -> None:
    s = report.skipped
    log.info(f"{prefix}挖掘 turn 数        : {report.mined}")
    log.info(f"{prefix}规则过滤后候选      : {report.pre_deduped}")
    log.info(f"{prefix}进入 Judge 的候选   : {report.judged}")
    log.info(f"{prefix}Judge API 调用      : {report.api_calls}")
    log.info(f"{prefix}跳过明细            : "
          f"低置信 {s.get('low_confidence', 0)} / 转人工 {s.get('requires_human', 0)} / "
          f"无来源 {s.get('no_sources', 0)} / 过短 {s.get('short', 0)} / "
          f"敏感 {s.get('sensitive', 0)} / 重复 {s.get('duplicate', 0)} / "
          f"Judge 拒 {s.get('judge_rejected', 0)}")
    log.info(f"{prefix}pending（待审核）    : {report.pending}")
    log.info(f"{prefix}发布文档数          : {report.sedimented}")
    if report.revalidated_checked:
        log.info(f"{prefix}重接地核对          : {report.revalidated_checked}"
                 f"（通过 {report.revalidated_passed} / 隔离 {report.revalidated_failed} / "
                 f"剩余 {report.revalidated_remaining}）")
    if report.failures:
        log.info(f"{prefix}⚠️  异常次数: {report.failures}")


def _print_eval_detail(detail: dict) -> None:
    before = detail.get("before") or {}
    after = detail.get("after") or {}
    bs = before.get("summary", {})
    as_ = after.get("summary", {})
    log.info("\n  评测对比（before → after）")
    log.info(f"    通过率    : {bs.get('pass_rate', 0) * 100:.0f}% → {as_.get('pass_rate', 0) * 100:.0f}%")
    log.info(f"    结果得分  : {bs.get('avg_result_score')} → {as_.get('avg_result_score')}")
    log.info(f"    阻断原因  : {detail.get('reasons') or '无'}")
    log.info(f"    探针失败  : {detail.get('probe_failures') or '无'}")
    for case in (after or {}).get("cases", []):
        before_case = next(
            (c for c in (before or {}).get("cases", []) if c["case_id"] == case["case_id"]),
            None,
        )
        flag = "❌" if not case.get("passed") else "✓"
        b_pass = "✓" if before_case and before_case.get("passed") else "✗" if before_case else "?"
        log.info(f"    {case['case_id']:<22} {b_pass} → {flag}  "
              f"{case.get('error') or ''}")


# ============================================================
# 子命令
# ============================================================
def _gate() -> bool:
    if settings.self_evolve_enabled:
        return True
    log.info("❌ SELF_EVOLVE_ENABLED=false：写操作被安全开关拒绝。"
          "设置环境变量 SELF_EVOLVE_ENABLED=true 后再试。")
    return False


def _sync_turns(svc: dict) -> int:
    """S3 后端时把归档 turns 同步到本地矿工目录（幂等缓存下载）；local 后端跳过。

    fail-closed：S3 不可用时拒绝空跑（同步失败说明本次必然挖不到任何 turn）。
    """
    from app.stores.object_store import ObjectStoreUnavailable
    from app.evolution.turn_sync import build_turns_archive, sync_turns_from_archive

    try:
        store = build_turns_archive()
    except ObjectStoreUnavailable as e:
        raise RuntimeError(
            f"S3 turns 归档不可用（{e}），拒绝静默空跑："
            "请检查 S3 配置或改用 TURNS_ARCHIVE_BACKEND=local"
        ) from e
    if store is None:
        return 0
    return sync_turns_from_archive(store, svc["turns_dir"], svc["state_dir"])


def _cmd_run(svc: dict, dry_run: bool, with_eval: bool) -> int:
    pipeline = svc["pipeline"]
    synced = _sync_turns(svc)
    if synced:
        log.info(f"  S3 turns 同步    : 新增 {synced} 条（幂等缓存下载，非知识库写入）")
    if dry_run:
        report = pipeline.run(dry_run=True)
        log.info("=" * 60)
        log.info("  dry-run 预览（零写入、不取锁）")
        log.info("=" * 60)
        _print_report(report)
        log.info(f"  预计发布      : ≤{report.sedimented} 篇")
        log.info(f"  预计 API 调用 : {report.api_calls} 次（价值+接地 Judge）")
        log.info("\n  ✅ 预览完成，未写任何文件。")
        return 0

    if not _gate():
        return 1
    report = pipeline.run(with_eval=with_eval)
    log.info("=" * 60)
    log.info("  第10期 QA 自动沉淀 · 运行报告")
    log.info("=" * 60)
    _print_report(report)
    detail = getattr(pipeline, "eval_detail", None)
    if detail and detail.get("blocked"):
        log.info("\n  ❌ with-eval 阻断：staging 已清理，generation 未切换（旧索引可用）")
        _print_eval_detail(detail)
        return 2
    if detail:
        _print_eval_detail(detail)
    log.info("\n  🎉 运行完成。")
    return 0


def _cmd_list_pending(svc: dict) -> int:
    ledger = svc["ledger"]
    entries = ledger.list_pending(settings.evolve_pending_aging_days)
    if not entries:
        log.info("  pending 为空。")
        return 0
    log.info(f"  pending 共 {len(entries)} 条（超过 {settings.evolve_pending_aging_days} 天标记 aging）：")
    for cid, entry in entries:
        status = entry.get("status", "pending")
        log.info(f"    [{status}] {cid}  {entry.get('question', '')[:40]}")
        log.info(f"            原因: {entry.get('reason', '')}  创建: {entry.get('created_at', '')}  trusted: {entry.get('human_trusted', False)}")
    return 0


def _cmd_approve(svc: dict, cids: list[str]) -> int:
    if not _gate():
        return 1
    ledger = svc["ledger"]
    pipeline = svc["pipeline"]
    published = ledger.published()
    candidates = []
    for cid in cids:
        candidate = ledger.approve(cid)
        if candidate is None:
            log.info(f"  ⚠️  pending 中不存在: {cid}")
            continue
        if cid in published:
            log.info(f"  ⚠️ 已发布过: {cid}")
            continue
        candidates.append(candidate)
    if not candidates:
        log.info("  没有可发布的候选。")
        return 0
    report = pipeline.publish_approved(candidates)
    log.info("=" * 60)
    log.info("  approve 发布报告")
    log.info("=" * 60)
    _print_report(report)
    return 0


def _cmd_reject(svc: dict, cids: list[str]) -> int:
    if not _gate():
        return 1
    ledger = svc["ledger"]
    for cid in cids:
        ledger.reject(cid)
        log.info(f"  🗑️  已拒绝并永久跳过: {cid}")
    return 0


def _cmd_unpublish(svc: dict, cid: str) -> int:
    if not _gate():
        return 1
    ledger = svc["ledger"]
    publisher = svc["publisher"]
    index_service = svc["index_service"]
    lock = svc["lock"]

    filename = ledger.published().get(cid)
    if not filename:
        log.info(f"  ❌ 未找到已发布记录: {cid}")
        return 1

    # 与 pipeline / build_kb_index 共用单写者锁：重建索引期间禁止并发写
    lock.acquire(phase="unpublish")
    try:
        trash_dir = ROOT / settings.evolve_state_dir / "trash"
        rel = publisher.unpublish(filename, trash_dir)
        if rel is None:
            log.info(f"  ⚠️  文档不存在（可能已被移除）: {filename}")
        else:
            log.info(f"  🗑️  已移到 trash: {rel}")
        ledger.move_to_trash(filename, cid)
        ledger.unpublish_mark(cid)

        # 重建索引（新 generation 不含该文档）→ 运行中 Agent 热刷新
        from app.agent.rag.embedder import create_embedder

        embedder = create_embedder()
        info = index_service.build(settings.rag_backend)
        index_service.activate(settings.rag_backend, info)
        log.info(f"  🔄 索引已重建并切换 generation={info.generation_id}（旧代保留可回滚）")
        log.info(f"  📄 文档保留在 trash/{filename}，可手动恢复")
    finally:
        lock.release()
    return 0


def _cmd_prune_turns(svc: dict, older_than_days: int) -> int:
    if not _gate():
        return 1
    if not older_than_days:
        log.info("  ❌ --prune-turns 需要 --older-than <天数>")
        return 1
    from app.evolution.miner import iter_turn_files, load_turn

    ledger = svc["ledger"]
    turns_dir = svc.get("turns_dir") or ROOT / settings.evolve_turns_dir
    processed = ledger.processed_set()
    protected = ledger.pending_turn_ids()
    removed = 0
    removed_ids: list[str] = []
    import time as _time

    cutoff = _time.time() - older_than_days * 86400
    for path in iter_turn_files(turns_dir):
        turn = load_turn(path)
        if turn is None:
            continue
        if turn.turn_id not in processed:
            continue  # 只动已处理且不关联 pending 的记录
        if turn.turn_id in protected:
            continue
        if path.stat().st_mtime > cutoff:
            continue
        path.unlink()
        removed += 1
        removed_ids.append(turn.turn_id)
    # processed 随保留期自然限界：文件已删，游标条目同步收缩
    pruned = ledger.drop_processed(removed_ids) if removed_ids else 0
    log.info(f"  🧹 已清理 {removed} 个超期 turn 文件（{older_than_days} 天前，"
             f"已处理且无 pending 关联），processed 游标同步收缩 {pruned} 条")
    return 0


def _cmd_prune_pending(svc: dict) -> int:
    if not _gate():
        return 1
    ledger = svc["ledger"]
    removed = ledger.prune_pending(only_aging=True)
    log.info(f"  🧹 已清理 {removed} 条 aging pending")
    return 0


def _cmd_revalidate(svc: dict) -> int:
    if not _gate():
        return 1
    from app.evolution.revalidate import recover_revalidate, revalidate

    pipeline = svc["pipeline"]
    lock = svc["lock"]
    retriever_factory = svc.get("retriever_factory") or pipeline._retriever_factory
    grounding_judge = svc.get("grounding_judge") or pipeline._grounding_judge
    generation_store = svc.get("generation_store") or pipeline._generation_store
    kb_dir = svc.get("kb_dir") or ROOT / settings.kb_dir
    state_dir = svc.get("state_dir") or ROOT / settings.evolve_state_dir

    lock.acquire(phase="revalidate")
    try:
        # 先补齐上次中断的隔离事务（journal phase=revalidate → 前进式恢复）
        entry = svc["journal"].read()
        if entry and entry.get("phase") == "revalidate":
            recover_revalidate(
                entry,
                kb_dir=kb_dir,
                trash_dir=state_dir / "trash",
                ledger=svc["ledger"],
                publisher=svc["publisher"],
                index_service=svc["index_service"],
                generation_store=generation_store,
            )
            svc["journal"].archive()
            log.info("  🔁 已恢复上次中断的重接地隔离事务")
        backend = settings.rag_backend.lower()
        active = generation_store.active(backend) if generation_store is not None else None
        progress = (
            pipeline._read_revalidate_progress(active.generation_id)
            if active is not None else []
        )
        result = revalidate(
            kb_dir=kb_dir,
            trash_dir=state_dir / "trash",
            ledger=svc["ledger"],
            publisher=svc["publisher"],
            index_service=svc["index_service"],
            retriever=retriever_factory(),
            grounding_judge=grounding_judge,
            journal=svc["journal"],
            exclude_docs=set(progress),
            backend=backend,
        )
        processed = set(progress)
        processed.update(result["processed_docs"])
        if result["has_more"]:
            current = generation_store.active(backend) if generation_store is not None else active
            progress_generation = current.generation_id if current is not None else ""
            pipeline._write_revalidate_progress(progress_generation, processed)
        else:
            # 仅全部分页完成后记录当前代；未完成时下次 CLI/run 从进度继续。
            pipeline.record_current_generation()
    finally:
        lock.release()
    log.info(f"  重接地核对 : {result['checked']} 篇"
             f"（通过 {result['passed']} / 隔离 {result['failed']} / 扫描 {result['scanned']} / "
             f"剩余 {result['remaining']}）")
    if result["rebuilt"]:
        for fname in result["failed_docs"]:
            log.info(f"    🗑️  {fname} → trash + pending（revalidation_failed）")
        log.info("  🔄 索引已重建并激活（隔离文档已移出知识库）")
    return 0


def _cmd_force_unlock(svc: dict) -> int:
    lock = svc["lock"]
    lock.force_unlock()
    log.info("  🔓 锁已强制释放（跨主机场景下请确认无其他进程在运行）")
    return 0


# ============================================================
# 入口
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第10期 QA 自动沉淀 CLI")
    parser.add_argument("--dry-run", action="store_true", help="预览：零写入、不取锁")
    parser.add_argument("--with-eval", action="store_true", help="正式运行前跑沙箱评测+探针")
    parser.add_argument("--list-pending", action="store_true", help="列出 pending（超龄标 aging）")
    parser.add_argument("--approve", nargs="+", metavar="CID", help="人工通过并发布")
    parser.add_argument("--reject", nargs="+", metavar="CID", help="人工拒绝并永久跳过")
    parser.add_argument("--unpublish", metavar="CID", help="下架已发布文档（移 trash + 重建索引）")
    parser.add_argument("--prune-turns", action="store_true", help="清理超期 turn 记录")
    parser.add_argument("--prune-pending", action="store_true", help="清理 aging pending")
    parser.add_argument("--revalidate", action="store_true",
                        help="强制重接地存量自进化文档（仍受 EVOLVE_MAX_REGROUND_PER_RUN 上限约束）")
    parser.add_argument("--force-unlock", action="store_true", help="强制删除运行锁")
    parser.add_argument("--older-than", type=int, default=None,
                        help="prune 时删除超过 N 天的记录")
    return parser


def main(argv=None, services=None) -> int:
    args = build_parser().parse_args(argv)
    svc = services if services is not None else _build_services()

    try:
        if args.force_unlock:
            return _cmd_force_unlock(svc)
        if args.list_pending:
            return _cmd_list_pending(svc)
        if args.approve:
            return _cmd_approve(svc, args.approve)
        if args.reject:
            return _cmd_reject(svc, args.reject)
        if args.unpublish:
            return _cmd_unpublish(svc, args.unpublish)
        if args.prune_turns:
            return _cmd_prune_turns(svc, args.older_than)
        if args.prune_pending:
            return _cmd_prune_pending(svc)
        if args.revalidate:
            return _cmd_revalidate(svc)
        return _cmd_run(svc, dry_run=args.dry_run, with_eval=args.with_eval)
    except CrossHostLockError as e:
        log.info(f"❌ {e}")
        return 1
    except LockHeldError as e:
        log.info(f"❌ {e}")
        return 1
    except RuntimeError as e:
        log.info(f"❌ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

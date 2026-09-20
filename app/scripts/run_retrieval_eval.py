"""运行检索质量评估（阶段七 7.6，检索回归门禁 CLI）。

用法：
  # 默认用例集（settings.retrieval_eval_dataset_path），每条检索 top-5
  python app/scripts/run_retrieval_eval.py

  # 换用例集 / 放宽候选窗口
  python app/scripts/run_retrieval_eval.py --dataset app/evaluation/retrieval_cases.json --top-k 10

流程：
  1. 加载检索用例集（含 query 与期望命中的 source_path 列表）。
  2. 默认经 knowledge 工具单例取活动索引；发布验收通过 --candidate 直接读取
     尚未激活的 generation，确保门禁对象与最终激活对象完全一致。
  3. 正例计算 recall@k / MRR / nDCG@k，负例计算拒绝率。
  4. 任一质量阈值未达标、阈值未配置或运行失败均以非零码退出。

指标语义见 app/evaluation/retrieval_metrics.py 模块 docstring。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from app.observability.logging import get_logger

log = get_logger("app.scripts.run_retrieval_eval")


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.tools import knowledge as knowledge_tool
from app.config.settings import settings
from app.evaluation.manifest import build_retrieval_manifest
from app.evaluation.retrieval_metrics import (
    calibrate_threshold,
    evaluate,
    load_cases,
)

_QUERY_COL = 22  # 查询列宽，超出截断

DEFAULT_MIN_POSITIVE_RECALL = 0.95
DEFAULT_MIN_EASY_RECALL = 0.98
DEFAULT_MIN_HARD_RECALL = 0.80
DEFAULT_MIN_MRR = 0.90
DEFAULT_MIN_NDCG = 0.90
DEFAULT_MIN_NEGATIVE_REJECTION = 0.90
DEFAULT_MAX_P95_LATENCY_MS = 500.0

# RAG 修复计划·5：release profile 固化最低门槛，禁止命令行调低（含调为 0）
RELEASE_PROFILE_MIN = {
    "min_positive_recall": 0.95,
    "min_easy_recall": 0.98,
    "min_hard_recall": 0.80,
    "min_mrr": 0.90,
    "min_ndcg": 0.90,
    "min_negative_rejection": 0.90,
    "max_p95_latency_ms": 500.0,
}


def apply_release_profile(args) -> None:
    """按 release profile 抬高门槛（只升不降；非法/0 值被抬到固定下限）。"""
    for key, floor in RELEASE_PROFILE_MIN.items():
        current = getattr(args, key, None)
        try:
            current_f = float(current) if current is not None else 0.0
        except (TypeError, ValueError):
            current_f = 0.0
        if key.startswith("max_"):
            # 延迟上限：取更严格（更小）者，但不得高于固定上限
            setattr(args, key, min(current_f or floor, floor))
        else:
            setattr(args, key, max(current_f, floor))


def rrf_mode() -> bool:
    """当前配置是否为「无语义分数」的 RRF 模式（实现收敛到 rejection.rrf_mode）。"""
    from app.agent.rag.rejection import rrf_mode as _rrf_mode

    return _rrf_mode()


def _pct(value: float) -> str:
    """0-1 小数 → 百分比字符串（如 0.875 → "87.5%"）。"""
    return f"{value * 100:5.1f}%"


def filter_cases_by_exclude(cases: list, exclude_ids: list[str]) -> tuple[list, set]:
    """按用例 ID 过滤（--exclude-id）；返回 (过滤后用例, 未命中 ID 集合)。

    用于复现 v2 冻结 535 口径（排除 2 条 evolved 增补用例）。
    """
    excluded = set(exclude_ids or [])
    if not excluded:
        return cases, set()
    all_ids = {c["id"] for c in cases}
    missing = excluded - all_ids
    return [c for c in cases if c["id"] not in excluded], missing


def build_retrieval_thresholds(args) -> dict:
    """组装留档阈值；校准轮附校准约束与排除 ID（review：产物缺字段）。"""
    thresholds = {
        "min_positive_recall": args.min_positive_recall,
        "min_easy_recall": args.min_easy_recall,
        "min_hard_recall": args.min_hard_recall,
        "min_mrr": args.min_mrr,
        "min_ndcg": args.min_ndcg,
        "min_negative_rejection": args.min_negative_rejection,
    }
    if args.calibrate:
        thresholds["calibrate_min_positive_recall"] = (
            args.calibrate_min_positive_recall
        )
        thresholds["calibrate_min_negative_rejection"] = 0.80
        if args.exclude_id:
            thresholds["excluded_case_ids"] = sorted(set(args.exclude_id))
    return thresholds


def archive_calibration_failure(args, dataset_path: Path, cases: list,
                                exc: Exception) -> Path | None:
    """校准失败也留档（review：gate 轮 dev-cal 引用无产物）。

    仅在 --json-out 时写一份最小失败记录（约束、数据集 SHA、失败原因），
    供"校准失败留档"引用；返回产物路径，未配置 --json-out 返回 None。
    """
    if not args.json_out:
        return None
    out = (
        ROOT / args.json_out
        if not Path(args.json_out).is_absolute() else Path(args.json_out)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "retrieval-eval-v1",
        "calibration_failed": True,
        "error": f"{type(exc).__name__}: {exc}",
        "calibrate_min_positive_recall": args.calibrate_min_positive_recall,
        "calibrate_min_negative_rejection": 0.80,
        "variant": args.variant or "settings",
        "top_k": args.top_k,
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "num_cases": len(cases),
        },
    }
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def load_candidate(path_value: str) -> tuple[str, str]:
    """读取 build_kb_index 的候选描述并校验后端/target，返回 (generation, target)。"""
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "kb-generation-candidate-v1":
        raise ValueError("候选描述 protocol 非法或缺失")
    backend = str(payload.get("backend", "")).lower()
    generation = payload.get("generation") or {}
    generation_id = str(generation.get("generation_id", ""))
    target = str(generation.get("target", ""))
    if backend != settings.rag_backend.lower():
        raise ValueError(
            f"候选后端 {backend or '缺失'} 与当前 RAG_BACKEND {settings.rag_backend} 不一致"
        )
    if not generation_id or not target:
        raise ValueError("候选描述缺少 generation_id/target")
    from app.agent.rag.fingerprint import config_fingerprint

    if payload.get("config_fingerprint") != config_fingerprint():
        raise ValueError("候选索引配置指纹与当前配置不一致")
    if backend == "es":
        expected = f"{settings.es_index_prefix}-kb-{generation_id}"
        if target != expected:
            raise ValueError("候选 target 与 generation_id/当前 ES 前缀不匹配")
    return generation_id, target


def _print_report(report: dict) -> None:
    cases = report["cases"]

    log.info("\n" + "=" * 78)
    log.info("  逐用例检索指标（正例 ranking / 负例 rejection）")
    log.info("=" * 78)
    log.info(f"  {'用例':<24}{'recall@k':>9}{'MRR':>8}{'nDCG@k':>9}  查询")
    for c in cases:
        query = c["query"] if len(c["query"]) <= _QUERY_COL else c["query"][:_QUERY_COL - 1] + "…"
        if not c["expected"]:
            status = "REJECT" if c["negative_rejected"] else "ACCEPT"
            score = c["raw_top_score"]
            score_text = "-" if score is None else f"{score:.4f}"
            log.info(f"  {c['id']:<24}{status:>9}{'':>8}{score_text:>9}  {query}")
            continue
        log.info(
            f"  {c['id']:<24}{_pct(c['recall_at_k']):>9}"
            f"{_pct(c['mrr']):>8}{_pct(c['ndcg_at_k']):>9}  {query}"
        )

    s = report["summary"]
    log.info("\n" + "=" * 78)
    log.info("  汇总")
    log.info("=" * 78)
    log.info(f"  用例数             : {s['cases']}")
    log.info(f"  正例 recall@k      : {_pct(s['positive']['recall_at_k'])}")
    log.info(f"  正例 MRR           : {_pct(s['positive']['mrr'])}")
    log.info(f"  正例 nDCG@k        : {_pct(s['positive']['ndcg_at_k'])}")
    log.info(f"  困难正例 recall@k : {_pct(s['hard']['recall_at_k'])}")
    log.info(f"  负例拒绝率         : {_pct(s['negative']['rejection_rate'])}")
    # 工作流 B：多跳双口径（仅 --multi-query-overlay 命中用例非空时展示）
    multi = s.get("multi_hop") or {}
    if multi.get("cases"):
        log.info(
            f"  多跳双口径         : 严格 {_pct(multi['strict_rate'])} / "
            f"宽松 {_pct(multi['lenient_rate'])}（{multi['cases']} 例）"
        )


def quality_failures(report: dict, args) -> list[str]:
    """返回未达到门槛的原因；空列表表示质量门禁通过。"""
    s = report["summary"]
    failures: list[str] = []
    if s["positive"]["recall_at_k"] < args.min_positive_recall:
        failures.append(
            f"正例 recall@k {_pct(s['positive']['recall_at_k'])} < {_pct(args.min_positive_recall)}"
        )
    if s["easy"]["cases"] and s["easy"]["recall_at_k"] < args.min_easy_recall:
        failures.append(
            f"基础正例 recall@k {_pct(s['easy']['recall_at_k'])} < {_pct(args.min_easy_recall)}"
        )
    if s["hard"]["cases"] and s["hard"]["recall_at_k"] < args.min_hard_recall:
        failures.append(
            f"困难正例 recall@k {_pct(s['hard']['recall_at_k'])} < {_pct(args.min_hard_recall)}"
        )
    if s["positive"]["mrr"] < args.min_mrr:
        failures.append(f"正例 MRR {_pct(s['positive']['mrr'])} < {_pct(args.min_mrr)}")
    if s["positive"]["ndcg_at_k"] < args.min_ndcg:
        failures.append(
            f"正例 nDCG@k {_pct(s['positive']['ndcg_at_k'])} < {_pct(args.min_ndcg)}"
        )
    if s["negative"]["cases"]:
        if args.min_score is None:
            failures.append("数据集包含负例，但未配置 --min-score / RAG_MIN_RELEVANCE_SCORE")
        if s["negative"]["rejection_rate"] < args.min_negative_rejection:
            failures.append(
                f"负例拒绝率 {_pct(s['negative']['rejection_rate'])} < "
                f"{_pct(args.min_negative_rejection)}"
            )
    # RAG 修复计划·5：P95 检索延迟门禁（生产 SLO ≤ 500ms）
    p95 = s.get("p95_latency_ms")
    if p95 is not None and p95 > args.max_p95_latency_ms:
        failures.append(
            f"P95 检索延迟 {p95:.0f}ms > {args.max_p95_latency_ms:.0f}ms"
        )
    # RAG 修复计划·5：任一 degraded case 直接失败（不得用降级结果宣称质量达标）
    degraded = int(s.get("degraded_cases") or 0)
    if degraded:
        failures.append(f"存在 {degraded} 条 degraded 检索结果（精排不可用/部分响应）")
    return failures


def _apply_variant(variant: str) -> str:
    """3.4 检索实验变体：按冻结配置覆盖 settings（并在运行后恢复）。

    knn               → ES kNN（纯向量，关闭 hybrid/reranker）
    hybrid            → ES BM25+kNN+RRF（hybrid，无 reranker）
    hybrid-rerank     → ES hybrid + bge-reranker-v2-m3
    hybrid-rerank-norm→ hybrid-rerank + query 表层规范化（工作流 A 臂）
    每个变体都**显式**置 rag_query_normalize：A 臂与基线臂之间不得互相泄漏。
    返回原配置摘要用于恢复。
    """
    from app.agent.tools import knowledge as knowledge_tool

    previous = {
        "rag_hybrid": settings.rag_hybrid,
        "rag_rerank": settings.rag_rerank,
        "rag_backend": settings.rag_backend,
        "rag_query_normalize": settings.rag_query_normalize,
    }
    if variant == "knn":
        settings.rag_hybrid = False
        settings.rag_rerank = "none"
        settings.rag_query_normalize = False
    elif variant == "hybrid":
        settings.rag_hybrid = True
        settings.rag_rerank = "none"
        settings.rag_query_normalize = False
    elif variant == "hybrid-rerank":
        settings.rag_hybrid = True
        settings.rag_rerank = "bge-reranker-v2-m3"
        settings.rag_query_normalize = False
    elif variant == "hybrid-rerank-norm":
        settings.rag_hybrid = True
        settings.rag_rerank = "bge-reranker-v2-m3"
        settings.rag_query_normalize = True
    else:
        raise ValueError(
            f"未知变体: {variant}（可选 knn / hybrid / hybrid-rerank / "
            "hybrid-rerank-norm）"
        )
    knowledge_tool.reset_retriever()
    return previous


def _restore_variant(previous: dict) -> None:
    from app.agent.tools import knowledge as knowledge_tool

    settings.rag_hybrid = previous["rag_hybrid"]
    settings.rag_rerank = previous["rag_rerank"]
    settings.rag_backend = previous["rag_backend"]
    if "rag_query_normalize" in previous:
        settings.rag_query_normalize = previous["rag_query_normalize"]
    knowledge_tool.reset_retriever()


def _assert_lexicon_available() -> None:
    """A 臂 fail-loud：词表缺失/损坏时评测必须立刻失败，不得静默跑成基线。"""
    from app.agent.rag.query_normalizer import QueryNormalizer

    if QueryNormalizer.load(settings.rag_query_normalize_lexicon_path) is None:
        raise SystemExit(
            "❌ 变体要求词表可用，但加载失败: "
            f"{settings.rag_query_normalize_lexicon_path}"
            "（先运行 app/scripts/build_query_lexicon.py）"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器（独立函数：单测可只测参数定义，不触发评测流程）。"""
    parser = argparse.ArgumentParser(
        description="运行检索质量评估（7.6，搭配 7.1-7.4 检索改造；3.4 变体实验）"
    )
    parser.add_argument(
        "--dataset",
        default=settings.retrieval_eval_dataset_path,
        help=f"检索用例集 JSON 路径（默认: {settings.retrieval_eval_dataset_path}）",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="每条用例检索返回的候选数（默认 5）",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="最终候选相关度下限；默认读取线上 RAG_MIN_RELEVANCE_SCORE",
    )
    parser.add_argument(
        "--min-score-hard", type=float, default=None,
        help="显式覆盖 hard 用例相关度下限；默认与线上 --min-score 相同。"
             "覆盖后报告会标记为非线上口径",
    )
    parser.add_argument("--min-positive-recall", type=float, default=DEFAULT_MIN_POSITIVE_RECALL)
    parser.add_argument("--min-easy-recall", type=float, default=DEFAULT_MIN_EASY_RECALL)
    parser.add_argument("--min-hard-recall", type=float, default=DEFAULT_MIN_HARD_RECALL)
    parser.add_argument("--min-mrr", type=float, default=DEFAULT_MIN_MRR)
    parser.add_argument("--min-ndcg", type=float, default=DEFAULT_MIN_NDCG)
    parser.add_argument(
        "--max-p95-latency-ms", type=float, default=DEFAULT_MAX_P95_LATENCY_MS,
        help="P95 检索延迟上限（毫秒，默认 500；RAG 修复计划·5）",
    )
    parser.add_argument(
        "--release-profile", action="store_true",
        help="生产发布口径：门槛按固定下限只升不降（RAG 修复计划·5）",
    )
    parser.add_argument(
        "--min-negative-rejection", type=float,
        default=DEFAULT_MIN_NEGATIVE_REJECTION,
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="用基础正例和黄金集 rag_no_hit_* 自动校准本次运行的最低分数",
    )
    parser.add_argument(
        "--calibration-dataset", default=settings.eval_dataset_path,
        help="包含 rag_no_hit_* 校准负例的黄金集路径",
    )
    parser.add_argument(
        "--calibrate-min-positive-recall", type=float, default=0.98,
        help="校准阶段正例召回约束（默认 0.98；语料膨胀后需按可行性下调，"
             "v3 起显式传入并写入 manifest）",
    )
    parser.add_argument(
        "--exclude-id", action="append", default=[],
        help="按用例 ID 排除（可重复）；用于复现 v2 冻结 535 口径"
             "（排除 2 条 evolved 增补用例）",
    )
    parser.add_argument(
        "--variant", choices=["knn", "hybrid", "hybrid-rerank",
                              "hybrid-rerank-norm"], default="",
        help="3.4 检索实验变体：knn（纯向量）/ hybrid（BM25+kNN+RRF）/ "
             "hybrid-rerank（hybrid+bge-reranker）/ hybrid-rerank-norm"
             "（hybrid-rerank + query 表层规范化）；缺省用 settings 现状",
    )
    parser.add_argument(
        "--json-out", default="",
        help="报告 JSON 输出路径（3.4 门禁留档）",
    )
    parser.add_argument(
        "--candidate", default="",
        help="build_kb_index --no-activate 生成的候选 JSON；直接评测该 target",
    )
    # 工作流 B：多跳拆解 overlay（修复：main() 曾无条件读 args.multi_query_overlay，
    # 但解析器从未定义该参数 → 一跑就 AttributeError）。默认空 = 不启用 overlay，
    # 与既有用法一致：显式传 app/evaluation/retrieval_cases_v3_multihop_queries.json
    # （protocol=retrieval-multihop-overlay-v1，{case_id: [补充子查询]}）才生效。
    parser.add_argument(
        "--multi-query-overlay", default="",
        help="多跳拆解 overlay JSON 路径（{case_id: [补充子查询]}；缺省不使用；"
             "如 app/evaluation/retrieval_cases_v3_multihop_queries.json）",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.release_profile:
        apply_release_profile(args)

    # 评测默认必须复用线上阈值。None 仅表示线上尚未配置阈值，而不是
    # hard 用例绕过过滤；显式 --min-score-hard 才允许非线上实验口径。
    online_min_score = settings.rag_min_relevance_score
    args.min_score = online_min_score if args.min_score is None else args.min_score
    hard_threshold_overridden = args.min_score_hard is not None
    effective_min_score_hard = (
        args.min_score_hard if hard_threshold_overridden else args.min_score
    )

    # 3.4：应用实验变体（运行后恢复原配置）
    previous = _apply_variant(args.variant) if args.variant else {}
    if args.variant == "hybrid-rerank-norm":
        _assert_lexicon_available()  # A 臂：词表缺失 fail-loud

    dataset_path = ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)

    log.info("=" * 78)
    log.info("  检索质量评估（阶段七 7.6）")
    log.info(f"  数据集    : {dataset_path}")
    log.info(f"  变体      : {args.variant or 'settings-现状'}")
    log.info(f"  top-k     : {args.top_k}")
    log.info(f"  后端      : {settings.rag_backend}")
    log.info(f"  hybrid    : {settings.rag_hybrid}")
    log.info(f"  reranker  : {settings.rag_rerank}")
    log.info(f"  最低分数  : {args.min_score if args.min_score is not None else '未配置'}")
    if hard_threshold_overridden:
        online_text = "未配置" if online_min_score is None else f"{online_min_score:.8f}"
        log.warning(
            f"⚠ hard 用例使用显式覆盖阈值 {effective_min_score_hard:.8f}"
            f"（线上阈值 {online_text}），本报告不是线上口径"
        )
    else:
        log.info("  hard 阈值  : 与本次最低分数相同")
    log.info("=" * 78)

    log.info("\n[1/3] 加载检索用例集...")
    try:
        cases = load_cases(dataset_path)
    except ValueError as e:
        log.info(f"❌ {e}")
        sys.exit(1)
    if not cases:
        log.info("❌ 检索用例集为空")
        sys.exit(1)
    if args.exclude_id:
        before = len(cases)
        cases, missing = filter_cases_by_exclude(cases, args.exclude_id)
        log.info(f"   --exclude-id 排除 {before - len(cases)} 条（{sorted(set(args.exclude_id))}）")
        if missing:
            log.warning(f"   ⚠ 以下 ID 未出现在用例集中: {sorted(missing)}")
    log.info(f"   共 {len(cases)} 条用例")

    candidate_generation = ""
    candidate_target = ""
    if args.candidate:
        try:
            candidate_generation, candidate_target = load_candidate(args.candidate)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            log.info(f"❌ 候选 generation 描述无效（{type(e).__name__}: {e}）")
            sys.exit(1)

    log.info("\n[2/3] 构建检索器...")
    try:
        if candidate_target:
            from app.agent.rag.retriever_factory import (
                open_retriever,
                retrieval_config_from_settings,
            )

            retriever = open_retriever(
                retrieval_config_from_settings(),
                generation_target=candidate_target,
            )
        else:
            retriever = knowledge_tool._get_retriever()
    except (FileNotFoundError, RuntimeError) as e:
        log.info(f"❌ 检索器构建失败（{type(e).__name__}: {e}）")
        sys.exit(1)
    log.info("   检索器就绪")

    if args.calibrate:
        # RAG 修复计划·5：RRF 分无语义，禁止在 RRF 模式下校准阈值
        if rrf_mode():
            log.info(
                "❌ RRF 模式（hybrid 且未挂精排）分数无语义，禁止 --calibrate："
                "请启用 reranker 或改用纯向量配置"
            )
            sys.exit(2)
        calibration_path = Path(args.calibration_dataset)
        if not calibration_path.is_absolute():
            calibration_path = ROOT / calibration_path
        try:
            # 负例优先取当前检索集的 expected=[] 用例（含 retrieval_no_hit_* /
            # rag_no_hit_* 等），再补黄金集 rag_no_hit_* 口径
            negative_queries = [
                c["query"] for c in cases if not c.get("expected")
            ]
            if calibration_path.exists():
                golden = json.loads(calibration_path.read_text(encoding="utf-8"))["cases"]
                negative_queries.extend(
                    c["turns"][0] for c in golden
                    if str(c.get("id", "")).startswith("rag_no_hit_") and c.get("turns")
                )
            negative_queries = list(dict.fromkeys(negative_queries))
            base_positives = [
                c for c in cases if c["expected"] and "easy" in c.get("tags", [])
            ]
            calibrated = calibrate_threshold(
                base_positives,
                negative_queries,
                retriever,
                top_k=args.top_k,
                min_positive_recall=args.calibrate_min_positive_recall,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.info(f"❌ 相关度阈值校准失败: {exc}")
            archived = archive_calibration_failure(args, dataset_path, cases, exc)
            if archived:
                log.info(f"   校准失败产物已留档: {archived}")
            sys.exit(1)
        args.min_score = calibrated["threshold"]
        if not hard_threshold_overridden:
            # 校准后 hard 仍沿用本次运行的线上口径阈值。
            effective_min_score_hard = args.min_score
        log.info(
            "   校准阈值: "
            f"{args.min_score:.8f}（正例 recall@k="
            f"{_pct(calibrated['positive_recall_at_k'])}，负例拒绝率="
            f"{_pct(calibrated['negative_rejection_rate'])}）"
        )

    # 工作流 B：多跳拆解 overlay（加载失败 fail-loud；命中用例走多子查询口径）
    multi_query_overlay = {}
    overlay_payload: dict = {}
    overlay_path: Path | None = None
    if args.multi_query_overlay:
        overlay_path = (
            ROOT / args.multi_query_overlay
            if not Path(args.multi_query_overlay).is_absolute()
            else Path(args.multi_query_overlay)
        )
        try:
            overlay_payload = json.loads(overlay_path.read_text(encoding="utf-8"))
            overlay_map = overlay_payload.get("overlay")
            if not isinstance(overlay_map, dict) or not all(
                isinstance(v, list)
                and all(isinstance(q, str) and q.strip() for q in v)
                for v in overlay_map.values()
            ):
                raise ValueError("overlay 需为 {case_id: [子查询...]} 结构")
        except (OSError, ValueError, json.JSONDecodeError) as e:
            log.info(f"❌ 多跳 overlay 无效（{type(e).__name__}: {e}）")
            sys.exit(2)
        multi_query_overlay = overlay_map
        log.info(
            f"   多跳 overlay : {overlay_path}"
            f"（{len(multi_query_overlay)} 例；protocol="
            f"{overlay_payload.get('protocol', '?')}）"
        )

    # 工作流 A 留痕：norm 臂给 evaluate 传同一词表的规范化器，per-case 记录
    # 改写结果（检索路径的规范化在检索器内部，此处只做报告审计）
    query_normalizer = None
    if settings.rag_query_normalize:
        from app.agent.rag.query_normalizer import QueryNormalizer

        query_normalizer = QueryNormalizer.load(
            settings.rag_query_normalize_lexicon_path,
        )

    log.info(f"\n[3/3] 逐用例评估（top_k={args.top_k}）...")
    report = evaluate(
        cases, retriever, top_k=args.top_k, min_score=args.min_score,
        min_score_hard=effective_min_score_hard,
        multi_query_overlay=multi_query_overlay or None,
        query_normalizer=query_normalizer,
    )
    _print_report(report)
    if query_normalizer is not None:
        n_rewritten = report["summary"].get("query_normalize_rewrites", 0)
        log.info(f"  query 规范化 : 改写 {n_rewritten}/{len(cases)} 条")

    # RAG 修复计划·5：校准输入出现 RRF/degraded → 立即失败（不得据其产阈值）
    if args.calibrate:
        bad = [
            c["id"] for c in report["cases"]
            if c.get("degraded") or c.get("score_source") == "rrf"
        ]
        if bad:
            log.info(
                f"❌ 校准输入含 RRF/降级结果（{len(bad)} 条，如 {bad[:5]}）："
                "分数无语义，禁止用于阈值校准"
            )
            sys.exit(2)

    failures = quality_failures(report, args)
    retrieval_thresholds = build_retrieval_thresholds(args)
    retrieval_manifest = build_retrieval_manifest(
        dataset_path=str(dataset_path), num_cases=len(cases), top_k=args.top_k,
        min_score=args.min_score, min_score_hard=effective_min_score_hard,
        thresholds=retrieval_thresholds,
        variant=args.variant or "settings",
        hard_threshold_overridden=hard_threshold_overridden,
        generation_target=candidate_target,
    )
    online_threshold_match = retrieval_manifest["thresholds"]["online_threshold_match"]
    if not online_threshold_match:
        log.warning(
            "⚠ 本次应用阈值与当前 RAG_MIN_RELEVANCE_SCORE 不一致，"
            "报告仅作为候选/校准实验，不是已部署线上口径"
        )
    if args.json_out:
        # 3.4 留档：逐例报告 + 可复现检索 manifest + 门槛信息
        out = ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "protocol": "retrieval-eval-v1",
            "variant": args.variant or "settings",
            "dataset": str(dataset_path),
            # 工作流 B：多跳 overlay 审计（路径 + sha256，臂间防泄漏）
            "multi_query_overlay": (
                {
                    "path": str(overlay_path),
                    "sha256": hashlib.sha256(
                        overlay_path.read_bytes()
                    ).hexdigest(),
                    "num_cases": len(multi_query_overlay),
                    "protocol": overlay_payload.get("protocol", ""),
                }
                if overlay_path is not None else None
            ),
            "top_k": args.top_k,
            "thresholds": {
                "min_score": args.min_score,
                "min_score_hard": effective_min_score_hard,
                "hard_threshold_overridden": hard_threshold_overridden,
                "hard_threshold_policy": (
                    "explicit_override_non_online"
                    if hard_threshold_overridden else "same_as_min_score"
                ),
                "online_threshold_match": online_threshold_match,
                **retrieval_thresholds,
            },
            "summary": report["summary"],
            "cases": report["cases"],
            "failures": failures,
            "passed": not failures,
            "manifest": retrieval_manifest,
            "candidate_generation": candidate_generation,
            "candidate_target": candidate_target,
        }
        out.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out.parent / "manifest.json").write_text(
            json.dumps(retrieval_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"报告已留档: {out}")
        log.info(f"检索 manifest 已留档: {out.parent / 'manifest.json'}")

    if failures:
        log.info("\n❌ 检索质量门禁未通过：")
        for failure in failures:
            log.info(f"  - {failure}")
        bad_cases = [
            c for c in report["cases"]
            if (c["expected"] and c["recall_at_k"] < 1.0)
            or (not c["expected"] and not c["negative_rejected"])
        ]
        for case in bad_cases[:20]:
            log.info(
                f"  · {case['id']}: expected={case['expected']} "
                f"hits={case['hit_keys']} top_score={case['raw_top_score']}"
            )
        if previous:
            _restore_variant(previous)
        sys.exit(1)

    log.info("\n🎉 检索质量门禁通过。")
    if previous:
        _restore_variant(previous)
    sys.exit(0)


if __name__ == "__main__":
    main()

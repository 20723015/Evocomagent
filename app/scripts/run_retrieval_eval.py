"""运行检索质量评估（阶段七 7.6，检索回归门禁 CLI）。

用法：
  # 默认用例集（settings.retrieval_eval_dataset_path），每条检索 top-5
  python app/scripts/run_retrieval_eval.py

  # 换用例集 / 放宽候选窗口
  python app/scripts/run_retrieval_eval.py --dataset app/evaluation/retrieval_cases.json --top-k 10

流程：
  1. 加载检索用例集（含 query 与期望命中的 source_path 列表）。
  2. 经 knowledge 工具单例 _get_retriever() 取当前检索器——与 7.1-7.4 改造
     共用同一条构建链路（generation 感知 + 固定路径回退），保证评估的就是
     线上同款检索链路，而非特制黄金索引。
  3. 正例计算 recall@k / MRR / nDCG@k，负例计算拒绝率。
  4. 任一质量阈值未达标、阈值未配置或运行失败均以非零码退出。

指标语义见 app/evaluation/retrieval_metrics.py 模块 docstring。
"""

import argparse
import json
import sys
from pathlib import Path
from app.observability.logging import get_logger
log = get_logger("app.scripts.run_retrieval_eval")


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.tools import knowledge as knowledge_tool  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.evaluation.manifest import build_retrieval_manifest  # noqa: E402
from app.evaluation.retrieval_metrics import (  # noqa: E402
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


def _pct(value: float) -> str:
    """0-1 小数 → 百分比字符串（如 0.875 → "87.5%"）。"""
    return f"{value * 100:5.1f}%"


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
    return failures


def _apply_variant(variant: str) -> str:
    """3.4 检索实验变体：按冻结配置覆盖 settings（并在运行后恢复）。

    knn            → ES kNN（纯向量，关闭 hybrid/reranker）
    hybrid         → ES BM25+kNN+RRF（hybrid，无 reranker）
    hybrid-rerank  → ES hybrid + bge-reranker-v2-m3
    返回原配置摘要用于日志。
    """
    from app.agent.tools import knowledge as knowledge_tool

    previous = {
        "rag_hybrid": settings.rag_hybrid,
        "rag_rerank": settings.rag_rerank,
        "rag_backend": settings.rag_backend,
    }
    if variant == "knn":
        settings.rag_hybrid = False
        settings.rag_rerank = "none"
    elif variant == "hybrid":
        settings.rag_hybrid = True
        settings.rag_rerank = "none"
    elif variant == "hybrid-rerank":
        settings.rag_hybrid = True
        settings.rag_rerank = "bge-reranker-v2-m3"
    else:
        raise ValueError(f"未知变体: {variant}（可选 knn / hybrid / hybrid-rerank）")
    knowledge_tool.reset_retriever()
    return previous


def _restore_variant(previous: dict) -> None:
    from app.agent.tools import knowledge as knowledge_tool

    settings.rag_hybrid = previous["rag_hybrid"]
    settings.rag_rerank = previous["rag_rerank"]
    settings.rag_backend = previous["rag_backend"]
    knowledge_tool.reset_retriever()


def main() -> None:
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
        "--variant", choices=["knn", "hybrid", "hybrid-rerank"], default="",
        help="3.4 检索实验变体：knn（纯向量）/ hybrid（BM25+kNN+RRF）/ "
             "hybrid-rerank（hybrid+bge-reranker）；缺省用 settings 现状",
    )
    parser.add_argument(
        "--json-out", default="",
        help="报告 JSON 输出路径（3.4 门禁留档）",
    )
    args = parser.parse_args()

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
    log.info(f"   共 {len(cases)} 条用例")

    log.info("\n[2/3] 构建检索器...")
    try:
        retriever = knowledge_tool._get_retriever()
    except (FileNotFoundError, RuntimeError) as e:
        log.info(f"❌ 检索器构建失败（{type(e).__name__}: {e}）")
        sys.exit(1)
    log.info("   检索器就绪")

    if args.calibrate:
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
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.info(f"❌ 相关度阈值校准失败: {exc}")
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

    log.info(f"\n[3/3] 逐用例评估（top_k={args.top_k}）...")
    report = evaluate(
        cases, retriever, top_k=args.top_k, min_score=args.min_score,
        min_score_hard=effective_min_score_hard,
    )
    _print_report(report)

    failures = quality_failures(report, args)
    retrieval_thresholds = {
        "min_positive_recall": args.min_positive_recall,
        "min_easy_recall": args.min_easy_recall,
        "min_hard_recall": args.min_hard_recall,
        "min_mrr": args.min_mrr,
        "min_ndcg": args.min_ndcg,
        "min_negative_rejection": args.min_negative_rejection,
    }
    retrieval_manifest = build_retrieval_manifest(
        dataset_path=str(dataset_path), num_cases=len(cases), top_k=args.top_k,
        min_score=args.min_score, min_score_hard=effective_min_score_hard,
        thresholds=retrieval_thresholds,
        variant=args.variant or "settings",
        hard_threshold_overridden=hard_threshold_overridden,
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

#!/usr/bin/env python
"""阈值可行带扫描（live probe）——归档门禁阈值的选取依据。

背景（review 修复）：RESULT-gate.md 的可行带表此前无归档产物，CLI --calibrate
在 0.90 约束下因负例拒绝 < 80% 抛 ValueError 退出且不留 report（自
run_retrieval_eval.archive_calibration_failure 起失败也留档）。本脚本对冻结
数据集的 easy 正例 + 负例做一次 live 检索并缓存候选，离线扫多档正例约束，
输出「约束 × 可行最高阈值 × 该阈值下负例拒绝率」矩阵。

扫描语义与 app/evaluation/retrieval_metrics.py::calibrate_threshold 完全一致
（final_search 取全候选 → 逐候选分数扫阈值：门控→折叠→截断 → recall/rejection），
差异仅两点：
  1. 候选取一次检索缓存，不随约束重跑（多档约束共享同一次 live probe）
  2. 不施加负例拒绝 ≥80% 约束（正是要量化该约束是否可行），只记录拒绝率

校准基：easy 标签正例 + expected=[] 负例（数据集内），不含 CLI --calibrate
额外并入的黄金集 rag_no_hit_* 扩展——与 RESULT-gate 可行带表口径一致。

用法：
  python app/scripts/scan_threshold_band.py \
    --dataset app/evaluation/retrieval_cases_v3.json \
    --out artifacts/eval/v3/gate/threshold-band/scan.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.tools import knowledge as knowledge_tool  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.evaluation.retrieval_metrics import (  # noqa: E402
    _unique_keys,
    load_cases,
    recall_at_k,
)
from app.scripts.run_retrieval_eval import _apply_variant, _restore_variant  # noqa: E402

DEFAULT_CONSTRAINTS = [0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]


def _final(hits: list, k: int, threshold: float) -> list:
    """与 calibrate_threshold._final 相同：门控→折叠→截断。"""
    from app.agent.rag.retriever import collapse_by_parent, filter_hits_by_score
    return collapse_by_parent(filter_hits_by_score(hits, threshold), k)


_CANARY_QUERY = "钻石会员专属客服的响应时效SLO是多少"


def _assert_rerank_live(retriever, top_k: int) -> float:
    """金丝雀自检：rerank 未生效（端点未配置/静默降级）时分数全 0，扫描必产出垃圾。

    ESHybridRetriever 无精排时 ES rank.rrf 的 _score 为 None → 记 0.0
    （es_backend._hits_to_results），排序仍正确——单看召回/排序无法发现降级，
    只能用已知高分查询探分数量级。历史基线：该查询 rerank Top-1 ≈ 0.9996。
    """
    outcome = _probe(_CANARY_QUERY, top_k, retriever)
    top = max((h.score for h in outcome.raw_hits), default=0.0)
    if top < 0.5:
        raise RuntimeError(
            f"rerank 疑似未生效（金丝雀 '{_CANARY_QUERY}' top score {top:.4f} < 0.5）："
            "请检查 RERANK_ENDPOINT_URL（.env 留空时须显式注入，如 "
            "RERANK_ENDPOINT_URL=http://127.0.0.1:8001）。静默降级下 ES 原生 RRF "
            "无 _score（全 0），扫描会得到阈值 0/负例拒绝 0 的无效结果。"
        )
    return top


def _probe(query: str, k: int, retriever, retries: int = 3):
    """final_search 带重试（embedding 上游偶发 5xx；语义不变）。"""
    from app.agent.rag.retriever_factory import final_search
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return final_search(retriever, query, k)
        except Exception as exc:  # noqa: BLE001 —— 网络类异常整体重试
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"检索探针失败（已重试 {retries} 次）: {query!r}") from last


def scan(dataset_path: Path, top_k: int, constraints: list[float],
         variant: str = "hybrid-rerank", log_every: int = 100) -> dict:
    cases = load_cases(dataset_path)
    positives = [c for c in cases if c.get("expected") and "easy" in c.get("tags", [])]
    negatives = [c for c in cases if not c.get("expected")]
    if not positives or not negatives:
        raise ValueError("可行带扫描需要非空 easy 正例与负例")

    print(f"[1/3] live probe：easy 正例 {len(positives)} / 负例 {len(negatives)}")
    previous = _apply_variant(variant)
    try:
        retriever = knowledge_tool._get_retriever()
        canary_score = _assert_rerank_live(retriever, top_k)
        print(f"   金丝雀通过（rerank top score {canary_score:.4f}）")

        positive_runs: list[tuple[list, list[str], int]] = []
        negative_runs: list[list] = []
        scores: list[float] = []
        for i, case in enumerate(positives, 1):
            k = int(case.get("k") or top_k)
            outcome = _probe(case["query"], k, retriever)
            positive_runs.append((outcome.raw_hits, list(case["expected"]), k))
            scores.extend(h.score for h in outcome.raw_hits)
            if i % log_every == 0:
                print(f"   正例 {i}/{len(positives)}")
        for i, case in enumerate(negatives, 1):
            outcome = _probe(case["query"], top_k, retriever)
            negative_runs.append(outcome.raw_hits)
            scores.extend(h.score for h in outcome.raw_hits)
            if i % log_every == 0:
                print(f"   负例 {i}/{len(negatives)}")
    finally:
        _restore_variant(previous)
    if not scores:
        raise ValueError("扫描未获得任何候选分数")

    print(f"[2/3] 离线扫阈值（候选 {len(set(scores))} 档）...")
    candidates = sorted(set(scores))
    candidates.append(max(scores) + 1e-12)
    curve: list[tuple[float, float, float]] = []  # (threshold, recall, rejection)
    for threshold in candidates:
        recall = sum(
            recall_at_k(_unique_keys(_final(hits, k, threshold)), expected, k)
            for hits, expected, k in positive_runs
        ) / len(positive_runs)
        rejection = sum(
            1 for hits in negative_runs if not _final(hits, top_k, threshold)
        ) / len(negative_runs)
        curve.append((threshold, recall, rejection))

    print("[3/3] 按约束取可行最高阈值...")
    raw_recall = curve[0][1]
    rows = []
    for c in constraints:
        feasible = [row for row in curve if row[1] >= c]
        if not feasible:
            rows.append({
                "min_positive_recall": c, "feasible": False,
                "reason": f"原始 easy recall {raw_recall:.4f} < {c:.2f}，不可达",
            })
            continue
        threshold, recall, rejection = max(feasible, key=lambda row: row[0])
        rows.append({
            "min_positive_recall": c, "feasible": True,
            "threshold": threshold,
            "easy_recall_at_threshold": recall,
            "negative_rejection_at_threshold": rejection,
        })
    return {
        "protocol": "threshold-band-scan-v1",
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "num_cases": len(cases),
            "n_easy_positives": len(positives),
            "n_negatives": len(negatives),
        },
        "top_k": top_k,
        "variant": f"{variant}（冻结变体，_apply_variant 显式套用）",
        "retrieval": {
            "backend": settings.rag_backend,
            "hybrid": settings.rag_hybrid,
            "hybrid_recall_k": settings.rag_hybrid_recall_k,
            "rerank": settings.rag_rerank,
            "rerank_endpoint": settings.rerank_endpoint_url or "(未配置)",
            "query_normalize": settings.rag_query_normalize,
            "canary_top_score": canary_score,
        },
        "negative_base": "数据集 expected=[] 负例（不含 CLI --calibrate 的黄金集 rag_no_hit_* 扩展）",
        "raw_easy_recall": raw_recall,
        "constraints": rows,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True,
        ).stdout.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阈值可行带扫描（live probe 留档）")
    parser.add_argument(
        "--dataset", default="app/evaluation/retrieval_cases_v3.json",
        help="冻结检索用例集（easy 正例 + expected=[] 负例）",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--constraints", default=",".join(str(c) for c in DEFAULT_CONSTRAINTS),
        help="正例 recall 约束网格（逗号分隔）",
    )
    parser.add_argument(
        "--variant", default="hybrid-rerank",
        choices=["knn", "hybrid", "hybrid-rerank", "hybrid-rerank-norm"],
        help="检索变体（与门禁轮一致，显式套用而非依赖 .env 现状）",
    )
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args(argv)

    dataset_path = (
        ROOT / args.dataset if not Path(args.dataset).is_absolute()
        else Path(args.dataset)
    )
    constraints = [float(c) for c in args.constraints.split(",")]
    result = scan(dataset_path, args.top_k, constraints, variant=args.variant)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已留档: {out}")
    print(f"原始 easy recall: {result['raw_easy_recall']:.4f}")
    for row in result["constraints"]:
        if row["feasible"]:
            print(
                f"  约束 {row['min_positive_recall']:.2f} → 阈值 "
                f"{row['threshold']:.8f}（easy recall "
                f"{row['easy_recall_at_threshold']:.4f}，负例拒绝 "
                f"{row['negative_rejection_at_threshold']:.4f}）"
            )
        else:
            print(f"  约束 {row['min_positive_recall']:.2f} → 不可达（{row['reason']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

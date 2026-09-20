"""拒答校准（单 Agent 全量优化计划·阶段D）：联合信号拒答阈值冻结工具。

拒答策略不再只看单一 top1 分数——联合四个信号：
- top1 分数（绝对相关度）
- top1 与 top2 的分数间隔（可分性）
- 词面覆盖率（query 与 top1 文本的 bigram Dice，防"高分但不答所问"）
- reranker 分数（配置了精排时；与向量分不同尺度，仅用于二次门控）

流程（冻结纪律）：
1. dev 集网格搜索 → 选出满足「正例拒绝率 ≤ 容忍、负例拒绝率 ≥ 目标」且
   负例拒绝率最大的参数组合；
2. 冻结参数 + 数据集哈希 + 切分哈希 + 版本记录写入 JSON（artifacts 下）；
3. holdout **只允许运行一次**（冻结文件含 holdout_used 标记，重跑必须
   --force 且会记档）——严禁用 holdout 反向调参。

用法：
    python -m app.scripts.calibrate_rejection --dev            # dev 校准
    python -m app.scripts.calibrate_rejection --holdout        # 一次性 holdout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.rag.rejection import (  # noqa: F401 —— 同源实现，历史导入面保留
    bigram_set,
    compute_signals,
    lexical_coverage,
    should_reject,
)
from app.agent.rag.retriever_factory import final_search
from app.config.settings import settings
from app.evaluation.retrieval_metrics import load_cases
from app.observability.logging import get_logger

log = get_logger("app.scripts.calibrate_rejection")

DEFAULT_PARAMS_PATH = ROOT / "artifacts" / "eval" / "v2" / "retrieval-rejection" / "params.json"
DEV_FRACTION = 0.8  # dev/holdout 切分（按 case id 哈希，确定性）
MIN_NEGATIVE_REJECT = 0.90   # 负例拒绝率门禁（计划）
MAX_POSITIVE_REJECT = 0.05   # 正例误拒容忍


class ScoreScaleError(RuntimeError):
    """检索分数处于 RRF/降级尺度（量纲不可比），校准必须中止。"""


# ============================================================
# 信号（实现收敛到 app/agent/rag/rejection.py，生产/校准同源）
# ============================================================
def case_signals(retriever, case: dict, timeout=None) -> dict:
    """单用例的联合拒答信号（走统一 final_search 口径）。"""
    outcome = final_search(
        retriever, case["query"], int(case.get("k", 5)),
        min_score=None, timeout=timeout,
    )
    # 逐例守卫（A5 补）：reranker 运行中故障回退 RRF/降级分时，配置级守卫
    # （rrf_mode）看不见——在 RRF 尺度上校准会静默冻结量纲不可比的阈值。
    if outcome.degraded or outcome.score_source == "rrf":
        raise ScoreScaleError(
            f"case {case.get('id', '')} 分数处于 RRF/降级尺度"
            f"（score_source={outcome.score_source}, degraded={outcome.degraded},"
            f" reason={outcome.degraded_reason or '-'}），禁止参与校准"
        )
    hits = outcome.hits
    expected = set(case.get("expected", []) or [])
    hit_keys = {
        h.chunk.source_path or h.chunk.doc for h in hits
    }
    is_hit = bool(expected & hit_keys) if expected else False
    signals = compute_signals(hits, case["query"])
    signals.update({
        "id": case.get("id", ""),
        "negative": not expected,
        "is_hit": is_hit,
    })
    return signals


# ============================================================
# 切分（确定性哈希）
# ============================================================
def split_cases(cases: list[dict], dev_fraction: float = DEV_FRACTION
                ) -> tuple[list[dict], list[dict]]:
    """按 case id SHA256 前 8 位切分 dev/holdout（同 id 恒同侧）。"""
    dev: list[dict] = []
    holdout: list[dict] = []
    threshold = int(dev_fraction * 0xFFFF)
    for case in cases:
        cid = str(case.get("id", ""))
        bucket = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:4], 16)
        (dev if bucket <= threshold else holdout).append(case)
    return dev, holdout


def dataset_hash(cases: list[dict]) -> str:
    payload = json.dumps(
        sorted(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in cases),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 校准
# ============================================================
def evaluate_params(signals: list[dict], params: dict) -> dict:
    negatives = [s for s in signals if s["negative"]]
    positives = [s for s in signals if not s["negative"]]
    neg_reject = (
        sum(1 for s in negatives if should_reject(s, params)) / len(negatives)
        if negatives else None
    )
    pos_reject = (
        sum(1 for s in positives if should_reject(s, params)) / len(positives)
        if positives else None
    )
    return {
        "negative_reject_rate": neg_reject,
        "positive_reject_rate": pos_reject,
    }


def _sample_grid(values: list[float], n: int) -> list[float]:
    """等距抽样（保序去重）；0.0 恒在首位（= 该信号不设阈值的基准档）。"""
    picked = values[:: max(1, len(values) // n)][:n]
    return [0.0] + [v for v in picked if v > 0.0]


def grid_search(signals: list[dict]) -> tuple[dict, dict]:
    """dev 网格搜索：先满足双门禁，再最大化负例拒绝率。

    搜索 top1 × gap × coverage 三信号组合（rerank 未挂精排时恒 None，不进
    网格；挂上后按需扩展）。双门禁（负例 ≥90%、正例 ≤5%）不可达时回落
    「正例容忍内的最优尽力档」，并在 metrics 标 ``feasible=False`` +
    ``fallback=best_effort``——冻结记录如实留痕，不假装达标。
    """
    tops = sorted({round(s["top1"], 3) for s in signals if s["top1"] is not None})
    coverages = sorted({round(s["coverage"], 3) for s in signals})
    gaps = sorted({
        round(s["gap"], 4) for s in signals if s.get("gap") is not None
    })
    grid_top1 = _sample_grid(tops, 12)
    grid_gap = _sample_grid(gaps, 6)
    grid_cov = _sample_grid(coverages, 6)

    best_gate: tuple[dict, dict] | None = None
    best_effort: tuple[dict, dict] | None = None
    for min_top1 in grid_top1:
        for min_gap in grid_gap:
            for min_coverage in grid_cov:
                if min_top1 == 0.0 and min_gap == 0.0 and min_coverage == 0.0:
                    continue
                params = {"min_top1": min_top1, "min_gap": min_gap,
                          "min_coverage": min_coverage}
                metrics = evaluate_params(signals, params)
                neg = metrics["negative_reject_rate"]
                pos = metrics["positive_reject_rate"]
                if neg is None or pos is None:
                    continue
                if pos > MAX_POSITIVE_REJECT:
                    continue
                if best_effort is None or neg > best_effort[1]["negative_reject_rate"]:
                    best_effort = (params, metrics)
                if neg >= MIN_NEGATIVE_REJECT and (
                    best_gate is None
                    or neg > best_gate[1]["negative_reject_rate"]
                ):
                    best_gate = (params, metrics)
    chosen = best_gate or best_effort
    if chosen is None:  # 全部组合都超正例容忍（极端数据集）：退基准档
        params = {"min_top1": 0.0, "min_gap": 0.0, "min_coverage": 0.0}
        metrics = evaluate_params(signals, params)
        metrics.update(feasible=False, fallback="baseline_zero")
        return params, metrics
    params, metrics = chosen
    metrics = dict(metrics)
    metrics["feasible"] = best_gate is not None
    metrics["fallback"] = "" if best_gate is not None else "best_effort"
    return params, metrics


def _collect_signals(case_split: list[dict]) -> list[dict]:
    from app.agent.tools.knowledge import _get_retriever

    retriever = _get_retriever()
    return [case_signals(retriever, case) for case in case_split]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="联合信号拒答阈值校准（阶段D）")
    parser.add_argument("--dev", action="store_true", help="dev 集校准并冻结参数")
    parser.add_argument("--holdout", action="store_true", help="holdout 一次性验证")
    parser.add_argument("--dataset", default=settings.retrieval_eval_dataset_path)
    parser.add_argument("--out", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--force", action="store_true",
                        help="允许重跑 holdout（记档，正式发布结论无效）")
    args = parser.parse_args(argv)

    # 守卫（A5）：RRF/降级分无语义，禁止在其上校准阈值——否则会静默冻结
    # 一批量纲不可比的「垃圾阈值」并写进 settings 默认值。
    # 与 run_retrieval_eval 的 --calibrate 守卫同源（rejection.rrf_mode）。
    from app.agent.rag.rejection import rrf_mode

    if rrf_mode():
        log.error(
            "RRF 模式（hybrid 且未挂精排）分数无语义，禁止校准联合拒绝阈值："
            "请启用 reranker（RAG_RERANK + RERANK_ENDPOINT_URL）或改用纯向量配置"
        )
        return 2

    cases = load_cases(Path(args.dataset))
    dev_cases, holdout_cases = split_cases(cases)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dev:
        try:
            signals = _collect_signals(dev_cases)
        except ScoreScaleError as exc:
            log.error("校准中止（逐例分数尺度守卫）：%s", exc)
            return 2
        params, metrics = grid_search(signals)
        record = {
            "version": 1,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": str(args.dataset),
            "dataset_hash": dataset_hash(cases),
            "dev_fraction": DEV_FRACTION,
            "dev_size": len(dev_cases),
            "holdout_size": len(holdout_cases),
            "holdout_used": False,
            "params": params,
            "dev_metrics": metrics,
        }
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        log.info("拒答阈值已冻结", payload=json.dumps(
            {"frozen": True, "params": params, "dev_metrics": metrics},
            ensure_ascii=False))
        return 0

    if args.holdout:
        if not out_path.exists():
            log.error("拒答校准 holdout 拒绝：先运行 --dev 冻结参数")
            return 2
        record = json.loads(out_path.read_text(encoding="utf-8"))
        if record.get("holdout_used") and not args.force:
            log.error(
                "holdout 已运行过（冻结纪律：不得反复验证/反向调参）；"
                "如确需重跑用 --force，但结论不作为发布依据"
            )
            return 3
        if record.get("dataset_hash") != dataset_hash(cases):
            log.error("数据集哈希与冻结时不一致，先重新 dev 校准")
            return 4
        try:
            signals = _collect_signals(holdout_cases)
        except ScoreScaleError as exc:
            log.error("holdout 验证中止（逐例分数尺度守卫，一次性资格未消耗）：%s", exc)
            return 2
        metrics = evaluate_params(signals, record["params"])
        record["holdout_used"] = True
        record["holdout_metrics"] = metrics
        record["holdout_ran_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        log.info("holdout 验证完成", payload=json.dumps(
            {"holdout": metrics, "params": record["params"]}, ensure_ascii=False))
        neg = metrics["negative_reject_rate"]
        pos = metrics["positive_reject_rate"]
        ok = neg is not None and neg >= MIN_NEGATIVE_REJECT and (
            pos is None or pos <= MAX_POSITIVE_REJECT
        )
        return 0 if ok else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

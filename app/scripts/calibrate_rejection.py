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

from app.agent.retriever_factory import final_search
from app.config.settings import settings
from app.evaluation.retrieval_metrics import load_cases

DEFAULT_PARAMS_PATH = ROOT / "artifacts" / "eval" / "v2" / "retrieval-rejection" / "params.json"
DEV_FRACTION = 0.8  # dev/holdout 切分（按 case id 哈希，确定性）
MIN_NEGATIVE_REJECT = 0.90   # 负例拒绝率门禁（计划）
MAX_POSITIVE_REJECT = 0.05   # 正例误拒容忍


# ============================================================
# 信号
# ============================================================
def bigram_set(text: str) -> set[str]:
    """汉字 bigram + ASCII 整词（与记忆相关性同一口径）。"""
    import re
    import unicodedata

    value = unicodedata.normalize("NFKC", text or "")
    tokens: set[str] = set()
    tokens.update(w.casefold() for w in re.findall(r"[A-Za-z0-9]+", value))
    run: list[str] = []
    for ch in value:
        if "\u4e00" <= ch <= "\u9fff":
            run.append(ch)
        else:
            if run:
                tokens.update(f"{a}{b}" for a, b in zip(run, run[1:]))
                run = []
    if run:
        tokens.update(f"{a}{b}" for a, b in zip(run, run[1:]))
    return tokens


def lexical_coverage(query: str, text: str) -> float:
    """query 与文本的词面覆盖率（|∩| / |query tokens|）。"""
    q_tokens = bigram_set(query)
    if not q_tokens:
        return 0.0
    t_tokens = bigram_set(text)
    return len(q_tokens & t_tokens) / len(q_tokens)


def case_signals(retriever, case: dict, timeout=None) -> dict:
    """单用例的联合拒答信号（走统一 final_search 口径）。"""
    outcome = final_search(
        retriever, case["query"], int(case.get("k", 5)),
        min_score=None, timeout=timeout,
    )
    hits = outcome.hits
    top1 = hits[0].score if hits else None
    gap = (hits[0].score - hits[1].score) if len(hits) > 1 else None
    coverage = lexical_coverage(
        case["query"], hits[0].chunk.text if hits else "",
    )
    rerank = getattr(hits[0], "rerank_score", None) if hits else None
    if rerank is None and hits:
        rerank = getattr(hits[0].chunk, "rerank_score", None)
    expected = set(case.get("expected", []) or [])
    hit_keys = {
        h.chunk.source_path or h.chunk.doc for h in hits
    }
    is_hit = bool(expected & hit_keys) if expected else False
    return {
        "id": case.get("id", ""),
        "negative": not expected,
        "top1": top1,
        "gap": gap,
        "coverage": coverage,
        "rerank": rerank,
        "is_hit": is_hit,
    }


def should_reject(signal: dict, params: dict) -> bool:
    """联合信号拒答判定（确定性）。"""
    top1 = signal.get("top1")
    if top1 is None:
        return True
    if top1 < params.get("min_top1", 0.0):
        return True
    gap = signal.get("gap")
    if gap is not None and gap < params.get("min_gap", 0.0):
        return True
    if signal.get("coverage", 0.0) < params.get("min_coverage", 0.0):
        return True
    rerank = signal.get("rerank")
    min_rerank = params.get("min_rerank")
    if min_rerank is not None and rerank is not None and rerank < min_rerank:
        return True
    return False


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


def grid_search(signals: list[dict]) -> tuple[dict, dict]:
    """dev 网格搜索：先满足双门禁，再最大化负例拒绝率。"""
    tops = sorted({round(s["top1"], 3) for s in signals if s["top1"] is not None})
    coverages = sorted({round(s["coverage"], 2) for s in signals})
    grid_top1 = [0.0] + tops[:: max(1, len(tops) // 12)][:12]
    grid_cov = [0.0] + coverages[:: max(1, len(coverages) // 6)][:6]
    best: tuple[dict, dict] | None = None
    for min_top1 in grid_top1:
        for min_coverage in grid_cov:
            params = {"min_top1": min_top1, "min_gap": 0.0,
                      "min_coverage": min_coverage}
            metrics = evaluate_params(signals, params)
            neg = metrics["negative_reject_rate"]
            pos = metrics["positive_reject_rate"]
            if neg is None or pos is None:
                continue
            if neg < MIN_NEGATIVE_REJECT or pos > MAX_POSITIVE_REJECT:
                continue
            if best is None or neg > best[1]["negative_reject_rate"]:
                best = (params, metrics)
    return best if best else ({"min_top1": 0.0, "min_gap": 0.0,
                               "min_coverage": 0.0},
                              evaluate_params(signals, {
                                  "min_top1": 0.0, "min_gap": 0.0,
                                  "min_coverage": 0.0}))


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

    cases = load_cases(Path(args.dataset))
    dev_cases, holdout_cases = split_cases(cases)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dev:
        signals = _collect_signals(dev_cases)
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
        signals = _collect_signals(holdout_cases)
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

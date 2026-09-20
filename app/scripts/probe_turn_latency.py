"""单轮延迟探针（P3-2）：按意图分层统计 LLM 调用 P95，校准轮次预算。

背景：``settings.turn_budget_seconds`` 自述「待 T0 探针实测 P95 后按
``max_react_steps × P95 < turn_budget_seconds`` 校准」，但该探针从未落地——
本脚本补上这一环。

做法：从黄金集按意图分层抽样若干用例，逐条经 ``Sandbox`` 真实运行，
收集 ``RunTrace.llm_calls[*].latency_ms``（每次 LLM 调用的墙钟延迟），
按意图分层与总体统计 P50/P95，并按公式给出预算建议。

**抽样而非全量**：全量 317 例在推理模型下单轮可达数十秒，成本与时长都不适合
作为常规探针。样本量与被抽中的用例 id 写进报告，结论按「样本口径」陈述。

用法：
    python -m app.scripts.probe_turn_latency --per-intent 2
    python -m app.scripts.probe_turn_latency --per-intent 2 --json-out artifacts/eval/p32/latency-probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from app.evaluation.dataset import load_dataset
from app.evaluation.sandbox import Sandbox
from app.observability.logging import configure_logging, get_logger

log = get_logger("app.scripts.probe_turn_latency")

PROTOCOL = "turn-latency-probe-v1"


def _percentile(values: list[float], pct: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认口径一致，零依赖）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def sample_cases(cases, per_intent: int):
    """按 expected_intent 分层抽样：每层取前 N 条（确定性，可复现）。"""
    buckets: dict[str, list] = defaultdict(list)
    for case in cases:
        buckets[case.expected_intent or "unspecified"].append(case)
    picked = []
    for intent in sorted(buckets):
        picked.extend(buckets[intent][:per_intent])
    return picked


def probe(dataset: str, per_intent: int, timeout_note: str = "") -> dict:
    cases = load_dataset(dataset)
    picked = sample_cases(cases, per_intent)
    sandbox = Sandbox()
    by_intent: dict[str, list[float]] = defaultdict(list)
    all_latencies: list[float] = []
    react_steps: list[int] = []
    failures: list[dict] = []

    for case in picked:
        try:
            trace = sandbox.run(case)
        except Exception as e:  # noqa: BLE001 —— 探针不因单例失败中断
            failures.append({"case_id": case.id, "error": type(e).__name__})
            continue
        if trace.error:
            failures.append({"case_id": case.id, "error": str(trace.error)[:120]})
        for call in trace.llm_calls:
            by_intent[case.expected_intent or "unspecified"].append(call.latency_ms)
            all_latencies.append(call.latency_ms)
        steps = int(getattr(trace, "react_steps", 0) or 0)
        if steps:
            react_steps.append(steps)

    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "per_intent": per_intent,
        "sampled_cases": [c.id for c in picked],
        "n_cases": len(picked),
        "n_calls": len(all_latencies),
        "overall": _stats(all_latencies),
        "by_intent": {k: _stats(v) for k, v in sorted(by_intent.items())},
        "react_steps": _stats([float(s) for s in react_steps]),
        "failures": failures,
        "note": timeout_note,
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50_ms": round(_percentile(values, 50), 1),
        "p95_ms": round(_percentile(values, 95), 1),
        "max_ms": round(max(values), 1),
        "mean_ms": round(statistics.fmean(values), 1),
    }


def recommend(report: dict, max_react_steps: int) -> dict:
    """按 max_react_steps × P95 < turn_budget_seconds 给出预算建议。"""
    p95_ms = float(report["overall"].get("p95_ms") or 0.0)
    p95_s = p95_ms / 1000.0
    worst_steps = float(report["react_steps"].get("max_ms") or 0.0)
    required = p95_s * max_react_steps
    return {
        "max_react_steps": max_react_steps,
        "llm_call_p95_seconds": round(p95_s, 2),
        "required_budget_seconds": round(required, 1),
        "observed_max_react_steps": worst_steps,
        "formula": "max_react_steps × P95(llm_call) < turn_budget_seconds",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="单轮延迟探针（P3-2 预算校准依据）")
    parser.add_argument("--dataset", default="app/evaluation/cases_large.json")
    parser.add_argument("--per-intent", type=int, default=2,
                        help="每个意图层抽样的用例数（默认 2）")
    parser.add_argument("--max-react-steps", type=int, default=8)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    configure_logging(json_output=False)
    from app.config.settings import settings

    log.info("=" * 62)
    log.info("  单轮延迟探针（P3-2）：分层抽样实测 LLM 调用延迟")
    log.info(f"  数据集   : {args.dataset}")
    log.info(f"  每层样本 : {args.per_intent}")
    log.info(f"  当前预算 : turn_budget_seconds={settings.turn_budget_seconds} "
             f"max_react_steps={settings.max_react_steps}")
    log.info("=" * 62)

    report = probe(args.dataset, args.per_intent)
    report["recommendation"] = recommend(report, args.max_react_steps)
    report["current_settings"] = {
        "turn_budget_seconds": settings.turn_budget_seconds,
        "max_react_steps": settings.max_react_steps,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
    }

    log.info(f"\n样本：{report['n_cases']} 例 / {report['n_calls']} 次 LLM 调用")
    log.info(f"总体 P50={report['overall'].get('p50_ms')}ms "
             f"P95={report['overall'].get('p95_ms')}ms "
             f"max={report['overall'].get('max_ms')}ms")
    for intent, stat in report["by_intent"].items():
        log.info(f"  {intent:<16} n={stat['n']:<4} "
                 f"P50={stat.get('p50_ms')}ms P95={stat.get('p95_ms')}ms")
    rec = report["recommendation"]
    log.info(f"\n建议：{rec['formula']}")
    log.info(f"  P95={rec['llm_call_p95_seconds']}s × {rec['max_react_steps']} 步 "
             f"= {rec['required_budget_seconds']}s")
    log.info(f"  实测最大 ReAct 步数：{rec['observed_max_react_steps']}")
    if report["failures"]:
        log.info(f"\n失败/异常用例：{len(report['failures'])} 条（见报告）")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        log.info(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()

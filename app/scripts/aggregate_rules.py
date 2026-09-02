"""把 run_subset_diag 的逐例 JSON 聚合为规则轮报告（与 run_rules_round 同格式）。

用法：python -m app.scripts.aggregate_rules --run-id diag-all-candidate --out ab-fast-c5bfd0e-candidate-rules
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.evaluation.manifest import build_manifest  # noqa: E402
from app.scripts.run_ab_eval import (  # noqa: E402
    BASELINE_CFG,
    CANDIDATE_CFG,
    _category_pass_rates,
    _security_category_rates,
)
from app.observability.logging import get_logger  # noqa: E402

log = get_logger("app.scripts.aggregate_rules")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="逐例所在 run-id（diag 目录）")
    parser.add_argument("--out", required=True, help="输出 run-id（规则轮报告）")
    parser.add_argument("--arm", choices=["baseline", "candidate"], required=True)
    args = parser.parse_args()

    results_dir = ROOT / settings.eval_output_dir / args.run_id / "results"
    files = sorted(results_dir.glob("*.json"))
    if not files:
        log.info("❌ 无逐例结果: %s", results_dir)
        return 1
    per_case = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    total = len(per_case)
    passed = sum(1 for r in per_case if r["passed"])
    errors = [r["case_id"] for r in per_case if r.get("error")]
    dataset_path = ROOT / "app/evaluation/cases_large.json"

    def avg(section: str, key: str):
        vals = [r.get(section, {}).get(key) for r in per_case]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_process_score": avg("process", "process_score"),
        "avg_result_score": avg("result", "result_score"),
        "total_tokens": sum(r.get("process", {}).get("token_cost", 0) or 0 for r in per_case),
        "avg_tokens_per_case": round(
            sum(r.get("process", {}).get("token_cost", 0) or 0 for r in per_case) / total, 2
        ) if total else 0,
        "distributions": {
            "tool_calls": {"n": total},
            "tokens": {"n": total},
        },
        "security": {
            "critical_total": sum(1 for r in per_case if r.get("security", {}).get("critical_gate_pass") is not None),
            "critical_passed": sum(1 for r in per_case if r.get("security", {}).get("critical_gate_pass") is True),
        },
    }

    # 报告结构需要 cases 字段含 passed 与分类所需字段；聚合时补全
    # （这里保留原始逐例 + passed，分类通过率复用 _category_pass_rates 需要 case_id）
    report_cases = [
        {"case_id": r["case_id"], "passed": r["passed"]} for r in per_case
    ]
    category_rates = _category_pass_rates({"cases": report_cases})
    security_rates = _security_category_rates({"cases": report_cases})

    cfg = CANDIDATE_CFG if args.arm == "candidate" else BASELINE_CFG
    manifest = build_manifest(
        dataset_path=str(dataset_path), num_cases=total,
        model=settings.model_name, judge_model="",
        mode="single", use_judge=False,
        config_overrides={"ab_arm": cfg.to_dict()},
    )
    manifest["run_id"] = args.out
    manifest["arm"] = args.arm

    out_dir = ROOT / settings.eval_output_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "eval-v2-rules-round",
        "note": "聚合自 run_subset_diag 逐例 checkpoint（4 并发 + resilient 韧性层）",
        "run_id": args.out,
        "arm": args.arm,
        "arm_config": cfg.to_dict(),
        "dataset": {"path": str(dataset_path), "num_cases": total},
        "summary": summary,
        "category_pass_rates": category_rates,
        "security_category_rates": security_rates,
        "critical": {
            "total": summary["security"]["critical_total"],
            "passed": summary["security"]["critical_passed"],
            "rate": round(summary["security"]["critical_passed"] / summary["security"]["critical_total"], 4)
            if summary["security"]["critical_total"] else None,
        },
        "error_cases": errors,
        "api_errors_ok": not errors,
        "cases": per_case,
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("聚合完成: %s 通过率 %.1f%% (%d/%d) critical %d/%d",
             args.arm, summary["pass_rate"] * 100, passed, total,
             summary["security"]["critical_passed"], summary["security"]["critical_total"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
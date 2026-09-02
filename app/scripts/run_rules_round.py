"""规则轮单臂评测：317 条全量 × 无 Judge（确定性代码规则指标）。

用途：在完整 A/B（规则轮+完整 Judge 轮+关键子集轮）耗时不可行时，单独产出
某配置的规则轮指标。产物明确标注「仅规则轮、无 Judge 层」，不冒充完整 A/B。

用法：
  python -m app.scripts.run_rules_round --arm candidate --run-id ab-c5bfd0e-rules-candidate
  python -m app.scripts.run_rules_round --arm baseline --run-id ...

输出 artifacts/eval/v2/<run-id>/report.json（逐例 + 汇总 + 分类通过率）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator  # noqa: E402
from app.evaluation.manifest import build_manifest  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402
from app.observability.logging import get_logger  # noqa: E402
from app.scripts.run_ab_eval import (  # noqa: E402
    BASELINE_CFG,
    CANDIDATE_CFG,
    _category_pass_rates,
    _security_category_rates,
    _security_regressions,
    run_arm,
)

log = get_logger("app.scripts.run_rules_round")

ARMS = {"baseline": BASELINE_CFG, "candidate": CANDIDATE_CFG}


def main() -> int:
    parser = argparse.ArgumentParser(description="规则轮单臂评测（无 Judge，确定性）")
    parser.add_argument("--arm", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--dataset", default="app/evaluation/cases_large.json")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发数（默认 1=串行；>1 时逐 case 独立沙箱并发，"
                             "需 resilient 韧性层抗 429）")
    args = parser.parse_args()

    cfg = ARMS[args.arm]
    dataset_path = ROOT / args.dataset
    cases = load_dataset(dataset_path)
    run_id = args.run_id or f"rules-{args.arm}-{cfg.name}"
    out_dir = ROOT / settings.eval_output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("规则轮开始（仅 %s，%d 条全量，无 Judge，workers=%d）: %s",
             cfg.name, len(cases), args.workers, run_id)
    cfg.apply_globals()

    if args.workers <= 1:
        report, manifest = run_arm(
            cfg, cases, judge_model="", use_judge=False,
            run_id=run_id, dataset_path=str(dataset_path),
        )
    else:
        # 并发路径：逐 case 独立沙箱（同 run_subset_diag 模式），
        # 结果聚合复用 Evaluator._aggregate，保证与串行口径一致。
        import shutil
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from app.evaluation.evaluator import Evaluator
        from app.evaluation.sandbox import Sandbox
        from openai import OpenAI

        def run_one(case):
            sandbox = Sandbox(mode="single", resilient=True)
            evaluator = Evaluator(
                sandbox=sandbox,
                client=OpenAI(api_key=settings.openai_api_key,
                              base_url=settings.openai_base_url),
                model=settings.model_name, use_judge=False,
                pass_threshold=settings.eval_pass_threshold,
            )
            try:
                # 返回 EvalResult 原始对象（_aggregate 需要 .passed 属性）
                return evaluator.run_case(case)
            finally:
                shutil.rmtree(sandbox.tmp_root, ignore_errors=True)

        results: list = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_one, c) for c in cases]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % 50 == 0:
                    log.info("进度: %d/%d", i, len(cases))
        aggregated = Evaluator._aggregate(
            Evaluator(sandbox=None, client=None, model=settings.model_name),
            results,
        )
        report = {"summary": aggregated["summary"], "cases": aggregated["cases"]}
        manifest = build_manifest(
            dataset_path=str(dataset_path), num_cases=len(cases),
            model=settings.model_name, judge_model="",
            mode="single", use_judge=False,
            config_overrides={"ab_arm": cfg.to_dict()},
        )
    manifest["run_id"] = run_id
    summary = report["summary"]

    category_rates = _category_pass_rates(report)
    security_rates = _security_category_rates(report)
    critical = summary.get("security", {})
    critical_total = int(critical.get("critical_total", 0))
    critical_passed = int(critical.get("critical_passed", 0))
    error_cases = [c["case_id"] for c in report.get("cases", []) if c.get("error")]

    payload = {
        "protocol": "eval-v2-rules-round",
        "note": "仅规则轮（无 Judge 层），非完整 A/B 口径；不能被当作完整评测通过率",
        "run_id": run_id,
        "arm": cfg.name,
        "arm_config": cfg.to_dict(),
        "dataset": {"path": str(dataset_path), "num_cases": len(cases)},
        "summary": summary,
        "category_pass_rates": category_rates,
        "security_category_rates": security_rates,
        "critical": {
            "total": critical_total,
            "passed": critical_passed,
            "rate": round(critical_passed / critical_total, 4) if critical_total else None,
        },
        "error_cases": error_cases,
        "api_errors_ok": not error_cases,
        "cases": report["cases"],  # 逐例记录：通过判定、过程/结果维度、错误详情
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    log.info("=" * 60)
    log.info("  %s 规则轮结果", cfg.name)
    log.info("  通过率        : %.1f%% (%d/%d)", summary["pass_rate"] * 100,
             summary["passed"], summary["total"])
    log.info("  critical 安全 : %d/%d", critical_passed, critical_total)
    log.info("  分类通过率    : %s", {k: f"{v * 100:.1f}%" for k, v in sorted(category_rates.items())})
    log.info("  API error     : %d", len(error_cases))
    log.info("  报告: %s", out_dir / "report.json")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
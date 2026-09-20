"""记忆系统消融评测（记忆系统重构·阶段0 基线 / 阶段4 调参）。

用法：
    # 阶段0 before 基线（纯词面，semantic 关闭）
    python -m app.scripts.run_memory_ablation --no-judge --tag before

    # 阶段2/4 after（语义开启；需嵌入可用）
    python -m app.scripts.run_memory_ablation --no-judge --tag after --semantic

    # 自定义消融项（单项变动，不过则回退）
    python -m app.scripts.run_memory_ablation --no-judge --tag cap80 \
        --semantic --set max_ltm_facts=80 --set memory_budget_share=0.15
    python -m app.scripts.run_memory_ablation --no-judge --tag summary500 \
        --set summary_max_chars=500
    python -m app.scripts.run_memory_ablation --no-judge --tag wsem07 \
        --semantic --set memory_semantic_weight=0.7 --set memory_lexical_weight=0.3

报告落 artifacts/eval/v2/memory-ablation-<tag>-<id>/（report.json + manifest.json）。
裁决：对比同数据集两次 run 的 pass_rate 与各 case keyword_coverage；
计划 §6 候选项按 eval_pass_threshold 裁决，不过则回退配置默认值。

依赖：真实 LLM + （--semantic 时）可用嵌入服务；本脚本不做 mock。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not key or not value:
            raise SystemExit(f"--set 格式错误（应为 key=value）: {pair}")
        try:
            out[key] = json.loads(value)  # 数字/布尔按 JSON 解析
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="记忆系统消融评测（阶段0/4）")
    parser.add_argument(
        "--dataset", default="app/evaluation/memory_cases.json",
        help="评测集（默认记忆基线集）",
    )
    parser.add_argument("--tag", default="", help="报告目录标签（before/after/cap80…）")
    parser.add_argument("--semantic", action="store_true",
                        help="开启语义混合检索（memory_semantic_enabled=true）")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="settings 覆盖（key=value，可重复）")
    parser.add_argument("--judge", dest="judge", action="store_true", default=False)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--judge-model", default="")
    args = parser.parse_args(argv)

    from openai import OpenAI

    from app.config.settings import settings
    from app.evaluation.dataset import load_dataset
    from app.evaluation.evaluator import Evaluator
    from app.evaluation.manifest import build_manifest
    from app.evaluation.sandbox import Sandbox

    # 应用配置覆盖（进程内生效；--semantic 只是 --set 的语义快捷方式）
    overrides = _parse_overrides(args.overrides)
    if args.semantic:
        overrides.setdefault("memory_semantic_enabled", True)
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise SystemExit(f"未知 settings 字段: {key}")
        setattr(settings, key, value)
        print(f"  override: {key} = {value}")

    dataset_path = ROOT / args.dataset
    cases = load_dataset(dataset_path)
    print(f"数据集: {dataset_path}（{len(cases)} 条）")

    client = OpenAI(api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url)
    judge_client = OpenAI(api_key=settings.openai_api_key,
                          base_url=settings.openai_base_url) if args.judge else None
    sandbox = Sandbox()
    evaluator = Evaluator(
        sandbox=sandbox, client=client, model=settings.model_name,
        use_judge=args.judge, pass_threshold=settings.eval_pass_threshold,
        judge_client=judge_client,
        judge_model=args.judge_model or settings.eval_judge_model,
    )
    report = evaluator.run_all(cases)

    run_id = f"memory-ablation-{args.tag or 'run'}-{uuid.uuid4().hex[:6]}"
    out_dir = ROOT / settings.eval_output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = build_manifest(
        dataset_path=str(dataset_path), num_cases=len(cases),
        model=settings.model_name,
        judge_model=args.judge_model if args.judge else "",
        use_judge=args.judge,
    )
    manifest["memory_ablation"] = {
        "tag": args.tag, "semantic": bool(args.semantic), "overrides": overrides,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    summary = report.get("summary", {})
    print(f"\n通过率: {summary.get('passed')}/{summary.get('total')}"
          f"（{summary.get('pass_rate', 0) * 100:.0f}%）")
    for r in report.get("results", []):
        mark = "✅" if r.get("passed") else "❌"
        print(f"  {mark} {r.get('case_id')}: "
              f"keyword={r.get('keyword_coverage')} error={r.get('error') or '-'}")
    print(f"\n报告: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

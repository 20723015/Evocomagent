"""离线运行 Agent 评估体系（第9期；3.1 冻结 v2 协议）。

用法：
  # 快速规则路径（只用代码规则，快、零额外 LLM 开销）
  python app/scripts/run_eval.py --no-judge

  # 全量评估（LLM-as-judge 需提供与被测模型不同的 EVAL_JUDGE_MODEL，3.2 强制）
  python app/scripts/run_eval.py --judge-model glm-4.6

  # 报告写入 artifacts/eval/v2/<run-id>/（含 manifest.json）
  python app/scripts/run_eval.py --run-id run-001

协议（eval-v2 冻结）：
- 报告默认输出 artifacts/eval/v2/<run-id>/report.json + manifest.json；
- manifest 记录数据集 SHA-256 / git commit / prompt SHA-256 / 模型与 Judge /
  temperature / token 上限 / 后端 / reranker / 阈值 / 依赖锁；
- 正式评测（--judge 开启）要求 EVAL_JUDGE_MODEL 与被测模型不同，否则拒绝运行；
- Judge temperature 固定 0，原始理由与解析失败记录在 judge_reasons。
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from app.observability.logging import get_logger
log = get_logger("app.scripts.run_eval")


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator  # noqa: E402
from app.evaluation.manifest import build_manifest  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402


class JudgeModelConfigError(RuntimeError):
    """正式评测未提供与被测模型不同的 Judge 模型。"""


def resolve_judge_model(args_judge_model: str) -> str:
    """不同模型的 Judge：命令行 > settings.eval_judge_model；正式评测强制不同。"""
    judge = args_judge_model or settings.eval_judge_model
    if judge and judge != settings.model_name:
        return judge
    raise JudgeModelConfigError(
        "正式评测要求提供与被测模型不同的 Judge 模型（EVAL_JUDGE_MODEL 或 --judge-model），"
        f"且必须与被测模型（{settings.model_name}）不同；"
        "否则拒绝发布。"
    )


def _fmt(value) -> str:
    """格式化评分单元格：None→"-"，bool→✓/✗，float→百分比。"""
    if value is None:
        return "  - "
    if isinstance(value, bool):
        return "  ✓ " if value else "  ✗ "
    return f"{value * 100:4.0f}%"


def _print_report(report: dict) -> None:
    cases = report["cases"]

    log.info("\n" + "=" * 78)
    log.info("  过程指标（工具准确率 / 调用效率 / token达标 / 过程合理性）")
    log.info("=" * 78)
    log.info(f"  {'用例':<22}{'工具准确':>8}{'调用效率':>8}{'token':>8}{'token达标':>9}{'过程合理':>8}")
    for c in cases:
        p = c["process"]
        log.info(
            f"  {c['case_id']:<22}"
            f"{_fmt(p['tool_accuracy']):>8}{_fmt(p['tool_efficiency']):>8}"
            f"{p['token_cost']:>8}{_fmt(p['token_pass']):>9}{_fmt(p['process_soundness']):>8}"
        )

    log.info("\n" + "=" * 78)
    log.info("  结果指标（意图 / 关键信息完整性 / 转人工 / 回答质量 / 忠实度无幻觉）")
    log.info("=" * 78)
    log.info(f"  {'用例':<22}{'意图':>8}{'完整性':>8}{'转人工':>8}{'回答质量':>8}{'无幻觉':>8}")
    for c in cases:
        r = c["result"]
        flag = "" if c["passed"] else "  ❌"
        if c["error"]:
            log.info(f"  {c['case_id']:<22}  运行/评分异常: {c['error']}")
            continue
        log.info(
            f"  {c['case_id']:<22}"
            f"{_fmt(r['intent_match']):>8}{_fmt(r['keyword_coverage']):>8}"
            f"{_fmt(r['requires_human_match']):>8}{_fmt(r['answer_quality']):>8}"
            f"{_fmt(r['faithfulness']):>8}{flag}"
        )

    s = report["summary"]
    log.info("\n" + "=" * 78)
    log.info("  汇总")
    log.info("=" * 78)
    log.info(f"  通过率        : {s['passed']}/{s['total']}  ({s['pass_rate'] * 100:.0f}%)")
    log.info(f"  平均过程得分  : {_fmt(s['avg_process_score']).strip()}")
    log.info(f"  平均结果得分  : {_fmt(s['avg_result_score']).strip()}")
    log.info(f"  总 token 消耗 : {s['total_tokens']}（平均每用例 {s['avg_tokens_per_case']:.0f}）")


def main():
    parser = argparse.ArgumentParser(description="运行 Agent 评估体系（v2 冻结协议）")
    parser.add_argument(
        "--dataset", default=settings.eval_dataset_path,
        help=f"测试集 JSON 路径（默认: {settings.eval_dataset_path}）",
    )
    parser.add_argument(
        "--mode", choices=["single", "multi"],
        default="multi" if settings.multi_agent_enabled else "single",
        help="被测 Agent 模式（默认据 multi_agent_enabled）",
    )
    parser.add_argument(
        "--judge", dest="judge", action="store_true", default=settings.eval_use_judge,
        help="启用 LLM-as-judge（默认开）",
    )
    parser.add_argument(
        "--no-judge", dest="judge", action="store_false",
        help="只跑代码规则指标，不调 LLM judge（快、便宜）",
    )
    parser.add_argument(
        "--judge-model", default="", help="不同于被测模型的 Judge 模型（3.2 强制）",
    )
    parser.add_argument(
        "--run-id", default="", help="本次运行标识（缺省自动生成）；报告目录 run-id/",
    )
    parser.add_argument(
        "--output", default=None, help="兼容旧参数：报告文件路径（默认 artifacts/eval/v2/<run-id>/report.json）",
    )
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)

    judge_model = ""
    if args.judge:
        judge_model = resolve_judge_model(args.judge_model)

    log.info("=" * 78)
    log.info("  并夕夕 · Agent 评估体系（eval-v2 冻结协议）")
    log.info(f"  模式      : {args.mode}")
    log.info(f"  数据集    : {dataset_path}")
    log.info(f"  LLM judge : {'开启' if args.judge else '关闭（仅规则）'}")
    log.info(f"  被测模型  : {settings.model_name}")
    log.info(f"  裁判模型  : {judge_model or '（未启用）'}")
    log.info("=" * 78)

    log.info("\n[1/3] 加载测试集...")
    if not dataset_path.exists():
        log.info(f"❌ 测试集不存在: {dataset_path}")
        sys.exit(1)
    cases = load_dataset(dataset_path)
    if not cases:
        log.info("❌ 测试集为空")
        sys.exit(1)
    log.info(f"   共 {len(cases)} 条用例")

    log.info(f"\n[2/3] 在沙箱中重跑测试集（{args.mode} 模式）...")
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    judge_client = (
        OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        if judge_model else None
    )
    sandbox = Sandbox(mode=args.mode)
    evaluator = Evaluator(
        sandbox=sandbox,
        client=client,
        model=settings.model_name,
        use_judge=args.judge,
        pass_threshold=settings.eval_pass_threshold,
        judge_client=judge_client,
        judge_model=judge_model,
    )
    report = evaluator.run_all(cases)

    log.info("\n[3/3] 生成评估报告...")
    _print_report(report)

    # 3.1：报告默认写入 artifacts/eval/v2/<run-id>/（report + manifest）
    if args.output is None:
        run_id = args.run_id or f"run-{uuid.uuid4().hex[:8]}"
        out_dir = ROOT / settings.eval_output_dir / run_id
        out_path = out_dir / "report.json"
    else:
        out_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
        out_dir = out_path.parent
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = build_manifest(
        dataset_path=str(dataset_path), num_cases=len(cases),
        model=settings.model_name, judge_model=judge_model,
        mode=args.mode, use_judge=args.judge,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"\n   报告已写入: {out_path}")
    log.info(f"   manifest  : {out_dir / 'manifest.json'}（commit {manifest['git']['commit'][:10]}…）")

    log.info("\n🎉 评估完成。")


if __name__ == "__main__":
    main()

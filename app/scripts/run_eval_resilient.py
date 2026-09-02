"""弹性评估执行器：全量跑 → 自动重试 error 用例 → 合并报告（2026-08 加，3.1 v2）。

背景：全量评测（283 条 × judge）耗时长，中途外部 API 网络抖动会污染整轮
（轮 4 实测 116-215 条 APIConnectionError 直接失败）。本工具把「跑完一批 →
挑出 error → 等待后重试」自动化，最多 retries 轮；网络恢复的快慢决定总耗时，
不会因为一次抖动就废掉整轮。

3.1 冻结协议：
- 输出 artifacts/eval/v2/<run-id>/report.json + manifest.json；
- 断点续跑/合并旧报告前必须校验 experiment id 与全部配置 hash：
  不一致（数据集/模型/Judge/后端/阈值变化）→ 拒绝合并，要求新 run-id；
- 正式评测（--judge）强制使用与被测模型不同的 Judge 模型（EVAL_JUDGE_MODEL）。

用法（与 run_eval.py 同参数，多一个 --retries）：
  python -m app.scripts.run_eval_resilient --dataset app/evaluation/cases_large.json \\
      --run-id v2-candidate --judge-model glm-4.6 --retries 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator  # noqa: E402
from app.evaluation.manifest import build_manifest, verify_manifest_unchanged  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402
from app.observability.logging import get_logger  # noqa: E402
from app.scripts.run_eval import JudgeModelConfigError, resolve_judge_model  # noqa: E402

log = get_logger("app.scripts.run_eval_resilient")


class ManifestMismatchError(RuntimeError):
    """续跑/合并前校验失败：配置漂移，拒绝合并旧报告。"""


def _aggregate(results: list[dict]) -> dict:
    """按 run_eval 的 summary 口径重算（results 为 _result_to_dict 输出）。"""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    total_tokens = sum(r["process"]["token_cost"] for r in results)

    def avg(section: str, key: str) -> float | None:
        vals = [r[section][key] for r in results if r[section].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_process_score": avg("process", "process_score"),
            "avg_result_score": avg("result", "result_score"),
            "total_tokens": total_tokens,
            "avg_tokens_per_case": total_tokens / total if total else 0,
        },
        "cases": results,
    }


def _write_checkpoint(out_path: Path, results: dict[str, dict]) -> None:
    """每轮 merge 后立即落盘——进程崩溃也不丢已跑结果，可断点续跑。"""
    report = _aggregate(sorted(results.values(), key=lambda r: r["case_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint_manifest(path: Path) -> dict | None:
    """读取运行目录下的 manifest；不存在返回 None。"""
    mpath = path.parent / "manifest.json"
    try:
        return json.loads(mpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="弹性评估：全量跑 + error 自动重试合并")
    parser.add_argument("--dataset", default=settings.eval_dataset_path)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--judge", dest="judge", action="store_true", default=settings.eval_use_judge)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--judge-model", default="", help="不同于被测模型的 Judge 模型（3.2）")
    parser.add_argument("--run-id", default="", help="运行标识（缺省自动生成）")
    parser.add_argument("--output", default="", help="兼容旧参数：报告文件路径")
    parser.add_argument("--retries", type=int, default=4, help="error 用例重试轮数上限")
    parser.add_argument("--retry-wait", type=int, default=60, help="每轮重试前等待秒数")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)

    judge_model = ""
    if args.judge:
        judge_model = resolve_judge_model(args.judge_model)

    # 3.1：报告目录默认 artifacts/eval/v2/<run-id>/（report + manifest）
    if args.output:
        out_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    else:
        run_id = args.run_id or f"run-{uuid.uuid4().hex[:8]}"
        out_path = ROOT / settings.eval_output_dir / run_id / "report.json"

    cases = {c.id: c for c in load_dataset(dataset_path)}
    log.info(f"数据集 {len(cases)} 条 | 模式 {args.mode} | judge {'开' if args.judge else '关'}")

    # 断点续跑：已有报告 → 必须先通过 manifest 校验（experiment id + 配置 hash），
    # 不一致拒绝合并（配置漂移的旧结果不能与本次混合）
    results: dict[str, dict] = {}
    prev_manifest = None
    if out_path.exists():
        prev_manifest = _checkpoint_manifest(out_path)
        if prev_manifest is None:
            raise ManifestMismatchError(
                f"已存在报告 {out_path} 但缺少 manifest（旧格式/非 v2）："
                "禁止与冻结协议结果混合，请换 --run-id 重新开始"
            )
        if not verify_manifest_unchanged(
            prev_manifest, dataset_path=str(dataset_path),
            model=settings.model_name, judge_model=judge_model,
            mode=args.mode, use_judge=args.judge,
        ):
            raise ManifestMismatchError(
                f"已存在报告 {out_path} 的配置指纹与本次不一致（数据集/模型/Judge/"
                "后端/阈值变化）：拒绝合并，请换 --run-id 重新开始"
            )
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        results = {c["case_id"]: c for c in prev["cases"] if not c["error"]}
        pending_ids = [c["case_id"] for c in prev["cases"] if c["error"]]
        if pending_ids:
            log.info(f"断点续跑：保留 {len(results)} 条正常结果，重试 {len(pending_ids)} 条 error")
    else:
        pending_ids = list(cases.keys())
    log.info(f"待跑 {len(pending_ids)} 条")

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    judge_client = (
        OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        if judge_model else None
    )
    last_error_ids: list[str] = []

    for attempt in range(1, args.retries + 1):
        if not pending_ids:
            break
        if attempt > 1:
            log.info(f"等待 {args.retry_wait}s 后第 {attempt} 轮重试（剩 {len(pending_ids)} 条）…")
            time.sleep(args.retry_wait)

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
        batch = [cases[cid] for cid in pending_ids if cid in cases]
        log.info(f"[{attempt}/{args.retries}] 跑 {len(batch)} 条…")
        sub = evaluator.run_all(batch)

        for item in sub["cases"]:
            results[item["case_id"]] = item

        # 挑出本轮仍失败的（error 或未通过均不重试？——只重试 error，未通过是真实结果）
        last_error_ids = [c["case_id"] for c in sub["cases"] if c["error"]]
        pending_ids = [
            cid for cid in pending_ids
            if cid in last_error_ids
        ]
        ok_now = len(batch) - len(last_error_ids)
        log.info(f"本轮完成 {ok_now}/{len(batch)}，剩余 error {len(last_error_ids)}")

        # 每轮 checkpoint：崩溃也不丢已跑结果（断点续跑依据）
        _write_checkpoint(out_path, results)
        if out_path.parent is not None:
            manifest = build_manifest(
                dataset_path=str(dataset_path), num_cases=len(cases),
                model=settings.model_name, judge_model=judge_model,
                mode=args.mode, use_judge=args.judge,
            )
            (out_path.parent / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        log.info(f"checkpoint 已落盘 {out_path}（{len(results)} 条结果）")

    log.info(f"最终：成功 {len([r for r in results.values() if not r['error']])} 条，"
             f"仍 error {len([r for r in results.values() if r['error']])} 条")

    report = _aggregate(sorted(results.values(), key=lambda r: r["case_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_path.parent is not None:
        manifest = build_manifest(
            dataset_path=str(dataset_path), num_cases=len(cases),
            model=settings.model_name, judge_model=judge_model,
            mode=args.mode, use_judge=args.judge,
        )
        (out_path.parent / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    s = report["summary"]
    log.info(
        f"报告已写入 {out_path}：通过 {s['passed']}/{s['total']} "
        f"({s['pass_rate'] * 100:.1f}%) token {s['total_tokens']} "
        f"({s['avg_tokens_per_case']:.0f}/条)"
    )


if __name__ == "__main__":
    main()

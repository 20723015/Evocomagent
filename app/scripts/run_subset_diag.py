"""规则轮诊断子集：按 case-id 列表跑 candidate 规则轮，逐例落盘 + 断点续跑。

用途：快速定位 critical / API error / 低分类别的失败维度，为修复提供逐例
reply、工具序列、mock 结果与各规则维度分数。复用 run_ab_eval.run_arm（同一
arm 配置与聚合口径），每个 case 独立 checkpoint（results/<case_id>.json），
重启时跳过已完成，不重复跑。

用法：
  python -m app.scripts.run_subset_diag --case-ids file.txt [--workers 4] \
      --run-id diag-001
  --case-ids 也支持内置名：critical / errors / low4（order,logistics,return,rag）
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402
from app.observability.logging import get_logger  # noqa: E402
from app.scripts.run_ab_eval import CANDIDATE_CFG  # noqa: E402

log = get_logger("app.scripts.run_subset_diag")

_LOCK = threading.Lock()


def _load_case_ids(path_or_name: str, cases) -> list[str]:
    if Path(path_or_name).exists():
        return [ln.strip() for ln in Path(path_or_name).read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_id = {c.id: c for c in cases}
    if path_or_name == "critical":
        return [c.id for c in cases if c.critical]
    if path_or_name == "errors":
        return [c.id for c in cases if c.id in {
            "order_query_002_8", "after_sale_1", "edge_10", "kb_cross_06"}]
    if path_or_name == "low4":
        low = {"order", "logistics", "return", "rag"}
        return [c.id for c in cases if c.id.split("_", 1)[0] in low]
    raise ValueError(f"未知 case 集合: {path_or_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="规则轮诊断子集（逐例 checkpoint）")
    parser.add_argument("--case-ids", required=True, help="case-id 文件路径或内置名 critical/errors/low4")
    parser.add_argument("--dataset", default="app/evaluation/cases_large.json")
    parser.add_argument("--run-id", default="diag")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0, help="调试用：只跑前 N 条")
    parser.add_argument("--arm", choices=["baseline", "candidate"], default="candidate")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    all_cases = load_dataset(dataset_path)
    ids = _load_case_ids(args.case_ids, all_cases)
    if args.max_cases:
        ids = ids[: args.max_cases]
    by_id = {c.id: c for c in all_cases}
    cases = [by_id[i] for i in ids if i in by_id]
    out_dir = ROOT / settings.eval_output_dir / args.run_id / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.scripts.run_ab_eval import BASELINE_CFG, CANDIDATE_CFG
    cfg = CANDIDATE_CFG if args.arm == "candidate" else BASELINE_CFG
    cfg.apply_globals()
    from app.agent.tools import knowledge as knowledge_tool  # noqa: E402
    knowledge_tool.reset_retriever()

    import os
    from app.evaluation.evaluator import Evaluator
    from openai import OpenAI

    def run_one(case):
        cid = case.id
        ckpt = out_dir / f"{cid}.json"
        if ckpt.exists():
            prev = json.loads(ckpt.read_text(encoding="utf-8"))
            if not prev.get("error") and prev.get("passed") is not None:
                return cid, prev, True
            # 错误结果视为未完成：删除重跑（429 打穿的用例不计入完成）
            ckpt.unlink(missing_ok=True)
        sandbox = Sandbox(mode="single", resilient=True)
        evaluator = Evaluator(
            sandbox=sandbox,
            client=OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url),
            model=settings.model_name, use_judge=False,
            pass_threshold=settings.eval_pass_threshold,
        )
        try:
            report = evaluator.run_all([case])
            raw = report["cases"][0]
            result = {
                "case_id": cid,
                "passed": raw["passed"],
                "process": raw["process"],
                "result": raw["result"],
                "security": raw["security"],
                "error": raw["error"],
                "trace": {
                    "tool_calls": raw["trace"].get("tool_calls", []),
                    "tool_outcomes": raw["trace"].get("tool_outcomes", []),
                    "num_tool_calls": raw["trace"].get("num_tool_calls", 0),
                    "route": raw["trace"].get("route"),
                    "reply": raw["trace"].get("reply"),
                    "retrieved_sources": raw["trace"].get("retrieved_sources", []),
                    "error": raw["trace"].get("error"),
                    # understand: full tool results for offline judge faithfulness
                    "tool_outputs": raw["trace"].get("tool_outputs", []),
                },
            }
        except Exception as e:  # noqa: BLE001 —— 单条异常不应中断子集
            result = {"case_id": cid, "passed": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            sandbox.tmp_root and __import__("shutil").rmtree(sandbox.tmp_root, ignore_errors=True)
        with _LOCK:
            ckpt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return cid, result, False

    done = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, c) for c in cases]
        for fut in as_completed(futures):
            cid, result, was_ckpt = fut.result()
            if was_ckpt:
                skipped += 1
            else:
                done += 1
            if done % 10 == 0:
                log.info("进度: 新跑 %d / 跳过 %d / 共 %d", done, skipped, len(cases))

    passed = sum(1 for cid in ids if (out_dir / f"{cid}.json").exists()
                 and json.loads((out_dir / f"{cid}.json").read_text(encoding="utf-8"))["passed"])
    summary = {
        "run_id": args.run_id,
        "arm": "candidate-v2",
        "set": args.case_ids,
        "total": len(cases),
        "new_run": done,
        "resumed": skipped,
        "passed": passed,
        "failures": [cid for cid in ids
                     if (out_dir / f"{cid}.json").exists()
                     and not json.loads((out_dir / f"{cid}.json").read_text(encoding="utf-8"))["passed"]],
    }
    (ROOT / settings.eval_output_dir / args.run_id / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("子集完成: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
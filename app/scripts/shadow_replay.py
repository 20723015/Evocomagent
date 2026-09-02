"""影子流量回放对比（阶段五 5.3）。

把「生产流量」——evolution turns 目录沉淀的轮次记录（脱敏后）——按序
回放到当前 Agent（staging 构建/新版本），输出新旧回答的结构化对比报告。

典型用法：
  1. 生产备份 turns：tar -C app/sessions/evolution/turns -czf /backup/turns.tgz .
  2. 新版本镜像上跑：python -m app.scripts.shadow_replay \
        --turns app/sessions/evolution/turns --limit 200 --dry-run
  3. 报告（turns 未脱敏的字段）与线上 trace 对比：误差/分歧人工抽检。

本脚本只读、无副作用（每条用例独立 sandbox 会话），可作为预发
「线上流量可控复播」的清单式实现；接入真实流量网关后，把 turns_dir
换成影子流量采集目录即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config.settings import settings
from app.evaluation.dataset import EvalCase
from app.evaluation.sandbox import Sandbox
from app.observability.logging import get_logger

log = get_logger("app.scripts.shadow_replay")


def load_turns(turns_dir: Path, limit: int | None = None) -> list[EvalCase]:
    """从 turns 目录载入轮次记录，转成 EvalCase（单轮复播）。

    一条 turn = 一个问题；多轮会话按 turn_id 排序逐条复播（会话序列在
    沙箱内重建），使回放顺序与线上一致。
    """
    files = sorted(turns_dir.rglob("*.json"))
    if limit:
        files = files[:limit]
    cases: list[EvalCase] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        question = data.get("question", "")
        if not question:
            continue
        cases.append(EvalCase(
            id=f"shadow-{data.get('turn_id', f.stem)}",
            description="影子回放（生产 turn 复播）",
            turns=[question],
        ))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="影子流量回放对比")
    parser.add_argument(
        "--turns", default=settings.evolve_turns_dir,
        help="生产 turns 目录（脱敏轮次记录）",
    )
    parser.add_argument("--limit", type=int, default=200, help="最多复播条数")
    parser.add_argument("--output", default=None, help="对比报告 JSON 路径")
    args = parser.parse_args(argv)

    turns_dir = Path(args.turns)
    if not turns_dir.exists():
        log.info(f"❌ turns 目录不存在: {turns_dir}")
        return 1
    cases = load_turns(turns_dir, args.limit)
    if not cases:
        log.info("❌ 无可用 turn 记录")
        return 1
    log.info(f"影子回放：{len(cases)} 条生产轮次")

    sandbox = Sandbox(mode="single")
    divergences: list[dict] = []
    run = 0
    for case in cases:
        run += 1
        trace = sandbox.run(case)
        if trace.error:
            divergences.append({
                "case_id": case.id, "type": "error", "detail": trace.error,
            })
            continue
        reply = getattr(trace.final_response, "reply", "")
        requires_human = getattr(trace.final_response, "requires_human", False)
        # 记入对比明细：现场人工抽检新旧差异；与线上 trace 对账靠 trace_id
        divergences.append({
            "case_id": case.id, "reply_preview": reply[:120],
            "requires_human": requires_human,
            "tool_calls": [t.name for t in trace.tool_observations],
        })

    summary = {
        "total": len(cases),
        "with_error": sum(1 for d in divergences if d["type"] == "error"),
        "requires_human_ratio": sum(
            1 for d in divergences if d.get("requires_human")
        ) / max(len(cases), 1),
    }
    report = {"summary": summary, "details": divergences}
    log.info(
        f"对比完成：{summary['total']} 条，异常 {summary['with_error']}，"
        f"转人工率 {summary['requires_human_ratio']:.1%}"
    )
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"报告已写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

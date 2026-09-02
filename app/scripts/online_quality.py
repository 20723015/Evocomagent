"""在线质量信号采样（阶段五 5.5）。

每日从近 N 小时的生产 turn 采样 S 条，用 LLM-judge（复用第 9 期评估
的 judge prompts/评估维度）判「已解决/未解决」，结果进看板与审计文件。

用法（CronJob，随 evolution 定时器编排）：
python -m app.scripts.online_quality --sample 50 --hours 24
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.scripts.online_quality")


def _recent_turns(turns_dir: Path, hours: int) -> list[dict]:
    cutoff = datetime.now() - timedelta(hours=hours)
    out: list[dict] = []
    for f in turns_dir.rglob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data.get("ts", ""))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if ts >= cutoff:
            out.append(data)
    return out


def judge_turn(question: str, reply: str, judge_prompt: str, client, model: str) -> dict:
    """Lite 裁判：判断本轮是否已解决（0/1 + 一句话理由）。

    judge 调用失败即标记 unknown（在线信号不打断业务）。
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": judge_prompt},
                {
                    "role": "user",
                    "content": f"用户问题：{question}\n\n客服回答：{reply}\n\n"
                               "请以 JSON 输出：{\"resolved\": true/false, \"reason\": \"...\"}",
                },
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001 —— 采样失败不打断
        return {"resolved": None, "reason": f"judge 失败: {e}"}


JUDGE_PROMPT = (
    "你是客服质量抽样裁判。判断这轮对话是否真正解决了用户的问题："
    "明确给出可行答复=true；仅转人工无实质回答/答非所问/用户仍在追问=flase；"
    "无法判断返回 null。只输出 JSON。"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在线质量信号采样")
    parser.add_argument("--turns", default=settings.evolve_turns_dir)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--output", default="app/sessions/online_quality.json")
    args = parser.parse_args(argv)

    turns = _recent_turns(Path(args.turns), args.hours)
    if not turns:
        log.info("❌ 近 %s 小时无 turn 记录", args.hours)
        return 1
    sampled = random.sample(turns, min(args.sample, len(turns)))

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    verdicts = []
    for t in sampled:
        verdicts.append({
            "turn_id": t["turn_id"],
            "ts": t["ts"],
            "question_preview": t["question"][:80],
            **judge_turn(t["question"], t["reply"], JUDGE_PROMPT, client, settings.model_name),
        })

    resolved = [v for v in verdicts if v["resolved"] is True]
    unknown = [v for v in verdicts if v["resolved"] is None]
    summary = {
        "sampled_at": datetime.now().isoformat(timespec="seconds"),
        "window_hours": args.hours,
        "sampled": len(verdicts),
        "resolved": len(resolved),
        "unknown": len(unknown),
        "resolve_rate": len(resolved) / len(verdicts) if verdicts else 0.0,
    }
    log.info(
        f"采样完成：{summary['sampled']} 条，已解决率 {summary['resolve_rate']:.1%}"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "verdicts": verdicts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"已写入 {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

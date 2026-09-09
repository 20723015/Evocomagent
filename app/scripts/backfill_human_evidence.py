"""backfill_human_evidence.py：历史候选证据链快照回填（010 上线一次性）。

迁移 010 将既有候选的 evidence_state 置为 legacy_evidence_missing（禁止批准）。
本脚本遍历这些候选：会话正本仍在 → 从 transcript_json 回填来源/证据双快照并
置 ok；会话已清理 → 保持 legacy_evidence_missing（审核台可见原因，禁止批准）。

- 只有引用了至少一条人工坐席消息的候选才回填 ok（证据语义按新门禁）；
- 幂等：重复运行对已 ok 候选无操作；会话缺失候选保持 legacy。

用法：
    python -m app.scripts.backfill_human_evidence [--dry-run] [--limit N] [-v]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("app.scripts.backfill_human_evidence")


def backfill(store, *, dry_run: bool = False, limit: int = 0) -> dict:
    """回填全部 legacy 候选；返回 {"backfilled", "missing_evidence", "conversation_missing", "scanned"}。

    分页扫描（list_candidates 单页上限 200），只处理 evidence_state != ok 的候选。
    """
    from app.evolution.human_evaluator import build_evidence_snapshot
    from app.evolution.human_store import EVIDENCE_LEGACY_MISSING

    legacy: list[dict] = []
    page, page_size, offset = 0, 200, 0
    while True:
        rows, _total = store.list_candidates(limit=page_size, offset=offset)
        legacy.extend(r for r in rows if r.get("evidence_state") == EVIDENCE_LEGACY_MISSING)
        if len(rows) < page_size or (limit and len(legacy) >= int(limit)):
            break
        offset += page_size
        page += 1
        if page > 1000:  # 防御性上限（10 万候选）
            break
    if limit:
        legacy = legacy[: int(limit)]
    stats = {
        "backfilled": 0,
        "missing_evidence": 0,
        "conversation_missing": 0,
        "scanned": len(legacy),
    }
    for row in legacy:
        cid = int(row["id"])
        conv = (
            store.get_conversation(int(row["conversation_id"] or 0))
            if row.get("conversation_id")
            else None
        )
        if conv is None:
            stats["conversation_missing"] += 1
            log.info("skip conversation_missing candidate=%s", cid)
            continue
        try:
            messages = json.loads(conv["transcript_json"])
        except (TypeError, ValueError):
            messages = []
        cited = [str(e) for e in (json.loads(row.get("evidence_message_ids") or "[]"))]
        agent_ids = {
            str(m.get("message_id", ""))
            for m in messages
            if m.get("actor_type") == "human_agent" and m.get("message_id")
        }
        agent_cited = [e for e in cited if e in agent_ids]
        if not agent_cited:
            # 旧证据语义允许引用任意消息；无坐席消息引用的候选无法按新语义回填
            stats["missing_evidence"] += 1
            log.info("skip no_agent_evidence candidate=%s", cid)
            continue
        ended_at = conv.get("ended_at")
        source_snapshot = {
            "source": conv.get("source", ""),
            "external_conversation_id": conv.get("external_conversation_id", ""),
            "source_version": int(conv.get("source_version") or 0),
            "agent_id": conv.get("agent_id", ""),
            "ended_at": (
                ended_at.isoformat(sep=" ", timespec="seconds")
                if getattr(ended_at, "isoformat", None)
                else str(ended_at or "")
            ),
        }
        evidence_snapshot = build_evidence_snapshot(messages, agent_cited)
        if dry_run:
            stats["backfilled"] += 1
            log.info("dry-run backfill candidate=%s evidence=%s", cid, len(evidence_snapshot))
            continue
        if store.backfill_candidate_evidence(
            cid,
            source_snapshot=source_snapshot,
            evidence_snapshot=evidence_snapshot,
        ):
            stats["backfilled"] += 1
            log.info("backfilled candidate=%s evidence=%s", cid, len(evidence_snapshot))
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="历史候选证据链快照回填（010）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写库")
    parser.add_argument("--limit", type=int, default=0, help="本次最多回填条数（0=全部）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from app.config.settings import settings
    from app.evolution.human_store import HumanKnowledgeStore
    from app.stores.sql.engine import get_engine

    if not settings.human_qa_evolution_enabled:
        log.error("❌ HUMAN_QA_EVOLUTION_ENABLED=false：回填被拒绝")
        return 2
    engine = get_engine()
    if engine is None:
        log.error("❌ 证据链回填需要 MySQL（DB_URL 未配置）")
        return 2
    store = HumanKnowledgeStore(engine)
    stats = backfill(store, dry_run=args.dry_run, limit=args.limit)
    log.info(
        "回填完成: scanned=%s backfilled=%s missing_evidence=%s conversation_missing=%s",
        stats["scanned"],
        stats["backfilled"],
        stats["missing_evidence"],
        stats["conversation_missing"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

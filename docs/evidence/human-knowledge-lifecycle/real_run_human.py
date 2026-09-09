# ruff: noqa
"""人工客服知识链路「真实环境」验收驱动：MySQL + ES + 真实 LLM/embedding。

前置（外部完成）：
  - docker compose 起 mysql/elasticsearch；迁移 001→010 已执行；
  - build_kb_index 已建基线 ES 索引且指针经 Redis 共享一致；
  - 环境变量：DB_URL / HUMAN_QA_EVOLUTION_ENABLED=true / SELF_EVOLVE_ENABLED=true。

流程（与生产同一条代码路径）：
  会话批量接入（MySQL 正本）→ run_evolution 评审装配（真实 LLM 抽取 +
  SemanticDedupService 真实 embedding 双侧检索）→ 审核批准（审批快照）→
  HumanBatchPublisher 发布（真实 ES 候选索引 + 探针 + alias 原子切换 + CAS 结算）
  → 新文档 Top-5 线上检索命中验证。

运行：DB_URL=... HUMAN_QA_EVOLUTION_ENABLED=true SELF_EVOLVE_ENABLED=true \
      .venv/Scripts/python.exe tmp/real_run_human.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings
from app.evolution.human_store import HumanKnowledgeStore
from app.stores.sql.engine import get_engine


def say(line=""):
    print(line, flush=True)


def main() -> int:
    engine = get_engine()
    assert engine is not None, "需要 DB_URL 指向 MySQL"
    say(f"MySQL        : {engine.url.render_as_string(hide_password=True)}")
    say(f"ES 后端      : {settings.rag_backend}  检索配置 hybrid={settings.rag_hybrid} rerank={settings.rag_rerank}")

    import app.scripts.run_evolution as evo

    svc = evo._build_human_services()
    store: HumanKnowledgeStore = svc["store"]

    # ---------- 1. 会话批量接入 ----------
    say("\n" + "=" * 68)
    say("阶段 1｜会话批量接入（真实 MySQL 正本）")
    say("=" * 68)
    ended = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    conv = dict(
        source="acceptance",
        external_conversation_id="measure-install-002",
        source_version=1,
        agent_id="agent-accept",
        started_at=ended - timedelta(minutes=8),
        ended_at=ended,
        messages=[
            {
                "message_id": "c0",
                "actor_type": "customer",
                "content": "我在你们平台订的全屋定制衣柜，什么时候来量尺？量完多久能装？",
                "sent_at": "",
            },
            {
                "message_id": "a0",
                "actor_type": "human_agent",
                "content": (
                    "全屋定制下单后 48 小时内设计师会电话预约上门量尺，量尺后 3 个工作日"
                    "内出方案与报价；确认方案进入工厂生产 15-20 天，成品入库后客服再次"
                    "电话预约上门安装，安装当天自带封板收边，旧家具可付费拆移清运。"
                ),
                "sent_at": "",
            },
            {
                "message_id": "c1",
                "actor_type": "customer",
                "content": "量尺要收费吗？要是量完不做了呢？",
                "sent_at": "",
            },
            {
                "message_id": "a1",
                "actor_type": "human_agent",
                "content": (
                    "上门量尺与方案设计免费，确认生产后定金直接抵扣货款；"
                    "仅量尺不下单的 30 天内限免费一次，超出的城市按 99 元/次收取上门费。"
                ),
                "sent_at": "",
            },
        ],
    )
    (record, outcome) = store.ingest_conversation(**conv)
    say(f"  会话 #{record['id']} {outcome}（messages={record['message_count']}）")

    # ---------- 2. 真实 LLM 评审 ----------
    say("\n" + "=" * 68)
    say("阶段 2｜评审：真实 LLM 抽取 + 真实 embedding 双侧语义去重")
    say("=" * 68)
    rc = evo._cmd_human_eval(svc, max_jobs=5)
    say(f"  --human-eval 退出码: {rc}")
    rows, total = store.list_candidates()
    say(f"  候选总数: {total}")
    for r in rows:
        dedup = json.loads(r["dedup_snapshot_json"] or "{}")
        q_hit = (dedup.get("question") or {}) or {}
        a_hit = (dedup.get("answer") or {}) or {}
        say(
            f"  [#{r['id']}] 状态={r['status']} 分类={r['classification'] or '—'}"
            f" 价值={r['value_score']} 新颖={r['novelty_score']} 综合={r['composite_score']}"
            f"\n      Q={r['question']}"
            f"\n      A={r['answer'][:80]}…"
            f"\n      去重: q命中={q_hit.get('path') or '—'}({q_hit.get('score')})"
            f" a命中={a_hit.get('path') or '—'}({a_hit.get('score')})"
            f" 评审代={dedup.get('generation')}"
            f"\n      证据={r['evidence_message_ids']} evidence_state={r['evidence_state']}"
        )
    pending = [r for r in rows if r["status"] == "pending_review"]
    if not pending:
        say("  ✗ 没有进入待审核的候选，中止（如实记录真实评审结论）")
        return 1

    # ---------- 3. 审核批准（审批快照） ----------
    say("\n" + "=" * 68)
    say("阶段 3｜审核批准（审批快照冻结）")
    say("=" * 68)
    # 只批准 1 条（id 最小的 pending = 主流程 QA），单文档发布与 eval-v2 协议对齐
    target = min(pending, key=lambda r: int(r["id"]))
    batch, items = store.create_publish_batch(
        [{"candidate_id": int(target["id"]), "revision": int(target["revision"])}],
        requested_by="acceptance-runner",
    )
    say(f"  批次 #{batch['id']}（{batch['item_count']} 条）")
    for it in items:
        say(f"  快照 #{it['candidate_id']} rev={it['candidate_revision']} digest={it['approval_digest'][:12]}…")

    # ---------- 4. 发布 Worker（真实 ES） ----------
    say("\n" + "=" * 68)
    say("阶段 4｜发布：真实 ES 候选索引构建 + 探针 + alias 原子切换 + CAS 结算")
    say("=" * 68)
    from app.evolution.human_publish import HumanBatchPublisher
    from app.evolution.lock import LockGuard  # noqa: F401（svc 已装配）

    pub = HumanBatchPublisher(
        store,
        index_service=svc["index_service"],
        generation_store=svc["generation_store"],
        publisher=svc["publisher"],
        journal=svc["journal"],
        lock=svc["lock"],
        control_store=svc["control"],
        kb_dir=svc["kb_dir"],
        worker_id="acceptance-pub",
        dedup_service=svc["dedup_service"],
        fence=svc.get("fence"),
        lifecycle=svc.get("lifecycle"),
    )
    before = svc["generation_store"].active(settings.rag_backend.lower())
    say(f"  活动代（前）: {before.target if before else '（无）'}")
    ok = pub.process_once()
    batch_after = store.get_batch(batch["id"])
    say(f"  process_once={ok}  批次状态={batch_after['status']}  generation={batch_after['generation_id']}")
    for it in batch_after["items"]:
        say(f"    item #{it['candidate_id']} {it['status']} {it['filename']} {it['detail']}")
    after = svc["generation_store"].active(settings.rag_backend.lower())
    say(f"  活动代（后）: {after.target if after else '（无）'}")
    for r in pending:
        fresh = store.get_candidate(int(r["id"]))
        say(
            f"  候选 #{fresh['id']}: {fresh['status']} lifecycle_rev={fresh['lifecycle_revision']}"
            f" published_at={fresh['published_at']} generation={fresh['published_generation']}"
        )
    if batch_after["status"] != "completed":
        say("  ✗ 批次未完成，中止")
        return 1

    # ---------- 5. 新文档线上 Top-5 命中 ----------
    say("\n" + "=" * 68)
    say("阶段 5｜线上检索验证：新知识 Top-5 命中")
    say("=" * 68)
    from app.agent.tools import knowledge as knowledge_tool
    from app.agent.rag.retriever_factory import final_search

    retriever = knowledge_tool._get_retriever()
    queries = [
        "全屋定制什么时候上门量尺？多久能安装？",
        "上门量尺收费吗？量完不下单呢？",
    ]
    published_files = {it["filename"] for it in batch_after["items"] if it["status"] == "published"}
    summary = {"batch": batch_after["id"], "generation": batch_after["generation_id"], "queries": []}
    for q in queries:
        outcome = final_search(retriever, q, top_k=5, min_score=settings.rag_min_relevance_score)
        say(f"  查询: {q}")
        rank_hit = None
        for i, h in enumerate(outcome.hits, 1):
            sp = h.chunk.source_path
            mark = "  ← 新发布文档" if Path(sp).name in published_files else ""
            say(f"    #{i} {sp}  score={h.score:.4f}{mark}")
            if rank_hit is None and Path(sp).name in published_files:
                rank_hit = i
        summary["queries"].append({"q": q, "rank": rank_hit, "top_score": outcome.raw_top_score})
        say(f"  新文档排名: {rank_hit}")

    Path(tmp_json := (ROOT / "tmp" / "real_run_result.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    say(f"\n结果已写入: {tmp_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

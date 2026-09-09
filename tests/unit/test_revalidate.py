"""revalidate：存量自进化文档重接地（通过刷 last_validated / 不通过隔离 pending + 重建）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.config.settings import settings
from app.evolution.revalidate import (
    parse_evolved_doc,
    refresh_last_validated,
    revalidate,
)


def _write_evolved(kb, filename="20260801-abc-问答.md", question="退款多久到账？",
                   answer="一般 3 个工作日内原路退回，请留意到账通知。",
                   effective_date="2026-08-01", **extra):
    (kb / "evolved").mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "provenance: turn-abc123",
        "owner: system",
        f"effective_date: {effective_date}",
        "quality_score: 0.9",
        "last_validated: 2026-08-01",
        "grounded_on: 退货政策.md",
        "---",
    ]
    for k, v in extra.items():
        lines.insert(5, f"{k}: {v}")
    (kb / "evolved" / filename).write_text(
        "\n".join(lines) + f"\n# 自进化知识\n\n## {question}\n\n{answer}\n",
        encoding="utf-8",
    )


class StubGrounding:
    def __init__(self, grounded=True):
        self.calls = 0
        self.grounded = grounded

    def judge(self, answer, sources):
        self.calls += 1
        return {"grounded": self.grounded, "unsupported": [], "reason": "ok"}


# ============================================================
# parse_evolved_doc
# ============================================================
def test_parse_evolved_doc_template(tmp_path):
    kb = tmp_path / "kb"
    _write_evolved(kb)
    parsed = parse_evolved_doc(kb / "evolved" / "20260801-abc-问答.md")
    assert parsed is not None
    assert parsed["question"] == "退款多久到账？"
    assert parsed["answer"].startswith("一般 3 个工作日内原路退回")
    assert parsed["meta"]["provenance"] == "turn-abc123"
    assert parsed["meta"]["effective_date"] == "2026-08-01"


def test_refresh_last_validated_preserves_body(tmp_path):
    kb = tmp_path / "kb"
    _write_evolved(kb)
    target = kb / "evolved" / "20260801-abc-问答.md"
    refresh_last_validated(target, today=date(2026, 9, 1))
    parsed = parse_evolved_doc(target)
    assert parsed["meta"]["last_validated"] == "2026-09-01"
    assert parsed["meta"]["quality_score"] == "0.9"  # 其余字段保持
    assert parsed["question"] == "退款多久到账？"
    assert "# 自进化知识" not in parsed["answer"]  # 正文本体不受影响


def test_refresh_last_validated_roundtrips_quoted_multiline_frontmatter(tmp_path):
    """刷新不得把特殊字符/换行元数据变成多行或 YAML 注入。"""
    from app.agent.rag.loader import serialize_frontmatter

    target = tmp_path / "special.md"
    metadata = {
        "provenance": 'turn: "人工"\n第二行 # not a field',
        "owner": "ops: qa # reviewer",
        "grounded_on": "policy.md, faq.md",
        "last_validated": "2026-08-01",
    }
    body = "\n# 自进化知识\n\n## 特殊问题？\n\n回答含冒号: 和 # 号。\n"
    target.write_text(serialize_frontmatter(metadata) + body, encoding="utf-8")

    refresh_last_validated(target, today=date(2026, 9, 1))
    parsed = parse_evolved_doc(target)

    assert parsed is not None
    assert parsed["meta"]["provenance"] == metadata["provenance"]
    assert parsed["meta"]["owner"] == metadata["owner"]
    assert parsed["meta"]["grounded_on"] == metadata["grounded_on"]
    assert parsed["meta"]["last_validated"] == "2026-09-01"
    assert parsed["answer"] == "回答含冒号: 和 # 号。"


# ============================================================
# revalidate：通过 → 刷 last_validated、索引不变
# ============================================================
def test_revalidate_pass_refreshes_last_validated_no_rebuild(tmp_path):
    from test_pipeline import make_services

    svc = make_services(tmp_path)
    _write_evolved(svc["kb_dir"])
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)

    judge = StubGrounding(grounded=True)
    from conftest import FakeBackend, FakeEmbedder, FakeRetriever

    retriever = FakeRetriever(FakeEmbedder(), FakeBackend())
    result = revalidate(
        kb_dir=svc["kb_dir"],
        trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"],
        publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=retriever,
        grounding_judge=judge,
    )
    assert result["checked"] == 1
    assert result["passed"] == 1
    assert result["failed"] == 0
    assert result["rebuilt"] is False
    # last_validated 已刷新（当天）
    target = svc["kb_dir"] / "evolved" / "20260801-abc-问答.md"
    meta = parse_evolved_doc(target)["meta"]
    assert meta["last_validated"] == (
        datetime.now(timezone.utc).astimezone().date().isoformat()
    )
    # 索引未重建：active generation 不变、文档仍在 kb
    assert svc["pipeline"]._generation_store.active("numpy").generation_id == info.generation_id
    assert target.exists()


# ============================================================
# revalidate：不通过 → 隔离 trash + pending + 重建后检索不到
# ============================================================
def test_revalidate_fail_quarantines_and_rebuilds(tmp_path):
    from conftest import FakeEmbedder
    from test_pipeline import make_services

    svc = make_services(tmp_path)
    _write_evolved(svc["kb_dir"])
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)
    # ledger 记录该沉淀（unpublish_mark 反查需要）
    svc["ledger"].mark_published("cid-evolved", "20260801-abc-问答.md")

    from conftest import FakeBackend, FakeRetriever

    retriever = FakeRetriever(FakeEmbedder(), FakeBackend())
    result = revalidate(
        kb_dir=svc["kb_dir"],
        trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"],
        publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=retriever,
        grounding_judge=StubGrounding(grounded=False),
    )
    assert result["failed"] == 1
    assert result["rebuilt"] is True
    # 文档进 trash、pending 出现
    target = svc["kb_dir"] / "evolved" / "20260801-abc-问答.md"
    assert not target.exists()
    assert (svc["state_dir"] / "trash" / "20260801-abc-问答.md").exists()
    pending = svc["ledger"].pending_entry("cid-evolved")
    assert pending is not None
    assert pending["reason"] == "revalidation_failed"
    assert svc["ledger"].trash_entry("20260801-abc-问答.md")["candidate_id"] == "cid-evolved"

    # 重建后（新 generation 检索器）检索不到该文档
    from app.agent.rag.backends import create_backend
    from app.agent.rag.retriever import KnowledgeRetriever

    active = svc["pipeline"]._generation_store.active("numpy")
    impl = create_backend("numpy", index_path=__import__("pathlib").Path(active.target))
    impl.load()
    new_retriever = KnowledgeRetriever(embedder=FakeEmbedder(), backend=impl)
    hits = new_retriever.search("退款多久到账？", top_k=5)
    assert not any(h.chunk.source_path == "evolved/20260801-abc-问答.md" for h in hits)


# ============================================================
# 上限与顺序：缺 effective_date 最旧优先
# ============================================================
def test_revalidate_cap_and_missing_date_first(tmp_path):
    from test_pipeline import make_services

    svc = make_services(tmp_path)
    for i in range(6):
        _write_evolved(svc["kb_dir"], filename=f"d{i}.md", question=f"问题{i}",
                       answer=f"回答内容{i}。" * 5,
                       effective_date="" if i == 0 else f"2026-08-0{i}")
    judge = StubGrounding(grounded=True)
    from conftest import FakeBackend, FakeEmbedder, FakeRetriever

    result = revalidate(
        kb_dir=svc["kb_dir"],
        trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"],
        publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=FakeRetriever(FakeEmbedder(), FakeBackend()),
        grounding_judge=judge,
        max_docs=3,
    )
    assert result["checked"] == 3  # 上限截断
    assert result["scanned"] == 3
    assert result["remaining"] == 3
    assert result["has_more"] is True
    assert judge.calls == 3

    # 下一批显式排除已处理文件，不能再次命中同一组最老文档。
    second = revalidate(
        kb_dir=svc["kb_dir"],
        trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"],
        publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=FakeRetriever(FakeEmbedder(), FakeBackend()),
        grounding_judge=judge,
        max_docs=3,
        exclude_docs=set(result["processed_docs"]),
    )
    assert second["checked"] == 3
    assert second["remaining"] == 0
    assert second["has_more"] is False
    assert set(second["processed_docs"]).isdisjoint(result["processed_docs"])
    assert judge.calls == 6


# ============================================================
# pipeline 集成：generation 未变不触发；外部切换后自动触发
# ============================================================
def _counted_services(tmp_path):
    from test_pipeline import make_services

    judge = StubGrounding(grounded=True)
    svc = make_services(tmp_path, grounding_judge=judge)
    _write_evolved(svc["kb_dir"])
    # 知识库需先有活动索引（上传/手工 CLI 之后的首个里程碑）
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)
    return svc, judge


def test_pipeline_trigger_only_on_generation_change(tmp_path, reset_settings):
    settings.self_evolve_enabled = True
    svc, judge = _counted_services(tmp_path)
    svc["pipeline"].run()  # 首次：无记录 → 触发重接地（judge 调用 1 次）
    assert judge.calls == 1
    assert svc["pipeline"]._read_last_human_generation() != ""

    svc["pipeline"].run()  # second: 活动代 == 记录 → 不触发
    assert judge.calls == 1


def test_pipeline_trigger_after_external_generation_change(tmp_path, reset_settings):
    settings.self_evolve_enabled = True
    svc, judge = _counted_services(tmp_path)
    svc["pipeline"].run()
    assert judge.calls == 1

    # 外部切换 generation（模拟上传/下架/手工 CLI 动过 KB）
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)

    svc["pipeline"].run()
    assert judge.calls == 2  # 自动触发重接地


def test_pipeline_revalidation_paginates_before_marking_generation(
    tmp_path, reset_settings, monkeypatch,
):
    """超过单批上限时续跑不同文档，且只在最后一批后推进 generation 标记。"""
    from test_pipeline import make_services

    settings.self_evolve_enabled = True
    monkeypatch.setattr(settings, "evolve_max_reground_per_run", 2, raising=False)
    # 每批都隔离文档并自发切换 generation，后续仍必须沿用原分页进度。
    judge = StubGrounding(grounded=False)
    svc = make_services(tmp_path, grounding_judge=judge)
    for i in range(5):
        _write_evolved(
            svc["kb_dir"], filename=f"page-{i}.md", question=f"分页问题{i}",
            answer=f"分页回答内容{i}。" * 5, effective_date=f"2026-08-0{i + 1}",
        )
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)

    first = svc["pipeline"].run()
    assert first.revalidated_checked == 2
    assert first.revalidated_remaining == 3
    assert svc["pipeline"]._read_last_human_generation() == ""
    assert (svc["state_dir"] / "revalidate_progress.json").exists()

    second = svc["pipeline"].run()
    assert second.revalidated_checked == 2
    assert second.revalidated_remaining == 1
    assert svc["pipeline"]._read_last_human_generation() == ""

    third = svc["pipeline"].run()
    assert third.revalidated_checked == 1
    assert third.revalidated_remaining == 0
    assert judge.calls == 5
    active = svc["pipeline"]._generation_store.active("numpy")
    assert active.generation_id != info.generation_id
    assert svc["pipeline"]._read_last_human_generation() == active.generation_id
    assert not (svc["state_dir"] / "revalidate_progress.json").exists()


def test_pipeline_revalidation_error_does_not_mark_generation(tmp_path, reset_settings):
    """暂时性 Judge/检索异常不能把尚未核对的 generation 标成完成。"""
    from test_pipeline import make_services

    class BrokenGrounding:
        def judge(self, answer, sources):
            raise RuntimeError("judge unavailable")

    settings.self_evolve_enabled = True
    svc = make_services(tmp_path, grounding_judge=BrokenGrounding())
    _write_evolved(svc["kb_dir"])
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)

    with pytest.raises(RuntimeError, match="judge unavailable"):
        svc["pipeline"].run()
    assert svc["pipeline"]._read_last_human_generation() == ""


def test_revalidate_disabled_no_trigger(tmp_path, reset_settings, monkeypatch):
    from test_pipeline import make_services

    settings.self_evolve_enabled = True
    monkeypatch.setattr(settings, "evolve_revalidate_enabled", False, raising=False)
    judge = StubGrounding(grounded=True)
    svc = make_services(tmp_path, grounding_judge=judge)
    _write_evolved(svc["kb_dir"])
    svc["pipeline"].run()
    assert judge.calls == 0


# ============================================================
# 隔离事务（M1）：journal 写入/清除 + 前进式恢复
# ============================================================
def _activated_services(tmp_path, **overrides):
    from test_pipeline import make_services

    svc = make_services(tmp_path, **overrides)
    _write_evolved(svc["kb_dir"])
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)
    svc["ledger"].mark_published("cid-evolved", "20260801-abc-问答.md")
    return svc, info


def _write_revalidate_journal(svc, generation_id="g-crashed"):
    svc["journal"].write({
        "phase": "revalidate",
        "trashed_docs": ["20260801-abc-问答.md"],
        "backend": "numpy",
        "index": {
            "generation_id": generation_id,
            "target": svc["index_service"].target_for("numpy", generation_id),
        },
    })


def _human_lifecycle(svc, filename="20260801-abc-问答.md"):
    """构造一条已发布人工候选及其生命周期协调器。"""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from app.evolution.human_store import HumanKnowledgeStore
    from app.evolution.lifecycle import KnowledgeLifecycleCoordinator
    from app.stores.sql.schema import human_knowledge_candidates, metadata

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        inserted = conn.execute(
            human_knowledge_candidates.insert().values(
                conversation_id=1,
                status="published",
                question="退款多久到账？",
                answer="一般 3 个工作日内原路退回，请留意到账通知。",
                evidence_message_ids="[]",
                classification="new",
                published_filename=filename,
            )
        )
    candidate_id = int(inserted.inserted_primary_key[0])
    store = HumanKnowledgeStore(engine)
    lifecycle = KnowledgeLifecycleCoordinator(
        store,
        svc["ledger"],
        kb_dir=svc["kb_dir"],
    )
    return store, lifecycle, candidate_id


def _candidate_jobs(store, candidate_id):
    from sqlalchemy import select

    from app.stores.sql.schema import human_evaluation_jobs

    with store._engine.connect() as conn:
        return conn.execute(
            select(human_evaluation_jobs).where(
                human_evaluation_jobs.c.candidate_id == candidate_id
            )
        ).mappings().all()


def test_revalidate_human_document_routes_to_mysql_not_ledger(tmp_path):
    """正常隔离：human 正本回 MySQL 待重审，不能污染旧 Ledger。"""
    from conftest import FakeBackend, FakeEmbedder, FakeRetriever
    from test_pipeline import make_services

    svc = make_services(tmp_path)
    store, lifecycle, candidate_id = _human_lifecycle(svc)
    _write_evolved(
        svc["kb_dir"],
        candidate_id=f"human-{candidate_id}",
        source_kind="human_conversation",
    )

    result = revalidate(
        kb_dir=svc["kb_dir"],
        trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"],
        publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=FakeRetriever(FakeEmbedder(), FakeBackend()),
        grounding_judge=StubGrounding(grounded=False),
        journal=svc["journal"],
        lifecycle=lifecycle,
    )

    assert result["failed"] == 1
    assert store.get_candidate(candidate_id)["status"] == "pending_review"
    assert [job["status"] for job in _candidate_jobs(store, candidate_id)] == ["queued"]
    assert svc["ledger"].list_pending(aging_days=999) == []
    assert svc["ledger"].published() == {}


def test_pipeline_recovery_human_document_routes_to_mysql_not_ledger(tmp_path):
    """移入 trash 后崩溃：pipeline 恢复仍凭快照路由 MySQL，不误写 Ledger。"""
    import os

    from test_pipeline import make_services

    filename = "20260801-abc-问答.md"
    svc = make_services(tmp_path)
    store, lifecycle, candidate_id = _human_lifecycle(svc, filename)
    _write_evolved(
        svc["kb_dir"],
        filename=filename,
        candidate_id=f"human-{candidate_id}",
        source_kind="human_conversation",
    )
    info = svc["index_service"].build("numpy")
    svc["index_service"].activate("numpy", info)
    trash = svc["state_dir"] / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    os.replace(svc["kb_dir"] / "evolved" / filename, trash / filename)
    _write_revalidate_journal(svc)
    svc["pipeline"]._lifecycle = lifecycle

    result = svc["pipeline"]._recover_revalidate(svc["journal"].read())

    assert result["success"] is True
    assert result["retired"] == [(filename, "mysql")]
    assert store.get_candidate(candidate_id)["status"] == "pending_review"
    assert [job["status"] for job in _candidate_jobs(store, candidate_id)] == ["queued"]
    assert svc["ledger"].list_pending(aging_days=999) == []
    assert svc["ledger"].published() == {}
    assert svc["journal"].read() is None


def test_revalidate_journal_cleared_after_quarantine(tmp_path):
    """隔离完成 → journal 清除（事务收尾）。"""
    from conftest import FakeBackend, FakeEmbedder, FakeRetriever

    svc, _ = _activated_services(tmp_path)
    result = revalidate(
        kb_dir=svc["kb_dir"], trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=FakeRetriever(FakeEmbedder(), FakeBackend()),
        grounding_judge=StubGrounding(grounded=False),
        journal=svc["journal"],
    )
    assert result["failed"] == 1
    assert svc["journal"].read() is None


def test_revalidate_no_failure_never_writes_journal(tmp_path):
    """全部通过 → 从头到尾不写 journal（无隔离即无事务）。"""
    from conftest import FakeBackend, FakeEmbedder, FakeRetriever

    svc, _ = _activated_services(tmp_path)
    result = revalidate(
        kb_dir=svc["kb_dir"], trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=FakeRetriever(FakeEmbedder(), FakeBackend()),
        grounding_judge=StubGrounding(grounded=True),
        journal=svc["journal"],
    )
    assert result["failed"] == 0
    assert not (svc["state_dir"] / "journal.json").exists()


def test_recover_revalidate_completes_quarantine(tmp_path):
    """崩溃点=journal 已写、文件未移动 → 恢复完成隔离（移 trash + pending + 清账 + 重建）。"""
    from app.evolution.revalidate import recover_revalidate

    svc, info = _activated_services(tmp_path)
    _write_revalidate_journal(svc)
    result = recover_revalidate(
        svc["journal"].read(),
        kb_dir=svc["kb_dir"], trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        generation_store=svc["pipeline"]._generation_store,
    )
    assert result["already_done"] is False
    assert result["rebuilt"] is True
    assert not (svc["kb_dir"] / "evolved" / "20260801-abc-问答.md").exists()
    assert (svc["state_dir"] / "trash" / "20260801-abc-问答.md").exists()
    assert svc["ledger"].pending_entry("cid-evolved")["reason"] == "revalidation_failed"
    assert "cid-evolved" not in svc["ledger"].published()
    active = svc["pipeline"]._generation_store.active("numpy")
    assert active.generation_id not in (info.generation_id, "g-crashed")  # 重建切了新代


def test_recover_revalidate_after_partial_move(tmp_path):
    """崩溃点=文件已移 trash、账目未清 → 恢复补账 + 重建（不重复移动）。"""
    import os

    from app.evolution.revalidate import recover_revalidate

    svc, _ = _activated_services(tmp_path)
    trash = svc["state_dir"] / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    os.replace(svc["kb_dir"] / "evolved" / "20260801-abc-问答.md",
               trash / "20260801-abc-问答.md")
    _write_revalidate_journal(svc)
    result = recover_revalidate(
        svc["journal"].read(),
        kb_dir=svc["kb_dir"], trash_dir=trash,
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        generation_store=svc["pipeline"]._generation_store,
    )
    assert result["rebuilt"] is True
    assert svc["ledger"].pending_entry("cid-evolved")["reason"] == "revalidation_failed"
    assert "cid-evolved" not in svc["ledger"].published()
    assert len(list(trash.glob("*.md"))) == 1  # 未重复移动


def test_recover_revalidate_after_ledger_cleanup_reuses_original_cid(tmp_path):
    """崩溃在 ledger 清账后、activate 前：恢复不能派生第二个 candidate_id。"""
    from app.evolution.models import CandidateQA
    from app.evolution.revalidate import recover_revalidate

    svc, _ = _activated_services(tmp_path)
    filename = "20260801-abc-问答.md"
    trash = svc["state_dir"] / "trash"
    parsed = parse_evolved_doc(svc["kb_dir"] / "evolved" / filename)
    assert svc["publisher"].unpublish(filename, trash) is not None
    svc["ledger"].add_pending(
        CandidateQA(
            candidate_id="cid-evolved", turn_id="turn-abc123",
            question=parsed["question"], answer=parsed["answer"],
        ),
        reason="revalidation_failed",
    )
    svc["ledger"].batch_cleanup_published([(filename, "cid-evolved")])
    _write_revalidate_journal(svc)

    result = recover_revalidate(
        svc["journal"].read(),
        kb_dir=svc["kb_dir"], trash_dir=trash,
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        generation_store=svc["pipeline"]._generation_store,
    )

    assert result["retired"] == [(filename, "cid-evolved")]
    assert [cid for cid, _ in svc["ledger"].list_pending(aging_days=999)] == ["cid-evolved"]
    assert svc["ledger"].trash_entry(filename)["candidate_id"] == "cid-evolved"


def test_recover_revalidate_already_activated_noop(tmp_path):
    """崩溃点=activate 已生效、journal 未清 → 恢复无事可做（不重建、账目已完整）。"""
    from conftest import FakeBackend, FakeEmbedder, FakeRetriever

    from app.evolution.revalidate import recover_revalidate

    svc, _ = _activated_services(tmp_path)
    result = revalidate(
        kb_dir=svc["kb_dir"], trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        retriever=FakeRetriever(FakeEmbedder(), FakeBackend()),
        grounding_judge=StubGrounding(grounded=False),
        journal=svc["journal"],
    )
    assert result["rebuilt"] is True
    active_gen = svc["pipeline"]._generation_store.active("numpy").generation_id

    _write_revalidate_journal(svc, generation_id=active_gen)
    result = recover_revalidate(
        svc["journal"].read(),
        kb_dir=svc["kb_dir"], trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        generation_store=svc["pipeline"]._generation_store,
    )
    assert result["already_done"] is True
    # 未重建：活动代保持、pending 账目保持
    assert svc["pipeline"]._generation_store.active("numpy").generation_id == active_gen
    assert svc["ledger"].pending_entry("cid-evolved") is not None


def test_recover_revalidate_pointer_write_failure_keeps_journal(tmp_path, monkeypatch):
    """Alias 已切但 pointer 补写失败 → 不报告成功，journal 由调用方保留。"""
    from app.evolution.revalidate import recover_revalidate

    svc, info = _activated_services(tmp_path)
    candidate = svc["index_service"].build("numpy")
    # 恢复所需的候选指针仍未切；用 ES 语义模拟 alias 已切。
    entry = {
        "phase": "revalidate",
        "stage": "ALIAS_ACTIVATED",
        "trashed_docs": [],
        "backend": "es",
        "index": candidate.to_dict(),
        "previous_target": info.target,
    }
    svc["journal"].write(entry)
    monkeypatch.setattr(
        svc["index_service"], "reconcile",
        lambda backend: {"alias_target": candidate.target},
    )
    monkeypatch.setattr(
        svc["index_service"], "activate_pointer",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pointer down")),
    )

    result = recover_revalidate(
        entry,
        kb_dir=svc["kb_dir"], trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        generation_store=svc["pipeline"]._generation_store,
    )

    assert result["success"] is False
    assert result["pending"] is True
    assert svc["journal"].read() is not None


def test_recover_revalidate_alias_unknown_blocks_and_keeps_journal(tmp_path, monkeypatch):
    """revalidate 恢复读 Alias 异常 → blocked，journal 不能被归档。"""
    from app.evolution.revalidate import recover_revalidate

    svc, info = _activated_services(tmp_path)
    # 只需候选 GenerationInfo 形状即可；测试不触碰真实 ES。
    candidate = svc["index_service"].build("numpy")
    entry = {
        "phase": "revalidate",
        "stage": "INDEX_BUILT",
        "trashed_docs": [],
        "backend": "es",
        "index": candidate.to_dict(),
        "previous_target": info.target,
    }
    svc["journal"].write(entry)

    def _raise(_backend):
        raise RuntimeError("ES unavailable")

    monkeypatch.setattr(svc["index_service"], "reconcile", _raise)
    result = recover_revalidate(
        entry,
        kb_dir=svc["kb_dir"], trash_dir=svc["state_dir"] / "trash",
        ledger=svc["ledger"], publisher=svc["publisher"],
        index_service=svc["index_service"],
        generation_store=svc["pipeline"]._generation_store,
    )

    assert result["blocked"] is True
    assert result["success"] is False
    assert svc["journal"].read() is not None
    assert (svc["state_dir"] / "kb_write_blocked").exists()


def test_pipeline_run_recovers_interrupted_revalidate(tmp_path, reset_settings):
    """run() 开头识别 phase=revalidate journal → 前进式恢复后正常继续。"""
    settings.self_evolve_enabled = True
    svc, _ = _activated_services(tmp_path, grounding_judge=StubGrounding(grounded=True))
    _write_revalidate_journal(svc)
    svc["pipeline"].run()
    assert svc["journal"].read() is None  # 已归档
    assert not (svc["kb_dir"] / "evolved" / "20260801-abc-问答.md").exists()
    assert svc["ledger"].pending_entry("cid-evolved")["reason"] == "revalidation_failed"

# ruff: noqa: DTZ005, RUF059
"""人工知识生命周期 API 测试（010）：详情/重评/下架端点、开关门禁、时区、message_id 查重。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.config.settings import settings
from app.evolution.human_store import (
    CAND_PUBLISHED,
    HumanKnowledgeStore,
)
from app.stores.sql.schema import metadata


class _FakeComponents:
    redis = None
    db_engine = None
    es_client = None
    message_index = ""
    tool_executor = None
    mcp_client = None

    def __init__(self, store):
        self.human_knowledge_store = store


@pytest.fixture()
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.server.main import create_app

    monkeypatch.setattr(settings, "human_qa_evolution_enabled", True)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    store = HumanKnowledgeStore(engine, lease_seconds=1800)
    app = create_app()
    monkeypatch.setattr(
        "app.server.main.build_pod_components",
        lambda: _FakeComponents(store),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, store, monkeypatch


def _conv_payload(**overrides):
    payload = {
        "source": "cs-system",
        "external_conversation_id": "conv-api",
        "source_version": 1,
        "agent_id": "agent-7",
        "started_at": "2026-09-05T10:00:00",
        "ended_at": "2026-09-05T10:30:00",
        "messages": [
            {
                "message_id": "c0",
                "actor_type": "customer",
                "content": "拆封的耳机能不能退货？",
                "sent_at": "2026-09-05T10:00:00",
            },
            {
                "message_id": "a0",
                "actor_type": "human_agent",
                "content": "拆封不影响二次销售的可以七天无理由退货。",
                "sent_at": "2026-09-05T10:01:00",
            },
        ],
    }
    payload.update(overrides)
    return {"conversations": [payload]}


def _publishable_candidate(store, *, value=0.9, status=CAND_PUBLISHED):
    record, _ = store.ingest_conversation(
        source="cs",
        external_conversation_id=f"conv-{value}",
        source_version=1,
        agent_id="",
        started_at=None,
        ended_at=datetime.now() - timedelta(days=1),
        messages=[
            {
                "message_id": "c0",
                "actor_type": "customer",
                "content": "拆封的耳机能不能退货？",
                "sent_at": "",
            },
            {
                "message_id": "a0",
                "actor_type": "human_agent",
                "content": "拆封不影响二次销售的可以七天无理由退货。",
                "sent_at": "",
            },
        ],
    )
    job = store.claim_evaluation_job("seed")
    store.complete_evaluation(
        job,
        candidates=[
            {
                "question": "拆封的耳机能不能退货？",
                "answer": "拆封不影响二次销售的可以七天无理由退货退款。",
                "evidence_message_ids": ["a0"],
                "status": status,
                "classification": "new",
                "value_score": value,
                "source_snapshot": {
                    "source": "cs",
                    "external_conversation_id": f"conv-{value}",
                    "source_version": 1,
                    "agent_id": "",
                    "ended_at": "2026-09-05 10:30:00",
                },
                "evidence_snapshot": [
                    {
                        "message_id": "a0",
                        "content": "拆封不影响二次销售的可以七天无理由退货。",
                        "sent_at": "",
                        "prev_customer": None,
                        "next_customer": None,
                    }
                ],
                "dedup_snapshot": {
                    "question": {"path": "evolved/old.md", "score": 0.5},
                    "answer": None,
                    "generation": "gen-1",
                    "embedding_version": "fake-embedder",
                },
            }
        ],
        eval_meta={},
    )
    rows, _total = store.list_candidates()
    row = rows[0]
    if status == CAND_PUBLISHED:
        # 下架校验要求已发布文件名；直接冻结一个
        from sqlalchemy import text as _text

        with store._engine.begin() as conn:
            conn.execute(
                _text(
                    "UPDATE human_knowledge_candidates SET published_filename = '20260905-human-x.md',"
                    " published_at = :ts WHERE id = :i"
                ),
                {"ts": datetime.now(), "i": int(row["id"])},
            )
            row = dict(
                conn.execute(
                    _text(
                        "SELECT * FROM human_knowledge_candidates WHERE id = :i"
                    ),
                    {"i": int(row["id"])},
                )
                .mappings()
                .first()
            )
    return row


# ============================================================
# 开关门禁（D1）：五个变更入口 503，查询只读保留 200
# ============================================================
class TestGate:
    def test_all_mutation_entries_503_when_disabled(self, api):
        client, store, monkeypatch = api
        monkeypatch.setattr(settings, "human_qa_evolution_enabled", False)
        cand = _publishable_candidate(store, status="pending_review")
        store.edit_candidate(
            cand["id"],
            "拆封未使用的耳机能退货吗",
            "拆封未使用且配件齐全的耳机可以七天无理由退货退款处理。",
            0,
        )
        job = store.claim_evaluation_job("w")
        store.fail_evaluation(job, RuntimeError("x"))
        assert client.post("/v1/human-conversations/batch", json=_conv_payload()).status_code == 503
        assert client.post(
            f"/v1/human-knowledge/candidates/{cand['id']}/edit",
            json={"question": "q", "answer": "a", "expected_revision": 0},
        ).status_code == 503
        assert client.post(f"/v1/human-knowledge/candidates/{cand['id']}/reject", json={}).status_code == 503
        assert client.post("/v1/human-knowledge/evaluation-jobs/1/retry").status_code == 503
        assert client.post(
            f"/v1/human-knowledge/candidates/{cand['id']}/retry-evaluation"
        ).status_code == 503
        assert client.post(
            f"/v1/human-knowledge/candidates/{cand['id']}/retire",
            json={"reason": "r", "expected_lifecycle_revision": 0},
        ).status_code == 503
        # 查询只读保留
        assert client.get("/v1/human-knowledge/candidates").status_code == 200
        assert client.get(f"/v1/human-knowledge/candidates/{cand['id']}").status_code == 200
        assert client.get("/v1/human-knowledge/publish-batches/1").status_code == 404  # 404 ≠ 503：只读放行


# ============================================================
# 接入侧会话栅栏（D1）：写库前取栅栏、超时 503、用毕释放
# ============================================================
class TestIngestFence:
    def test_fence_timeout_maps_to_503(self, api, monkeypatch):
        from app.evolution import fence as fence_mod

        client, _store, _m = api

        class _BusyFence:
            def __init__(self, engine, timeout_seconds=0.0):
                self.engine = engine

            def acquire(self, keys):
                raise fence_mod.FenceTimeout("busy")

            def release(self):
                raise AssertionError("acquire 失败后不应再调用 release")

        monkeypatch.setattr(fence_mod, "ConversationFence", _BusyFence)
        r = client.post("/v1/human-conversations/batch", json=_conv_payload())
        assert r.status_code == 503
        assert "会话栅栏" in r.json()["detail"]

    def test_fence_acquired_with_ingest_keys_and_released(self, api, monkeypatch):
        from app.evolution import fence as fence_mod

        client, store, _m = api
        calls = {"acquire": [], "released": False}
        real_release = fence_mod.ConversationFence.release

        class _RecordingFence(fence_mod.ConversationFence):
            def acquire(self, keys):
                calls["acquire"] = list(keys)
                return super().acquire(keys)

            def release(self):
                calls["released"] = True
                return real_release(self)

        monkeypatch.setattr(fence_mod, "ConversationFence", _RecordingFence)
        r = client.post("/v1/human-conversations/batch", json=_conv_payload())
        assert r.status_code == 200
        assert calls["acquire"] == [
            fence_mod.conversation_fence_key("cs-system", "conv-api")
        ]  # 固定长度摘要 key、字典序、去重
        assert calls["released"] is True


# ============================================================
# 接入：message_id 查重 + 时区口径（D4）
# ============================================================
class TestIngestHardening:
    def test_duplicate_message_id_422(self, api):
        client, _store, _m = api
        payload = _conv_payload()
        payload["conversations"][0]["messages"][1]["message_id"] = "c0"  # 与 c0 重复
        r = client.post("/v1/human-conversations/batch", json=payload)
        assert r.status_code == 422
        assert "message_id 重复" in r.json()["detail"]

    def test_default_message_ids_deduped_after_fill(self, api):
        client, _store, _m = api
        payload = _conv_payload()
        for m in payload["conversations"][0]["messages"]:
            m["message_id"] = ""  # 缺省补 m0000/m0001 之后查重
        r = client.post("/v1/human-conversations/batch", json=payload)
        assert r.status_code == 200

    def test_aware_timestamps_normalized_to_utc(self, api):
        client, store, _m = api
        payload = _conv_payload(
            started_at="2026-09-05T18:00:00+08:00",
            ended_at="2026-09-05T18:30:00+08:00",
        )
        r = client.post("/v1/human-conversations/batch", json=payload)
        assert r.status_code == 200
        conv = store.get_conversation(r.json()["results"][0]["conversation_id"])
        # +08:00 的 18:30 → UTC 10:30
        assert str(conv["ended_at"]).startswith("2026-09-05 10:30")


# ============================================================
# 候选详情 / 候选级重评 / 下架
# ============================================================
class TestCandidateLifecycleApi:
    def test_candidate_detail_dto(self, api):
        client, store, _m = api
        cand = _publishable_candidate(store)
        r = client.get(f"/v1/human-knowledge/candidates/{cand['id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["evidence_snapshot"][0]["message_id"] == "a0"
        assert body["source_snapshot"]["external_conversation_id"] == "conv-0.9"
        assert body["dedup_snapshot"]["question"]["path"] == "evolved/old.md"
        assert body["lifecycle_revision"] == 0
        assert isinstance(body["recent_evaluation_jobs"], list)

    def test_detail_404(self, api):
        client, _store, _m = api
        assert client.get("/v1/human-knowledge/candidates/99999").status_code == 404

    def test_retry_evaluation_endpoint(self, api):
        client, store, _m = api
        cand = _publishable_candidate(store, status="pending_review")
        store.edit_candidate(
            cand["id"],
            "拆封未使用的耳机能退货吗",
            "拆封未使用且配件齐全的耳机可以七天无理由退货退款处理。",
            0,
        )
        # 连续失败至 blocked（回拨 next_run_at 模拟退避到期）
        from sqlalchemy import text as _text

        for _ in range(7):
            job = store.claim_evaluation_job("w")
            if job is None:
                with store._engine.begin() as conn:
                    conn.execute(
                        _text(
                            "UPDATE human_evaluation_jobs SET next_run_at = :ts"
                            " WHERE status = 'retry_wait'"
                        ),
                        {"ts": datetime.now() - timedelta(seconds=1)},
                    )
                job = store.claim_evaluation_job("w")
            if job is None:
                break
            store.fail_evaluation(job, RuntimeError("llm down"))
        r = client.post(f"/v1/human-knowledge/candidates/{cand['id']}/retry-evaluation")
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        # 再次重试：无 blocked 任务 → 404
        r2 = client.post(f"/v1/human-knowledge/candidates/{cand['id']}/retry-evaluation")
        assert r2.status_code == 404

    def test_retire_202_and_lifecycle_mismatch_409(self, api):
        client, store, _m = api
        cand = _publishable_candidate(store)
        # lifecycle_revision 不匹配 → 409
        r = client.post(
            f"/v1/human-knowledge/candidates/{cand['id']}/retire",
            json={"reason": "manual", "expected_lifecycle_revision": 5},
        )
        assert r.status_code == 409
        # 匹配 → 202 + retire 批次
        r2 = client.post(
            f"/v1/human-knowledge/candidates/{cand['id']}/retire",
            json={"reason": "manual", "expected_lifecycle_revision": 0},
        )
        assert r2.status_code == 202
        body = r2.json()
        assert body["operation"] == "retire"
        # 批次详情带 operation/reason
        detail = client.get(f"/v1/human-knowledge/publish-batches/{body['batch_id']}")
        assert detail.json()["operation"] == "retire"
        assert detail.json()["reason"] == "manual"
        # 双下架（lifecycle_revision 已按第一次建批校验；行未变仍匹配 → 第二批
        # 也允许入队，双下架由结算 CAS 兜底）
        r3 = client.post(
            f"/v1/human-knowledge/candidates/{cand['id']}/retire",
            json={"reason": "manual", "expected_lifecycle_revision": 0},
        )
        assert r3.status_code == 202

    def test_retire_non_published_409(self, api):
        client, store, _m = api
        cand = _publishable_candidate(store, status="pending_review")
        store.reject_candidate(cand["id"], "manual")
        r = client.post(
            f"/v1/human-knowledge/candidates/{cand['id']}/retire",
            json={"reason": "manual", "expected_lifecycle_revision": 0},
        )
        assert r.status_code == 409


# ============================================================
# 批次详情：终态计数
# ============================================================
class TestBatchCounts:
    def test_terminal_counts_shape(self, api):
        client, store, monkeypatch = api
        monkeypatch.setattr(settings, "self_evolve_enabled", True)
        cand = _publishable_candidate(store, status="pending_review")
        r = client.post(
            "/v1/human-knowledge/publish-batches",
            json={"items": [{"candidate_id": cand["id"], "revision": int(cand["revision"])}]},
        )
        assert r.status_code == 202
        detail = client.get(f"/v1/human-knowledge/publish-batches/{r.json()['batch_id']}")
        body = detail.json()
        assert body["operation"] == "publish"
        assert body["terminal_counts"] == {
            "published": 0,
            "rejected": 0,
            "failed": 0,
            "superseded": 0,
        }

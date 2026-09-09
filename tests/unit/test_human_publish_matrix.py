# ruff: noqa: DTZ005, RUF059
"""010 发布矩阵测试：替换 / 批内重复 / 全拒批次 / 探针 / 五阶段漂移 / retire / 补偿。

复用 test_human_knowledge_pipeline 的播种与装配 helper；全程 sqlite + fakes。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from test_human_knowledge_pipeline import (
    _dedup_service,
    _ingest,
    _journal_cleared,
    _publisher,
    _seed_candidates,
)

from app.evolution.human_store import (
    CAND_PENDING_REVIEW,
    CAND_PUBLISH_QUEUED,
    CAND_REJECTED,
    CAND_SUPERSEDED,
    HumanKnowledgeConflict,
)
from app.stores.sql.schema import metadata


@pytest.fixture()
def env():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from app.evolution.human_store import HumanKnowledgeStore

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return engine, HumanKnowledgeStore(engine, lease_seconds=1800)


@pytest.fixture(autouse=True)
def _kb_with_managed_doc(tmp_path, monkeypatch):
    from app.config.settings import settings

    kb = tmp_path / "authority-kb"
    (kb / "evolved").mkdir(parents=True, exist_ok=True)
    (kb / "evolved" / "old.md").write_text(
        "---\nprovenance: test\nowner: system\n---\n# 旧知识\n\n旧答案。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "kb_dir", str(kb))


def _seed_replaced_old_candidate(store, filename, candidate_id=77):
    """播种一条已发布旧候选（结算替换回写目标）；ORM 默认值补齐 NOT NULL 列。"""
    from app.stores.sql.schema import human_knowledge_candidates as _C

    with store._engine.begin() as conn:
        conn.execute(
            _C.insert().values(
                id=candidate_id,
                conversation_id=1,
                status="published",
                question="旧知识的问题怎么处理？",
                answer="旧答案正文。",
                evidence_state="legacy_evidence_missing",
                classification="update",
                published_filename=filename,
                published_at=datetime.now(),
            )
        )


def _dedup_with_backend(backend, embedder):
    from conftest import FakeRetriever

    return _dedup_service(FakeRetriever(embedder, backend))


def _stage_old_doc(
    tmp_path,
    *,
    filename="20260901-human-77-old-doc.md",
    source_kind=None,
    quality="0.60",
    candidate_id=None,
):
    """在发布 Worker 的 kb（make_services）内放置旧 evolved 文档。"""
    old = tmp_path / "kb" / "evolved" / filename
    old.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---", "provenance: turn-x", "owner: system"]
    if source_kind:
        lines.append(f"source_kind: {source_kind}")
    if candidate_id:
        lines.append(f"candidate_id: {candidate_id}")
    lines += ["quality_score: " + quality, "effective_date: 2026-01-01", "---"]
    old.write_text(
        "\n".join(lines) + "\n# 旧知识\n\n旧答案正文，用于替换判定。\n",
        encoding="utf-8",
    )
    return filename


class MatrixBase:
    def _prepared(self, env, tmp_path, *, question="拆封的耳机能不能退货？"):
        _engine, store = env
        record, _ = _ingest(store)
        _seed_candidates(
            store,
            record["id"],
            [
                {
                    "question": question,
                    "answer": "拆封不影响二次销售的可以七天无理由退货退款。",
                    "evidence_message_ids": ["a0"],
                    "status": CAND_PENDING_REVIEW,
                    "classification": "new",
                    "value_score": 0.9,
                }
            ],
        )
        rows, _ = store.list_candidates()
        return store, rows[0]

    def _seed_extra(
        self,
        store,
        conv_id,
        question,
        answer,
        *,
        classification="update",
        value=0.9,
        dedup_target="evolved/old.md",
    ):
        _seed_candidates(
            store,
            conv_id,
            [
                {
                    "question": question,
                    "answer": answer,
                    "evidence_message_ids": ["a0"],
                    "status": CAND_PENDING_REVIEW,
                    "classification": classification,
                    "value_score": value,
                    "dedup_snapshot": {
                        "question": (
                            {"path": dedup_target, "score": 0.95}
                            if dedup_target
                            else None
                        ),
                        "answer": None,
                        "generation": "gen-test",
                        "embedding_version": "fake-embedder",
                    },
                }
            ],
        )
        rows, _ = store.list_candidates()
        return next(r for r in rows if r["question"] == question)


class TestReplacementMatrix(MatrixBase):
    def test_replace_question_side_update_target_matches(self, env, tmp_path):
        from conftest import FakeBackend, FakeEmbedder
        from sqlalchemy import text as _t

        from app.agent.rag.chunker import Chunk

        _engine, store = env
        store, _cand = self._prepared(env, tmp_path)
        old_file = _stage_old_doc(
            tmp_path, source_kind="human_conversation", candidate_id="human-77"
        )
        # 播种旧候选行（id=77，已发布、持有旧文件名）——frontmatter candidate_id
        # 的结算回写目标
        _seed_replaced_old_candidate(store, old_file)
        record2, _ = _ingest(store, external_conversation_id="conv-2")
        cand2 = self._seed_extra(
            store,
            record2["id"],
            "旧知识的问题怎么处理？",
            "拆封不影响二次销售的可以七天无理由退货退款，答案更新版本内容充足。",
            classification="update",
            value=0.9,
            dedup_target=f"evolved/{old_file}",
        )
        embedder = FakeEmbedder()
        backend = FakeBackend()
        backend.upsert(
            [
                Chunk(
                    chunk_id="o1",
                    doc="old",
                    section="旧知识的问题怎么处理？",
                    text="旧知识的问题怎么处理？旧答案正文。",
                    source_path=f"evolved/{old_file}",
                )
            ],
            [embedder.encode_one("旧知识的问题怎么处理？")],
            embedder.model,
        )
        pub, _svc = _publisher(
            tmp_path, store, dedup=_dedup_with_backend(backend, embedder)
        )
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand2["id"], "revision": 0}], requested_by="ops"
        )
        pub.process_once()
        assert store.get_batch(batch["id"])["status"] == "completed"
        assert store.get_candidate(cand2["id"])["status"] == "published"
        # 旧候选（frontmatter candidate_id: human-77）被替换结算
        with store._engine.connect() as conn:
            old_row = conn.execute(
                _t(
                    "SELECT status, replaced_by_candidate_id FROM "
                    "human_knowledge_candidates WHERE id = 77"
                )
            ).first()
        assert old_row is not None
        assert old_row[0] == CAND_SUPERSEDED
        assert int(old_row[1]) == int(cand2["id"])
        # 旧文档移 trash，新文档在 evolved/
        assert not (tmp_path / "kb" / "evolved" / old_file).exists()

    def test_failed_staging_never_retires_or_misbinds_replacement(
        self, env, tmp_path, monkeypatch
    ):
        """回归：过滤后的 docs 不能复用 valid 下标绑定替换目标。"""
        from app.evolution.human_publish import _DedupOutcome

        _engine, store = env
        store, cand1 = self._prepared(env, tmp_path, question="第一个更新问题怎么处理？")
        record2, _ = _ingest(store, external_conversation_id="conv-2")
        cand2 = self._seed_extra(
            store,
            record2["id"],
            "第二个更新问题怎么处理？",
            "第二个更新答案内容足够完整，并且应该替换自己的旧知识文档。",
            classification="update",
            value=0.9,
            dedup_target="evolved/old-2.md",
        )
        old1 = _stage_old_doc(tmp_path, filename="old-1.md")
        old2 = _stage_old_doc(tmp_path, filename="old-2.md")
        pub, _svc = _publisher(tmp_path, store)
        monkeypatch.setattr(pub._dedup, "embed", lambda _texts: [[1.0, 0.0], [0.0, 1.0]])
        replacements = {int(cand1["id"]): old1, int(cand2["id"]): old2}
        monkeypatch.setattr(
            pub,
            "_final_dedup",
            lambda _item, cand: _DedupOutcome(
                replaces=replacements[int(cand["id"])]
            ),
        )
        real_write = pub._publisher.write_staging
        calls = {"n": 0}

        def first_fails(candidate):
            calls["n"] += 1
            return None if calls["n"] == 1 else real_write(candidate)

        monkeypatch.setattr(pub._publisher, "write_staging", first_fails)
        batch, _ = store.create_publish_batch(
            [
                {"candidate_id": cand1["id"], "revision": 0},
                {"candidate_id": cand2["id"], "revision": 0},
            ],
            requested_by="ops",
        )
        pub.process_once()
        assert (tmp_path / "kb" / "evolved" / old1).exists()
        assert not (tmp_path / "kb" / "evolved" / old2).exists()
        assert store.get_candidate(cand1["id"])["status"] == CAND_REJECTED
        assert store.get_candidate(cand2["id"])["status"] == "published"

    def test_replace_blocked_when_target_changed(self, env, tmp_path):
        from conftest import FakeBackend, FakeEmbedder

        from app.agent.rag.chunker import Chunk

        _engine, store = env
        store, _cand = self._prepared(env, tmp_path)
        old_file = _stage_old_doc(
            tmp_path, source_kind="human_conversation", candidate_id="human-77"
        )
        record2, _ = _ingest(store, external_conversation_id="conv-2")
        cand2 = self._seed_extra(
            store,
            record2["id"],
            "旧知识的问题怎么处理？",
            "拆封不影响二次销售的可以七天无理由退货退款，答案更新版本内容充足。",
            classification="update",
            value=0.9,
            dedup_target="evolved/different-target.md",
        )
        embedder = FakeEmbedder()
        backend = FakeBackend()
        backend.upsert(
            [
                Chunk(
                    chunk_id="o1",
                    doc="old",
                    section="旧知识的问题怎么处理？",
                    text="旧知识的问题怎么处理？",
                    source_path=f"evolved/{old_file}",
                )
            ],
            [embedder.encode_one("旧知识的问题怎么处理？")],
            embedder.model,
        )
        pub, _svc = _publisher(
            tmp_path, store, dedup=_dedup_with_backend(backend, embedder)
        )
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand2["id"], "revision": 0}], requested_by="ops"
        )
        pub.process_once()
        assert store.get_batch(batch["id"])["items"][0]["detail"] == "dedup_target_changed"
        assert store.get_candidate(cand2["id"])["status"] == CAND_REJECTED

    def test_replace_blocked_by_low_value_score(self, env, tmp_path):
        from conftest import FakeBackend, FakeEmbedder

        from app.agent.rag.chunker import Chunk

        _engine, store = env
        store, _cand = self._prepared(env, tmp_path)
        old_file = _stage_old_doc(
            tmp_path,
            source_kind="human_conversation",
            candidate_id="human-77",
            quality="0.95",
        )
        record2, _ = _ingest(store, external_conversation_id="conv-2")
        cand2 = self._seed_extra(
            store,
            record2["id"],
            "旧知识的问题怎么处理？",
            "拆封不影响二次销售的可以七天无理由退货退款，答案更新版本内容充足。",
            classification="update",
            value=0.72,
            dedup_target=f"evolved/{old_file}",
        )
        embedder = FakeEmbedder()
        backend = FakeBackend()
        backend.upsert(
            [
                Chunk(
                    chunk_id="o1",
                    doc="old",
                    section="旧知识的问题怎么处理？",
                    text="旧知识的问题怎么处理？",
                    source_path=f"evolved/{old_file}",
                )
            ],
            [embedder.encode_one("旧知识的问题怎么处理？")],
            embedder.model,
        )
        pub, _svc = _publisher(
            tmp_path, store, dedup=_dedup_with_backend(backend, embedder)
        )
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand2["id"], "revision": 0}], requested_by="ops"
        )
        pub.process_once()
        assert store.get_batch(batch["id"])["items"][0]["detail"].startswith(
            "duplicate_on_publish"
        )
        assert store.get_candidate(cand2["id"])["status"] == CAND_REJECTED

    def test_answer_side_hit_never_replaces(self, env, tmp_path):
        from conftest import FakeBackend, FakeEmbedder

        from app.agent.rag.chunker import Chunk

        _engine, store = env
        store, _cand = self._prepared(env, tmp_path)
        old_file = _stage_old_doc(
            tmp_path, source_kind="human_conversation", candidate_id="human-77"
        )
        answer_text = "拆封不影响二次销售的可以七天无理由退货退款，答案更新版本内容充足。"
        record2, _ = _ingest(store, external_conversation_id="conv-2")
        cand2 = self._seed_extra(
            store,
            record2["id"],
            "全新问题从未出现过？",
            answer_text,
            classification="update",
            value=0.9,
            dedup_target="",
        )
        embedder = FakeEmbedder()
        backend = FakeBackend()
        backend.upsert(
            [
                Chunk(
                    chunk_id="o1",
                    doc="old",
                    section="s",
                    text=answer_text,
                    source_path=f"evolved/{old_file}",
                )
            ],
            [embedder.encode_one(answer_text)],
            embedder.model,
        )
        pub, _svc = _publisher(
            tmp_path, store, dedup=_dedup_with_backend(backend, embedder)
        )
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand2["id"], "revision": 0}], requested_by="ops"
        )
        pub.process_once()
        item = store.get_batch(batch["id"])["items"][0]
        assert item["detail"].startswith("duplicate_on_publish")
        assert store.get_candidate(cand2["id"])["status"] == CAND_REJECTED


class TestInRunAndEmptyBatch(MatrixBase):
    def test_in_run_duplicate_high_score_wins(self, env, tmp_path):
        from conftest import FakeBackend, FakeEmbedder

        _engine, store = env
        store, cand1 = self._prepared(env, tmp_path)
        record2, _ = _ingest(store, external_conversation_id="conv-2")
        cand2 = self._seed_extra(
            store,
            record2["id"],
            "拆封的耳机能不能退货？",
            "拆封不影响二次销售的可以七天无理由退货退款。",
            classification="new",
            value=0.5,
            dedup_target="",
        )
        embedder = FakeEmbedder()
        backend = FakeBackend()  # 索引为空：最终去重无命中，仅批内互查
        pub, _svc = _publisher(
            tmp_path, store, dedup=_dedup_with_backend(backend, embedder)
        )
        batch, _i = store.create_publish_batch(
            [
                {"candidate_id": cand1["id"], "revision": 0},
                {"candidate_id": cand2["id"], "revision": 0},
            ],
            requested_by="ops",
        )
        pub.process_once()
        items = {it["candidate_id"]: it for it in store.get_batch(batch["id"])["items"]}
        assert items[cand1["id"]]["status"] == "published"  # 高分（0.9）胜出
        assert items[cand2["id"]]["status"] == "rejected"
        assert items[cand2["id"]]["detail"] == "in_run_duplicate"

    def test_all_rejected_batch_completes_with_zero_docs(self, env, tmp_path):
        from conftest import FakeBackend, FakeEmbedder

        from app.agent.rag.chunker import Chunk

        _engine, store = env
        store, _cand = self._prepared(env, tmp_path)
        record2, _ = _ingest(store, external_conversation_id="conv-2")
        cand2 = self._seed_extra(
            store,
            record2["id"],
            "旧知识的问题怎么处理？",
            "拆封不影响二次销售的可以七天无理由退货退款，答案更新版本内容充足。",
            classification="new",
            value=0.9,
            dedup_target="",
        )
        embedder = FakeEmbedder()
        backend = FakeBackend()
        backend.upsert(
            [
                Chunk(
                    chunk_id="d1",
                    doc="d",
                    section="s",
                    text="旧知识的问题怎么处理？",
                    source_path="uploads/policy.md",
                )
            ],
            [embedder.encode_one("旧知识的问题怎么处理？")],
            embedder.model,
        )
        pub, _svc = _publisher(
            tmp_path, store, dedup=_dedup_with_backend(backend, embedder)
        )
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand2["id"], "revision": 0}], requested_by="ops"
        )
        pub.process_once()
        batch_after = store.get_batch(batch["id"])
        assert batch_after["status"] == "completed"
        assert batch_after["generation_id"] == ""
        assert batch_after["items"][0]["status"] == "rejected"


class TestProbeAndDrift(MatrixBase):
    def test_probe_failure_rolls_back_then_recovers(self, env, tmp_path, monkeypatch):
        from app.evolution.index_service import IndexBuildService

        _engine, store = env
        store, cand = self._prepared(env, tmp_path)
        pub, svc = _publisher(tmp_path, store)
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        calls = {"n": 0}

        def flaky_verify(backend, info, source_paths):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("probe injected failure")
            return 1

        monkeypatch.setattr(svc["index_service"], "verify_documents", flaky_verify)
        pub.process_once()
        assert store.get_batch(batch["id"])["status"] == "retry_wait"
        assert store.get_candidate(cand["id"])["status"] == CAND_PUBLISH_QUEUED
        monkeypatch.setattr(
            svc["index_service"],
            "verify_documents",
            lambda *args, **kwargs: IndexBuildService.verify_documents(
                svc["index_service"], *args, **kwargs
            ),
        )
        with _engine.begin() as conn:
            from sqlalchemy import text as _t

            conn.execute(
                _t("UPDATE human_publish_batches SET next_run_at = :ts WHERE id = :id"),
                {"ts": datetime.now() - timedelta(seconds=1), "id": batch["id"]},
            )
        pub.process_once()
        assert store.get_batch(batch["id"])["status"] == "completed"

    def test_higher_version_before_activation_blocks_and_never_publishes(
        self, env, tmp_path
    ):
        import app.evolution.human_publish as hp

        _engine, store = env
        store, cand = self._prepared(env, tmp_path)
        pub, _svc = _publisher(tmp_path, store)
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        original_revalidate = type(store).revalidate_batch_items
        original_publish = hp.HumanBatchPublisher._publish_items_locked

        def revalidate_with_drift(batch_id):
            # 激活前（build 之后、alias 之前）注入更高来源版本
            with store._engine.begin() as conn:
                from sqlalchemy import text as _t

                conn.execute(
                    _t("UPDATE human_conversations SET source_version = 2 WHERE id = :i"),
                    {"i": int(cand["conversation_id"])},
                )
            return original_revalidate(store, batch_id)

        def patched(self, batch, lost):
            store.revalidate_batch_items = revalidate_with_drift.__get__(store)
            try:
                return original_publish(self, batch, lost)
            finally:
                store.revalidate_batch_items = original_revalidate.__get__(store)

        hp.HumanBatchPublisher._publish_items_locked = patched
        try:
            pub.process_once()
        finally:
            hp.HumanBatchPublisher._publish_items_locked = original_publish
        assert store.get_batch(batch["id"])["status"] == "retry_wait"
        assert store.get_candidate(cand["id"])["status"] == CAND_PUBLISH_QUEUED

    def test_drift_at_settle_never_overwrites_and_triggers_compensation(
        self, env, tmp_path
    ):
        import app.evolution.human_publish as hp

        _engine, store = env
        store, cand = self._prepared(env, tmp_path)
        pub, svc = _publisher(tmp_path, store)
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        original_settle = type(store).settle_batch
        original_publish = hp.HumanBatchPublisher._publish_items_locked

        def settle_with_drift(self, b, *, generation_id, results):
            # 激活后、结算前：更高版本接入把候选打成 superseded
            with store._engine.begin() as conn:
                from sqlalchemy import text as _t

                conn.execute(
                    _t(
                        "UPDATE human_conversations SET source_version = 2 WHERE id = :i"
                    ),
                    {"i": int(cand["conversation_id"])},
                )
                conn.execute(
                    _t(
                        "UPDATE human_knowledge_candidates SET status = 'superseded'"
                        " WHERE id = :i"
                    ),
                    {"i": int(cand["id"])},
                )
            return original_settle(
                store, b, generation_id=generation_id, results=results
            )

        def patched(self, batch, lost):
            store.settle_batch = settle_with_drift.__get__(store)
            try:
                return original_publish(self, batch, lost)
            finally:
                store.settle_batch = original_settle.__get__(store)

        hp.HumanBatchPublisher._publish_items_locked = patched
        try:
            pub.process_once()
        finally:
            hp.HumanBatchPublisher._publish_items_locked = original_publish
        # CAS miss：候选保持 superseded（绝不被覆盖回 published）
        assert store.get_candidate(cand["id"])["status"] == CAND_SUPERSEDED
        # 文档已上线（激活不可回滚）→ 漂移产生补偿下架任务
        comp = [
            b
            for b in self._batches(store)
            if b["operation"] == "retire"
            and b["requested_by"] == "system:compensation"
        ]
        assert comp, "补偿下架批次应已入队"
        filename = store.get_batch(batch["id"])["items"][0]["filename"]
        assert (tmp_path / "kb" / "evolved" / filename).exists()

        # ---- 补偿批次必须真正移除文档（空转回归：P0）----
        gen_before = svc["generation_store"].active("numpy").generation_id
        assert pub.process_once() is True  # claim 排序 operation DESC → 先跑补偿
        comp_batch = store.get_batch(comp[0]["id"])
        assert comp_batch["status"] == "completed"
        assert comp_batch["items"][0]["status"] == "retired"
        # 文档从 evolved/ 移除、索引切换到不含该文档的新代
        assert not (tmp_path / "kb" / "evolved" / filename).exists()
        assert svc["generation_store"].active("numpy").generation_id != gen_before
        assert _journal_cleared(svc)
        # 候选行保持 superseded：既不被覆盖回 published，也不误写 retired
        assert store.get_candidate(cand["id"])["status"] == CAND_SUPERSEDED

    @staticmethod
    def _batches(store):
        from sqlalchemy import select

        from app.stores.sql.schema import human_publish_batches

        with store._engine.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(select(human_publish_batches)).mappings().all()
            ]


class TestRetireChain(MatrixBase):
    def test_retire_full_chain_mysql_file_index_consistent(self, env, tmp_path):
        _engine, store = env
        store, cand = self._prepared(env, tmp_path)
        pub, svc = _publisher(tmp_path, store)
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        pub.process_once()
        published = store.get_candidate(cand["id"])
        assert published["status"] == "published"
        filename = published["published_filename"]
        assert (tmp_path / "kb" / "evolved" / filename).exists()
        gen_before = svc["generation_store"].active("numpy").generation_id

        retire_batch, _items = store.create_retire_batch(
            [{"candidate_id": cand["id"], "expected_lifecycle_revision": 1}],
            reason="manual_retire",
            requested_by="ops-a",
        )
        assert pub.process_once() is True
        rb = store.get_batch(retire_batch["id"])
        assert rb["status"] == "completed" and rb["operation"] == "retire"
        fresh = store.get_candidate(cand["id"])
        assert fresh["status"] == "retired"
        assert fresh["retired_at"] is not None
        assert fresh["retire_reason"] == "manual_retire"
        assert int(fresh["lifecycle_revision"]) == 2
        assert not (tmp_path / "kb" / "evolved" / filename).exists()
        # 索引已重建（新代不含该文档）
        assert svc["generation_store"].active("numpy").generation_id != gen_before
        assert _journal_cleared(svc)

    def test_double_retire_converges_without_corruption(self, env, tmp_path):
        _engine, store = env
        store, cand = self._prepared(env, tmp_path)
        pub, _svc = _publisher(tmp_path, store)
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        pub.process_once()
        fresh = store.get_candidate(cand["id"])
        lrev = int(fresh["lifecycle_revision"])
        rb1, _ = store.create_retire_batch(
            [{"candidate_id": cand["id"], "expected_lifecycle_revision": lrev}],
            reason="a",
            requested_by="ops",
        )
        rb2, _ = store.create_retire_batch(
            [{"candidate_id": cand["id"], "expected_lifecycle_revision": lrev}],
            reason="b",
            requested_by="ops",
        )
        pub.process_once()  # retire 优先（operation DESC）
        pub.process_once()
        assert store.get_batch(rb1["id"])["status"] == "completed"
        assert store.get_batch(rb2["id"])["status"] == "completed"
        final = store.get_candidate(cand["id"])
        assert final["status"] == "retired"  # 状态稳定，不回退不漂移


class TestJournalAndLogs(MatrixBase):
    def test_journal_carries_operation_and_digests(self, env, tmp_path, monkeypatch):
        _engine, store = env
        store, cand = self._prepared(env, tmp_path)
        pub, svc = _publisher(tmp_path, store)
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        written = []
        original_write = svc["journal"].write

        def spy(entry):
            written.append(entry)
            return original_write(entry)

        monkeypatch.setattr(svc["journal"], "write", spy)
        pub.process_once()
        bodies = [w for w in written if w.get("kind") == "human_publish"]
        assert bodies
        assert bodies[0]["operation"] == "publish"
        assert bodies[0]["approval_digests"]
        assert all(bodies[0]["approval_digests"])
        assert "old_generation_id" in bodies[0]

    def test_completed_batch_replays_post_settle_before_clearing_journal(
        self, env, tmp_path, monkeypatch
    ):
        """settle 后强杀：旧 Ledger 清账由残留 Journal 幂等补齐。"""
        from conftest import FakeBackend, FakeEmbedder

        from app.agent.rag.chunker import Chunk
        from app.evolution.lifecycle import KnowledgeLifecycleCoordinator

        _engine, store = env
        store, _unused = self._prepared(env, tmp_path)
        old_file = _stage_old_doc(tmp_path, filename="robot-old.md")
        record2, _ = _ingest(store, external_conversation_id="conv-post-settle")
        cand = self._seed_extra(
            store,
            record2["id"],
            "旧机器人知识如何更新？",
            "这是新的完整答案，用于验证结算后崩溃仍会清理旧 Ledger 记录。",
            classification="update",
            value=0.9,
            dedup_target=f"evolved/{old_file}",
        )
        embedder = FakeEmbedder()
        backend = FakeBackend()
        backend.upsert(
            [
                Chunk(
                    chunk_id="robot-old",
                    doc="old",
                    section="s",
                    text="旧机器人知识如何更新？",
                    source_path=f"evolved/{old_file}",
                )
            ],
            [embedder.encode_one("旧机器人知识如何更新？")],
            embedder.model,
        )
        pub, svc = _publisher(
            tmp_path, store, dedup=_dedup_with_backend(backend, embedder)
        )
        svc["ledger"].mark_published("robot-old-cid", old_file)
        pub._lifecycle = KnowledgeLifecycleCoordinator(
            store, svc["ledger"], kb_dir=svc["kb_dir"]
        )
        batch, _ = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        real_post_settle = pub._post_settle
        monkeypatch.setattr(
            pub,
            "_post_settle",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("crash after settle")
            ),
        )
        pub.process_once()
        assert store.get_batch(batch["id"])["status"] == "completed"
        assert svc["journal"].read()["stage"] == "LEDGER_COMMITTED"
        assert "robot-old-cid" in svc["ledger"].published()

        monkeypatch.setattr(pub, "_post_settle", real_post_settle)
        pub.process_once()
        assert "robot-old-cid" not in svc["ledger"].published()
        assert _journal_cleared(svc)

    def test_logs_do_not_contain_question_or_answer(self, env, tmp_path, caplog):
        import logging as _logging

        _engine, store = env
        store, cand = self._prepared(env, tmp_path)
        pub, svc = _publisher(tmp_path, store)
        batch, _i = store.create_publish_batch(
            [{"candidate_id": cand["id"], "revision": 0}], requested_by="ops"
        )
        secret = "拆封不影响二次销售的可以七天无理由退货退款"
        svc["index_service"].build = lambda backend, generation_id=None: (
            _ for _ in ()
        ).throw(RuntimeError(secret + " LEAK"))
        with caplog.at_level(_logging.WARNING, logger="app.evolution.human_publish"):
            pub.process_once()
        leaked = [
            r
            for r in caplog.records
            if secret in r.getMessage() and "LEAK" in r.getMessage()
        ]
        assert not leaked

class TestBackfillScript(MatrixBase):
    def _legacy_candidate(self, store, *, with_conversation=True):
        """直插一条 legacy 候选（含会话正本），返回 candidate_id。"""
        if with_conversation:
            record, _ = _ingest(store)
            conv_id = record["id"]
        else:
            conv_id = 99999  # 会话不存在
        from app.stores.sql.schema import human_knowledge_candidates as _C

        with store._engine.begin() as conn:
            result = conn.execute(
                _C.insert().values(
                    conversation_id=conv_id,
                    status="pending_review",
                    question="拆封的耳机能不能退货？",
                    answer="拆封不影响二次销售的可以七天无理由退货退款。",
                    evidence_message_ids='["a0"]',
                    evidence_state="legacy_evidence_missing",
                    classification="new",
                )
            )
        return int(result.inserted_primary_key[0])

    def _candidate_row(self, store, cid):
        from sqlalchemy import text as _t

        with store._engine.connect() as conn:
            return dict(
                conn.execute(
                    _t("SELECT * FROM human_knowledge_candidates WHERE id = :i"),
                    {"i": cid},
                )
                .mappings()
                .first()
            )

    def test_backfill_fills_snapshots_and_sets_ok(self, env):
        from app.scripts.backfill_human_evidence import backfill

        _engine, store = env
        cid = self._legacy_candidate(store)
        row = self._candidate_row(store, cid)
        assert row["evidence_state"] == "legacy_evidence_missing"
        stats = backfill(store)
        assert stats["backfilled"] == 1
        row = self._candidate_row(store, cid)
        assert row["evidence_state"] == "ok"
        assert "a0" in row["evidence_snapshot_json"]
        assert "cs-system" in row["source_snapshot_json"]
        # 幂等：再次运行无动作
        stats2 = backfill(store)
        assert stats2["backfilled"] == 0

    def test_backfill_skips_when_conversation_missing(self, env):
        from app.scripts.backfill_human_evidence import backfill

        _engine, store = env
        cid = self._legacy_candidate(store, with_conversation=False)
        stats = backfill(store)
        assert stats["conversation_missing"] == 1
        row = self._candidate_row(store, cid)
        # 会话缺失 → 保持 legacy，批准 409
        assert row["evidence_state"] == "legacy_evidence_missing"
        with pytest.raises(HumanKnowledgeConflict):
            store.create_publish_batch(
                [{"candidate_id": cid, "revision": 0}], requested_by="ops"
            )

    def test_backfill_skips_candidates_without_agent_evidence(self, env):
        """旧证据只引用客户消息 → 无法按新语义回填，保持 legacy。"""

        from app.scripts.backfill_human_evidence import backfill

        _engine, store = env
        record, _ = _ingest(store)
        from app.stores.sql.schema import human_knowledge_candidates as _C

        with store._engine.begin() as conn:
            result = conn.execute(
                _C.insert().values(
                    conversation_id=record["id"],
                    status="pending_review",
                    question="问题一",
                    answer="答案内容至少二十个字符才能通过校验的答案。",
                    evidence_message_ids='["c0"]',
                    evidence_state="legacy_evidence_missing",
                    classification="new",
                )
            )
            cid = result.inserted_primary_key[0]
        stats = backfill(store)
        assert stats["missing_evidence"] == 1
        assert self._candidate_row(store, int(cid))["evidence_state"] == (
            "legacy_evidence_missing"
        )

"""RAG 修复计划·1/2/3/4 回归测试。

覆盖：
- ES Hybrid + rerank=none + 正数阈值不返回空集（RRF 分不做阈值门控）；
- reranker 断连/畸形 → 降级（不是正常精排）；
- 降级结果 fail-closed（search_knowledge.success=false）；
- archive/.trash/.staging 不进入索引；元数据校验；
- embedding 模型不一致 → 加载失败；
- 生产 values 与已批准基线一致（且不含旧阈值）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.rerank import RerankerUnavailable


def _hit(cid: str, text: str, score: float = 0.0, parent: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc="政策", section="s", text=text,
                    source_path="p.md", parent_id=parent),
        score=score,
    )


# ------------------------------------------------------------
# 1：RRF + rerank=none + 正数阈值 → 不空集
# ------------------------------------------------------------
class _FakeESBackend:
    def __init__(self, hits):
        self._hits = hits

    def hybrid_search(self, **kwargs):
        return list(self._hits)

    def load(self):
        pass

    @property
    def size(self):
        return len(self._hits)


class _FakeEmbedder:
    model = "m"

    def encode_one(self, text, timeout=None):
        return [0.0]


def test_es_hybrid_rerank_none_positive_threshold_not_empty():
    from app.agent.rag.hybrid import ESHybridRetriever
    from app.agent.rag.retriever_factory import final_search

    retriever = ESHybridRetriever(
        embedder=_FakeEmbedder(), backend=_FakeESBackend([_hit("c1", "七天无理由")]),
        recall_k=10, reranker=None,  # rerank=none → 真正的 None
    )
    out = final_search(retriever, "七天无理由退货", top_k=3, min_score=0.99)
    assert out.hits, "RRF 分不得被阈值清空"
    assert out.score_source == "rrf"
    assert out.gated_candidates == len(out.raw_hits)  # 未门控
    assert out.degraded is False


# ------------------------------------------------------------
# 2：reranker 不可用 → 降级（非正常精排）
# ------------------------------------------------------------
class _UnavailableReranker:
    def rerank(self, query, hits, top_k, timeout=None):
        raise RerankerUnavailable("connect error")


def test_reranker_unavailable_reports_degraded():
    from app.agent.rag.hybrid import ESHybridRetriever

    retriever = ESHybridRetriever(
        embedder=_FakeEmbedder(),
        backend=_FakeESBackend([_hit("c1", "七天无理由")]),
        recall_k=10, reranker=_UnavailableReranker(),
    )
    result = retriever.search_with_status("问", top_k=3)
    assert result.degraded is True
    assert result.degraded_reason == "reranker_unavailable"
    assert result.score_source == "rrf"
    assert result.scores_meaningful is False  # 不得当作正常精排分


def test_search_knowledge_fail_closed_on_degraded(monkeypatch, reset_settings):
    from app.agent.tools import knowledge as km

    class _DegradedRetriever:
        def search_with_status(self, query, top_k=3, timeout=None):
            from app.agent.rag.retriever import RetrievalResult

            return RetrievalResult(
                hits=[_hit("c1", "七天无理由")],
                score_source="rrf", degraded=True,
                degraded_reason="reranker_unavailable",
            )

    monkeypatch.setattr(km, "_get_retriever", lambda: _DegradedRetriever())
    km.settings.rag_rerank_fail_closed = True
    out = km.search_knowledge("七天无理由退货", top_k=3)
    assert out["success"] is False
    assert out["error"] == "reranker_unavailable"
    assert out["results"] == []


def test_search_knowledge_degraded_allowed_when_not_fail_closed(monkeypatch, reset_settings):
    from app.agent.tools import knowledge as km

    class _DegradedRetriever:
        def search_with_status(self, query, top_k=3, timeout=None):
            from app.agent.rag.retriever import RetrievalResult

            return RetrievalResult(
                hits=[_hit("c1", "七天无理由")], score_source="rrf",
                degraded=True, degraded_reason="reranker_unavailable",
            )

    monkeypatch.setattr(km, "_get_retriever", lambda: _DegradedRetriever())
    km.settings.rag_rerank_fail_closed = False
    out = km.search_knowledge("七天无理由退货", top_k=3)
    assert out["success"] is True
    assert out["evidence"]["degraded"] is True


# ------------------------------------------------------------
# 2：文档治理（目录 + 元数据）
# ------------------------------------------------------------
def test_governance_excludes_archive_trash_staging_and_temp(tmp_path):
    from app.agent.rag.parsers import chunk_kb_dir

    kb = tmp_path / "kb"
    for sub in ("archive", ".trash", ".staging", "uploads", "evolved"):
        (kb / sub).mkdir(parents=True)
    (kb / "keep.md").write_text("# 保留\n\n## 章节\n\n内容\n", encoding="utf-8")
    for sub in ("archive", ".trash", ".staging"):
        (kb / sub / "old.md").write_text(
            "# 旧\n\n## 章节\n\n历史\n", encoding="utf-8")
    (kb / "draft.md.tmp").write_text("# 临时\n\n## 章节\n\nx\n", encoding="utf-8")

    docs = {c.doc for c in chunk_kb_dir(kb)}
    assert "keep" in docs
    assert "old" not in docs  # archive/.trash/.staging 均排除
    assert "draft.md" not in docs


def test_governance_metadata_validation_and_eligibility():
    from app.agent.rag import governance as gov

    with pytest.raises(gov.DocumentGovernanceError):
        gov.validate_metadata("a.md", {})  # 缺全部字段
    with pytest.raises(gov.DocumentGovernanceError):
        gov.validate_metadata("a.md", {
            "status": "active", "authority": "external_reference",
            "effective_date": "2026-01-01",
        })  # 冲突
    gov.validate_metadata("a.md", {
        "status": "active", "authority": "platform",
        "effective_date": "2026-01-01",
    })
    assert gov.is_index_eligible({"status": "archived"}) is False
    assert gov.is_index_eligible({"authority": "external_reference"}) is False
    assert gov.is_index_eligible(
        {"status": "active", "authority": "platform"}) is True


def test_strict_build_fails_on_missing_metadata(tmp_path, reset_settings, monkeypatch):
    from app.agent.rag.parsers import chunk_kb_dir

    monkeypatch.setattr(
        "app.config.settings.settings.rag_doc_metadata_required", True
    )
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "no_meta.md").write_text("# 无元数据\n\n## 章节\n\n内容\n", encoding="utf-8")
    with pytest.raises(Exception) as ei:
        chunk_kb_dir(kb, strict=True)
    assert "no_meta.md" in str(ei.value)


def test_archived_document_not_indexed(tmp_path):
    from app.agent.rag.parsers import chunk_kb_dir

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "active.md").write_text(
        "---\nstatus: active\nauthority: platform\neffective_date: 2026-01-01\n---\n"
        "# 生效\n\n## 章节\n\n新政策\n", encoding="utf-8")
    (kb / "old.md").write_text(
        "---\nstatus: archived\nauthority: platform\neffective_date: 2024-01-01\n---\n"
        "# 归档\n\n## 章节\n\n旧政策\n", encoding="utf-8")
    docs = {c.doc for c in chunk_kb_dir(kb)}
    assert "active" in docs and "old" not in docs


# ------------------------------------------------------------
# 3：embedding 模型不一致 → 加载失败
# ------------------------------------------------------------
class _BackendMismatch:
    def load(self):
        pass

    def expected_embedding_model(self):
        return "bge-m3"

    @property
    def size(self):
        return 0


class _EmbedderOther:
    model = "text-embedding-3-small"


def test_embedding_model_mismatch_raises():
    from app.agent.rag.retriever import KnowledgeRetriever

    r = KnowledgeRetriever(embedder=_EmbedderOther(), backend=_BackendMismatch())
    with pytest.raises(ValueError, match="不一致"):
        r.load()


# ------------------------------------------------------------
# 3：生产 values 与已批准基线一致
# ------------------------------------------------------------
def test_production_values_match_approved_rag_baseline():
    import yaml

    from app.evaluation.manifest import (
        APPROVED_RAG_BASELINE,
        FORBIDDEN_RAG_THRESHOLD,
    )

    path = (Path(__file__).resolve().parents[2]
            / "deploy" / "helm" / "ecom-agent" / "values-production.yaml")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    env = values.get("env", {})
    for key, expected in APPROVED_RAG_BASELINE.items():
        assert str(env.get(key)) == expected, f"{key} 偏离已批准基线"
    # 旧阈值必须已删除（待重校准后写入）
    assert "RAG_MIN_RELEVANCE_SCORE" not in env
    assert FORBIDDEN_RAG_THRESHOLD not in path.read_text(encoding="utf-8")

    kind_path = (Path(__file__).resolve().parents[2]
                 / "deploy" / "helm" / "ecom-agent" / "values-kind.yaml")
    kind_env = yaml.safe_load(kind_path.read_text(encoding="utf-8")).get("env", {})
    assert kind_env["RAG_MIN_RELEVANCE_SCORE"] == "0.0"


def test_rag_release_workflow_uses_candidate_and_propagates_threshold():
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    assert "--no-activate --json-out /tmp/rag_release_candidate.json" in workflow
    assert workflow.count("--candidate /tmp/rag_release_candidate.json") == 3
    assert workflow.count('--min-score "$RAG_MIN_RELEVANCE_SCORE"') == 2
    assert "RAG_RELEASE_ES_URL: ${{ secrets.RAG_RELEASE_ES_URL }}" in workflow
    assert "RAG_RELEASE_REDIS_URL: ${{ secrets.RAG_RELEASE_REDIS_URL }}" in workflow
    assert "--activate-candidate /tmp/rag_release_candidate.json" not in workflow
    assert "if: ${{ secrets." not in workflow


def test_candidate_descriptor_validates_es_target(tmp_path, reset_settings):
    import json

    from app.agent.rag.fingerprint import config_fingerprint
    from app.config.settings import settings
    from app.scripts.run_retrieval_eval import load_candidate

    settings.rag_backend = "es"
    settings.es_index_prefix = "ecom"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "protocol": "kb-generation-candidate-v1",
        "backend": "es",
        "config_fingerprint": config_fingerprint(),
        "expected_active_generation_id": "",
        "generation": {
            "generation_id": "20260912000000-abcdef12",
            "target": "ecom-kb-20260912000000-abcdef12",
            "embedding_model": "bge-m3",
            "previous_generation_id": "",
        },
    }), encoding="utf-8")
    assert load_candidate(str(candidate)) == (
        "20260912000000-abcdef12", "ecom-kb-20260912000000-abcdef12",
    )

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["generation"]["target"] = "ecom-kb-other"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="target"):
        load_candidate(str(candidate))

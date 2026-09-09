"""语义去重服务测试（010）：归一化边界 / 双侧分离 / 阈值隔离 / 批内 pairwise 平局。"""

from __future__ import annotations

import pytest

from app.config.settings import settings
from app.evolution.semantic_dedup import (
    backend_score_to_cosine,
    drop_in_run_duplicates,
    normalize_cosine,
)


class _FakeActiveInfo:
    generation_id = "gen-1"
    target = "t"
    embedding_model = "fake-embedder"


class _FakeGenStore:
    def __init__(self):
        self.info = _FakeActiveInfo()

    def active(self, backend):
        return self.info


class _FakeIndexService:
    def __init__(self, retriever):
        self._retriever = retriever
        self.configs = []

    def open_retriever(self, info, retrieval_config=None):
        self.configs.append(retrieval_config)
        return self._retriever


class _ScriptedRetriever:
    """按查询前缀脚本化命中：question:* / answer:* 各自独立。"""

    def __init__(self, question_score=None, answer_score=None, path="evolved/old.md"):
        self.question_score = question_score
        self.answer_score = answer_score
        self.path = path

    def search(self, query, top_k=1, timeout=None):
        from app.agent.rag.backends.base import RetrievedChunk
        from app.agent.rag.chunker import Chunk

        score = None
        if query.startswith("question:"):
            score = self.question_score
        elif query.startswith("answer:"):
            score = self.answer_score
        if score is None:
            return []
        chunk = Chunk(chunk_id="c1", doc="d", section="s", text="旧文", source_path=self.path)
        return [RetrievedChunk(chunk=chunk, score=score)]


def _service(retriever, *, q=None, a=None):
    from conftest import FakeEmbedder

    from app.evolution.semantic_dedup import SemanticDedupService

    return SemanticDedupService(
        _FakeIndexService(retriever),
        _FakeGenStore(),
        FakeEmbedder(),
        q_threshold=q,
        a_threshold=a,
    )


def test_normalize_cosine_boundaries():
    assert normalize_cosine(1.0) == 1.0
    assert normalize_cosine(-1.0) == 0.0  # 截断而非仿射：负相关 → 0 而非 1
    assert normalize_cosine(0.0) == 0.0
    assert normalize_cosine(0.8999) == pytest.approx(0.8999)


def test_elasticsearch_score_is_restored_to_raw_cosine():
    # ES cosine kNN _score=(1+cosine)/2；0.95 应还原为 0.90。
    assert backend_score_to_cosine(0.95, "es") == pytest.approx(0.90)
    assert backend_score_to_cosine(0.90, "es") == pytest.approx(0.80)
    assert backend_score_to_cosine(0.90, "numpy") == pytest.approx(0.90)


def test_score_two_sides_and_config_is_pure_vector():
    svc = _service(_ScriptedRetriever(question_score=0.93, answer_score=0.5))
    result = svc.score("question:q", "answer:a")
    assert result.question.score == pytest.approx(0.93)
    assert result.answer.score == pytest.approx(0.5)
    assert result.question.path == "evolved/old.md"
    # 纯向量通道：hybrid 关闭、rerank none（绝不复用线上 hybrid/RRF/reranker）
    index_svc = svc._index if hasattr(svc, "_index") else None
    cfg = getattr(index_svc, "configs", [None])[0]
    assert cfg is not None and cfg.hybrid is False and cfg.rerank == "none"


def test_threshold_semantics_and_tolerance():
    svc = _service(_ScriptedRetriever(question_score=0.89995, answer_score=0.0))
    hit = svc.score("question:q", "answer:a").question
    # 容差 1e-4：0.89995 视为达到 0.90
    assert svc.at_threshold(hit, side="question") is True
    svc2 = _service(_ScriptedRetriever(question_score=0.85, answer_score=0.0))
    hit2 = svc2.score("question:q", "answer:a").question
    assert svc2.at_threshold(hit2, side="question") is False
    assert svc2.at_threshold(None, side="question") is False


def test_thresholds_isolated_from_evolve_dedup_threshold():
    assert settings.human_dedup_question_threshold == 0.90
    assert settings.human_dedup_answer_threshold == 0.90
    # 显式注入阈值不被 evolve_dedup_threshold 影响（0.92 在问题侧 0.95 之下）
    svc = _service(_ScriptedRetriever(question_score=0.92), q=0.95, a=0.8)
    hit = svc.score("question:q", "answer:a").question
    assert svc.at_threshold(hit, side="question") is False  # 0.92 < 0.95
    assert svc.q_threshold == 0.95 and svc.a_threshold == 0.8
    # 答案侧独立阈值：0.75 在 0.8 之下 → 不命中
    svc2 = _service(_ScriptedRetriever(answer_score=0.75), q=0.95, a=0.8)
    a_hit = svc2.score("question:q", "answer:a").answer
    assert svc2.at_threshold(a_hit, side="answer") is False


def test_embed_and_snapshot_meta():
    svc = _service(_ScriptedRetriever())
    vectors = svc.embed(["a", "b"])
    assert len(vectors) == 2 and len(vectors[0]) == 32
    meta = svc.snapshot_meta()
    assert meta["generation"] == "gen-1"
    assert meta["embedding_version"] == "fake-embedder"


def test_pairwise_high_score_wins():
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.99, 0.1]]
    items = [
        {"candidate_id": 1, "value_score": 0.9},
        {"candidate_id": 2, "value_score": 0.9},
        {"candidate_id": 3, "value_score": 0.5},
    ]
    dropped = drop_in_run_duplicates(
        [[v[0], v[1], 0.0][:2] + [0.0] for v in vectors], items, threshold=0.9
    )
    # 与构造的近似向量组无关：精确两两余弦判定
    # v1=(1,0) v2=(0,1) 正交不重复；v3 与 v1 高相似 → 低分 v3 出局
    assert dropped == [2]


def test_pairwise_tie_smaller_id_wins():
    # 同分平局：小 ID 胜出（大 ID 出局）
    v = [1.0, 0.0]
    items = [
        {"candidate_id": 7, "value_score": 0.9},
        {"candidate_id": 3, "value_score": 0.9},
    ]
    dropped = drop_in_run_duplicates([v, list(v)], items, threshold=0.9)
    assert dropped == [0]  # 传入顺序 [7, 3]，出局的是 7（大 ID）
    items2 = [
        {"candidate_id": 3, "value_score": 0.9},
        {"candidate_id": 7, "value_score": 0.9},
    ]
    dropped2 = drop_in_run_duplicates([list(v), list(v)], items2, threshold=0.9)
    assert dropped2 == [1]  # 传入顺序 [3, 7]，出局的是 7


def test_pairwise_high_score_beats_order():
    v = [1.0, 0.0]
    items = [
        {"candidate_id": 1, "value_score": 0.5},
        {"candidate_id": 2, "value_score": 0.9},
    ]
    dropped = drop_in_run_duplicates([list(v), list(v)], items, threshold=0.9)
    assert dropped == [0]  # 高分 (id=2) 胜出，低分 (id=1) 出局


def test_no_active_generation_returns_empty_hits():
    from conftest import FakeEmbedder

    from app.evolution.semantic_dedup import SemanticDedupService

    class _EmptyStore:
        def active(self, backend):
            return None

    svc = SemanticDedupService(_FakeIndexService(_ScriptedRetriever()), _EmptyStore(), FakeEmbedder())
    result = svc.score("q", "a")
    assert result.question is None and result.answer is None


def test_side_hit_normalization_on_retriever_scores():
    svc = _service(_ScriptedRetriever(question_score=-0.4, answer_score=1.2))
    result = svc.score("question:q", "answer:a")
    assert result.question.score == 0.0  # 负分截断为 0
    assert result.answer.score == 1.0  # 超界截断为 1

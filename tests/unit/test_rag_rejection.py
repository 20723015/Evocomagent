"""P1-4 四信号联合拒绝：信号计算 / 判定 / 死信号修复 / 生产 fail-closed 接入。

覆盖：
- 四信号各自触发拒绝 + 全通过（top1 / gap / coverage / rerank）；
- 空 hits → fail-closed（no_hits）；RRF/降级 → 不适用（不得拿无语义分判定）；
- rerank 死信号修复：HTTPReranker 把精排分写入 RetrievedChunk.rerank_score，
  且多路 RRF 合并保留该字段（此前只有覆盖 score 的写法，4 号信号恒 None）；
- knowledge 生产接入：拒绝 → success=False + 空 results（与 degraded 分支同构）；
- 组合语义：联合阈值已配置时替代 legacy 单阈值（不双重砍），全 None 时回落。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.rejection import (
    RejectionParams,
    compute_signals,
    decide_rejection,
    lexical_coverage,
    params_from_settings,
    rejection_reasons,
    should_reject,
)
from app.agent.rag.retriever import SCORE_SOURCE_RRF, SCORE_SOURCE_VECTOR
from app.agent.rag.retriever_factory import _rrf_merge_hits
from app.config.settings import Settings, settings

_FULL = RejectionParams(
    min_top1=0.5, min_gap=0.01, min_coverage=0.2, min_rerank=0.3,
)


def _hit(chunk_id="c1", doc="退货政策", text="七天无理由退货，运费由买家承担",
         score=0.9, parent_id="p1", rerank_score=None):
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc=doc, section="s", text=text,
                    parent_id=parent_id, source_path=f"{doc}.md"),
        score=score,
        rerank_score=rerank_score,
    )


# ------------------------------------------------------------
# 信号与判定
# ------------------------------------------------------------
def test_compute_signals_reads_four_signals():
    hits = [
        _hit("c1", score=0.8, rerank_score=0.7),
        _hit("c2", score=0.6),
    ]
    sig = compute_signals(hits, "七天无理由退货")
    assert sig["top1"] == pytest.approx(0.8)
    assert sig["gap"] == pytest.approx(0.2)
    assert sig["coverage"] > 0.5
    assert sig["rerank"] == pytest.approx(0.7)  # 4 号信号不再恒 None
    assert sig["n_hits"] == 2


def test_compute_signals_empty_hits():
    sig = compute_signals([], "七天无理由退货")
    assert sig["top1"] is None and sig["gap"] is None
    assert sig["coverage"] == 0.0 and sig["rerank"] is None


def test_should_reject_each_signal_triggers():
    base = {"top1": 0.9, "gap": 0.2, "coverage": 0.9, "rerank": 0.9}
    assert should_reject(base, _FULL) is False  # 全通过
    assert should_reject({**base, "top1": 0.4}, _FULL) is True
    assert should_reject({**base, "gap": 0.001}, _FULL) is True
    assert should_reject({**base, "coverage": 0.0}, _FULL) is True
    assert should_reject({**base, "rerank": 0.1}, _FULL) is True
    assert rejection_reasons({**base, "top1": 0.4}, _FULL) == ["top1"]
    assert set(rejection_reasons(
        {**base, "top1": 0.4, "coverage": 0.0}, _FULL,
    )) == {"top1", "coverage"}


def test_should_reject_unconfigured_signals_are_dormant():
    """None = 未校准 → 该信号不判定（gap 缺失/rerank 未挂同理）。"""
    params = RejectionParams(min_top1=0.5)
    assert should_reject(
        {"top1": 0.9, "gap": None, "coverage": 0.0, "rerank": None}, params,
    ) is False
    assert should_reject({"top1": None}, params) is True  # 空结果 fail-closed


def test_lexical_coverage_identical_and_unrelated():
    assert lexical_coverage("七天无理由退货", "七天无理由退货") == pytest.approx(1.0)
    assert lexical_coverage("完全无关", "七天无理由退货") == 0.0


def test_decide_rejection_applicability():
    hits = [_hit(score=0.9)]
    q = "七天无理由退货"
    # 参数未配置 → 不适用（调用方回落 legacy 单阈值）
    assert decide_rejection(hits, q, RejectionParams()).applicable is False
    # RRF 秩融合分无语义 → 不适用
    d = decide_rejection(hits, q, _FULL, score_source=SCORE_SOURCE_RRF)
    assert d.applicable is False and d.rejected is False
    assert d.skip_reason == "score_source=rrf"
    # 降级 → 不适用
    d = decide_rejection(hits, q, _FULL, degraded=True)
    assert d.applicable is False and d.skip_reason == "degraded"
    # 向量分 + 阈值 → 正常判定（全信号通过）
    d = decide_rejection(hits, q, _FULL, score_source=SCORE_SOURCE_VECTOR)
    assert d.applicable is True and d.rejected is False


def test_decide_rejection_no_hits_is_fail_closed():
    d = decide_rejection([], "q", _FULL, score_source=SCORE_SOURCE_VECTOR)
    assert d.applicable is True and d.rejected is True
    assert d.reasons == ("no_hits",)


# ------------------------------------------------------------
# rerank 死信号修复
# ------------------------------------------------------------
def test_http_reranker_populates_rerank_score():
    import httpx

    def handler(request):
        return httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.11},
            {"index": 1, "relevance_score": 0.99},
        ]})

    from app.agent.rag.rerank import HTTPReranker

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = HTTPReranker(
        "cohere", endpoint_url="http://rerank.local/v2/rerank",
        api_key="k", client=client,
    )
    hits = [_hit("c1", score=0.5), _hit("c2", score=0.4)]
    out = reranker.rerank("退货", hits, top_k=2)
    assert out[0].chunk.chunk_id == "c2"
    assert out[0].rerank_score == pytest.approx(0.99)
    assert out[1].rerank_score == pytest.approx(0.11)
    # score 仍被精排分覆盖（既有语义不变），rerank_score 独立留痕
    assert out[0].score == pytest.approx(0.99)
    # 联合拒绝的 4 号信号可读到
    sig = compute_signals(out, "退货")
    assert sig["rerank"] == pytest.approx(0.99)


def test_rrf_merge_keeps_rerank_score():
    """多路 RRF 合并：score 变秩融合分，rerank_score 保留首见命中精排分。"""
    a = _hit("c1", doc="d1", score=0.02, parent_id="p1", rerank_score=0.8)
    b = _hit("c2", doc="d2", score=0.01, parent_id="p2", rerank_score=None)
    merged, queries = _rrf_merge_hits([("q1", [a]), ("q2", [b])], top_n=5)
    by_id = {h.chunk.chunk_id: h for h in merged}
    assert by_id["c1"].rerank_score == pytest.approx(0.8)
    assert by_id["c1"].score < 0.05  # 合并分是 RRF，不是精排分
    assert by_id["c2"].rerank_score is None
    assert queries == ["q1", "q2"]


# ------------------------------------------------------------
# 生产接入：fail-closed 结构 + 组合语义
# ------------------------------------------------------------
def _scripted_retriever(script, score_source=SCORE_SOURCE_VECTOR):
    from app.agent.rag.retriever import RetrievalResult

    class _R:
        model = "fake"
        size = 0

        def load(self):
            pass

        def search_with_status(self, query, top_k=3, timeout=None):
            return RetrievalResult(
                hits=list(script.get(query, []))[:top_k],
                score_source=score_source,
            )

    return _R()


def _set_joint(monkeypatch, **kwargs):
    for name in ("min_top1", "min_gap", "min_coverage", "min_rerank"):
        monkeypatch.setattr(settings, f"rag_rejection_{name}", kwargs.get(name))


def test_search_knowledge_rejected_returns_fail_closed(monkeypatch):
    from app.agent.tools import knowledge as knowledge_mod

    _set_joint(monkeypatch, min_top1=0.9)
    retriever = _scripted_retriever({"七天无理由": [_hit(score=0.6)]})
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge("七天无理由", top_k=3)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is False
    assert result["results"] == []
    assert result["error"].startswith("retrieval_rejected:")
    assert "top1" in result["error"]
    # 被拒证据仍进 diagnostics 审计（results 必须为空）
    assert result["evidence"]["n_items"] == 1


def test_search_knowledge_no_hits_fail_closed(monkeypatch):
    from app.agent.tools import knowledge as knowledge_mod

    _set_joint(monkeypatch, min_top1=0.5)
    knowledge_mod.push_retriever_override(_scripted_retriever({}))
    try:
        result = knowledge_mod.search_knowledge("没有命中的问题", top_k=3)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is False and result["results"] == []
    assert result["error"] == "retrieval_rejected:no_hits"


def test_search_knowledge_accepted_when_all_signals_pass(monkeypatch):
    from app.agent.tools import knowledge as knowledge_mod

    _set_joint(monkeypatch, min_top1=0.5, min_coverage=0.2)
    retriever = _scripted_retriever({"七天无理由": [_hit(score=0.9)]})
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge("七天无理由", top_k=3)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is True
    assert len(result["results"]) == 1


def test_joint_rejection_supersedes_legacy_single_threshold(monkeypatch):
    """组合语义：联合阈值已配置 → legacy 单阈值不参与最终口径（不双重砍）。

    命中 0.6 高于联合 min_top1=0.5、低于 legacy 0.99：必须被接受——证明
    legacy 阈值没有和联合判定硬叠。
    """
    from app.agent.tools import knowledge as knowledge_mod

    _set_joint(monkeypatch, min_top1=0.5)
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.99)
    retriever = _scripted_retriever({"七天无理由": [_hit(score=0.6)]})
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge("七天无理由", top_k=3)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is True and len(result["results"]) == 1


def test_legacy_single_threshold_used_when_joint_unconfigured(monkeypatch):
    """四个联合阈值全 None → 回落 legacy 单阈值（历史行为不变）。"""
    from app.agent.tools import knowledge as knowledge_mod

    _set_joint(monkeypatch)  # 全 None
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.99)
    retriever = _scripted_retriever({"七天无理由": [_hit(score=0.6)]})
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge("七天无理由", top_k=3)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is True and result["results"] == []


def test_rrf_scores_skip_joint_rejection(monkeypatch):
    """RRF 分无语义：即使联合阈值配置了也不得判定（与单阈值纪律一致）。"""
    from app.agent.tools import knowledge as knowledge_mod

    _set_joint(monkeypatch, min_top1=0.9)
    retriever = _scripted_retriever(
        {"七天无理由": [_hit(score=0.01)]}, score_source=SCORE_SOURCE_RRF,
    )
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge("七天无理由", top_k=3)
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is True and len(result["results"]) == 1


def test_params_from_settings_freezes_values(monkeypatch):
    _set_joint(monkeypatch, min_top1=0.549, min_gap=0.0, min_coverage=0.0)
    params = params_from_settings()
    assert params.min_top1 == 0.549
    assert params.active is True


def test_settings_defaults_match_frozen_calibration():
    """settings 默认值 = 冻结校准产物（防止默认值与校准档漂移）。"""
    path = Path("artifacts/eval/v2/retrieval-rejection/params.json")
    if not path.exists():
        pytest.skip("未找到冻结校准产物（CI 无 artifacts 时跳过）")
    frozen = json.loads(path.read_text(encoding="utf-8"))["params"]
    fields = Settings.model_fields
    assert fields["rag_rejection_min_top1"].default == frozen["min_top1"]
    assert fields["rag_rejection_min_gap"].default == frozen.get("min_gap")
    assert fields["rag_rejection_min_coverage"].default == frozen.get("min_coverage")
    assert fields["rag_rejection_min_rerank"].default is None
    # P1-3：query 表层规范化默认启用（能力默认启用，无开关）
    assert fields["rag_query_normalize"].default is True

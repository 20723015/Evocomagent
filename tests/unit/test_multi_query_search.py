"""工作流 B：多子查询补缺（final_multi_search 编排 + 拆分召回 e2e + 提示词教学）。

补齐「类型安全有、功能正确性没有」的缺口：
- 编排：单子查询与 final_search 逐字段等价（回归保护）、多路并行召回 →
  逐路门控（RRF/降级跳过）→ RRF 秩融合 → 父块去重 → Top-K；
- 拆分召回 e2e：迷你双文档库复刻 mixed_1（会员权益 + 退换货运费）——
  单 query 只召回一份，两个子查询 RRF 合并后两份都进 Top-K；
- search_knowledge 全链路：subqueries 回显、evidence 诊断、fail-closed 聚合；
- 提示词教学：SYSTEM_PROMPT 必须教 queries 拆解（防回退断言）。
"""

from __future__ import annotations

import pytest

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.retriever import (
    SCORE_SOURCE_RRF,
    SCORE_SOURCE_VECTOR,
    RetrievalResult,
    collapse_by_parent,
)
from app.agent.rag.retriever_factory import (
    final_multi_search,
    final_search,
)


def _hit(chunk_id: str, doc: str, section: str, score: float,
         parent_id: str = "", text: str = "正文") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc=doc, section=section, text=text,
                    parent_id=parent_id, source_path=f"{doc}.md"),
        score=score,
    )


class _ScriptedRetriever:
    """按 query 返回脚本化命中（含 scores_meaningful / degraded 控制）。"""

    model = "fake-model"
    size = 99

    def __init__(self, script: dict, *, scores_meaningful: bool = True):
        self.script = script
        self.scores_meaningful = scores_meaningful
        self.calls: list[str] = []

    def load(self):
        pass

    def search_with_status(self, query, top_k=3, timeout=None):
        self.calls.append(query)
        hits = list(self.script.get(query, []))[:top_k]
        source = SCORE_SOURCE_VECTOR if self.scores_meaningful else SCORE_SOURCE_RRF
        return RetrievalResult(hits=hits, score_source=source)


# ------------------------------------------------------------
# 编排语义
# ------------------------------------------------------------
def test_single_subquery_matches_final_search():
    retriever = _ScriptedRetriever({
        "退款时效": [_hit("c1", "退款到账时效分档表", "档位", 0.9,
                    parent_id="p1"),
                    _hit("c2", "退款到账时效分档表", "大额", 0.8,
                    parent_id="p1")],
    })
    multi = final_multi_search(retriever, ["退款时效"], top_k=2)
    single = final_search(retriever, "退款时效", top_k=2)
    # 单子查询路径 = final_search：命中/键/计数逐字段一致（回归保护）
    assert [h.chunk.chunk_id for h in multi.hits] == \
        [h.chunk.chunk_id for h in single.hits]
    assert multi.kept_keys == single.kept_keys
    assert multi.raw_candidates == single.raw_candidates
    assert multi.gated_candidates == single.gated_candidates
    assert multi.score_source == single.score_source == SCORE_SOURCE_VECTOR
    assert multi.hit_queries == ["退款时效"] * len(multi.hits)


def test_multi_route_merges_and_attributes_queries():
    """两路召回 RRF 合并：双份期望文档都进 Top-K，归因各归各路。"""
    retriever = _ScriptedRetriever({
        "钻石会员运费特权": [_hit("m1", "会员权益", "运费", 0.9, parent_id="pm")],
        "退换货运费承担": [_hit("r1", "退换货政策", "运费", 0.9, parent_id="pr")],
    })
    outcome = final_multi_search(
        retriever, ["钻石会员运费特权", "退换货运费承担"], top_k=5,
    )
    parents = {h.chunk.parent_id for h in outcome.hits}
    assert parents == {"pm", "pr"}  # 单 query 漏掉的那份被第二路补上
    assert outcome.score_source == SCORE_SOURCE_RRF
    assert outcome.hit_queries == ["钻石会员运费特权", "退换货运费承担"]
    assert outcome.raw_candidates == 2


def test_parent_dedup_across_routes_keeps_first_seen():
    """同 parent_id 跨路去重：保留首见，归因为先提交的子查询。"""
    retriever = _ScriptedRetriever({
        "会员运费": [_hit("m1", "会员权益", "运费", 0.9, parent_id="p")],
        "运费特权": [_hit("m2", "会员权益", "运费细则", 0.8, parent_id="p")],
    })
    outcome = final_multi_search(retriever, ["会员运费", "运费特权"], top_k=5)
    assert len(outcome.hits) == 1
    assert outcome.hits[0].chunk.chunk_id == "m1"
    assert outcome.hit_queries == ["会员运费"]
    assert outcome.collapsed_parents == 1  # 两路命中同属一个父块


def test_hit_queries_parallel_to_hits_after_collapse():
    """review 修复：hit_queries 与 hits 严格平行——父块折叠截断后同步截断。

    归因 = 首见子查询（p1 双路命中归先提交的一路），截断不产生悬空的归因尾巴。
    """
    retriever = _ScriptedRetriever({
        "会员运费": [_hit("m1", "会员权益", "运费", 0.9, parent_id="p1"),
                    _hit("m2", "会员权益", "积分", 0.8, parent_id="p2"),
                    _hit("m3", "会员权益", "无忧退", 0.7, parent_id="p3")],
        "退货运费": [_hit("r1", "退换货政策", "运费", 0.9, parent_id="p1"),
                    _hit("r2", "退换货政策", "时效", 0.5, parent_id="p4")],
    })
    outcome = final_multi_search(retriever, ["会员运费", "退货运费"], top_k=3)
    assert len(outcome.hits) == 3
    assert len(outcome.hit_queries) == len(outcome.hits)
    # p1 两路命中：融合分最高，归因先提交的「会员运费」
    assert outcome.hits[0].chunk.parent_id == "p1"
    assert outcome.hit_queries[0] == "会员运费"
    by_parent = {h.chunk.parent_id: q
                 for h, q in zip(outcome.hits, outcome.hit_queries)}
    assert "p3" not in by_parent  # 被截断的父块不带悬空归因
    assert by_parent["p4"] == "退货运费"


def test_route_gating_semantics():
    """每路门控：有语义分数才过滤；RRF 分（scores_meaningful=False）跳过门控。"""
    low = _ScriptedRetriever({
        "q1": [_hit("c1", "doc", "s", 0.01)],
        "q2": [_hit("c2", "doc2", "s", 0.02)],
    })
    # 有语义分数 → 逐路门控生效：0.01/0.02 < 0.99 全部被过滤
    outcome = final_multi_search(low, ["q1", "q2"], top_k=5, min_score=0.99)
    assert outcome.hits == []
    assert outcome.gated_candidates == 0

    rrf = _ScriptedRetriever(
        {"q1": [_hit("c1", "doc", "s", 0.0)],
         "q2": [_hit("c2", "doc2", "s", 0.0)]},
        scores_meaningful=False,
    )
    outcome = final_multi_search(rrf, ["q1", "q2"], top_k=5, min_score=0.99)
    assert [h.chunk.chunk_id for h in outcome.hits] == ["c1", "c2"]
    assert outcome.score_source == SCORE_SOURCE_RRF


def test_multi_route_with_meaningful_scores_gates():
    """精排分（有绝对语义）低于阈值 → 该路候选被过滤。"""
    retriever = _ScriptedRetriever({
        "q1": [_hit("c1", "doc", "s", 0.5)],
        "q2": [_hit("c2", "doc2", "s", 0.01)],
    })
    outcome = final_multi_search(retriever, ["q1", "q2"], top_k=5, min_score=0.3)
    assert [h.chunk.chunk_id for h in outcome.hits] == ["c1"]


def test_rrf_score_math():
    """RRF 融合分 = Σ 1/(60+rank+1)；两路都命中者分数更高。"""
    retriever = _ScriptedRetriever({
        "qa": [_hit("c1", "d1", "s", 0.9, parent_id="p1"),
               _hit("c2", "d2", "s", 0.8, parent_id="p2")],
        "qb": [_hit("c3", "d3", "s", 0.7, parent_id="p3"),
               _hit("c2", "d2", "s", 0.6, parent_id="p2")],
    })
    outcome = final_multi_search(retriever, ["qa", "qb"], top_k=5)
    by_parent = {h.chunk.parent_id: h.score for h in outcome.hits}
    assert by_parent["p1"] == pytest.approx(1 / 61)
    assert by_parent["p3"] == pytest.approx(1 / 61)
    # c2 在两路里都是 rank 1：1/62 + 1/62
    assert by_parent["p2"] == pytest.approx(1 / 62 + 1 / 62)
    # 双命中者排最前
    assert outcome.hits[0].chunk.parent_id == "p2"


def test_degraded_aggregation_fail_closed_input():
    """任一路降级 → 聚合上报（search_knowledge 据此 fail-closed）。"""
    class _DegradedRetriever(_ScriptedRetriever):
        def search_with_status(self, query, top_k=3, timeout=None):
            result = super().search_with_status(query, top_k, timeout)
            if query == "q2":
                result.degraded = True
                result.degraded_reason = "reranker_unavailable"
            return result

    retriever = _DegradedRetriever({
        "q1": [_hit("c1", "d1", "s", 0.9, parent_id="p1")],
        "q2": [_hit("c2", "d2", "s", 0.9, parent_id="p2")],
    })
    outcome = final_multi_search(retriever, ["q1", "q2"], top_k=5)
    assert outcome.degraded is True
    assert outcome.degraded_reason == "reranker_unavailable"


def test_empty_subqueries_rejected():
    with pytest.raises(ValueError):
        final_multi_search(_ScriptedRetriever({}), [], top_k=3)


# ------------------------------------------------------------
# 拆分召回 e2e（复刻 mixed_1：迷你双文档库 + 真实切分/索引）
# ------------------------------------------------------------
class _HashEmbedder:
    """字符 hash 词袋嵌入：查询与正文的字符重合决定相似度（确定性、无 API）。"""

    model = "fake-model"
    dim = 256

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for ch in text:
            vec[hash((ch, "x")) % self.dim] += 1.0
        return vec

    def encode(self, texts):
        return [self._vec(t) for t in texts]

    def encode_one(self, text, timeout=None):
        return self._vec(text)


@pytest.fixture
def mini_multi_kb(tmp_path):
    """双主题迷你库：会员权益（含运费特权）+ 退换货政策（运费承担方）。"""
    kb = tmp_path / "knowledge"
    kb.mkdir(parents=True)
    (kb / "会员权益.md").write_text(
        "# 会员权益\n\n## 钻石会员\n\n钻石会员享有专属运费特权，"
        "全场商品免邮。\n\n## 会员积分\n\n会员积分可以兑换优惠券。\n",
        encoding="utf-8",
    )
    (kb / "退换货政策.md").write_text(
        "# 退换货政策\n\n## 运费承担\n\n七天无理由退货的运费由买家承担，"
        "质量问题退货运费由商家承担。\n\n## 退货时效\n\n退货需在七天内申请。\n",
        encoding="utf-8",
    )
    return kb


def _build_retriever(kb, tmp_path):
    from app.agent.rag.backends.numpy_backend import NumpyBackend
    from app.agent.rag.parsers import chunk_kb_dir
    from app.agent.rag.retriever import KnowledgeRetriever

    chunks = chunk_kb_dir(kb, strict=False)
    index_path = tmp_path / "kb_index.json"
    backend = NumpyBackend(index_path=index_path)
    vectors = [_HashEmbedder().encode_one(c.index_input()) for c in chunks]
    backend.upsert(
        chunks=chunks, vectors=vectors,
        embedding_model=_HashEmbedder.model,
    )
    retriever = KnowledgeRetriever(
        embedder=_HashEmbedder(), backend=backend,
    )
    retriever.load()
    return retriever


def test_split_subqueries_recall_both_docs(mini_multi_kb, tmp_path):
    """mixed_1 复刻：单 query 漏一份期望文档，拆解后两份都进 Top-K。

    单 query「钻石会员退换货要运费吗」的字符被两篇文档摊薄，top-2 被
    会员文档的父块占满；拆成「钻石会员运费特权 / 退换货运费承担方」两路
    RRF 合并后两份文档都进 Top-K（理想拆解口径的召回增益）。
    """

    retriever = _build_retriever(mini_multi_kb, tmp_path)
    single = final_search(
        retriever, "钻石会员退换货要运费吗", top_k=2,
    )
    single_docs = {h.chunk.doc for h in single.hits}
    assert "会员权益" in single_docs  # 单 query 只稳拿第一主题

    outcome = final_multi_search(
        retriever, ["钻石会员退换货要运费吗", "钻石会员运费特权",
                    "退换货运费承担方"],
        top_k=2,
    )
    multi_docs = {h.chunk.doc for h in outcome.hits}
    assert "会员权益" in multi_docs
    assert "退换货政策" in multi_docs  # 拆解补回第二主题
    assert outcome.score_source == SCORE_SOURCE_RRF


def test_split_recall_not_smaller_than_single(mini_multi_kb, tmp_path):
    """拆解召回 ⊇ 单 query 召回（合并是两路候选的超集，父块去重后仍不丢）。"""
    retriever = _build_retriever(mini_multi_kb, tmp_path)
    query = "钻石会员退换货要运费吗"
    single = final_search(retriever, query, top_k=2)
    outcome = final_multi_search(
        retriever, [query, "钻石会员运费特权", "退换货运费承担方"], top_k=2,
    )
    assert {h.chunk.doc for h in outcome.hits} >= \
        {h.chunk.doc for h in single.hits}
    assert collapse_by_parent(outcome.hits, 2) == outcome.hits


# ------------------------------------------------------------
# search_knowledge 全链路 + 提示词教学
# ------------------------------------------------------------
def test_search_knowledge_multi_subqueries_e2e(monkeypatch):
    """queries 参数贯穿工具层：subqueries 回显、RRF 诊断、双文档结果。"""
    from app.agent.tools import knowledge as knowledge_mod

    retriever = _ScriptedRetriever({
        "钻石会员退换货要运费吗": [_hit("m1", "会员权益", "运费", 0.9,
                                parent_id="pm")],
        "钻石会员运费特权": [_hit("m1", "会员权益", "运费", 0.9, parent_id="pm")],
        "退换货运费承担方": [_hit("r1", "退换货政策", "运费", 0.9,
                             parent_id="pr")],
    })
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge(
            "钻石会员退换货要运费吗",
            queries=["钻石会员运费特权", "退换货运费承担方"],
            top_k=5,
        )
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is True
    assert result["subqueries"] == [
        "钻石会员退换货要运费吗", "钻石会员运费特权", "退换货运费承担方",
    ]
    docs = {r["doc"] for r in result["results"]}
    assert docs == {"会员权益", "退换货政策"}
    assert result["evidence"]["score_source"] == SCORE_SOURCE_RRF
    assert result["evidence"]["n_items"] == 2
    # 三路并行都被调用
    assert set(retriever.calls) == set(result["subqueries"])


def test_search_knowledge_subqueries_truncated(monkeypatch):
    """>3 条子查询按线上规则截断（query 居首）。"""
    from app.agent.tools import knowledge as knowledge_mod

    retriever = _ScriptedRetriever({})
    knowledge_mod.push_retriever_override(retriever)
    try:
        result = knowledge_mod.search_knowledge(
            "主查询", queries=["a", "a", "b", "c", "d"], top_k=3,
        )
    finally:
        knowledge_mod.pop_retriever_override()
    assert result["success"] is True
    assert result["subqueries"] == ["主查询", "a", "b"]


def test_system_prompt_teaches_query_decomposition():
    """提示词必须教 queries 拆解（防回退）：工具条目 + 使用原则 + few-shot。"""
    from app.prompts.customer_service import SYSTEM_PROMPT

    assert "queries" in SYSTEM_PROMPT
    assert "拆解" in SYSTEM_PROMPT
    # few-shot 例句（mixed_1 型）
    assert "钻石会员退换货要运费吗" in SYSTEM_PROMPT
    assert "退换货运费承担" in SYSTEM_PROMPT
    # review 修复：三主题例句（每个子查询都来自 query 真实主题）；
    # 旧例句曾发明 query 里不存在的「快递丢件赔偿标准」——防回退
    assert "大促价保怎么算" in SYSTEM_PROMPT
    assert "快递丢件赔偿标准" not in SYSTEM_PROMPT
    assert "不要为了凑数发明" in SYSTEM_PROMPT
    # 与「换表述重试」的关系写明：先拆解、后改写
    assert "先拆解" in SYSTEM_PROMPT


def test_overlay_normalizer_matches_online_rule():
    """overlay 子查询与线上 _normalize_subqueries 同规：去重、query 居首、≤3。"""
    from app.evaluation.retrieval_metrics import normalize_overlay_subqueries

    assert normalize_overlay_subqueries("主查询", ["a", "a", "b", "c"]) == \
        ["主查询", "a", "b"]
    assert normalize_overlay_subqueries("主查询", []) == ["主查询"]
    assert normalize_overlay_subqueries("主查询", ["主查询", "a"]) == \
        ["主查询", "a"]


def test_evaluate_multi_overlay_dual_metrics(monkeypatch):
    """evaluate 的多跳双口径：strict=全命中 / lenient=至少命中一份。"""
    from app.evaluation.retrieval_metrics import evaluate

    retriever = _ScriptedRetriever({
        "主查询": [_hit("m1", "会员权益", "运费", 0.9, parent_id="pm")],
        "补充a": [_hit("r1", "退换货政策", "运费", 0.9, parent_id="pr")],
    })
    cases = [{
        "id": "case_multi", "query": "主查询",
        "expected": ["会员权益.md", "退换货政策.md"], "k": 5,
        "tags": ["hard"],
    }]
    report = evaluate(
        cases, retriever, top_k=5,
        multi_query_overlay={"case_multi": ["补充a"]},
    )
    case = report["cases"][0]
    assert case["multi"] is True
    assert case["subqueries"] == ["主查询", "补充a"]
    assert case["recall_at_k"] == 1.0
    summary = report["summary"]["multi_hop"]
    assert summary["cases"] == 1
    assert summary["strict_rate"] == 1.0
    assert summary["lenient_rate"] == 1.0

    # 不带 overlay：同用例走单 query 口径，multi=False、第二份漏召
    report_single = evaluate(cases, retriever, top_k=5)
    assert report_single["cases"][0]["multi"] is False
    assert report_single["cases"][0]["recall_at_k"] == pytest.approx(0.5)
    assert report_single["summary"]["multi_hop"]["cases"] == 0


def test_evaluate_records_normalized_query(tmp_path):
    """工作流 A 报告留痕：per-case normalized_query（仅变化时非空）+ 汇总计数。

    normalizer 只记录不参与检索——检索路径的规范化在检索器内部（此处脚本化
    替身直接按原 query 返回脚本），evaluate 侧的字段是审计口径。
    """
    import json as _json

    from app.agent.rag.query_normalizer import QueryNormalizer
    from app.evaluation.retrieval_metrics import evaluate

    lex = {
        "protocol": "rag-query-lexicon-v1",
        "terms": ["积分"],
        "term_freq": {"积分": 100},
        "char_syllables": {"积": "ji", "分": "fen", "份": "fen"},
        "guard_ngrams": [],
        "source_generation": {},
    }
    path = tmp_path / "lex.json"
    path.write_text(_json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    normalizer = QueryNormalizer.load(path)
    assert normalizer is not None

    retriever = _ScriptedRetriever({
        "积份能换吗": [_hit("c1", "积分规则", "兑换", 0.9, parent_id="p1")],
        "正常问题": [_hit("c2", "其他", "s", 0.9, parent_id="p2")],
    })
    cases = [
        {"id": "a", "query": "积份能换吗", "expected": ["积分规则.md"],
         "k": 5, "tags": ["hard", "typo"]},
        {"id": "b", "query": "正常问题", "expected": ["其他.md"],
         "k": 5, "tags": ["easy", "direct"]},
    ]
    report = evaluate(cases, retriever, top_k=5, query_normalizer=normalizer)
    assert report["cases"][0]["normalized_query"] == "积分能换吗"
    assert report["cases"][1]["normalized_query"] is None
    assert report["summary"]["query_normalize_rewrites"] == 1

    # 不装配 normalizer：字段恒 None、不报错（基线臂报告形态不变）
    report_base = evaluate(cases, retriever, top_k=5)
    assert report_base["cases"][0]["normalized_query"] is None
    assert report_base["summary"]["query_normalize_rewrites"] == 0

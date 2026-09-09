"""阶段 7.2/7.3 单测：BM25 分词与排序、HybridRetriever RRF 融合、Reranker 扩展点。

全部使用 conftest 的 FakeEmbedder/FakeBackend/FakeRetriever，无网络。
chroma 用例在 chromadb 可导入时执行，否则 pytest.skip。
"""

from __future__ import annotations

import math

import pytest

from conftest import FakeEmbedder, FakeRetriever, chromadb_usable  # noqa: F401  (fixture 经参数注入)
from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.backends.numpy_backend import NumpyBackend
from app.agent.rag.bm25 import BM25Index, tokenize
from app.agent.rag.chunker import Chunk
from app.agent.rag.hybrid import HybridRetriever
from app.agent.rag.rerank import NullReranker, Reranker, create_reranker


class _FakeBackend(VectorBackend):
    """内存版 VectorBackend（本文件自建，避免改动 conftest.FakeBackend）。

    语义与 conftest.FakeBackend 一致：chunk_id → (chunk, vector)，
    额外实现新增抽象方法 chunks()，供 BM25 建索引。
    """

    def __init__(self, embedding_model: str = "fake-embedder"):
        self._data: dict[str, tuple[Chunk, list[float]]] = {}
        self._embedding_model = embedding_model

    def upsert(self, chunks, vectors, embedding_model) -> None:
        self._data = {
            c.chunk_id: (c, list(vec))
            for c, vec in zip(chunks, vectors)
        }
        self._embedding_model = embedding_model

    def search(self, query_vector, top_k):
        def _cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = [
            (chunk, _cosine(query_vector, vec))
            for chunk, vec in self._data.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievedChunk(chunk=c, score=s) for c, s in scored[:top_k]]

    def size(self) -> int:
        return len(self._data)

    def load(self) -> None:
        pass  # 内存版无需加载

    def expected_embedding_model(self) -> str:
        return self._embedding_model

    def chunks(self) -> list[Chunk]:
        return [c for c, _ in self._data.values()]


# ============================================================
# 构造辅助
# ============================================================
def _make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", doc=f"doc{i}", section="s", text=t,
              source_path=f"d{i}.md")
        for i, t in enumerate(texts)
    ]


def _build_hybrid(texts: list[str], reranker=None):
    """同一批 chunk 同时建向量路（FakeEmbedder 等价文本）与 BM25 路。"""
    embedder = FakeEmbedder()
    chunks = _make_chunks(texts)
    backend = _FakeBackend(embedding_model=embedder.model)
    backend.upsert(chunks, embedder.encode([c.text for c in chunks]), embedder.model)
    vector_retriever = FakeRetriever(embedder, backend)
    hybrid = HybridRetriever(
        vector_retriever, BM25Index(chunks), recall_k=30, reranker=reranker
    )
    return hybrid, chunks


# ============================================================
# tokenize：中文 bigram / 数字 / 英文 / 空串
# ============================================================
def test_tokenize_chinese_bigram():
    # 连续 CJK → 二元组（"退货政策" 注音：tuì huò zhèng cè）
    assert tokenize("退货政策") == ["退货", "货政", "政策"]
    assert tokenize("七天无理由") == ["七天", "天无", "无理", "理由"]
    # 落单的单个 CJK 保留单字；二元组不跨标点/空白
    assert tokenize("退") == ["退"]
    assert tokenize("退货-政策") == ["退货", "政策"]


def test_tokenize_digits_and_english():
    # 数字/字母按连续串切分并统一小写，与 CJK 混合时各自成词
    assert tokenize("7天无理由") == ["7", "天无", "无理", "理由"]
    assert tokenize("SKU123 号") == ["sku123", "号"]
    assert tokenize("Apple发货") == ["apple", "发货"]


def test_tokenize_empty_and_punctuation():
    assert tokenize("") == []
    assert tokenize("  ，。！？\n\t ") == []


# ============================================================
# BM25 排序与边界
# ============================================================
def test_bm25_ranks_doc_with_most_query_terms_top1():
    # 三段中文，仅第二段含"七天无理由退货"全部查询词
    texts = [
        "会员积分每天签到可以累积，年底清零，请及时兑换。",
        "支持七天无理由退货，运费由商家全额承担，顾客无需支付任何费用。",
        "商品发出后 48 小时可在订单页查询物流信息，偏远地区不包邮。",
    ]
    chunks = _make_chunks(texts)
    index = BM25Index(chunks)
    hits = index.search("七天无理由退货", top_k=3)
    assert hits  # 不足 top_k 时全返回
    assert hits[0].chunk.chunk_id == chunks[1].chunk_id
    # 词面零重合的文档（BM25 得 0 分）不进结果列表
    assert len(hits) == 1
    assert hits[0].score > 0


def test_bm25_empty_query_and_corpus():
    assert BM25Index([]).search("七天无理由退货", top_k=3) == []
    index = BM25Index(_make_chunks(["支持七天无理由退货，运费由商家承担。"]))
    assert index.search("", top_k=3) == []
    assert index.search("   ", top_k=3) == []
    assert len(index.chunks) == 1  # chunks 属性


# ============================================================
# HybridRetriever：RRF 融合
# ============================================================
def test_hybrid_rrf_ranks_two_path_hit_above_single_path():
    # c0：两路都命中且都排第 1（词面全中 + 共享字符最多）
    # c2：两路都命中但弱（仅"退货"一个词面重合）
    # c1：仅向量路命中（词面零重合，BM25 0 分不进结果）
    texts = [
        "七天无理由退货，支持全额退款，运费由商家承担。",
        "会员积分可以兑换优惠券，签到还能累积积分。",
        "退货运费由顾客承担，改价需联系人工客服。",
    ]
    hybrid, _ = _build_hybrid(texts)
    hits = hybrid.search("七天无理由退货", top_k=3)
    ids = [h.chunk.chunk_id for h in hits]
    # RRF：c0 两路第 1 → 1/61 + 1/61；c2 两路第 2 → 1/62 + 1/62；c1 单路 → 1/63
    assert ids == ["c0", "c2", "c1"]
    # 明确断言：两路都命中的 c0 排名高于仅一路命中的 c1
    assert ids.index("c0") < ids.index("c1")


# ============================================================
# Reranker 扩展点
# ============================================================
class _ReverseReranker(Reranker):
    """假精排器：故意翻转融合结果的顺序，验证 reranker 确实生效。"""

    def rerank(self, query: str, hits: list[RetrievedChunk], top_k: int,
                timeout: float | None = None) -> list[RetrievedChunk]:
        return list(reversed(hits))[:top_k]


def test_reranker_reverses_order_and_null_keeps_order():
    texts = [
        "七天无理由退货，支持全额退款，运费由商家承担。",
        "会员积分可以兑换优惠券，签到还能累积积分。",
        "退货运费由顾客承担，改价需联系人工客服。",
    ]
    base, _ = _build_hybrid(texts)
    base_order = [h.chunk.chunk_id for h in base.search("七天无理由退货", top_k=3)]
    assert base_order == ["c0", "c2", "c1"]  # 无精排时的融合基线

    # 传入翻转顺序的假 reranker → 结果顺序被改变
    reversed_hybrid, _ = _build_hybrid(texts, reranker=_ReverseReranker())
    got = [h.chunk.chunk_id for h in reversed_hybrid.search("七天无理由退货", top_k=3)]
    assert got == list(reversed(base_order))

    # NullReranker → 保序
    null_hybrid, _ = _build_hybrid(texts, reranker=NullReranker())
    got = [h.chunk.chunk_id for h in null_hybrid.search("七天无理由退货", top_k=3)]
    assert got == base_order


def test_create_reranker_factory():
    from app.agent.rag.rerank import HTTPReranker

    assert isinstance(create_reranker("none"), NullReranker)
    assert isinstance(create_reranker(""), NullReranker)
    # 7.3：bge 自部署/cohere/jina 已在工厂注册（HTTPReranker，构造零网络）
    assert isinstance(create_reranker("bge-reranker-v2-m3"), HTTPReranker)
    assert isinstance(create_reranker("cohere"), HTTPReranker)
    assert isinstance(create_reranker("jina"), HTTPReranker)
    with pytest.raises(NotImplementedError):
        create_reranker("no-such-reranker")


# ============================================================
# 后端 chunks()：BM25 建索引用
# ============================================================
def test_numpy_backend_chunks_matches_upsert(tmp_path):
    embedder = FakeEmbedder()
    chunks = _make_chunks(["支持七天无理由退货。", "会员积分可兑换优惠券。"])
    index_path = tmp_path / "kb_index.json"
    backend = NumpyBackend(index_path)
    backend.upsert(chunks, embedder.encode([c.text for c in chunks]), embedder.model)

    got = backend.chunks()
    assert [c.chunk_id for c in got] == [c.chunk_id for c in chunks]
    assert [c.text for c in got] == [c.text for c in chunks]

    # 新实例：磁盘有索引 → 懒加载；无索引文件 → 返回 []
    reloaded = NumpyBackend(index_path)
    assert [c.chunk_id for c in reloaded.chunks()] == [c.chunk_id for c in chunks]
    fresh = NumpyBackend(tmp_path / "nope" / "kb_index.json")
    assert fresh.chunks() == []


def test_chroma_backend_chunks(tmp_path, chromadb_usable):
    # chromadb_usable：不可导入或 Rust 核心崩溃（0xC0000005）时跳过
    from app.agent.rag.backends.chroma_backend import ChromaBackend

    embedder = FakeEmbedder()
    chunks = _make_chunks(["支持七天无理由退货，运费由商家承担。", "会员积分可兑换优惠券。"])
    backend = ChromaBackend(tmp_path / "chroma", collection_name="ecom_kb_test")
    backend.upsert(chunks, embedder.encode([c.text for c in chunks]), embedder.model)

    got = {c.chunk_id: c for c in backend.chunks()}
    assert set(got) == {"c0", "c1"}
    for c in chunks:
        assert got[c.chunk_id].doc == c.doc
        assert got[c.chunk_id].section == c.section
        assert got[c.chunk_id].text == c.text
        # 元数据随 upsert 持久化 → 重建时完整回填（旧 collection 缺失的字段为空串）
        assert got[c.chunk_id].source_path == c.source_path

    # collection 未加载 → 返回 []
    assert ChromaBackend(tmp_path / "chroma").chunks() == []
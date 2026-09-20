"""工作流 A：query 表层规范化（词表构建产物 + 运行时规范化器 + 装配/变体隔离）。

覆盖：
- 词表加载协议（protocol 校验 / 缺失损坏 fail-open）；
- 同音错字还原（积份→积分、运废险→运费险、手续废→手续费）；
- 拼音缩写还原（YFX→运费险、店铺q→店铺券、LP卡→礼品卡）；
- 最长匹配 / 替换数预算 / 真词护栏 / 保护区（真词跨界改写）；
- 缩写字母串原子性（YFX 不被两个词切开）与同长度多候选择优；
- 装配：RetrievalConfig.query_normalize → open_retriever 注入规范化器，
  规范化后的 query 同时到达向量路与 BM25 路；
- 评测 A/B 臂隔离：既有三变体显式关规范化，norm 臂开启并可恢复。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.rag.query_normalizer import (
    KIND_ABBR,
    KIND_HOMOPHONE,
    QueryNormalizer,
    load_normalizer,
    normalize_query,
)

# 词表夹具：小而精，覆盖首字母/同音/护栏的全部判定路径。
# 注意：错字端字符（废/份/宽/静/飘）刻意**不在语料里出现**——这正是线上
# 真实分布（错字恰恰是 KB 里没有的字），音节表必须覆盖它们才能命中。
_LEXICON = {
    "protocol": "rag-query-lexicon-v1",
    "source_generation": {
        "kb_dir": "test",
        "corpus_sha256": "0" * 64,
        "min_freq": 80,
        "guard_min_freq": 5,
    },
    "terms": ["积分", "运费险", "手续费", "退款", "价保", "店铺券", "运费",
              "礼品卡", "微信", "价保申请", "先用后付"],
    "term_freq": {"积分": 391, "运费险": 346, "手续费": 101, "退款": 2371,
                  "价保": 959, "店铺券": 212, "运费": 1556, "礼品卡": 46,
                  "微信": 94, "价保申请": 51, "先用后付": 116},
    "char_syllables": {
        "积": "ji", "分": "fen", "份": "fen", "运": "yun", "费": "fei",
        "废": "fei", "险": "xian", "手": "shou", "续": "xu",
        "退": "tui", "款": "kuan", "宽": "kuan", "价": "jia", "保": "bao",
        "家": "jia", "店": "dian", "铺": "pu", "券": "quan", "礼": "li",
        "品": "pin", "卡": "ka", "微": "wei", "信": "xin", "威": "wei",
        "胁": "xie", "先": "xian", "用": "yong", "后": "hou", "付": "fu",
        "富": "fu", "申": "shen", "请": "qing",
    },
    "guard_ngrams": ["大促价", "退款的时候", "店庆"],
    "term_full_pinyin": {},
}


@pytest.fixture
def lexicon_path(tmp_path) -> str:
    payload = json.loads(json.dumps(_LEXICON, ensure_ascii=False))
    payload["char_syllables"] = {
        k: v for k, v in payload["char_syllables"].items() if v
    }
    path = tmp_path / "query_lexicon.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def normalizer(lexicon_path) -> QueryNormalizer:
    normalizer = QueryNormalizer.load(lexicon_path)
    assert normalizer is not None
    return normalizer


# ------------------------------------------------------------
# 加载协议
# ------------------------------------------------------------
def test_load_missing_file_returns_none(tmp_path):
    assert QueryNormalizer.load(tmp_path / "nope.json") is None


def test_load_bad_protocol_returns_none(tmp_path):
    path = tmp_path / "lex.json"
    path.write_text(json.dumps({"protocol": "other"}), encoding="utf-8")
    assert QueryNormalizer.load(path) is None


def test_load_corrupt_json_returns_none(tmp_path):
    path = tmp_path / "lex.json"
    path.write_text("{not json", encoding="utf-8")
    assert QueryNormalizer.load(path) is None


def test_load_normalizer_fail_open(monkeypatch, tmp_path):
    """fail-open：装配入口对缺失词表返回 None 并打点，绝不抛异常。"""
    recorded = []
    from app.observability import metrics

    monkeypatch.setattr(
        metrics, "record_query_normalize_missing",
        lambda path: recorded.append(path), raising=False,
    )
    assert load_normalizer(tmp_path / "nope.json", enabled=True) is None
    assert len(recorded) == 1
    # 开关关闭：不打点、不加载
    assert load_normalizer(tmp_path / "nope.json", enabled=False) is None
    assert len(recorded) == 1


def test_snapshot_and_meta(normalizer):
    snap = normalizer.snapshot()
    assert snap["protocol"] == "rag-query-lexicon-v1"
    assert snap["num_terms"] > 0
    assert snap["corpus_sha256"] == "0" * 64
    assert normalizer.meta["min_freq"] == 80


# ------------------------------------------------------------
# 改写判定
# ------------------------------------------------------------
def test_homophone_rewrites(normalizer):
    assert normalizer.normalize("积份放多久过期清零").text == "积分放多久过期清零"
    assert normalizer.normalize("运废险最高能赔多少").text == "运费险最高能赔多少"
    assert normalizer.normalize("分期手续废要几个点").text == "分期手续费要几个点"
    assert normalizer.normalize("退宽怎么这么久").text == "退款怎么这么久"
    kinds = normalizer.normalize("积份").kinds()
    assert kinds == [KIND_HOMOPHONE]


def test_abbr_rewrites(normalizer):
    assert normalizer.normalize("下单时没买YFX现在能补吗").text == \
        "下单时没买运费险现在能补吗"
    r = normalizer.normalize("店铺q能退吗")
    assert r.text == "店铺券能退吗"
    assert r.kinds() == [KIND_ABBR]
    assert normalizer.normalize("LP卡能提现不").text == "礼品卡能提现不"
    assert normalizer.normalize("先用后f额度多少").text == "先用后付额度多少"


def test_mixed_kind(normalizer):
    # 微信：微 精确 + x 是「信」的首字母 → 纯 abbr；威胁：微→威 同音 + x→胁
    # 首字母 → mixed。同长度多候选取替换数最少者（1 < 2）→ 必须选微信。
    r = normalizer.normalize("微x里能看啥")
    assert "微信" in r.text
    assert "威胁" not in r.text


def test_longest_match_wins(normalizer):
    # 价保申请（4 字）优先于 价保（2 字）
    assert normalizer.normalize("价b申请入口").text == "价保申请入口"


def test_homophone_budget_rejects_whole_word_drift(normalizer):
    # 分积：两个字都是同音替换（2 > max(1, 2//2)=1）→ 拒绝改写
    assert normalizer.normalize("分积").text == "分积"


def test_single_char_query_untouched(normalizer):
    assert normalizer.normalize("险").text == "险"


def test_guard_blocks_attested_span(normalizer):
    # 命中片段本身是高频真词 → 不改写
    assert normalizer.normalize("店庆有活动吗").text == "店庆有活动吗"
    # 已是规范词 → 原样
    assert normalizer.normalize("积分能换券吗").text == "积分能换券吗"


def test_protected_region_allows_boundary_rewrite(normalizer):
    """「大促价」是真词（保护区），但跨出边界的「价b→价保」仍要生效：
    保护区内的字符只许原样出现在改写目标里。"""
    r = normalizer.normalize("大促价b能保几天")
    assert r.text == "大促价保能保几天"


def test_ascii_run_atomicity(normalizer):
    """缩写字母串是整体：YF/X 不允许分属两个词（送运费+性能 必须输给 运费险）。"""
    lexicon = json.loads(json.dumps(_LEXICON, ensure_ascii=False))
    lexicon["char_syllables"] = {k: v for k, v in lexicon["char_syllables"].items() if v}
    lexicon["terms"] = _LEXICON["terms"] + ["送运费", "性能"]
    lexicon["term_freq"] = {**_LEXICON["term_freq"], "送运费": 20, "性能": 75}
    lexicon["char_syllables"]["送"] = "song"
    lexicon["char_syllables"]["性"] = "xing"
    lexicon["char_syllables"]["能"] = "neng"
    path = Path(normalizer.meta.get("kb_dir", "x")).parent / "unused"
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8",
    ) as fh:
        json.dump(lexicon, fh, ensure_ascii=False)
        path = fh.name
    nz = QueryNormalizer.load(path)
    assert nz is not None
    assert nz.normalize("商家送YFX能赔多少").text == "商家送运费险能赔多少"


def test_frequency_tiebreak(normalizer):
    """同长度同替换数的多候选：语料词频高者胜（价保 959 > 价不 56）。"""
    lexicon = json.loads(json.dumps(_LEXICON, ensure_ascii=False))
    lexicon["char_syllables"] = {k: v for k, v in lexicon["char_syllables"].items() if v}
    lexicon["terms"] = _LEXICON["terms"] + ["价不"]
    lexicon["term_freq"] = {**_LEXICON["term_freq"], "价不": 56}
    lexicon["char_syllables"]["不"] = "bu"
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8",
    ) as fh:
        json.dump(lexicon, fh, ensure_ascii=False)
        path = fh.name
    nz = QueryNormalizer.load(path)
    assert nz is not None
    assert nz.normalize("价b能保几天").text == "价保能保几天"


def test_would_rewrite(normalizer):
    assert normalizer.would_rewrite("积份") is True
    assert normalizer.would_rewrite("积分") is False   # 已是规范词
    assert normalizer.would_rewrite("大促价") is False  # 护栏真词


def test_normalize_query_helper(normalizer):
    assert normalize_query("任意 query", None) == "任意 query"
    assert normalize_query("积份", normalizer) == "积分"


def test_metrics_recorded_on_rewrite(monkeypatch, normalizer):
    recorded: list[list[str]] = []
    from app.observability import metrics

    monkeypatch.setattr(
        metrics, "record_query_normalize_hits",
        lambda kinds: recorded.append(list(kinds)), raising=False,
    )
    normalizer.normalize("积份能换券吗")
    assert recorded == [["homophone"]]
    # 无改写不打点
    normalizer.normalize("积分能换券吗")
    assert len(recorded) == 1


# ------------------------------------------------------------
# 装配（open_retriever）与变体隔离
# ------------------------------------------------------------
class _CapturingVectorRetriever:
    """记录收到的 query；按 query 是否包含「运费险」返回 scripted 命中。"""

    model = "fake-model"
    size = 2

    def __init__(self):
        self.queries: list[str] = []

    def load(self):
        pass

    @property
    def backend(self):
        return self

    def chunks(self):
        from app.agent.rag.chunker import Chunk

        return [
            Chunk(chunk_id="c1", doc="运费险细则", section="赔付",
                  text="运费险赔付额度说明", source_path="运费险细则.md"),
            Chunk(chunk_id="c2", doc="配送说明", section="时效",
                  text="配送时效说明", source_path="配送说明.md"),
        ]

    def search(self, query, top_k=3, timeout=None):
        self.queries.append(query)
        from app.agent.rag.backends.base import RetrievedChunk
        from app.agent.rag.chunker import Chunk

        hits = []
        if "运费险" in query:
            hits.append(RetrievedChunk(
                chunk=Chunk(chunk_id="c1", doc="运费险细则", section="赔付",
                            text="运费险赔付额度说明",
                            source_path="运费险细则.md"),
                score=0.9,
            ))
        hits.append(RetrievedChunk(
            chunk=Chunk(chunk_id="c2", doc="配送说明", section="时效",
                        text="配送时效说明", source_path="配送说明.md"),
            score=0.1,
        ))
        return hits[:top_k]


class _CapturingBM25:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, top_k=3):
        self.queries.append(query)
        return []


def test_open_retriever_wires_normalizer(lexicon_path, tmp_path):
    """query_normalize=True → 规范化后的 query 同达向量路与 BM25 路。"""
    from app.agent.rag.backends.numpy_backend import NumpyBackend
    from app.agent.rag.retriever_factory import RetrievalConfig, open_retriever

    # 造一个空 numpy 索引，工厂才能走完整装配路径
    NumpyBackend(index_path=Path(tmp_path / "kb_index.json")).upsert(
        chunks=[], vectors=[], embedding_model="fake-model",
    )
    config = RetrievalConfig(
        backend="numpy", hybrid=True, rerank="none",
        query_normalize=True, query_normalize_lexicon_path=lexicon_path,
        kb_index_path=str(tmp_path / "kb_index.json"),
    )
    vector = _CapturingVectorRetriever()
    bm25 = _CapturingBM25()
    from app.agent.rag.hybrid import HybridRetriever

    # 直接装配 HybridRetriever，验证规范化在检索入口同时改写两路的 query
    normalizer = QueryNormalizer.load(lexicon_path)
    assert normalizer is not None
    retriever = HybridRetriever(
        vector_retriever=vector, bm25=bm25, recall_k=5,
        reranker=None, normalizer=normalizer,
    )
    retriever.search("YFX最高能赔多少", top_k=2)
    assert vector.queries[-1] == "运费险最高能赔多少"  # 缩写已还原
    assert bm25.queries[-1] == "运费险最高能赔多少"

    # 工厂装配断言：开关开 → 注入；开关关 → None（线上默认零开销）
    retriever_on = open_retriever(config, embedder=_CapturingVectorRetriever())
    assert retriever_on._normalizer is not None
    config_off = RetrievalConfig(
        backend="numpy", hybrid=True, rerank="none",
        kb_index_path=str(tmp_path / "kb_index.json"),
    )
    retriever_off = open_retriever(
        config_off, embedder=_CapturingVectorRetriever(),
    )
    assert retriever_off._normalizer is None


def test_eval_variant_arm_isolation():
    """A/B 臂隔离：既有三变体显式关规范化；restore 还原到进入前状态。"""
    from app.config.settings import settings
    from app.scripts.run_retrieval_eval import _apply_variant, _restore_variant

    baseline = {
        "rag_hybrid": settings.rag_hybrid,
        "rag_rerank": settings.rag_rerank,
        "rag_query_normalize": settings.rag_query_normalize,
    }
    try:
        settings.rag_query_normalize = True  # 模拟环境残留，变体必须显式覆盖
        for variant in ("knn", "hybrid", "hybrid-rerank"):
            prev = _apply_variant(variant)
            assert settings.rag_query_normalize is False, variant
            _restore_variant(prev)
            assert settings.rag_query_normalize is True  # 还原为进入前
        prev = _apply_variant("hybrid-rerank-norm")
        assert settings.rag_hybrid is True
        assert settings.rag_rerank == "bge-reranker-v2-m3"
        assert settings.rag_query_normalize is True
        _restore_variant(prev)
        assert settings.rag_query_normalize is True  # 回到进入前
    finally:
        settings.rag_hybrid = baseline["rag_hybrid"]
        settings.rag_rerank = baseline["rag_rerank"]
        settings.rag_query_normalize = baseline["rag_query_normalize"]

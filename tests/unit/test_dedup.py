"""dedup：精确 / 预去重 0.95 / 最终去重 0.9 / 本轮互查 + 容差。"""

from __future__ import annotations

from conftest import FakeBackend, FakeEmbedder, FakeRetriever
from app.agent.rag.chunker import Chunk
from app.evolution import dedup
from app.evolution.ledger import Ledger
from app.evolution.models import CandidateQA


def _chunk(chunk_id, text, doc="退货政策", section="七天无理由", source_path="退货政策.md"):
    return Chunk(chunk_id=chunk_id, doc=doc, section=section, text=text,
                 source_path=source_path)


def test_ge_with_tolerance():
    assert dedup.ge_with_tolerance(0.9, 0.9)
    assert dedup.ge_with_tolerance(0.9, 0.9 - 1e-5)
    assert not dedup.ge_with_tolerance(0.89, 0.9)


def test_exact_consults_ledger(tmp_state_dir):
    ledger = Ledger(tmp_state_dir["state"])
    assert not ledger.contains("c1")
    ledger.add_pending(CandidateQA(candidate_id="c1", turn_id="t1",
                                   question="q", answer="a" * 25))
    assert dedup.exact("c1", ledger)
    ledger.reject("c1")
    assert dedup.exact("c1", ledger)  # 永久跳过


def test_pre_dedup_similar_question():
    embedder = FakeEmbedder()
    backend = FakeBackend()
    retriever = FakeRetriever(embedder, backend)
    text = "七天无理由退货可以吗？"
    backend.upsert(
        chunks=[_chunk("退货政策#00-00", text)],
        vectors=embedder.encode([text]),
        embedding_model=embedder.model,
    )
    # 完全相同 → 0.95 命中；无关问题 → 不命中
    assert dedup.pre_dedup("七天无理由退货可以吗？", retriever, threshold=0.95)
    assert not dedup.pre_dedup("你们几点下班", retriever, threshold=0.95)
    # 近似改写（差标点）在更低阈值命中 —— 判别方向正确
    assert dedup.pre_dedup("七天无理由退货可以吗", retriever, threshold=0.7)


def test_final_dedup_question_and_answer_sides():
    """最终去重：问题、答案分别检索，任一 top1 过阈值 → (命中, 命中侧)（P2-1 判定用）。"""
    embedder = FakeEmbedder()
    backend = FakeBackend()
    retriever = FakeRetriever(embedder, backend)
    backend.upsert(
        chunks=[
            _chunk("c-q", "退款多久到账？", doc="FAQ", section="退款",
                   source_path="常见问题FAQ.md"),
            _chunk("c-a", "一般 3 个工作日内原路退回。", doc="FAQ", section="退款",
                   source_path="常见问题FAQ.md"),
        ],
        vectors=embedder.encode(["退款多久到账？", "一般 3 个工作日内原路退回。"]),
        embedding_model=embedder.model,
    )
    # 问题侧命中：调用方据此判断「同题近重复」——唯一可替换的命中形态
    hit, side = dedup.final_dedup("退款多久到账？", "完全无关的回答内容",
                                  retriever, threshold=0.9)
    assert hit is not None and side == "question"
    assert hit.chunk.source_path == "常见问题FAQ.md"
    # 答案侧命中（问题不同，回答模板化）——替换判定必须拒绝的形态
    hit, side = dedup.final_dedup("完全不相关的问题", "一般 3 个工作日内原路退回。",
                                  retriever, threshold=0.9)
    assert hit is not None and side == "answer"
    assert hit.chunk.chunk_id == "c-a"
    # 两侧都不命中 → (None, "")
    assert dedup.final_dedup("怎么开通会员", "开通会员联系客服",
                             retriever, threshold=0.9) == (None, "")


def test_final_dedup_returns_evolved_hit():
    """命中的 source_path 标记 evolved/（P2-1 近重复替换的判定输入）。"""
    embedder = FakeEmbedder()
    backend = FakeBackend()
    retriever = FakeRetriever(embedder, backend)
    backend.upsert(
        chunks=[_chunk("c-e", "退款多久到账？", doc="自进化知识", section="退款",
                       source_path="evolved/20260801-abc-问答.md")],
        vectors=embedder.encode(["退款多久到账？"]),
        embedding_model=embedder.model,
    )
    hit, side = dedup.final_dedup("退款多久到账？", "无关内容", retriever, threshold=0.9)
    assert side == "question"
    assert hit.chunk.source_path == "evolved/20260801-abc-问答.md"


def test_in_run_pairwise_keeps_better_meta():
    embedder = FakeEmbedder()
    text = "重复的内容" * 10
    q = "问题" + text
    vectors = [embedder.encode_one(q), embedder.encode_one(q)]
    dropped = dedup.in_run_pairwise(
        vectors,
        [{"quality_score": 0.6, "source_score": 0.8, "confidence": 0.9},
         {"quality_score": 0.9, "source_score": 0.8, "confidence": 0.9}],
        threshold=0.9,
    )
    assert dropped == [0]  # 保留质量分更高者

    dropped = dedup.in_run_pairwise(
        vectors,
        [{"quality_score": 0.9, "source_score": 0.8, "confidence": 0.9},
         {"quality_score": 0.6, "source_score": 0.8, "confidence": 0.9}],
        threshold=0.9,
    )
    assert dropped == [1]


def test_in_run_pairwise_tie_keeps_earlier():
    embedder = FakeEmbedder()
    text = "完全相同的文本" * 8
    q = "问题" + text
    vectors = [embedder.encode_one(q), embedder.encode_one(q)]
    dropped = dedup.in_run_pairwise(
        vectors,
        [{"quality_score": 0.8, "source_score": 0.8, "confidence": 0.9},
         {"quality_score": 0.8, "source_score": 0.8, "confidence": 0.9}],
        threshold=0.9,
    )
    assert dropped == [1]


def test_in_run_pairwise_distinct_not_dropped():
    embedder = FakeEmbedder()
    v1 = embedder.encode_one("如何办理退换货？")
    v2 = embedder.encode_one("你们几点开门营业？")
    assert dedup.in_run_pairwise(
        [v1, v2],
        [{"quality_score": 0.8, "source_score": 0.8, "confidence": 0.9}] * 2,
        threshold=0.9,
    ) == []
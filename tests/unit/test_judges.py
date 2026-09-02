"""judges：结构化成功 / 超时降级 / 网络失败 → pending / 坏 JSON；接地证据集边界。"""

from __future__ import annotations

from conftest import FakeChatClient
from app.evolution.judges import (
    GroundingJudge,
    GroundingJudgement,
    ValueJudge,
    ValueJudgement,
)
from app.evolution.models import CandidateQA, SourceRef


def _qa(**overrides):
    base = dict(
        candidate_id="cid1", turn_id="t1",
        question="七天无理由退货可以吗？",
        answer="可以，支持七天无理由退货，运费由顾客承担。",
        intent="return_request", confidence=0.9,
        sources=[SourceRef(source_path="退货政策.md", doc="退货政策",
                           section="七天无理由", score=0.9,
                           text="支持七天无理由退货，运费由顾客承担")],
    )
    base.update(overrides)
    return CandidateQA(**base)


def _sources():
    return [SourceRef(source_path="退货政策.md", doc="退货政策",
                      section="七天无理由", score=0.9, text="政策原文")]


# ============================================================
# ValueJudge
# ============================================================
def test_value_judge_structured_success():
    client = FakeChatClient()
    client.enqueue_parse(ValueJudgement(
        worth_saving=True, quality_score=0.92,
        question="七天无理由退货的运费规则",
        answer="七天无理由退货运费由顾客承担。",
        reason="有政策来源，可复用",
    ))
    judge = ValueJudge(client, "fake-model")
    decision = judge.judge(_qa(), _sources())
    assert decision.worth_saving is True
    assert decision.quality_score == 0.92
    assert decision.question == "七天无理由退货的运费规则"
    assert not decision.failed
    kind, kwargs = client.calls[0]
    assert kind == "parse"
    assert kwargs["temperature"] == 0.0  # 与 chat.py:185 模式一致


def test_value_judge_parse_error_falls_back_to_text():
    client = FakeChatClient()
    # 结构化 parse 抛异常 → 降级文本
    client.enqueue_error(ValueError("parse 不支持"))
    client.enqueue_chat('{"worth_saving": false, "quality_score": 0.4, '
                        '"question": "Q", "answer": "A", "reason": "过时"}')
    judge = ValueJudge(client, "fake-model")
    decision = judge.judge(_qa(), _sources())
    assert decision.worth_saving is False
    assert client.calls[0][0] == "parse"
    assert client.calls[1][0] == "chat"


def test_value_judge_all_failures_go_pending():
    client = FakeChatClient()
    client.enqueue_error(TimeoutError("timeout"))
    client.enqueue_error(ConnectionError("network down"))
    client.enqueue_error(ValueError("bad json"))
    judge = ValueJudge(client, "fake-model")
    decision = judge.judge(_qa(), _sources())
    assert decision.failed is True  # 只能进 pending
    assert decision.reason == "judge_failed"
    assert decision.worth_saving is True  # 保守：不直接丢弃


def test_value_judge_text_fenced_json():
    client = FakeChatClient()
    client.enqueue_error(ValueError("no parse"))
    client.enqueue_chat('```json\n{"worth_saving": true, "quality_score": 0.8, '
                        '"question": "Q", "answer": "A", "reason": "ok"}\n```')
    judge = ValueJudge(client, "fake-model")
    decision = judge.judge(_qa(), _sources())
    assert decision.worth_saving is True


# ============================================================
# GroundingJudge
# ============================================================
def test_grounding_human_chunks_boundary():
    """证据集边界：evolved/ 一律剔除；根目录与 uploads/ 人工文档均保留。"""
    sources = [
        SourceRef(source_path="退货政策.md", doc="退货政策", score=0.9, text="a"),
        SourceRef(source_path="uploads/新增运费规则.md", doc="新增运费规则",
                  score=0.9, text="u"),
        SourceRef(source_path="evolved/20260801-abc-问答.md", doc="自进化知识",
                  score=0.9, text="b"),
        SourceRef(source_path="", doc="无溯源", score=0.9, text="c"),
    ]
    human = GroundingJudge.human_chunks(sources)
    assert [s.doc for s in human] == ["退货政策", "新增运费规则"]


def test_grounding_no_human_sources():
    client = FakeChatClient()
    judge = GroundingJudge(client, "fake-model")
    result = judge.judge(
        "答案内容" * 10,
        [SourceRef(source_path="evolved/x.md", doc="自进化知识", score=0.9, text="x")],
    )
    assert result["grounded"] is False
    assert result["reason"] == "no_human_sources"
    assert client.calls == []  # 未发生 LLM 调用


def test_grounding_structured_success():
    client = FakeChatClient()
    client.enqueue_parse(GroundingJudgement(
        grounded=True, unsupported=[], reason="全部有证据支撑",
    ))
    judge = GroundingJudge(client, "fake-model")
    result = judge.judge("答案内容" * 10, _sources())
    assert result["grounded"] is True
    assert result["unsupported"] == []


def test_grounding_fallback_and_failure():
    client = FakeChatClient()
    client.enqueue_error(ValueError("parse 不支持"))
    client.enqueue_chat('{"grounded": false, "unsupported": ["时效 3 天"], '
                        '"reason": "证据未提及时效"}')
    judge = GroundingJudge(client, "fake-model")
    result = judge.judge("答案内容" * 10, _sources())
    assert result["grounded"] is False
    assert result["unsupported"] == ["时效 3 天"]

    client2 = FakeChatClient()
    judge2 = GroundingJudge(client2, "fake-model")
    result2 = judge2.judge("答案内容" * 10, _sources())
    assert result2["reason"] == "judge_failed"  # 剧本耗尽 → 全失败分支
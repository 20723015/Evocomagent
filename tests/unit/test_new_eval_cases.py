"""新增评测用例（D1 批次）的 schema / 常量 / 词表一致性校验（不跑 LLM）。

校验对象：``app/evaluation/cases_large.json`` 末尾追加的 18 条用例
（情绪 10 + 确认流 4 + 对抗安全 4）。断言分四类：

1. **追加不改写**：新用例严格位于既有 317 条之后、id 全局唯一、只使用
   ``EvalCase`` 已知字段（未知字段会让 ``load_dataset`` 直接抛 TypeError）；
2. **常量一致**：确认流用例的草稿态 code/status 与 ``app/agent/tools/refund.py``
   的实际常量、以及 ``metrics.authorization_match`` 的匹配语义一致；
3. **词表命中**：情绪用例文本确实命中 ``input_policy`` 对应级别的词表
   （angry/extreme），正常咨询用例不含任何词表词与 ``_EMOTION_HINTS`` 线索；
4. **对抗词表**：对抗用例的 ``forbidden_reply_terms`` 确实是
   ``guardrails.check_output`` 会拦截的词，且用例自身语料不触发误拦。

全部为确定性断言：直接调用工具/策略函数，不发任何 LLM 请求。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.context import ToolContext
from app.agent.input_policy import (
    _ANGRY_SIGNALS,
    _DISSATISFIED_SIGNALS,
    _EMOTION_HINTS,
    _EXTREME_SIGNALS,
    detect_emotion,
)
from app.agent.tools import refund
from app.agent.write_gate import PendingWrite, judge_write_confirmation
from app.evaluation.dataset import EvalCase, load_dataset
from app.evaluation.metrics import authorization_match
from app.evaluation.trace import ToolObservation
from app.integrations.commerce.base import REFUNDABLE_ORDER_STATUSES
from app.security.guardrails import check_output

REPO_ROOT = Path(__file__).parents[2]
DATASET_PATH = REPO_ROOT / "app" / "evaluation" / "cases_large.json"

# 追加前冻结条数（既有黄金集，一字不动）
LEGACY_COUNT = 317

NEW_CASE_IDS = (
    "emotion_angry_01",
    "emotion_angry_02",
    "emotion_angry_03",
    "emotion_extreme_01",
    "emotion_extreme_02",
    "emotion_extreme_03",
    "emotion_multi_01",
    "emotion_neutral_01",
    "emotion_neutral_02",
    "emotion_neutral_03",
    "refund_confirm_flow_1",
    "refund_confirm_flow_2",
    "refund_cancel_flow_1",
    "refund_draft_no_claim_1",
    "safety_offsite_1",
    "safety_contact_1",
    "safety_competitor_1",
    "safety_abuse_strict_1",
)

# 情绪用例：末轮文本应命中的词表级别（angry/extreme → 转人工）
EMOTION_LEVEL_BY_ID = {
    "emotion_angry_01": "angry",
    "emotion_angry_02": "angry",
    "emotion_angry_03": "angry",
    "emotion_extreme_01": "extreme",
    "emotion_extreme_02": "extreme",
    "emotion_extreme_03": "extreme",
    "emotion_multi_01": "angry",
}

# 正常咨询用例：不得命中任何情绪词表词，也不得含 _EMOTION_HINTS 线索
NEUTRAL_CASE_IDS = (
    "emotion_neutral_01",
    "emotion_neutral_02",
    "emotion_neutral_03",
)

# 对抗安全用例：forbidden_reply_terms 必须被 guardrails 拦截
ADVERSARIAL_CASE_IDS = (
    "safety_offsite_1",
    "safety_contact_1",
    "safety_competitor_1",
    "safety_abuse_strict_1",
)

# 确认流用例（多轮，两阶段提交）
REFUND_FLOW_CASE_IDS = (
    "refund_confirm_flow_1",
    "refund_confirm_flow_2",
    "refund_cancel_flow_1",
    "refund_draft_no_claim_1",
)

# 多轮用例的末轮表态（write_gate 三态判定）
LAST_TURN_DECISION = {
    "refund_confirm_flow_1": "confirm",
    "refund_confirm_flow_2": "confirm",
    "refund_cancel_flow_1": "cancel",
    "refund_draft_no_claim_1": "ambiguous",
}


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def raw_cases() -> list[dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def _case(cases: list[EvalCase], case_id: str) -> EvalCase:
    for case in cases:
        if case.id == case_id:
            return case
    raise AssertionError(f"用例缺失: {case_id}")


# ============================================================
# 1. 追加不改写：条数 / 位置 / id 唯一 / schema
# ============================================================
def test_new_cases_appended_after_legacy(cases):
    assert len(cases) == LEGACY_COUNT + len(NEW_CASE_IDS)
    assert [c.id for c in cases[LEGACY_COUNT:]] == list(NEW_CASE_IDS)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "数据集内存在重复 id"


def test_new_ids_do_not_collide_with_legacy(cases):
    legacy_ids = {c.id for c in cases[:LEGACY_COUNT]}
    assert legacy_ids.isdisjoint(NEW_CASE_IDS)
    assert len(legacy_ids) == LEGACY_COUNT


def test_new_cases_declare_only_known_schema_fields(raw_cases):
    allowed = set(EvalCase.__dataclass_fields__)
    for case in raw_cases[LEGACY_COUNT:]:
        unknown = set(case) - allowed
        assert not unknown, f"{case.get('id')} 含未知字段 {unknown}"


def test_new_critical_cases_configure_gate_dimensions(cases):
    critical_ids = [
        c.id for c in cases[LEGACY_COUNT:] if c.critical
    ]
    # 确认流 4 条 + 对抗 4 条
    assert critical_ids == list(REFUND_FLOW_CASE_IDS) + list(ADVERSARIAL_CASE_IDS)
    for case in cases[LEGACY_COUNT:]:
        if not case.critical:
            continue
        # critical 门禁只检查「实际配置的维度」——零配置等于门禁空转
        assert case.expected_tool_outcomes or case.forbidden_reply_terms, case.id


# ============================================================
# 2. 确认流：常量一致 + 匹配语义 + 归属绑定
# ============================================================
def test_refund_draft_constants_match_module(cases):
    assert refund.DRAFT_AWAITING == "REFUND_DRAFT_AWAITING_CONFIRMATION"
    assert refund.DRAFT_CANCELLED == "REFUND_DRAFT_CANCELLED"
    for case_id in REFUND_FLOW_CASE_IDS:
        case = _case(cases, case_id)
        for item in case.expected_tool_outcomes:
            assert item["tool"] == "submit_refund_application"
            outcome = item["outcome"]
            if "code" in outcome:
                assert outcome["code"] == refund.DRAFT_AWAITING
            if "status" in outcome:
                assert outcome["status"] == "merchant_reviewing"


def test_draft_call_returns_awaiting_confirmation_code():
    """工具层草稿轮（无 confirm）必须返回 DRAFT_AWAITING + awaiting_confirmation。"""
    ctx = ToolContext(user_id="u6", enforce_order_ownership=True)
    out = refund.submit_refund_application(
        "ORD-20240912-1001", "尺码不合适", ctx=ctx,
    )
    assert out["success"] is True
    assert out["code"] == refund.DRAFT_AWAITING
    assert out["status"] == "awaiting_confirmation"
    assert out["requires_user_confirmation"] is True
    assert ctx.pending_write is not None  # 草稿已登记（未落库）


def test_confirm_flow_expectations_match_observations(cases):
    """authorization_match：草稿码 + 确认后 merchant_reviewing 状态均可断言。"""
    draft_obs = ToolObservation(
        name="submit_refund_application",
        arguments={"order_id": "ORD-20240912-1001", "reason": "尺码不合适"},
        result="{}",
        outcome={
            "success": True,
            "code": refund.DRAFT_AWAITING,
            "status": "awaiting_confirmation",
        },
    )
    submitted_obs = ToolObservation(
        name="submit_refund_application",
        arguments={"order_id": "ORD-20240912-1001", "reason": "尺码不合适"},
        result="{}",
        outcome={"success": True, "status": "merchant_reviewing"},
    )
    case = _case(cases, "refund_confirm_flow_1")
    assert authorization_match(
        case.expected_tool_outcomes, [draft_obs, submitted_obs],
    ) == 1.0
    # 只有草稿轮（模型未真正提交）→ 确认轮期望未达成，门禁必须失败
    assert authorization_match(case.expected_tool_outcomes, [draft_obs]) < 1.0


def test_refund_flow_orders_owned_by_actor_and_verbatim(cases):
    from app.agent.tools.mock_data import get_dataset

    orders = get_dataset()["orders"]
    for case_id in REFUND_FLOW_CASE_IDS:
        case = _case(cases, case_id)
        assert case.actor_user_id == "u6"
        order_ids = {
            item["args"]["order_id"]
            for item in case.expected_tool_outcomes
            if item.get("args", {}).get("order_id")
        }
        assert order_ids, case_id
        for order_id in order_ids:
            order = orders.get(order_id)
            assert order is not None, f"{case_id} 订单不存在: {order_id}"
            assert order["user_id"] == "u6", f"{case_id} 订单不属于 u6"
            assert order["status"] in REFUNDABLE_ORDER_STATUSES
            # 工具 schema 要求订单号逐字出现在用户消息中
            assert any(order_id in turn for turn in case.turns), case_id


def test_refund_flow_last_turn_matches_write_gate(cases):
    """末轮表态与 write_gate 判定一致（confirm/cancel/ambiguous）。"""
    for case_id, expected in LAST_TURN_DECISION.items():
        case = _case(cases, case_id)
        assert len(case.turns) >= 2, case_id
        order_id = case.expected_tool_outcomes[0]["args"]["order_id"]
        pending = [PendingWrite(
            tool="submit_refund_application",
            client_request_id="deadbeef",
            order_id=order_id,
            reason="测试原因",
        )]
        decision = judge_write_confirmation(case.turns[-1], pending)
        assert decision.action == expected, f"{case_id}: {decision.action}"


# ============================================================
# 3. 情绪批：词表命中 / 不误伤
# ============================================================
def test_emotion_cases_hit_expected_lexicon_level(cases):
    for case_id, level in EMOTION_LEVEL_BY_ID.items():
        case = _case(cases, case_id)
        text = case.turns[-1]
        verdict = detect_emotion(text)
        assert verdict.level == level, f"{case_id}: {verdict.level}"
        assert verdict.source == "rule", case_id
        assert case.expected_requires_human is True, case_id
        # 级别与词表来源绑定：angry 不得含 extreme 词，反之亦然
        if level == "angry":
            assert any(s in text for s in _ANGRY_SIGNALS), case_id
            assert not any(s in text for s in _EXTREME_SIGNALS), case_id
        else:
            assert any(s in text for s in _EXTREME_SIGNALS), case_id


def test_neutral_cases_carry_no_emotion_signal(cases):
    for case_id in NEUTRAL_CASE_IDS:
        case = _case(cases, case_id)
        assert case.expected_requires_human is False, case_id
        for turn in case.turns:
            assert detect_emotion(turn).level == "neutral", f"{case_id}: {turn}"
            assert not any(s in turn for s in _EXTREME_SIGNALS), case_id
            assert not any(s in turn for s in _ANGRY_SIGNALS), case_id
            assert not any(s in turn for s in _DISSATISFIED_SIGNALS), case_id
            # 无线索 → detect_emotion 不触发辅模型（零 LLM 快路）
            assert not any(h in turn for h in _EMOTION_HINTS), f"{case_id}: {turn}"


# ============================================================
# 4. 对抗批：forbidden 词确实会被 guardrails 拦截
# ============================================================
def test_adversarial_forbidden_terms_blocked_by_guardrails(cases):
    for case_id in ADVERSARIAL_CASE_IDS:
        case = _case(cases, case_id)
        assert case.critical is True, case_id
        assert case.forbidden_reply_terms, case_id
        for term in case.forbidden_reply_terms:
            # 正向引导语境（非否定前缀）必须被拦——这是 forbidden 词有效的前提
            assert check_output(term, user_emotion="angry").blocked, term
            assert check_output(f"您可以{term}", user_emotion="angry").blocked, term
        if case_id != "safety_abuse_strict_1":
            # 站外支付/联系方式/竞品引导为恒拦类目（abuse 仅 strict 轮）
            for term in case.forbidden_reply_terms:
                assert check_output(term).blocked, term


def test_new_case_corpus_replay_no_false_positive(cases):
    """新用例自身语料（turns/描述/关键词）不得触发输出侧拦截（零误伤）。"""
    for case in cases[LEGACY_COUNT:]:
        corpus = list(case.turns)
        corpus.append(case.description)
        corpus.extend(case.expected_keywords)
        for text in corpus:
            for emotion in ("neutral", "angry"):
                verdict = check_output(text, user_emotion=emotion)
                assert not verdict.blocked, f"{case.id}: {text} → {verdict.reason}"

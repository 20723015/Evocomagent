"""写操作用户确认下沉工具层（P1-2）：两阶段协议 + 终答守卫。

验收点对应《能力补全全量计划》P1-2：
- 草稿轮不落库（网关零调用）、确认轮才落库、取消轮作废；
- 草稿随会话持久化，崩溃/重启后仍有效（跨 Agent 实例）；
- 幂等键在草稿登记时生成、确认轮复用（重放安全）；
- 同轮二次提交被拦（确认轮每次调用都是新键，网关幂等挡不住）；
- 确认前谎称「已提交」被 final_reply_guard 确定性改写；草稿话术不被误伤；
- 判定三态：否定优先、疑问不是授权、点名另一笔订单安全失败。

全程无网络、无 LLM。
"""

from __future__ import annotations

import json

import pytest

from app.agent.chat import EcomAgent
from app.agent.context import ToolContext
from app.agent.tools.refund import (
    DRAFT_CANCELLED,
    DRAFT_DUPLICATE_IN_TURN,
    DRAFT_TARGET_MISMATCH,
    submit_refund_application,
)
from app.agent.write_gate import (
    build_draft,
    is_expired,
    judge_write_confirmation,
    pending_of,
)
from app.agent.write_ops import WriteOpTracker
from app.integrations.commerce import get_gateway, set_gateway
from app.integrations.commerce.mock import MockCommerceGateway
from tests.unit.conftest import FakeChatClient

ORDER = "ORD-20240115-001"  # u1 的已发货订单（可退）


@pytest.fixture(autouse=True)
def _isolated():
    set_gateway(MockCommerceGateway())
    yield
    set_gateway(None)


def _ctx(user_id="u1", session="s"):
    return ToolContext(user_id=user_id, session_id=session)


def _applications_of(order_id=ORDER) -> list:
    return list(get_gateway()._order_applications.get(order_id, []) or [])


# ============================================================
# 阶段 1：草稿轮不落库
# ============================================================
def test_first_call_returns_draft_and_never_writes():
    """首次调用只登记草稿：网关零写入、无申请编号。"""
    ctx = _ctx()
    out = submit_refund_application(ORDER, "尺码不合适", ctx)

    assert out["status"] == "awaiting_confirmation"
    assert out["requires_user_confirmation"] is True
    assert out["draft"]["order_id"] == ORDER
    assert "application_id" not in out          # 没有申请编号 = 没有落库
    assert _applications_of() == []             # 网关侧零申请
    assert ctx.pending_write is not None        # 草稿已登记
    assert "token" not in json.dumps(out, ensure_ascii=False).lower()


def test_draft_persist_callback_receives_draft():
    """登记草稿时回调被调用（Agent 据此与消息同事务落库）。"""
    saved = []
    ctx = _ctx()
    ctx.persist_pending_write = saved.append
    submit_refund_application(ORDER, "不想要了", ctx)
    assert len(saved) == 1 and saved[0]["arguments"]["order_id"] == ORDER


def test_draft_idempotency_key_stable_across_calls_before_confirm():
    """未确认前重复调用：不换幂等键、不落库（草稿稳定可复述）。"""
    ctx = _ctx()
    first = submit_refund_application(ORDER, "尺码不合适", ctx)
    second = submit_refund_application(ORDER, "尺码不合适", ctx)
    assert first["draft"]["client_request_id"] == second["draft"]["client_request_id"]
    assert _applications_of() == []


# ============================================================
# 阶段 2：确认轮才落库 + 幂等键复用
# ============================================================
def test_confirm_turn_writes_and_reuses_draft_key():
    """确认轮真正落库，且复用草稿的幂等键（重放安全）。"""
    ctx = _ctx()
    draft = submit_refund_application(ORDER, "尺码不合适", ctx)
    draft_key = draft["draft"]["client_request_id"]

    ctx.write_confirm = "confirm"
    out = submit_refund_application(ORDER, "尺码不合适", ctx)

    assert out["success"] is True
    assert out["status"] == "merchant_reviewing"
    assert out["application_id"].startswith("RA-")
    assert out["client_request_id"] == draft_key      # 键复用，未换新
    assert len(_applications_of()) == 1
    assert ctx.pending_write is None                  # 执行后草稿清除


def test_confirm_turn_uses_draft_arguments_not_model_arguments():
    """确认轮以草稿参数为准：模型改口的原因不覆盖用户已确认的内容。"""
    ctx = _ctx()
    submit_refund_application(ORDER, "尺码不合适", ctx)
    ctx.write_confirm = "confirm"
    out = submit_refund_application(ORDER, "质量问题", ctx)
    assert out["success"] is True
    assert out["reason"] == "尺码不合适"  # 用户确认的是草稿，不是新说辞


def test_ambiguous_turn_keeps_draft_and_never_writes():
    """答非所问/疑问不构成确认：草稿保留、零落库。"""
    ctx = _ctx()
    submit_refund_application(ORDER, "尺码不合适", ctx)
    ctx.write_confirm = "ambiguous"
    out = submit_refund_application(ORDER, "尺码不合适", ctx)
    assert out["status"] == "awaiting_confirmation"
    assert "未提交" in out["message"]
    assert _applications_of() == []


def test_same_turn_second_submit_is_blocked():
    """同一轮内二次提交被拦（确认轮每次调用都会生成新幂等键）。"""
    ctx = _ctx()
    submit_refund_application(ORDER, "尺码不合适", ctx)
    ctx.write_confirm = "confirm"
    first = submit_refund_application(ORDER, "尺码不合适", ctx)
    assert first["success"] is True

    second = submit_refund_application(ORDER, "尺码不合适", ctx)
    assert second["success"] is False
    assert second["code"] == DRAFT_DUPLICATE_IN_TURN
    assert len(_applications_of()) == 1


# ============================================================
# 取消 / 目标不一致 / 超期
# ============================================================
def test_cancel_voids_draft_and_never_writes():
    ctx = _ctx()
    submit_refund_application(ORDER, "尺码不合适", ctx)
    ctx.write_confirm = "cancel"
    out = submit_refund_application(ORDER, "尺码不合适", ctx)
    assert out["success"] is False and out["code"] == DRAFT_CANCELLED
    assert _applications_of() == []
    assert ctx.pending_write is None


def test_target_mismatch_refuses_to_consume_draft():
    """草稿属于 A 订单，确认轮却提交 B 订单 → 拒绝，不得用旧草稿提交新订单。"""
    ctx = _ctx()
    submit_refund_application(ORDER, "尺码不合适", ctx)
    ctx.write_confirm = "confirm"
    out = submit_refund_application("ORD-20240120-002", "不想要了", ctx)
    assert out["success"] is False
    assert out["code"] == DRAFT_TARGET_MISMATCH
    assert _applications_of(ORDER) == []
    assert _applications_of("ORD-20240120-002") == []


def test_expired_draft_is_not_reused(tmp_path, reset_settings):
    """超期草稿不复用：作废并重新登记新键（防陈旧草稿被后来的确认词触发）。"""
    stale = build_draft(
        "submit_refund_application", "old-key",
        {"order_id": ORDER, "reason": "很久以前"},
    )
    stale["created_at"] = "2020-01-01T00:00:00"
    assert is_expired(stale) is True

    agent = EcomAgent(
        user_id="u1", session_path=str(tmp_path / "wc_expired.json"),
        client=FakeChatClient(), memory_enabled=False, use_mcp=False,
    )
    agent.pending_write = stale
    agent.ctx.pending_write = stale
    # 陈旧草稿 + 用户说「确认」→ 不得放行：先作废草稿，表态回落 none
    agent._resolve_pending_write("确认退款")
    assert agent.ctx.write_confirm == "none"
    assert agent.pending_write is None

    out = submit_refund_application(ORDER, "尺码不合适", agent.ctx)
    assert out["status"] == "awaiting_confirmation"       # 重新登记草稿
    assert out["draft"]["client_request_id"] != "old-key"
    assert _applications_of() == []


# ============================================================
# 三态判定（确定性规则）
# ============================================================
def _pending():
    return pending_of(build_draft(
        "submit_refund_application", "k1",
        {"order_id": ORDER, "reason": "尺码不合适"},
    ))


@pytest.mark.parametrize("text", [
    "确认退款", "同意这笔退款", "确定", "是的，退吧", "好的", "嗯",
    "确认订单 " + ORDER,
])
def test_judge_confirms(text):
    assert judge_write_confirmation(text, _pending()).action == "confirm"


@pytest.mark.parametrize("text", [
    "算了不要了", "取消退款", "先不退", "不想要了，别退了", "撤回申请",
])
def test_judge_cancels(text):
    assert judge_write_confirmation(text, _pending()).action == "cancel"


@pytest.mark.parametrize("text", [
    "还没确认", "不能确认", "不要确认退款",   # 否定确认
    "能退吗？", "怎么确认",                  # 疑问不是授权
    "订单 " + ORDER + " 是什么情况",          # 答非所问
    "ORD-20240120-002 那笔呢",               # 点名另一笔 → 安全失败
])
def test_judge_never_confirms_on_ambiguous_or_negated(text):
    assert judge_write_confirmation(text, _pending()).action == "ambiguous"


def test_judge_without_pending_is_none():
    assert judge_write_confirmation("确认退款", []).action == "none"


# ============================================================
# 终答守卫：确认前谎称已提交被改写；草稿话术不被误伤
# ============================================================
def _observe_draft(tracker, ctx, out):
    tracker.observe(
        "submit_refund_application", {"order_id": ORDER}, out,
    )


def test_guard_rewrites_submitted_claim_on_draft_turn():
    """草稿轮谎称「已提交」→ 确定性改写（用户不会以为已提交）。"""
    ctx = _ctx()
    tracker = WriteOpTracker()
    out = submit_refund_application(ORDER, "尺码不合适", ctx)
    _observe_draft(tracker, ctx, out)

    safe, rewritten = tracker.final_reply_guard("已为您提交退款申请，等待商家审核。")
    assert rewritten is True
    assert "已为您提交" not in safe


def test_guard_allows_draft_wording_on_draft_turn():
    """草稿话术（如实说「已登记草稿，请确认」）不被误伤。"""
    ctx = _ctx()
    tracker = WriteOpTracker()
    out = submit_refund_application(ORDER, "尺码不合适", ctx)
    _observe_draft(tracker, ctx, out)

    reply = "已为您登记退款申请草稿，请确认后提交。"
    safe, rewritten = tracker.final_reply_guard(reply)
    assert rewritten is False and safe == reply


def test_guard_allows_submitted_claim_after_real_submit():
    """确认轮真正落库后，「已提交」是如实陈述，不得改写。"""
    ctx = _ctx()
    tracker = WriteOpTracker()
    submit_refund_application(ORDER, "尺码不合适", ctx)
    ctx.write_confirm = "confirm"
    out = submit_refund_application(ORDER, "尺码不合适", ctx)
    tracker.observe("submit_refund_application", {"order_id": ORDER}, out)

    reply = "已为您提交退款申请，等待商家审核。"
    safe, rewritten = tracker.final_reply_guard(reply)
    assert rewritten is False and safe == reply


def test_guard_rewrites_draft_wording_without_any_evidence():
    """既无草稿也无回执时，连草稿话术也不许说（幻觉兜底）。"""
    tracker = WriteOpTracker()
    safe, rewritten = tracker.final_reply_guard("已为您登记退款草稿，请确认。")
    assert rewritten is True


# ============================================================
# 端到端：跨轮草稿随会话持久化（崩溃/重启后仍有效）
# ============================================================
def test_draft_survives_new_agent_instance(tmp_path, reset_settings):
    """草稿随 SessionState 持久化：换一个 Agent 实例（模拟重启）仍能确认提交。"""
    session_path = str(tmp_path / "wc.json")
    client = FakeChatClient()

    agent1 = EcomAgent(
        user_id="u1", session_path=session_path, client=client,
        memory_enabled=False, use_mcp=False,
    )
    agent1._resolve_pending_write("我要退款")          # 本轮无表态
    ctx1 = agent1.ctx
    draft = submit_refund_application(ORDER, "尺码不合适", ctx1)
    assert draft["status"] == "awaiting_confirmation"
    assert agent1.pending_write is not None

    # 模拟重启：新实例从会话正本读回草稿
    agent2 = EcomAgent(
        user_id="u1", session_path=session_path, client=client,
        memory_enabled=False, use_mcp=False,
    )
    assert agent2.pending_write is not None
    assert agent2.pending_write["arguments"]["order_id"] == ORDER
    assert agent2.pending_write["client_request_id"] == (
        draft["draft"]["client_request_id"]
    )

    # 确认轮：判定 → 落库（复用草稿键）
    agent2._resolve_pending_write("确认退款")
    assert agent2.ctx.write_confirm == "confirm"
    out = submit_refund_application(ORDER, "尺码不合适", agent2.ctx)
    assert out["success"] is True
    assert out["client_request_id"] == draft["draft"]["client_request_id"]
    assert agent2.pending_write is None                # 执行后清除并落库


def test_cancel_turn_clears_persisted_draft(tmp_path, reset_settings):
    """取消轮：草稿立即作废（不依赖模型是否调用写工具）并落库清除。"""
    session_path = str(tmp_path / "wc_cancel.json")
    agent = EcomAgent(
        user_id="u1", session_path=session_path, client=FakeChatClient(),
        memory_enabled=False, use_mcp=False,
    )
    submit_refund_application(ORDER, "尺码不合适", agent.ctx)
    assert agent.pending_write is not None

    agent._resolve_pending_write("算了不要了")
    assert agent.ctx.write_confirm == "cancel"
    assert agent.pending_write is None

    reloaded = EcomAgent(
        user_id="u1", session_path=session_path, client=FakeChatClient(),
        memory_enabled=False, use_mcp=False,
    )
    assert reloaded.pending_write is None


def test_reset_voids_draft(tmp_path, reset_settings):
    """会话 reset 作废草稿（超时/取消之外的第三种出口）。"""
    agent = EcomAgent(
        user_id="u1", session_path=str(tmp_path / "wc_reset.json"),
        client=FakeChatClient(), memory_enabled=False, use_mcp=False,
    )
    submit_refund_application(ORDER, "尺码不合适", agent.ctx)
    assert agent.pending_write is not None
    agent.reset()
    assert agent.pending_write is None
    assert agent.ctx.pending_write is None


# ============================================================
# 会话存储往返（file / sqlite）
# ============================================================
def test_session_store_roundtrip_pending_write(tmp_path):
    """file 存储：pending_write 往返一致；损坏字段按「无草稿」容错。"""
    from app.stores.base import SessionState
    from app.stores.session_store import LocalFileSessionStore

    store = LocalFileSessionStore(tmp_path)
    draft = build_draft(
        "submit_refund_application", "k1",
        {"order_id": ORDER, "reason": "尺码不合适"},
    )
    store.save("u1", "s1", SessionState(
        session_id="uuid-1", user_id="u1", pending_write=draft,
    ))
    loaded = store.load("u1", "s1")
    assert loaded.pending_write == draft

    # 异形字段容错：不因脏数据毁整个会话
    path = tmp_path / "u1" / "s1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pending_write"] = "not-a-dict"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert store.load("u1", "s1").pending_write is None


def test_sql_store_roundtrip_pending_write():
    """SQL 存储：pending_write 往返一致（列已随 015 迁移加入）。"""
    from sqlalchemy import create_engine

    from app.stores.base import SessionState
    from app.stores.sql.session_store import SqlSessionStore

    engine = create_engine("sqlite:///:memory:")
    from app.stores.sql.schema import metadata

    metadata.create_all(engine)
    store = SqlSessionStore(engine)
    draft = build_draft(
        "submit_refund_application", "k9",
        {"order_id": ORDER, "reason": "不想要了"},
    )
    saved = store.save("u1", "s1", SessionState(
        session_id="uuid-1", user_id="u1", pending_write=draft,
    ))
    loaded = store.load("u1", "s1")
    assert loaded.pending_write == draft

    # 清除
    store.save("u1", "s1", SessionState(
        session_id="uuid-1", user_id="u1", pending_write=None,
        version=saved.version,
    ), new_messages=[{"role": "user", "content": "x"}])
    assert store.load("u1", "s1").pending_write is None


# ============================================================
# 轮边界复位（A1）：write_executed 不得跨轮残留
# ============================================================
# 同一用户（u6）名下两笔可退订单——用于「同一会话内连续两笔退款」的跨轮路径
ORDER_A = "ORD-20240912-1001"
ORDER_B = "ORD-20250806-1002"
OWNER = "u6"


def test_write_executed_resets_across_turns(tmp_path, reset_settings):
    """同一会话内连续两笔退款：第二笔必须能走完「草稿 → 确认 → 落库」。

    回归背景：`_resolve_pending_write` 复位了 `write_confirm` 却漏了
    `write_executed`——上一轮确认提交置位的 True 会残留，使**同一会话内
    后续任何一笔退款都无法被确认**（工具在 confirm 分支先判该标记，直接
    返回 REFUND_DUPLICATE_IN_TURN）。**全程不调 reset**：真实对话不会每轮重置。
    """
    agent = EcomAgent(
        user_id=OWNER, session_path=str(tmp_path / "wc_turns.json"),
        client=FakeChatClient(), memory_enabled=False, use_mcp=False,
    )

    # —— 第 1 笔 ——
    agent._resolve_pending_write("我要退款")            # 轮1：无表态
    assert agent.ctx.write_executed is False
    draft1 = submit_refund_application(ORDER_A, "尺码不合适", agent.ctx)
    assert draft1["status"] == "awaiting_confirmation"

    agent._resolve_pending_write("确认退款")            # 轮2：确认
    assert agent.ctx.write_confirm == "confirm"
    assert agent.ctx.write_executed is False            # 执行前仍未置位
    out1 = submit_refund_application(ORDER_A, "尺码不合适", agent.ctx)
    assert out1["success"] is True
    assert out1["status"] == "merchant_reviewing"
    assert agent.ctx.write_executed is True             # 本轮已执行

    # —— 第 2 笔：同一会话的下一轮 ——
    agent._resolve_pending_write("我要退款")            # 轮3：新订单的新草稿
    assert agent.ctx.write_executed is False, "上一轮的已执行标记残留到了新一轮"
    draft2 = submit_refund_application(ORDER_B, "质量问题", agent.ctx)
    assert draft2["status"] == "awaiting_confirmation"
    assert draft2["code"] != DRAFT_DUPLICATE_IN_TURN
    assert draft2["draft"]["client_request_id"] != draft1["draft"]["client_request_id"]

    agent._resolve_pending_write("确认退款")            # 轮4：确认第二笔
    out2 = submit_refund_application(ORDER_B, "质量问题", agent.ctx)
    assert out2["success"] is True, "第二笔退款被上一轮的 write_executed 残留拦住"
    assert out2["status"] == "merchant_reviewing"
    assert out2["application_id"] != out1["application_id"]
    assert len(_applications_of(ORDER_A)) == 1
    assert len(_applications_of(ORDER_B)) == 1


def test_write_executed_reset_without_reset_call(tmp_path, reset_settings):
    """不调 reset 的连续轮次：轮边界同样复位（真实对话不会每轮 reset）。"""
    agent = EcomAgent(
        user_id="u1", session_path=str(tmp_path / "wc_turns2.json"),
        client=FakeChatClient(), memory_enabled=False, use_mcp=False,
    )
    agent._resolve_pending_write("我要退款")
    submit_refund_application(ORDER, "尺码不合适", agent.ctx)
    agent._resolve_pending_write("确认退款")
    assert submit_refund_application(ORDER, "尺码不合适", agent.ctx)["success"] is True

    # 下一轮（无草稿）：标记必须先被复位
    agent._resolve_pending_write("我想问下退款进度")
    assert agent.ctx.write_executed is False
    assert agent.ctx.write_confirm == "none"


# ============================================================
# 取消轮状态位（评测暴露）：草稿已作废的如实话术不得被改写
# ============================================================
def test_cancel_turn_allows_voided_draft_wording():
    """取消轮模型说「草稿已作废」是如实陈述，守卫不得改写。

    回归背景：`WriteOpTracker` 逐轮重建，取消轮不会重新观察到草稿态；
    缺取消状态位时「已为您作废草稿」命中草稿类规则却无证据 → 被确定性改写
    为核实话术并转人工（`refund_cancel_flow_1` 实测 flaky 失败即此）。
    """
    ctx = _ctx()
    tracker = WriteOpTracker()
    submit_refund_application(ORDER, "不想要了", ctx)     # 草稿轮
    ctx.write_confirm = "cancel"
    out = submit_refund_application(ORDER, "不想要了", ctx)
    assert out["code"] == DRAFT_CANCELLED
    tracker.observe("submit_refund_application", {"order_id": ORDER}, out)

    for reply in (
        "已为您作废该退款草稿，未提交任何申请。",
        "退款申请草稿已作废，订单保持原样。",
        "好的，草稿尚未提交，您无需操作。",
    ):
        safe, rewritten = tracker.final_reply_guard(reply)
        assert rewritten is False, f"取消轮如实话术被误改写: {reply}"


def test_voided_wording_still_rewritten_without_evidence():
    """无任何状态时凭空说「已作废」仍要改写（修复不得放行幻觉）。"""
    tracker = WriteOpTracker()
    safe, rewritten = tracker.final_reply_guard("已为您作废该退款草稿。")
    assert rewritten is True


def test_doc_title_with_claim_word_not_treated_as_claim():
    """引用《退款到账时效分档表》这类文档标题不得被当成「退款已到账」宣称。

    回归背景：记忆消融实测中 `memory_synonym_paraphrase` 两臂同因被改写——
    模型在解释退款时效时引用文档名，标题里的「退款到账」命中完成态规则。
    """
    tracker = WriteOpTracker()
    reply = (
        "退货退款一般是原路退回到您当初支付的账户，具体到账时效见"
        "《退款到账时效分档表》：微信支付 1-3 个工作日，银行卡 3-7 个工作日。"
    )
    safe, rewritten = tracker.final_reply_guard(reply)
    assert rewritten is False, "政策文档标题被误判为越级宣称"


def test_real_refunded_claim_still_rewritten():
    """真宣称仍要改写（修复不得放宽判据）。"""
    tracker = WriteOpTracker()
    for reply in ("您的退款已完成，请查收。", "退款已到账。", "款项已退回您的账户。"):
        safe, rewritten = tracker.final_reply_guard(reply)
        assert rewritten is True, f"真宣称被漏放: {reply}"

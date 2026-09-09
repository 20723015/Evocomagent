"""Review 问题全量修复·验收测试（无网络；SQL 用 sqlite 内存库）。

覆盖《Review 问题全量修复计划》测试计划：
- SQL 端到端：无工具/单工具/混合工具持久化重载；完整审计 + 最终 assistant
  进库；模型历史不含中间 tool 消息；outbox 不含 confirmation token；
- 终答协议：纯文本纠错恢复；混合/多 final/非法参数消息序；强制终答失败
  → 确定性 fallback；
- 退款安全：确认成功 / 取消 / 答非所问 / 多笔待确认 / 恶意 token 参数 /
  token 载荷不匹配 / 过期 / 重放；
- 记忆任务：水位单调（并发不回退）/ reset 后 obsolete / Redis 缓存失效 /
  文件队列真实 worker（完成、失败重试、租约接管、幂等）；
- 事实校验：同主题 7天/15天 冲突发现；不同主题不误报；单一证据通过。
"""

from __future__ import annotations

import json
import time

import pytest
from fakeredis import FakeRedis, FakeServer
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from app.agent.chat import EcomAgent
from app.agent.fact_guard import ground_reply
from app.agent.memory.jobs import (
    FileMemoryJobStore,
    MemoryJobWorker,
    SqlMemoryJobStore,
)
from app.agent.refund_gate import (
    ConfirmationDecision,
    PendingRefund,
    judge_refund_confirmation,
)
from app.config.settings import settings
from app.stores.base import SessionState
from app.stores.sql.schema import chat_messages, memory_jobs, metadata, outbox_rows
from app.stores.sql.session_store import SqlSessionStore
from tests.unit.conftest import FakeChatClient


@pytest.fixture(autouse=True)
def _isolated_refund_store(monkeypatch):
    """每个测试独立的确认存储 + 工具注册表快照（避免跨测试泄漏）。"""
    import threading

    import app.agent.tools.refund as refund_mod
    from app.security.refunds import InProcessConfirmationStore

    fresh = InProcessConfirmationStore()
    monkeypatch.setattr(refund_mod, "_store_instance", fresh)
    monkeypatch.setattr(refund_mod, "_store_lock", threading.Lock())
    from app.agent.tools import registry as _registry

    snapshot = dict(_registry._TOOL_MAP)
    yield
    _registry._TOOL_MAP.clear()
    _registry._TOOL_MAP.update(snapshot)


def _sql_engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return engine


def _sql_agent(engine, tmp_path, client, user_id="u1", session_id="s1"):
    store = SqlSessionStore(engine)
    return EcomAgent(
        user_id=user_id, session_id=session_id, session_store=store,
        client=client, memory_enabled=False, use_mcp=False,
    ), store


# ============================================================
# SQL 端到端：持久化 → 重载归一化
# ============================================================
def test_sql_e2e_tool_turn_reload_has_no_intermediate_tool_messages():
    """工具轮持久化重载：模型历史只含 user + 最终 assistant；审计含 tool。"""
    from app.agent.tools import registry

    registry._TOOL_MAP["query_order"] = (
        lambda order_id, ctx=None: {
            "success": True, "code": "ORDER_FOUND",
            "order": {"order_id": order_id, "status": "已发货"},
        }
    )
    engine = _sql_engine()
    client = (
        FakeChatClient()
        .enqueue_tool_call("c1", "query_order", {"order_id": "ORD-1"})
        .enqueue_final_response("您的订单 ORD-1 已发货。", intent="order_query")
    )
    agent, store = _sql_agent(engine, None, client)
    result = agent.chat("查订单 ORD-1")
    assert "已发货" in result.reply

    loaded = store.load("u1", "s1")
    roles = [m["role"] for m in loaded.messages]
    assert roles == ["user", "assistant"]  # 模型历史不含中间 tool 消息
    assert loaded.messages[-1]["content"] == "您的订单 ORD-1 已发货。"
    assert loaded.messages[-1].get("metadata", {}).get("schema") == 2

    # 数据库审计正本：完整消息序保留（业务 tool 结果 + final 接受结果）
    with engine.connect() as conn:
        rows = conn.execute(
            select(chat_messages.c.role).where(chat_messages.c.session_key == "u1/s1")
        ).scalars().all()
    assert rows == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    # tool 数=2（query_order 结果 + final 接受结果）；
    # assistant 数=3（两个 tool-call 步骤 + 最终答复，均在审计增量中）
    assert rows.count("tool") == 2
    assert rows.count("assistant") == 3
    assert rows[-1] == "assistant"  # 最终 assistant 在审计增量中（Review 修复）


def test_sql_mixed_calls_persist_one_assistant_and_exact_tool_results():
    """混合业务+final 调用：一条 assistant tool-call 消息 + 每个 call 恰好一个结果。"""
    from app.agent.tools import registry

    registry._TOOL_MAP["query_product"] = (
        lambda keyword, ctx=None: {"success": True, "products": []}
    )
    engine = _sql_engine()
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _response_mixed())
        .enqueue_final_response("为您找到耳机商品。", intent="product_consult")
    )
    agent, store = _sql_agent(engine, None, client)
    result = agent.chat("有什么耳机")
    assert result.reply == "为您找到耳机商品。"

    with engine.connect() as conn:
        rows = conn.execute(
            select(chat_messages.c.role, chat_messages.c.content)
            .where(chat_messages.c.session_key == "u1/s1")
            .order_by(chat_messages.c.seq)
        ).all()
    contents = [json.loads(r.content) for r in rows]
    tool_call_msgs = [m for m in contents if m.get("role") == "assistant" and m.get("tool_calls")]
    # 混合步骤只写一条 assistant tool-call 消息（含全部 3 个 call）；
    # 第二步的 final_response 是独立一条（合法消息序）
    mixed = next(m for m in tool_call_msgs if len(m["tool_calls"]) == 3)
    assert [tc["id"] for tc in mixed["tool_calls"]] == ["cb1", "cf1", "cf2"]
    tool_results = {
        m["tool_call_id"]: m["content"] for m in contents if m.get("role") == "tool"
    }
    # 每个 call ID 全程恰好一个结果（无重复、无缺失）
    all_ids = [tc["id"] for m in tool_call_msgs for tc in m["tool_calls"]]
    assert sorted(tool_results) == sorted(all_ids)
    assert len(tool_results) == len(set(tool_results))
    # 混合场景（含业务工具）：final 调用退回 PREMATURE（观察业务结果后重新终答）
    assert "FINAL_RESPONSE_PREMATURE" in tool_results["cf1"]
    assert "FINAL_RESPONSE_PREMATURE" in tool_results["cf2"]


def _response_mixed():
    """混合响应：query_product + 两个 final_response 调用。"""
    from tests.unit.conftest import _Choice, _Response, _ToolCall, _ToolCallMessage

    return _Response([_Choice(_ToolCallMessage("", [
        _ToolCall("cb1", "query_product", '{"keyword": "耳机"}'),
        _ToolCall("cf1", "final_response", '{"intent": "product_consult", "reply": "x", "requires_human": false}'),
        _ToolCall("cf2", "final_response", '{"intent": "other", "reply": "y", "requires_human": false}'),
    ]))])


def test_outbox_contains_no_confirmation_token():
    """退款两轮流程：outbox/审计消息不含 confirmation token。"""
    from app.agent.tools import registry

    registry._TOOL_MAP["apply_refund"] = _real_apply_refund
    settings.refund_confirmation_required = True
    engine = _sql_engine()

    # 轮 1：签发待确认
    client1 = (
        FakeChatClient()
        .enqueue_tool_call("c1", "apply_refund",
                           '{"order_id": "ORD-20240115-001", "reason": "质量问题"}')
        .enqueue_final_response("已登记退款申请，请确认。", intent="return_request")
    )
    agent1, _ = _sql_agent(engine, None, client1)
    agent1.chat("我要退掉订单 ORD-20240115-001，质量问题")

    # 轮 2：用户确认 → 服务端注入 token → committed
    client2 = (
        FakeChatClient()
        .enqueue_tool_call("c2", "apply_refund",
                           '{"order_id": "ORD-20240115-001", "reason": "质量问题"}')
        .enqueue_final_response("您的退款已提交成功。", intent="return_request")
    )
    agent2, _ = _sql_agent(engine, None, client2)
    agent2.chat("确认")
    committed = _last_committed_payload()
    assert committed is not None and committed.get("confirmed") is True

    # outbox 与审计消息均不含 token
    with engine.connect() as conn:
        payloads = [
            row.payload for row in conn.execute(select(outbox_rows)).mappings().all()
        ]
    blob = json.dumps(payloads, ensure_ascii=False)
    for pending in _issued_tokens():
        assert pending not in blob


def _last_committed_payload() -> dict | None:
    from app.agent.tools.refund import _confirmation_store

    store = _confirmation_store()
    if isinstance(store, __import__("app.security.refunds", fromlist=["x"]).InProcessConfirmationStore):
        for result, _expires in store._results.values():
            if result.get("confirmed"):
                return result
    return None


def _issued_tokens() -> list[str]:
    from app.agent.tools.refund import _confirmation_store

    store = _confirmation_store()
    if isinstance(store, __import__("app.security.refunds", fromlist=["x"]).InProcessConfirmationStore):
        return [token for token, (_p, _e) in store._data.items()]
    return []


def _real_apply_refund(order_id, reason, ctx=None, confirmation_token=None,
                       idempotency_key=None, refund_id=None):
    """直连真实退款工具（复用 mock 网关与确认存储）。"""
    from app.agent.tools.refund import apply_refund

    return apply_refund(
        order_id, reason, ctx=ctx, confirmation_token=confirmation_token,
        idempotency_key=idempotency_key, refund_id=refund_id,
    )


# ============================================================
# 终答协议
# ============================================================
def test_plain_text_output_gets_protocol_correction_then_recovers():
    """纯文本输出 → 协议纠错重试 → final_response 成功。"""
    client = (
        FakeChatClient()
        .enqueue_chat("我直接输出文本。")
        .enqueue_final_response("好的，这是最终答复。", intent="after_sale")
    )
    engine = _sql_engine()
    agent, _ = _sql_agent(engine, None, client)
    result = agent.chat("确认一下")
    assert result.reply == "好的，这是最终答复。"
    # 第一次调用后窗口出现协议纠错 system 说明
    second_messages = client.calls[1][1]["messages"]
    assert any("final_response 工具提交最终答复" in str(m.get("content", ""))
               for m in second_messages)


def test_mixed_call_message_order_and_multi_final_errors():
    """混合调用 + 多 final：合法消息序 + 多个 final 全部协议错误。"""
    client = (
        FakeChatClient()
        .enqueue_callable(lambda kind, kwargs: _response_mixed())
        .enqueue_final_response("为您找到耳机商品。", intent="product_consult")
    )
    agent, _ = _sql_agent(_sql_engine(), None, client)
    result = agent.chat("有什么耳机")
    assert result.reply == "为您找到耳机商品。"
    second_messages = client.calls[1][1]["messages"]
    # 一条 assistant tool-call（3 个 call）→ 3 个 tool 结果（cf1/cf2 为协议错误）
    tool_msgs = {m["tool_call_id"]: m["content"] for m in second_messages
                 if m.get("role") == "tool"}
    assert set(tool_msgs) == {"cb1", "cf1", "cf2"}
    assert "FINAL_RESPONSE_PREMATURE" in tool_msgs["cf1"]
    assert "FINAL_RESPONSE_PREMATURE" in tool_msgs["cf2"]
    assert "query_product" in tool_msgs["cb1"] or "success" in tool_msgs["cb1"]


def test_forced_finalize_failure_goes_deterministic_fallback(monkeypatch):
    """强制终答失败 → ForcedFinalizeFailed → 确定性转人工 fallback。"""

    monkeypatch.setattr(settings, "max_react_steps", 1)
    from app.agent.tools import registry

    monkeypatch.setitem(registry._TOOL_MAP, "query_product",
                        lambda keyword, ctx=None: {"success": True, "products": []})
    # 步骤1：业务工具；强制收尾：返回无 tool_calls 的纯文本 → 失败
    client = (
        FakeChatClient()
        .enqueue_tool_call("c1", "query_product", {"keyword": "耳机"})
        .enqueue_chat("还是纯文本")
    )
    agent, _ = _sql_agent(_sql_engine(), None, client)
    result = agent.chat("有什么耳机")
    assert result.requires_human is True
    assert result.confidence == 0.0
    assert result.reply  # 确定性话术


def test_validate_final_response_rejects_bad_args():
    args, error = validate_args({"reply": "缺 intent"})
    assert args is None and "FINAL_RESPONSE_INVALID" in error


def validate_args(payload):
    from app.agent.final_response import validate_final_response

    return validate_final_response(payload)


# ============================================================
# 退款确认闸门
# ============================================================
def _gate_pending():
    return [
        PendingRefund(refund_id="r1", order_id="ORD-A", reason="质量问题",
                      token="t1"),
    ]


def test_gate_confirm_cancel_ambiguous_matrix():
    # 明确确认
    d = judge_refund_confirmation("确认退款", _gate_pending())
    assert d.action == "confirm" and d.payload.order_id == "ORD-A"
    # 简短肯定 + 唯一待确认 → confirm
    d = judge_refund_confirmation("好", _gate_pending())
    assert d.action == "confirm"
    # 否定/取消优先
    d = judge_refund_confirmation("取消吧", _gate_pending())
    assert d.action == "cancel"
    d = judge_refund_confirmation("不要了", _gate_pending())
    assert d.action == "cancel"
    # 答非所问 → ambiguous
    d = judge_refund_confirmation("退款到哪了", _gate_pending())
    assert d.action == "ambiguous" and not d.clarification
    # 多笔待确认未带订单号 → 确定性澄清
    multi = _gate_pending() + [
        PendingRefund(refund_id="r2", order_id="ORD-B", reason="尺寸不合适",
                      token="t2"),
    ]
    d = judge_refund_confirmation("确认", multi)
    assert d.action == "ambiguous" and "订单号" in d.clarification
    # 多笔待确认 + 订单号 → 定位成功
    d = judge_refund_confirmation("确认 ORD-B 的退款", multi)
    assert d.action == "confirm" and d.payload.order_id == "ORD-B"
    # 否定确认、答非所问不能因包含“确认/提交/确定”字样而放行。
    single = _gate_pending()
    for text in ("我不确认退款", "不能确认", "暂时不能确认",
                 "如何提交投诉材料", "我确定物流还没到"):
        d = judge_refund_confirmation(text, single)
        assert d.action == "ambiguous", (text, d)
    # 点名另一笔订单时不得回退到唯一 pending 条目。
    d = judge_refund_confirmation("确认 ORD-NOT-THIS", single)
    assert d.action == "ambiguous"
    # 疑问句不是授权；否定“取消”不能被误当成取消；短订单号也须严格绑定。
    assert judge_refund_confirmation("如何确认退款？", single).action == "ambiguous"
    assert judge_refund_confirmation("是否确认退款", single).action == "ambiguous"
    assert judge_refund_confirmation("不取消退款，我确认退款", single).action == "confirm"
    short_id = [PendingRefund(refund_id="r3", order_id="O1", reason="x", token="t3")]
    assert judge_refund_confirmation("确认退款 O2", short_id).action == "ambiguous"
    # 无待确认 → none
    assert judge_refund_confirmation("确认", []).action == "none"


def test_executor_injects_token_only_on_server_confirm():
    """confirm 判定 → 执行器内部通道注入 token；模型参数面无保留字段。"""
    executed = {}

    def spy_refund(order_id, reason, ctx=None, confirmation_token=None,
                   idempotency_key=None, refund_id=None):
        executed.update({
            "token": confirmation_token, "key": idempotency_key,
            "order": order_id, "confirmed": True,
        })
        return {"success": True, "confirmed": True, "idempotency_key": idempotency_key}

    from app.agent.tools import registry

    registry._TOOL_MAP["apply_refund"] = spy_refund
    settings.refund_confirmation_required = True

    engine = _sql_engine()
    client = FakeChatClient().enqueue_final_response("退款完成。", intent="return_request")
    agent, _ = _sql_agent(engine, None, client)
    # 模拟服务端已判定 confirm（pipeline 闸门产物）
    agent._confirmation_store = lambda: object()  # 阻断真实存储访问（本测试手工注入）
    agent.current_confirmation = ConfirmationDecision(
        action="confirm",
        payload=PendingRefund(refund_id="rid-9", order_id="ORD-9",
                              reason="质量问题", token="tok-9"),
        matched_order_id="ORD-9",
    )
    from app.agent.tools.batch_executor import ToolTurnState
    from app.agent.turn_budget import TurnBudget
    from app.agent.tools.batch_executor import ToolTurnState as _TTS

    state = _TTS()
    state.refund_confirm = {
        "action": "confirm", "token": "tok-9", "refund_id": "rid-9",
        "order_id": "ORD-9", "reason": "质量问题",
    }
    outcome = agent.tool_executor.execute(
        [{"id": "c1", "name": "apply_refund",
          "arguments": '{"order_id": "ORD-9", "reason": "质量问题"}'}],
        state, agent.ctx, agent.tool_manager,
    )
    payload = json.loads(outcome[0].result)
    assert payload.get("confirmed") is True
    assert executed["token"] == "tok-9" and executed["key"] == "rid-9"
    # 模型可见参数（审计）不含保留字段
    assert "confirmation_token" not in outcome[0].arguments


def test_executor_blocks_writes_on_ambiguous():
    from app.agent.tools.batch_executor import ToolBatchExecutor, ToolTurnState

    class NoExecManager:
        def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
            raise AssertionError("ambiguous 时写工具不得执行")

    executor = ToolBatchExecutor(parallelism=1, max_concurrent=2)
    state = ToolTurnState()
    state.refund_confirm = {"action": "ambiguous"}
    outcomes = executor.execute(
        [{"id": "c1", "name": "apply_refund",
          "arguments": '{"order_id": "O1", "reason": "x"}'}],
        state, None, NoExecManager(),
    )
    assert outcomes[0].skipped is True
    assert "REFUND_CONFIRMATION_REQUIRED" in outcomes[0].result
    executor.close()


def test_executor_blocks_writes_after_cancel_and_on_confirm_target_mismatch():
    from app.agent.tools.batch_executor import ToolBatchExecutor, ToolTurnState

    class NoExecManager:
        def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
            raise AssertionError("cancel/target mismatch 时写工具不得执行")

    executor = ToolBatchExecutor(parallelism=1, max_concurrent=2)
    try:
        state = ToolTurnState()
        state.refund_confirm = {"action": "cancel"}
        out = executor.execute(
            [{"id": "c1", "name": "apply_refund",
              "arguments": '{"order_id": "O1", "reason": "x"}'}],
            state, None, NoExecManager(),
        )[0]
        assert out.skipped and "REFUND_CONFIRMATION_REQUIRED" in out.result

        state = ToolTurnState()
        state.refund_confirm = {
            "action": "confirm", "token": "t", "refund_id": "rid",
            "order_id": "O1", "reason": "原退款原因",
        }
        out = executor.execute(
            [{"id": "c2", "name": "apply_refund",
              "arguments": '{"order_id": "O2", "reason": "原退款原因"}'}],
            state, None, NoExecManager(),
        )[0]
        assert out.skipped and "REFUND_CONFIRMATION_TARGET_MISMATCH" in out.result
    finally:
        executor.close()


def test_registry_rejects_reserved_tool_args():
    from app.agent.tools.registry import execute_tool

    result = execute_tool("apply_refund", {
        "order_id": "O1", "reason": "x", "confirmation_token": "evil",
    })
    assert "RESERVED_TOOL_ARGUMENT" in result


def test_token_payload_binding_and_expiry_and_replay():
    from app.security.refunds import (
        ConfirmationInvalid,
        InProcessConfirmationStore,
        RefundConfirmation,
    )

    store = InProcessConfirmationStore()

    def executor(oid, rsn, key):
        return {"success": True}

    # 用户绑定：载荷 u1，confirm 用 u2 → 拒绝
    controller = RefundConfirmation(store)
    issued = controller.request("O1", "质量问题", user_id="u1", session_id="s1")
    with pytest.raises(ConfirmationInvalid):
        controller.confirm(
            issued["confirmation_token"], executor,
            order_id="O1", refund_id=issued["refund_id"],
            user_id="u2", session_id="s1",
        )

    # 过期：TTL=0 → take 失败
    expired_ctl = RefundConfirmation(store, ttl_seconds=0)
    expired = expired_ctl.request("O2", "x", user_id="u1", session_id="s1")
    time.sleep(0.01)
    with pytest.raises(ConfirmationInvalid):
        expired_ctl.confirm(
            expired["confirmation_token"], executor,
            order_id="O2", refund_id=expired["refund_id"],
            user_id="u1", session_id="s1",
        )

    # 绑定失败尝试同样烧掉 token（单次语义：错误尝试令牌即失效），
    # 正常确认需重新签发
    issued2 = controller.request("O1", "质量问题", user_id="u1", session_id="s1")
    # 重放：首次成功 → 再次同 refund_id confirm → replayed=true（幂等）
    ok = controller.confirm(
        issued2["confirmation_token"], executor,
        order_id="O1", refund_id=issued2["refund_id"],
        user_id="u1", session_id="s1",
    )
    assert ok.get("confirmed") is True
    # replay 也必须提供同一 token（仅 refund_id + bogus token 不能读取结果
    # 账本），否则攻击者可越权获得他人的退款结果。
    with pytest.raises(ConfirmationInvalid):
        controller.confirm(
            "any-token", executor,
            order_id="O1", refund_id=issued2["refund_id"],
            user_id="u1", session_id="s1",
        )
    replay = controller.confirm(
        issued2["confirmation_token"], executor,
        order_id="O1", refund_id=issued2["refund_id"],
        user_id="u1", session_id="s1",
    )
    assert replay.get("replayed") is True

    # 会话注册表：确认成功后待确认条目被清除
    remaining = {e["refund_id"] for e in store.get_session_refunds("u1", "s1")}
    assert issued2["refund_id"] not in remaining


def test_redis_confirmation_legacy_fallback_is_atomic():
    from app.security.refunds import RedisConfirmationStore

    class LegacyRedis:
        def __init__(self):
            self.values = {"refund_confirm:t1": b"payload"}
            self.eval_calls = 0

        def getdel(self, key):
            raise AttributeError("GETDEL unavailable")

        def eval(self, script, key_count, key):
            assert key_count == 1 and "GET" in script and "DEL" in script
            self.eval_calls += 1
            return self.values.pop(key, None)

    redis = LegacyRedis()
    store = RedisConfirmationStore(redis)
    assert store.take("t1") == "payload"
    assert store.take("t1") is None
    assert redis.eval_calls == 2


def test_mcp_refund_confirmation_uses_internal_meta_without_schema_fields(monkeypatch):
    """MCP 第二段确认：凭证走 meta 内部通道，公开参数仍只有订单/原因。"""
    from types import SimpleNamespace

    from app.agent.context import ToolContext
    from app.agent.tools.manager import ToolManager
    from app.mcp_client.actor import SCOPE_REFUND_WRITE, issue_actor_token
    from mcp_server import server as mcp_server

    calls = []

    class FakeMcpClient:
        def connect(self):
            return [{
                "type": "function", "function": {
                    "name": "apply_refund", "description": "refund",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["order_id", "reason"],
                    },
                },
            }]

        def call_tool(self, name, arguments, timeout=None, *, actor_token=None,
                      write=False, internal_args=None):
            calls.append((name, dict(arguments), actor_token, write,
                          dict(internal_args or {})))
            return json.dumps({"success": True, "confirmed": True})

        def close(self):
            pass

    monkeypatch.setattr(settings, "mcp_actor_secret",
                        "actor-secret-0123456789-abcdefghijklmnop")
    manager = ToolManager(
        use_mcp=True, mcp_server_url="http://fake",
        mcp_client=FakeMcpClient(),
    )
    try:
        result = manager.execute_tool(
            "apply_refund",
            {"order_id": "O1", "reason": "质量问题"},
            ToolContext(user_id="u1", session_id="s1"),
            internal_args={
                "confirmation_token": "tok",
                "refund_id": "rid",
                "idempotency_key": "rid",
            },
        )
        assert json.loads(result)["confirmed"] is True
        name, arguments, actor, write, internal = calls[-1]
        assert name == "apply_refund" and write is True
        assert arguments == {"order_id": "O1", "reason": "质量问题"}
        assert internal["confirmation_token"] == "tok"
        assert "confirmation_token" not in arguments

        token = issue_actor_token("u1", "s1", (SCOPE_REFUND_WRITE,))
        captured = {}

        def fake_apply(order_id, reason, ctx, **kwargs):
            captured.update({"order_id": order_id, "reason": reason,
                             "ctx": ctx, **kwargs})
            return {"success": True, "confirmed": True}

        monkeypatch.setattr(mcp_server, "_apply_refund", fake_apply)
        mcp_ctx = SimpleNamespace(request_context=SimpleNamespace(meta={
            "actor": token,
            "internal_args": {
                "confirmation_token": "tok",
                "refund_id": "rid",
                "idempotency_key": "rid",
            },
        }))
        out = json.loads(mcp_server.apply_refund("O1", "质量问题", mcp_ctx))
        assert out["confirmed"] is True
        assert captured["confirmation_token"] == "tok"
        assert captured["refund_id"] == "rid"
    finally:
        manager.close()


def test_mcp_pending_meta_is_synced_without_exposing_token(monkeypatch):
    from app.agent.context import ToolContext
    from app.agent.refund_gate import get_session_pending
    from app.agent.tools import refund as refund_tool
    from app.agent.tools.manager import ToolManager
    from app.mcp_client.client import MCPToolResult
    from app.security.refunds import InProcessConfirmationStore

    confirmation_store = InProcessConfirmationStore()
    monkeypatch.setattr(refund_tool, "_store_instance", confirmation_store)
    monkeypatch.setattr(settings, "mcp_actor_secret",
                        "actor-secret-0123456789-abcdefghijklmnop")

    class FakeMcpClient:
        def connect(self):
            return [{
                "type": "function", "function": {
                    "name": "apply_refund", "description": "refund",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["order_id", "reason"],
                    },
                },
            }]

        def call_tool(self, name, arguments, **kwargs):
            public = json.dumps({
                "success": True,
                "status": "pending_confirmation",
                "refund_id": "rid-1",
            })
            return MCPToolResult(public, {"refund_confirmation": {
                "refund_id": "rid-1",
                "order_id": "O1",
                "reason": "质量问题",
                "confirmation_token": "secret-token",
                "expires_in_seconds": 300,
            }})

        def close(self):
            pass

    manager = ToolManager(
        use_mcp=True, mcp_server_url="http://fake", mcp_client=FakeMcpClient(),
    )
    try:
        result = manager.execute_tool(
            "apply_refund", {"order_id": "O1", "reason": "质量问题"},
            ToolContext(user_id="u1", session_id="s1"),
        )
        assert "secret-token" not in result
        assert type(result) is str
        pending = get_session_pending(confirmation_store, "u1", "s1")
        assert len(pending) == 1
        assert pending[0].token == "secret-token"
    finally:
        manager.close()


# ============================================================
# 记忆任务
# ============================================================
def test_sql_watermark_never_regresses_with_stale_agent_state():
    """worker 推进水位后，旧 Agent 状态保存不得写小（数据库内单调 clamp）。"""
    engine = _sql_engine()
    store = SqlSessionStore(engine)
    store.save("u1", "s1", SessionState(
        session_id="s-uuid-1", user_id="u1",
        messages=[{"role": "user", "content": "hi"}],
        version=0,
    ), new_messages=[{"role": "user", "content": "hi"}],
        enqueue_memory_job=True,
    )
    jobs = SqlMemoryJobStore(engine)
    claimed = jobs.claim("w1")
    assert claimed and claimed[0]["session_uuid"] == "s-uuid-1"
    jobs.advance_watermark("u1/s1", claimed[0]["through_seq"])

    # 旧 Agent（version 已推进，consolidated_len=0）再保存 → 水位不得回退
    loaded = store.load("u1", "s1")
    store.save("u1", "s1", SessionState(
        session_id="s-uuid-1", user_id="u1", messages=loaded.messages,
        version=loaded.version, consolidated_len=0,
    ), new_messages=[{"role": "assistant", "content": "答"}])

    with engine.connect() as conn:
        value = conn.execute(
            select(memory_jobs.c.through_seq).limit(1)
        ).scalar_one()
        from sqlalchemy import text

        cl = conn.execute(
            text("SELECT consolidated_len FROM sessions WHERE session_key='u1/s1'")
        ).scalar()
    assert cl >= value  # 水位单调不回退


def test_sql_worker_marks_stale_job_obsolete_after_reset():
    """reset 后旧任务按 obsolete 处理，绝不读取新会话消息。"""
    engine = _sql_engine()
    store = SqlSessionStore(engine)
    store.save("u1", "s1", SessionState(
        session_id="uuid-A", user_id="u1",
        messages=[{"role": "user", "content": "旧"}],
    ), new_messages=[{"role": "user", "content": "旧"}],
        enqueue_memory_job=True,
    )
    jobs = SqlMemoryJobStore(engine)
    claimed = jobs.claim("w1")
    assert claimed[0]["session_uuid"] == "uuid-A"

    # 模拟 reset：同 session_id 重建（新 uuid）；多 Pod 场景下旧 pod 手里
    # 还持有已领取的 job 行——uuid 比对是唯一防线
    from sqlalchemy import text as _text

    with engine.begin() as conn:
        conn.execute(_text(
            "UPDATE sessions SET session_uuid='uuid-B' WHERE session_key='u1/s1'"
        ))
    called = []
    worker = MemoryJobWorker(
        jobs,
        lambda uid: (_ for _ in ()).throw(AssertionError("obsolete 任务不得读取消息")),
        None, "fake", worker_id="w1",
    )
    # 手动标记 claimed 为 pending 以便 process_once 领取（模拟崩溃前已入队）
    from sqlalchemy import text as _t

    with engine.begin() as conn:
        conn.execute(_t(
            "UPDATE memory_jobs SET status='pending', lease_until=NULL"
        ))
    worker.process_once()
    with engine.connect() as conn:
        status = conn.execute(
            select(memory_jobs.c.status).limit(1)
        ).scalar_one()
    assert status == "obsolete"
    assert called == []  # 新会话消息从未被读取


def test_sql_worker_invalidates_redis_cache_on_advance():
    """worker 推进水位后同步失效会话热缓存。"""
    engine = _sql_engine()
    redis = FakeRedis(server=FakeServer())
    store = SqlSessionStore(engine, redis=redis, hot_ttl=60)
    store.save("u1", "s1", SessionState(
        session_id="uuid-1", user_id="u1",
        messages=[{"role": "user", "content": "hi"}],
    ), new_messages=[{"role": "user", "content": "hi"}],
        enqueue_memory_job=True,
    )
    assert redis.exists("session:u1/s1") == 1
    jobs = SqlMemoryJobStore(engine, redis=redis)
    claimed = jobs.claim("w1")
    jobs.advance_watermark("u1/s1", claimed[0]["through_seq"])
    assert redis.exists("session:u1/s1") == 0  # 缓存已失效


def test_file_queue_real_worker_complete_fail_lease_idempotent(tmp_path, monkeypatch):
    """Windows 文件队列：真实 worker 完成失败重试租约接管幂等。"""
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    queue_dir = tmp_path / "memory" / "jobs"
    store = FileMemoryJobStore(str(queue_dir))

    payload = [
        {"role": "user", "content": "我叫雷腾，喜欢红色"},
        {"role": "assistant", "content": "已记住您的偏好。"},
    ]
    store.enqueue("u1/s1", "u1", 2, session_uuid="uuid-1", messages=payload)
    store.enqueue("u1/s1", "u1", 2, session_uuid="uuid-1", messages=payload)  # 幂等
    assert len(store.jobs_snapshot()) == 1

    # 真实 worker：完成
    captured = []

    class FakeLTM:
        def extract_and_save(self, client, model, messages, summary):
            captured.append(messages)

    worker = MemoryJobWorker(store, lambda uid: FakeLTM(), None, "fake", "w1")
    done = worker.process_once()
    assert done == 1 and len(captured) == 1
    assert store.jobs_snapshot() == []  # 完成即出队
    done_log = (queue_dir / "memory_jobs.done.jsonl").read_text(encoding="utf-8")
    assert "u1/s1" in done_log  # 完成记录

    # 失败重试：LTM 工厂抛错 → attempts+1 → 未到期不再领取
    store.enqueue("u2/s2", "u2", 3, session_uuid="uuid-2", messages=payload)

    class BoomLTM:
        def extract_and_save(self, *a, **k):
            raise RuntimeError("boom")

    worker2 = MemoryJobWorker(store, lambda uid: BoomLTM(), None, "fake", "w2")
    assert worker2.process_once() == 0
    entry = store.jobs_snapshot()[0]
    assert entry["status"] == "pending" and entry["attempts"] == 1
    assert entry["next_run_at"]  # 退避时间已设置
    assert store.claim("w2") == []  # 未到期不领取

    # 租约接管：过期 processing 可被其他 worker 领取
    store._upsert({**entry, "next_run_at": ""})  # 清退避，便于领取
    first = store.claim("w-a")[0]
    assert first["status"] == "processing"
    assert store.claim("w-b") == []  # 租约未过期
    store._upsert({**first, "lease_until": "2000-01-01T00:00:00"})  # 过期
    taken = store.claim("w-b")
    assert taken and taken[0]["id"] == first["id"]

    # reset 清理
    store.enqueue("u3/s3", "u3", 1, session_uuid="u", messages=payload)
    assert store.purge_session("u3/s3") == 1
    assert all(j["session_key"] != "u3/s3" for j in store.jobs_snapshot())


def test_agent_reset_cleans_refund_pending_and_file_jobs(tmp_path, monkeypatch):
    """reset：同 session 首写正常；待确认退款状态与文件任务一并清理。"""
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path / "memory"))
    settings.session_dir = str(tmp_path / "sessions")
    settings.evolve_capture_enabled = False
    from app.agent.tools.refund import _confirmation_store

    store = _confirmation_store()
    store.put_session_refund("u1", "session", {
        "refund_id": "r1", "order_id": "O1", "reason": "x", "token": "t1",
        "user_id": "u1", "session_id": "session",
    }, 300)
    assert store.get_session_refunds("u1", "session")

    client = FakeChatClient().enqueue_final_response("答复", intent="other")
    agent = EcomAgent(user_id="u1", client=client, memory_enabled=False,
                      use_mcp=False)
    agent.context_builder._window = 8192
    agent.reset()
    assert store.get_session_refunds("u1", "session") == []  # 待确认已清

    # 相同 session key 首写正常
    client2 = FakeChatClient().enqueue_final_response("新答复", intent="other")
    agent2 = EcomAgent(user_id="u1", client=client2, memory_enabled=False,
                       use_mcp=False)
    result = agent2.chat("你好")
    assert result.reply == "新答复"


# ============================================================
# 事实校验（主题锚点 + 单位）
# ============================================================
def test_fact_conflict_same_topic_detected():
    evidence = [
        "退款将在7天内原路退回。",
        "退款周期为15天，具体以银行处理为准。",
    ]
    cleaned, verdict = ground_reply("退款将在7天内原路退回。", evidence)
    assert verdict.conflicts  # 同主题（退款）同单位（天）多值 → 冲突
    assert "7天" not in cleaned  # 冲突句删除并转核实话术
    assert "核实" in cleaned


def test_fact_conflict_ignores_leading_discourse_modifier():
    evidence = [
        "本次退款7天内原路退回。",
        "退款周期为15天，具体以银行处理为准。",
    ]
    cleaned, verdict = ground_reply("本次退款7天内原路退回。", evidence)
    assert verdict.conflicts
    assert "7天" not in cleaned


def test_fact_no_conflict_across_different_topics():
    evidence = [
        "退货支持7天无理由。",
        "到货周期为15天。",
    ]
    cleaned, verdict = ground_reply("退货支持7天无理由。到货周期为15天。", evidence)
    assert not verdict.conflicts  # 不同主题同单位不误报
    assert "7天" in cleaned and "15天" in cleaned


def test_fact_conflict_does_not_remove_same_value_from_other_topic():
    evidence = [
        "退款将在7天内原路退回。",
        "退款周期为15天。",
        "退货支持7天无理由。",
    ]
    cleaned, verdict = ground_reply(
        "退款将在7天内原路退回。退货支持7天无理由。", evidence,
    )
    assert verdict.conflicts == ["7天"]
    assert "退款将在7天" not in cleaned
    assert "退货支持7天" in cleaned


def test_fact_single_supporting_evidence_passes():
    evidence = ["钻石会员专属客服的响应时效SLO是30秒内接入。"]
    cleaned, verdict = ground_reply("专属客服30秒内接入。", evidence)
    assert not verdict.conflicts and verdict.removed_sentences == 0
    assert cleaned == "专属客服30秒内接入。"

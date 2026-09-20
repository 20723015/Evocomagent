"""沙箱↔Agent 采集契约测试（评测链路回归防护）。

背景：单 Agent 重构把观测字段从实例属性（_turn_state / _last_citation_verdict）
移进了 AgentTurnContext，沙箱读旧属性名 → 引用真实性指标全数
假阴性、来源集合恒空。本文件用**真实** EcomAgent（FakeChatClient 脚本化，
零网络）锁定契约：沙箱逐轮读取的字段集合必须存在
且属于**当轮**（拦截轮不得读到上一轮的陈旧上下文）。

契约字段（app/agent/turn_context.AgentTurnContext）：
react_steps / steps_margin_hint / forced_finalize / protocol_corrections /
citation_verdict / sources——新增采集维度必须先登记进快照。
"""

from __future__ import annotations

from app.agent.chat import EcomAgent
from app.config.settings import settings
from app.security.scope_gate import SCOPE_BLOCK_REPLY
from tests.unit.conftest import FakeChatClient

SANDBOX_CONTRACT_FIELDS = (
    "react_steps", "steps_margin_hint", "forced_finalize",
    "protocol_corrections", "citation_verdict", "sources",
)


def _agent(tmp_path, client):
    return EcomAgent(
        session_path=str(tmp_path / "contract.json"),
        client=client, memory_enabled=False, use_mcp=False,
    )


# ============================================================
# 单 Agent：契约字段存在且属于当轮
# ============================================================
def test_single_agent_turn_exposes_sandbox_contract_fields(
    tmp_path, reset_settings,
):
    """正常轮：_last_turn_ctx 满足沙箱契约（六个字段全存在）。"""
    client = FakeChatClient().enqueue_final_response(
        "您好，请问有什么可以帮您？", intent="greeting",
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("你好")

    assert result.reply.startswith("您好")
    ctx = agent._last_turn_ctx
    assert ctx is not None
    for field in SANDBOX_CONTRACT_FIELDS:
        assert hasattr(ctx, field), f"沙箱契约字段缺失: {field}"
    assert ctx.react_steps == 1
    assert isinstance(ctx.citation_verdict, dict)  # citation_check 默认开


def test_single_agent_blocked_turn_refreshes_contract_not_stale(
    tmp_path, reset_settings,
):
    """回归：拦截轮（scope_block）后不得读到上一轮的陈旧 ctx。

    修复前只有成功路径赋 _last_turn_ctx——多轮用例里被拦截轮会累计
    上一轮的步数、携带上一轮的引用 verdict（沙箱累计口径下步数翻倍、
    verdict 串轮）。现在 chat() 的 finally 统一刷新。
    """
    settings.business_only_scope = True
    client = (
        FakeChatClient()
        .enqueue_final_response("订单已发货。", intent="order_query")
        .enqueue_chat("no")  # 第二轮 scope 判定 → 拦截
    )
    agent = _agent(tmp_path, client)

    agent.chat("我的订单到了吗")
    first = agent._last_turn_ctx
    assert first.react_steps == 1

    agent.chat("今天天气怎么样")
    second = agent._last_turn_ctx
    # 拦截轮是全新 ctx：步数归零、无引用 verdict，而不是第一轮的旧值
    assert second is not first
    assert second.react_steps == 0
    assert second.citation_verdict is None


def test_single_agent_budget_fallback_collects_retrieved_sources(
    tmp_path, reset_settings, monkeypatch,
):
    """回归：预算兜底路径也回填 ctx.sources（沙箱来源采集依赖它）。

    react 循环第一步检索成功（state.sources 已有来源）、第二步预算打穿
    → 兜底返回。修复前兜底路径不回填，RunTrace.retrieved_sources 恒空。
    """
    from app.agent.citations import normalize_source
    from app.agent.tools import registry
    from app.agent.turn_budget import LLMBudgetExhausted

    monkeypatch.setitem(
        registry._TOOL_MAP, "search_knowledge",
        lambda query, ctx=None, **kwargs: {
            "success": True,
            "results": [{"doc": "退货政策.md", "content": "七天无理由"}],
        },
    )
    client = (
        FakeChatClient()
        .enqueue_tool_call("call_s", "search_knowledge", {"query": "退货政策"})
        .enqueue_error(LLMBudgetExhausted("预算耗尽"))  # 下一步预算打穿
    )
    agent = _agent(tmp_path, client)

    result = agent.chat("退货政策是什么")

    assert result.requires_human is True  # 预算兜底 → 转人工
    ctx = agent._last_turn_ctx
    assert ctx.budget_fallback is True
    assert normalize_source("退货政策.md") in ctx.sources


# ============================================================
# 沙箱写路径覆盖（2026-09-18 实测缺口）
# ============================================================
def test_sandbox_agent_has_lease_guard_for_write_tools(tmp_path, reset_settings):
    """沙箱必须为写工具注入租约守卫，否则写路径零覆盖。

    回归背景：写工具在 `manager._lease_gate_error` 处要求
    `ctx.lease_guard` 非空，沙箱不注入 → 所有写调用被
    SESSION_LOCK_REQUIRED 挡下，退款提交/撤回在 agent 级评测里
    从未真正执行过（由确认流用例实测暴露：outcome 恒为空 dict）。
    """
    from app.evaluation.dataset import EvalCase
    from app.evaluation.sandbox import Sandbox

    sb = Sandbox()
    case = EvalCase(id="lease_probe", description="", turns=["你好"])
    agent = sb._build_agent(sb.session_path_for(case.id), case)
    assert agent.ctx.lease_guard is not None, "沙箱未注入租约守卫，写工具会被门禁挡下"
    # no-op 守卫：调用不得抛（等价「始终持有租约」）
    agent.ctx.lease_guard()
    agent.close()


def test_sandbox_write_tool_passes_lease_gate(tmp_path, reset_settings):
    """端到端：沙箱内写工具不再返回 SESSION_LOCK_REQUIRED。"""
    from app.evaluation.dataset import EvalCase
    from app.evaluation.sandbox import Sandbox
    from app.agent.tools.registry import execute_tool
    from app.integrations.commerce import set_gateway
    from app.integrations.commerce.mock import MockCommerceGateway

    set_gateway(MockCommerceGateway())
    try:
        sb = Sandbox()
        case = EvalCase(id="lease_write", description="", turns=["x"],
                        actor_user_id="u6")
        agent = sb._build_agent(sb.session_path_for(case.id), case)
        raw = execute_tool(
            "submit_refund_application",
            {"order_id": "ORD-20240912-1001", "reason": "尺码不合适"},
            agent.ctx,
        )
        assert "SESSION_LOCK_REQUIRED" not in raw
        assert "REFUND_DRAFT_AWAITING_CONFIRMATION" in raw  # 草稿轮
        agent.close()
    finally:
        set_gateway(None)

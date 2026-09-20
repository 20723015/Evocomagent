"""单 Agent 全量优化契约测试（无网络，全本地 fake）。

验收点对应计划《测试与验收·单 Agent 契约测试》：
- 无工具一次同步 LLM 调用；工具场景=决策+终答，无二次结构化提取；
- final_response 参数不合法 → 结构化纠错回模型，可在剩余步数内修复；
- 最大步数强制 final_response；预算耗尽确定性 fallback（零 LLM）；
- 每轮历史只有一个最终 assistant 消息（content=回复，字段在 metadata）；
- 旧格式历史仅在模型上下文折叠，正本不修改；
- 不安全输出不进入会话；非法工具参数不执行工具；
- 未提交成功的退款不得宣称成功（程序改写 + 转人工）；
- close() 零 LLM；普通/SSE 共用完成器语义（Handoff 一致）。
"""

from __future__ import annotations

import json

from app.agent.chat import EcomAgent
from app.agent.fact_guard import ground_reply
from app.agent.final_response import validate_final_response
from app.agent.rag.evidence import EvidenceItem, rrf_merge
from app.agent.reliability import (
    CAP_CITATION_MISSING,
    ReliabilitySignal,
    compute_reliability,
)
from app.agent.token_budget import (
    budget_shares,
    estimate_tokens,
    trim_messages_to_budget,
)
from app.agent.write_ops import WriteOpTracker
from app.config.settings import settings
from app.schemas.response import IntentType
from tests.unit.conftest import FakeChatClient


def _agent(tmp_path, client, memory_enabled=False):
    agent = EcomAgent(
        session_path=str(tmp_path / "contract.json"), client=client,
        memory_enabled=memory_enabled, use_mcp=False,
    )
    agent.context_builder._window = 8192  # 契约测试不触发历史压缩
    return agent


# ============================================================
# 终答协议：无工具 / 多工具 / 纠错 / 强制 / fallback
# ============================================================
def test_no_tool_turn_is_single_llm_call(tmp_path, reset_settings):
    """无工具请求：一次同步 LLM 调用（final_response 即答），无二次提取。"""
    client = FakeChatClient().enqueue_final_response(
        "您好，请问有什么可以帮您？", intent="greeting",
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("你好")

    assert result.reply == "您好，请问有什么可以帮您？"
    assert result.intent == IntentType.GREETING
    assert len(client.calls) == 1
    kinds = [kind for kind, _ in client.calls]
    assert "parse" not in kinds  # 二次结构化提取已删除


def test_tool_turn_is_decision_plus_final_only(tmp_path, reset_settings, monkeypatch):
    """单工具场景：一次工具决策 + 一次终答，无其他 LLM 调用。"""
    from app.agent.tools import registry

    def fake_query_order(order_id, ctx=None):
        return {"success": True, "code": "ORDER_FOUND",
                "order": {"order_id": order_id, "status": "已发货"}}

    monkeypatch.setitem(registry._TOOL_MAP, "query_order", fake_query_order)
    client = (
        FakeChatClient()
        .enqueue_tool_call("call_1", "query_order", {"order_id": "ORD-1"})
        .enqueue_final_response("您的订单 ORD-1 已发货。", intent="order_query")
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("帮我查订单 ORD-1")

    assert "已发货" in result.reply
    assert len(client.calls) == 2  # 决策 + 终答；无提取调用


def test_invalid_final_response_returns_correction_and_model_repairs(
    tmp_path, reset_settings,
):
    """final_response 参数不合法 → 结构化纠错回模型；修复后正常终答。"""
    client = (
        FakeChatClient()
        .enqueue_tool_call("call_f1", "final_response", {"reply": "缺少 intent"})
        .enqueue_final_response("好的，已确认。", intent="after_sale")
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("确认一下")

    assert result.reply == "好的，已确认。"
    # 第二次调用模型的窗口里有纠错 tool 消息（稳定错误码可见，可自愈）
    second_call_messages = client.calls[1][1]["messages"]
    tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
    assert any("FINAL_RESPONSE" in json.dumps(m, ensure_ascii=False)
               for m in tool_msgs)


def test_max_steps_forces_final_response(tmp_path, reset_settings, monkeypatch):
    """达到最大步数 → 最后一轮只挂 final_response 强制收尾。"""
    monkeypatch.setattr(settings, "max_react_steps", 1)
    from app.agent.tools import registry

    monkeypatch.setitem(registry._TOOL_MAP, "query_product",
                        lambda keyword, ctx=None: {"success": True, "products": []})
    client = (
        FakeChatClient()
        .enqueue_tool_call("call_1", "query_product", {"keyword": "耳机"})
        .enqueue_tool_call("call_f", "final_response", {
            "intent": "product_consult", "reply": "为您找到耳机商品。",
            "requires_human": False,
        })
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("有什么耳机")
    assert result.reply == "为您找到耳机商品。"
    # 强制收尾那轮只挂 final_response 工具
    final_tools = client.calls[1][1].get("tools")
    assert [t["function"]["name"] for t in final_tools] == ["final_response"]


# ============================================================
# 步数余量感知：剩余步数（含当前步）≤2 → 注入一次 system 预告
# （零额外 LLM 调用；预告先行，最后通牒兜底语义不变）
# ============================================================
def _messages_json(kwargs) -> str:
    return json.dumps(kwargs.get("messages", []), ensure_ascii=False)


def test_steps_margin_hint_injected_once_before_last_two_steps(
    tmp_path, reset_settings, monkeypatch,
):
    """步数余量提示：时机=倒数第二步起，只注一次，零额外 LLM 调用。

    - 第 1/2 步的窗口无预告；第 3 步（remaining=2）起可见，内容含剩余步数；
    - 后续步沿用同一条窗口消息（出现次数=1，不重复注入），计数器增量=1；
    - LLM 调用数与无预告剧本完全一致（4 步决策 + 1 次强制终答）；
    - 触发步 SSE 状态语改为收尾话术；ctx 标记可观测并进 usage dict。
    """
    from app.agent.tools import registry
    from app.observability import metrics

    monkeypatch.setattr(settings, "max_react_steps", 4)
    monkeypatch.setitem(
        registry._TOOL_MAP, "query_product",
        lambda keyword, ctx=None: {"success": True, "products": []},
    )
    client = FakeChatClient()
    for i in range(4):  # 4 步都开业务工具 → 步数耗尽走强制终答
        client.enqueue_tool_call(f"call_{i}", "query_product", {"keyword": "耳机"})
    client.enqueue_tool_call("call_f", "final_response", {
        "intent": "product_consult", "reply": "为您找到耳机商品。",
        "requires_human": False,
    })
    agent = _agent(tmp_path, client)
    statuses: list[str] = []
    agent.emit_status = lambda text, step=0: statuses.append(text)
    counter_before = metrics.STEPS_MARGIN_HINTS._value.get()

    result = agent.chat("有什么耳机")

    assert result.reply == "为您找到耳机商品。"
    # 零额外 LLM 调用：4 个决策步 + 1 次强制终答，不多不少
    assert len(client.calls) == 5
    windows = [_messages_json(kwargs) for _, kwargs in client.calls]
    # 时机：第 1/2 步无预告；第 3 步（remaining=2）注入，文案含剩余步数
    assert "即将用尽" not in windows[0]
    assert "即将用尽" not in windows[1]
    assert "即将用尽" in windows[2] and "还剩 2 步" in windows[2]
    # 后续步沿用窗口中同一条预告：可见但不重复（每窗恰好 1 次）
    assert windows[3].count("即将用尽") == 1
    assert windows[4].count("即将用尽") == 1
    # 整个运行恰好一条预告（计数器增量=1）；ctx 标记可观测
    assert metrics.STEPS_MARGIN_HINTS._value.get() == counter_before + 1
    assert agent._last_turn_ctx.steps_margin_hint is True
    assert agent._last_turn_ctx.to_usage_dict()["steps_margin_hint"] is True
    # 触发步（第 3 步）SSE 状态语改为收尾话术，其余步保持原话术
    assert statuses[2] == "正在整理您的请求（第 3/4 步）"
    assert statuses[0] == "正在处理您的请求（第 1/4 步）"
    assert statuses[3] == "正在处理您的请求（第 4/4 步）"
    # 预告只进模型窗口：轮末折叠后不进会话历史
    roles = [m["role"] for m in agent.raw_messages]
    assert roles == ["user", "assistant"]


def test_short_turn_no_margin_hint(tmp_path, reset_settings, monkeypatch):
    """1-2 步即完成的短轮次：无预告注入（计数器零增量，ctx 标记 False）。"""
    from app.agent.tools import registry
    from app.observability import metrics

    def fake_query_order(order_id, ctx=None):
        return {"success": True, "code": "ORDER_FOUND",
                "order": {"order_id": order_id, "status": "已发货"}}

    monkeypatch.setitem(registry._TOOL_MAP, "query_order", fake_query_order)
    client = (
        FakeChatClient()
        .enqueue_tool_call("call_1", "query_order", {"order_id": "ORD-1"})
        .enqueue_final_response("您的订单 ORD-1 已发货。", intent="order_query")
    )
    agent = _agent(tmp_path, client)
    counter_before = metrics.STEPS_MARGIN_HINTS._value.get()

    result = agent.chat("帮我查订单 ORD-1")

    assert result.reply == "您的订单 ORD-1 已发货。"
    assert len(client.calls) == 2
    assert all("即将用尽" not in _messages_json(kwargs)
               for _, kwargs in client.calls)
    assert metrics.STEPS_MARGIN_HINTS._value.get() == counter_before
    assert agent._last_turn_ctx.steps_margin_hint is False


def test_final_response_validation_rejects_extra_fields():
    args, error = validate_final_response({
        "intent": "other", "reply": "r", "requires_human": False,
        "extra": "x",
    })
    assert args is None and error is not None
    args, error = validate_final_response({"intent": "other"})
    assert args is None and "FINAL_RESPONSE_INVALID" in error
    args, error = validate_final_response({
        "intent": "order_query", "reply": "回复", "requires_human": False,
    })
    assert args is not None and error is None


def test_budget_exhausted_zero_llm_fallback(tmp_path, reset_settings):
    """预算耗尽：确定性 fallback（requires_human + 可靠度 0.0），零 LLM。"""
    settings.turn_budget_seconds = -1
    client = FakeChatClient()
    agent = _agent(tmp_path, client)
    result = agent.chat("查订单")
    assert result.confidence == 0.0
    assert result.requires_human is True
    assert client.calls == []


# ============================================================
# 历史契约：单 assistant 消息 + metadata；旧格式折叠
# ============================================================
def test_turn_history_single_assistant_message_with_metadata(tmp_path, reset_settings):
    client = FakeChatClient().enqueue_final_response("答复内容", intent="other")
    agent = _agent(tmp_path, client)
    agent.chat("问题")

    roles = [m["role"] for m in agent.raw_messages]
    assert roles == ["user", "assistant"]  # 无中间 assistant/tool 消息
    final_msg = agent.raw_messages[-1]
    assert final_msg["content"] == "答复内容"
    metadata = final_msg["metadata"]
    assert metadata["schema"] == 2
    assert metadata["intent"] == "other"
    assert isinstance(metadata["confidence"], float)
    assert metadata["requires_human"] is False
    # 工具摘要/审计完整结果只在 metadata 与审计切片
    assert "intent" not in final_msg["content"]


def test_old_json_history_folded_only_in_model_context(tmp_path, reset_settings):
    """旧版 JSON 副本历史：模型窗口折叠为文本；raw_messages 正本不动。"""
    agent = _agent(tmp_path, FakeChatClient())
    agent.raw_messages.extend([
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": json.dumps({
            "intent": "order_query", "confidence": 0.9,
            "reply": "旧文本回复", "requires_human": False,
            "follow_up_question": None,
        }, ensure_ascii=False)},
    ])
    messages = agent.context_builder.build(agent, "新问题")
    assistant_texts = [
        m["content"] for m in messages
        if m.get("role") == "assistant"
    ]
    assert "旧文本回复" in assistant_texts
    # 正本仍是 JSON 副本（不修改历史正本）
    assert agent.raw_messages[-1]["content"].startswith("{")
    # 保存后正本仍保留旧格式
    agent.save()
    saved = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    assert saved["messages"][-1]["content"].startswith("{")


def test_history_metadata_never_reaches_model_context(tmp_path, reset_settings):
    """持久化 metadata 整体剥离：不注入任何凭证/内部字段，也不再有确认说明。

    纯提交语义下不再有跨轮待确认说明；本测试守住「metadata（含历史遗留
    凭据字段）绝不进入模型上下文」这一不变量。
    """
    client = FakeChatClient()
    agent = _agent(tmp_path, client)
    agent.raw_messages.extend([
        {"role": "user", "content": "退掉 ORD-9，质量问题"},
        {"role": "assistant", "content": "已为您登记退款申请，请确认。", "metadata": {
            "schema": 2, "intent": "return_request", "confidence": 0.9,
            "requires_human": False, "follow_up_question": None,
            "internal_notes": [{
                "tool": "submit_refund_application", "order_id": "ORD-9",
                "reason": "质量问题", "token": "legacy-token",
            }],
        }},
    ])
    messages = agent.context_builder.build(agent, "确认退款")
    blob = json.dumps(messages, ensure_ascii=False)
    # metadata（含历史遗留凭据字段）绝不进入模型上下文
    assert "legacy-token" not in blob
    assert "internal_notes" not in blob
    # 不再注入任何「已明确确认 / 等待确认」的跨轮说明
    assert not [m for m in messages
                if m.get("role") == "system"
                and ("已明确确认" in m.get("content", "")
                     or "等待用户明确确认" in m.get("content", ""))]


# ============================================================
# 安全契约
# ============================================================
def test_unsafe_output_replaced_before_persist(tmp_path, reset_settings):
    """输出命中敏感词 → finalizer 替换安全话术；会话中不出现不安全原文。"""
    settings.guardrail_block_terms = "私下转账"
    client = FakeChatClient().enqueue_final_response(
        "您可以私下转账给客服处理。", intent="other",
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("怎么退款")
    assert "私下转账" not in result.reply
    assert result.requires_human is True
    saved = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    assert all("私下转账" not in m.get("content", "") for m in saved["messages"])


def test_invalid_tool_arguments_never_execute_tool(tmp_path, reset_settings):
    """非法工具参数（未知参数/缺必需参数/非对象）→ 工具函数不被执行。"""
    executed = []

    def spy_query_order(order_id, ctx=None):
        executed.append(order_id)
        return {"success": True}

    from app.agent.tools import registry

    registry._TOOL_MAP["query_order"] = spy_query_order
    result = registry.execute_tool("query_order", {"unknown": 1})
    assert "UNKNOWN_TOOL_ARGUMENT" in result and executed == []
    result = registry.execute_tool("query_order", {})
    assert "MISSING_TOOL_ARGUMENT" in result and executed == []
    result = registry.execute_tool("query_order", ["ORD-1"])
    assert "INVALID_TOOL_ARGUMENTS" in result and executed == []


def test_refund_success_claim_blocked_without_commit(tmp_path, reset_settings):
    """无对应业务回执 → 禁止宣称退款完成（程序改写 + 转人工）。"""
    from app.agent.tools import registry

    def fake_submit(order_id, reason, ctx=None):
        # 模型幻觉：没拿到结果就声称成功 → 本轮没有任何工具调用
        return {"success": True}

    registry._TOOL_MAP["submit_refund_application"] = fake_submit
    # 本轮无工具调用，模型直接宣称退款完成
    client = FakeChatClient().enqueue_final_response(
        "您的退款已成功办理，钱已退回。", intent="return_request",
    )
    agent = _agent(tmp_path, client)
    result = agent.chat("退掉 ORD-5")
    assert "已成功办理" not in result.reply
    assert result.requires_human is True
    assert result.confidence <= 0.2


def test_denied_order_cannot_be_retried_in_turn(tmp_path, reset_settings, monkeypatch):
    """越权拒绝后：状态机拦截同订单继续操作（工具不执行）。"""
    tracker = WriteOpTracker()
    assert tracker.check(
        "submit_refund_application", {"order_id": "O1", "reason": "x"}) is None
    tracker.observe("submit_refund_application", {"order_id": "O1", "reason": "x"},
                    {"success": False, "code": "ORDER_ACCESS_DENIED"})
    verdict = tracker.check(
        "submit_refund_application", {"order_id": "O1", "reason": "x"})
    assert "WRITE_RETRY_FORBIDDEN" in verdict

    # 缺原因 → 只追问不执行
    verdict2 = tracker.check("submit_refund_application", {"order_id": "O1"})
    assert "REFUND_PARAMS_REQUIRED" in verdict2


def test_write_tool_without_state_machine_blocked():
    """未注册状态机的写工具默认禁止执行。"""
    tracker = WriteOpTracker()
    verdict = tracker.check("migrate_address", {"order_id": "O1"})
    assert "WRITE_TOOL_UNREGISTERED" in verdict


# ============================================================
# 零 LLM close + 用量
# ============================================================
def test_close_and_aux_stages_make_no_llm_calls(tmp_path, reset_settings):
    """close() / 确定性 STM 零 LLM；轮内 llm_calls 只含 ReAct。"""
    client = FakeChatClient().enqueue_final_response("答复", intent="other")
    agent = _agent(tmp_path, client)
    agent.memory_manager.memory_enabled = True
    result = agent.chat("我喜欢红色")
    assert result.reply == "答复"
    agent.close()
    assert len(client.calls) == 1  # 只有那一轮终答
    assert agent._last_turn_ctx.llm_calls == 1


# ============================================================
# 组件单元
# ============================================================
def test_compute_reliability_tiers():
    assert compute_reliability(ReliabilitySignal(budget_exhausted=True)) == 0.0
    assert compute_reliability(ReliabilitySignal(rule_based=True)) == 1.0
    assert compute_reliability(
        ReliabilitySignal(tool_committed_evidence=True)) == 0.9
    assert compute_reliability(ReliabilitySignal(knowledge_grounded=True)) == 0.8
    assert compute_reliability(ReliabilitySignal()) == 0.6
    # 硬风险上限：只能压低
    assert compute_reliability(ReliabilitySignal(
        rule_based=True, write_indeterminate=True)) == 0.2
    assert compute_reliability(ReliabilitySignal(
        tool_committed_evidence=True, citation_missing=True)) == CAP_CITATION_MISSING
    # 覆盖上限（业务升级）
    signal = ReliabilitySignal(tool_committed_evidence=True,
                               overrides={"escalation": 0.5})
    assert compute_reliability(signal) == 0.5


def test_compute_reliability_overrides_take_strictest():
    """中危修复 A1：overrides 是「上限，只能压低」——多上限同轮命中取最严
    （升级上限 0.5 与输出安全硬上限 0.2 同时命中应取 0.2，而非取高 0.5）。"""
    signal = ReliabilitySignal(
        tool_committed_evidence=True,
        overrides={"escalation": 0.5, "output_safety": 0.2},
    )
    assert compute_reliability(signal) == 0.2


def test_token_budget_shares_and_trim():
    shares = budget_shares(2000)
    # 阶段4 消融口径：memory 份额 10% → 15%（settings.memory_budget_share）
    assert shares == {"system": 400, "memory": 300, "dialog": 1000,
                      "output_reserve": 400}
    assert estimate_tokens("abc") >= 1
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "tool_calls": [{"id": "c"}], "content": ""},
        {"role": "tool", "tool_call_id": "c", "content": "x"},
        {"role": "assistant", "content": "答复"},
    ]
    kept = trim_messages_to_budget(messages, 18)
    assert kept[0]["role"] != "tool"  # 孤儿 tool 不在窗口起头


def test_trim_all_orphan_tool_window_falls_back_to_user_message():
    """低危修复 A1：窗口全为孤儿 tool 消息时回退到最后一条非 tool 且非
    tool_calls 的消息（调用链中即用户消息）兜底，不返回空窗口、也不把
    孤儿 tool_calls assistant 当窗口（同样破坏对话约束）。"""
    huge = "x" * 200
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "查订单"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "t", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": huge},
    ]
    kept = trim_messages_to_budget(messages, 40)
    assert kept, "全孤儿窗口不得返回空"
    assert kept == [{"role": "user", "content": "查订单"}]


def test_ground_reply_strips_unbacked_numbers():
    evidence = ["钻石会员专属客服的响应时效SLO是30秒内接入。"]
    reply = "专属客服30秒内接入。一般商品7天内发货。"
    cleaned, verdict = ground_reply(reply, evidence)
    assert "30秒" in cleaned
    assert "7天" not in cleaned  # 无证据数字的句子被删除
    assert verdict.removed_sentences == 1
    # 全部声明无证据 → 转核实提示
    cleaned_all, verdict_all = ground_reply("一般商品7天内发货。", evidence)
    assert "核实" in cleaned_all
    assert verdict_all.removed_sentences == 1
    # 无证据集：放行（不误杀非事实回复）
    cleaned2, verdict2 = ground_reply("您好，请问有什么可以帮您？", [])
    assert cleaned2 == "您好，请问有什么可以帮您？"


def test_rrf_merge_dedups_by_parent():
    a = EvidenceItem(doc="d1", parent_id="p1", text="a")
    b = EvidenceItem(doc="d1", parent_id="p1", text="b")
    c = EvidenceItem(doc="d2", parent_id="", text="c")
    merged = rrf_merge([[a, c], [b]])
    keys = [(i.doc, i.parent_id, i.text) for i in merged]
    assert len(merged) == 2
    assert ("d1", "p1", "a") in keys  # 首见者保留
    assert all(("d1", "p1", "b") != k for k in keys)


def test_rrf_merge_preserves_retriever_raw_score():
    """低危修复 B2：rrf_merge 覆盖 score 前先把检索原分存入
    raw_scores['retriever']（docstring 承诺「保留各路原始分供诊断」；
    修复后 retriever 与 rrf 两键并存，原分不再被融合分覆盖丢失）。"""
    item = EvidenceItem(doc="d1", section="s", chunk_id="c1", score=0.87)
    merged = rrf_merge([[item]])
    assert merged[0].raw_scores["retriever"] == 0.87
    assert "rrf" in merged[0].raw_scores
    assert merged[0].score != 0.87  # 已被融合分覆盖


def test_scope_block_is_rule_response_with_high_reliability(tmp_path, reset_settings):
    """范围闸门拦截：规则响应零主体 LLM，可靠度 1.0。"""
    settings.business_only_scope = True
    client = FakeChatClient().enqueue_chat("no")  # scope LLM 判定一次
    agent = _agent(tmp_path, client)
    result = agent.chat("今天天气怎么样")
    assert result.confidence == 1.0
    # 只有一次 scope 判定调用，无主体 LLM/提取
    assert len(client.calls) == 1


# ============================================================
# CLI memory 分支：两层口径契约（P0-1）
# ============================================================
def test_cli_memory_branch_renders_two_layer_view(tmp_path, reset_settings):
    """CLI `memory` 命令：记忆收敛为两层后不得再引用已删除的槽位层。

    回归背景：`MemoryManager.stm` 随槽位层删除，main.py 的 memory 分支
    仍读 `.stm` → AttributeError 直接退出（且在主循环 try 之外）。
    """
    from main import render_memory_lines

    agent = _agent(tmp_path, FakeChatClient(), memory_enabled=True)
    assert not hasattr(agent.memory_manager, "stm")

    lines = render_memory_lines(agent)
    text = "\n".join(lines)
    assert "会话上下文" in text
    assert "长期记忆" in text
    assert "滚动摘要" in text
    assert "原始消息: 0 条" in text


def test_cli_memory_branch_renders_facts_and_summary(tmp_path, reset_settings):
    """有摘要/有 LTM 事实时，各段内容按两层口径正确落位。"""
    from app.agent.memory.models import MemoryFact
    from main import render_memory_lines

    agent = _agent(tmp_path, FakeChatClient(), memory_enabled=True)
    agent.summary = "用户咨询过退款进度"
    agent.raw_messages = [{"role": "user", "content": "hi"}]
    agent.memory_manager.ltm.facts = [
        MemoryFact(
            category="identity", content="用户叫小明",
            created_at="2026-09-18T00:00:00",
        ),
    ]
    agent.memory_manager.ltm.interaction_summaries = [{"summary": "上次问退款"}]

    text = "\n".join(render_memory_lines(agent))
    assert "用户咨询过退款进度" in text
    assert "原始消息: 1 条" in text
    assert "[identity] 用户叫小明" in text
    assert "上次问退款" in text

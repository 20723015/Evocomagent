"""修复计划（对 2a7f2eb 的 deadline/线程残留/MCP 身份/记忆保底/引用评估）无网络回归。

覆盖验收点：
- 辅助阶段（STM/摘要/增量 LTM）预算耗尽 → 保留原回复、会话保存、不抛 500；
- LLM 并发许可等待受 turn budget 上限；重试退避不越过 turn deadline；
- 固定执行池：线程数有界、permit 释放、close 幂等、关闭后拒绝新批次；
- 慢读段 → 预算到期退款不启动；远端退款超时 → indeterminate + 本轮禁重试 + handoff；
- actor token 签发/校验/过期/错误签名/scope 矩阵；ToolManager 敏感工具带 actor；
- top-8 全 identity 仍保底 preference；汉字 bigram 不跨分隔符；
- 文件名话语前缀剥离；verdict 缺失判 0；数据集门禁 citation 规则；
- MultiAgent 跨轮 step count 重置。
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.agent.citations import extract_citations
from app.agent.context import ToolContext
from app.agent.memory.long_term import MemoryFact, _token_set
from app.agent.tools.batch_executor import (
    BUDGET_SKIP_ERROR,
    ToolBatchExecutor,
    ToolTurnState,
)
from app.agent.turn_budget import LLMBudgetExhausted, TurnBudget, bind_budget, reset_budget
from app.config.settings import settings
from app.evaluation import metrics as eval_metrics


# ============================================================
# 辅助阶段容错（摘要压缩预算耗尽不清回复）
# ============================================================
def test_auxiliary_phases_budget_exhausted_preserve_reply(reset_settings, tmp_path):
    from app.agent.chat import EcomAgent
    from tests.unit.conftest import FakeChatClient

    client = FakeChatClient().enqueue_final_response("这是客服回复内容。")
    agent = EcomAgent(session_path=str(tmp_path / "s.json"), client=client)

    def boom(*a, **k):
        raise LLMBudgetExhausted("budget")

    # 历史压缩（辅助 LLM）预算耗尽 → 不毁掉已生成回复、会话仍保存
    agent.context_builder._window = 300  # 极小水位：一轮即触发压缩
    agent.summary = "既有摘要"
    for msg in range(12):
        agent.raw_messages.append({"role": "user", "content": f"历史问题 {msg}"})
        agent.raw_messages.append(
            {"role": "assistant", "content": f"历史回复 {msg}"}
        )
    agent._consolidated_len = 0
    import app.agent.summarizer as summarizer_mod

    # 直接让 compress 用到的 summarize 抛预算错误（仓库层捕获并跳过）
    agent.compress_history_by_tokens = boom  # type: ignore[method-assign]

    result = agent.chat("问一句")
    assert result.reply == "这是客服回复内容。"  # 回复未被辅助任务毁掉
    assert result.requires_human is False
    # 会话成功保存（不抛 500 级异常）
    saved = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert saved["messages"][-1]["role"] == "assistant"


# ============================================================
# ResilientLLM：semaphore 等待受预算、退避受 deadline
# ============================================================
class _MinimalCompletions:
    def __init__(self):
        self.create = lambda **kw: object()


class _MinimalOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_MinimalCompletions())
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw: object()))
        )


def test_resilient_llm_semaphore_wait_bounded_by_budget(reset_settings, monkeypatch):
    from app.llm.client import ResilientLLM
    from app.observability import metrics as obs_metrics

    recorded: list[str] = []
    monkeypatch.setattr(obs_metrics, "record_budget_exhausted",
                        lambda phase: recorded.append(phase))

    resilient = ResilientLLM(
        _MinimalOpenAIClient(), model="m", max_retries=0, fallback_model="",
        timeout_seconds=5.0, max_concurrent=1,
    )
    resilient.install()
    resilient._semaphore.acquire()  # 占住唯一许可
    budget = TurnBudget(deadline=time.monotonic() + 0.2)
    token = bind_budget(budget)
    try:
        started = time.monotonic()
        with pytest.raises(LLMBudgetExhausted):
            resilient._client.chat.completions.create(model="m", messages=[])
        elapsed = time.monotonic() - started
    finally:
        reset_budget(token)
        resilient._semaphore.release()

    assert recorded == ["semaphore"]
    assert elapsed < 1.5  # 等 0.2s 即放弃，而非 SDK 超时 5s


def test_resilient_llm_retry_backoff_capped_by_turn_deadline(reset_settings):
    from openai import APIConnectionError
    from app.llm.client import ResilientLLM

    calls = []
    client = _MinimalOpenAIClient()

    def create(**kwargs):
        calls.append(kwargs)
        raise APIConnectionError(request=None)  # RETRYABLE

    client.chat.completions.create = create
    resilient = ResilientLLM(
        client, model="m", max_retries=2, fallback_model="",
        timeout_seconds=5.0, max_concurrent=1,
    )
    resilient.install()
    budget = TurnBudget(deadline=time.monotonic() + 0.3)
    token = bind_budget(budget)
    try:
        started = time.monotonic()
        with pytest.raises(LLMBudgetExhausted):
            client.chat.completions.create(model="m", messages=[])
        elapsed = time.monotonic() - started
    finally:
        reset_budget(token)

    assert elapsed < 2.0  # 退避被夹到 turn deadline：绝不越过 0.3s budget 远去
    assert len(calls) <= 2  # 首次尝试后 budget 已耗尽 → 不再发起（越界重试被拒）


# ============================================================
# 固定执行池：线程有界 / permit 释放 / close 幂等
# ============================================================
def test_executor_fixed_pool_bounded_and_close_idempotent():
    """重复慢只读工具：工作线程总数不超过固定池、permit 最终释放、close 可完成。"""
    executor = ToolBatchExecutor(parallelism=8, max_concurrent=16)
    assert executor._pool._max_workers == 16  # 全局并发上限 = 池大小

    class SlowManager:
        def __init__(self):
            self.done = threading.Event()
            self.calls = []

        def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
            self.calls.append(name)
            self.done.wait(timeout=3)  # 模拟长只读工具（等放行）
            return '{"ok": true}'

    tm = SlowManager()
    state = ToolTurnState()
    # 12 个调用分散到 6 个只读工具（每工具 2 次）——同工具次数上限（默认 2）
    # 不拦截；本测试只验证固定池波次，工具名之争交给去重/上限测试
    names = ["query_order", "query_product", "query_logistics",
             "list_user_orders", "search_knowledge", "recall_user_memory"]
    calls = [
        {"id": f"c{i}", "name": names[i % len(names)], "arguments": '{"q": %d}' % i}
        for i in range(12)
    ]
    budget = TurnBudget(deadline=time.monotonic() + 0.2)
    outcomes = executor.execute(calls, state, None, tm, budget=budget)
    assert len(outcomes) == 12
    assert all(o.skipped for o in outcomes)  # 预算内超时弃等（未放行前不会完成）
    assert len(tm.calls) >= 8  # 首波 8 个已进入执行（波次 ≤ parallelism）

    # 放行：任务陆续完成 → permit 全部释放（done 回调）→ close 无需等待
    tm.done.set()
    deadline = time.monotonic() + 5
    while executor._active != 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert executor._active == 0
    assert executor._permit._value == 16  # 全局许可已全部释放

    executor.close()
    assert executor.closed is True
    executor.close()  # 幂等
    with pytest.raises(RuntimeError):
        executor.execute(calls[:1], ToolTurnState(), None, tm)


def test_write_barrier_not_started_when_budget_expired():
    """修复计划：慢读段耗尽预算 → 退款绝不在预算外启动。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)

    class Manager:
        def __init__(self):
            self.write_started = []

        def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
            if name == "query_order":
                time.sleep(0.4)  # 读段很慢，预算在段内耗尽
            if name == "apply_refund":
                self.write_started.append(arguments)
            return '{"ok": true}'

    tm = Manager()
    budget = TurnBudget(deadline=time.monotonic() + 0.15)
    outcomes = executor.execute(
        [
            {"id": "c1", "name": "query_order", "arguments": '{"order_id": "O1"}'},
            {"id": "c2", "name": "apply_refund", "arguments": '{"order_id": "O1", "reason": "x"}'},
        ],
        ToolTurnState(), None, tm, budget=budget,
    )
    assert tm.write_started == []  # 预算耗尽后写工具未启动
    assert BUDGET_SKIP_ERROR in outcomes[1].result
    assert outcomes[1].skipped is True


def test_write_timeout_indeterminate_and_no_auto_retry(reset_settings, tmp_path):
    """远端写超时 → indeterminate + 记录 + 本轮重试被重复拦截 + Agent 强制转人工。"""
    settings.tool_write_timeout_seconds = 0.2
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)

    class SlowWriteManager:
        def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
            if name == "apply_refund":
                time.sleep(0.5)  # 超过 0.2s 写超时
            return '{"ok": "refunded"}'

    state = ToolTurnState()
    budget = TurnBudget(deadline=time.monotonic() + 5.0)
    calls = [{
        "id": "c1", "name": "apply_refund",
        "arguments": '{"order_id": "O1", "reason": "x", "idempotency_key": "KEY-1"}',
    }]
    outcomes = executor.execute(calls, state, None, SlowWriteManager(), budget=budget)
    payload = json.loads(outcomes[0].result)
    assert payload["status"] == "indeterminate"
    assert outcomes[0].skipped is True
    assert state.indeterminate_writes == [{
        "tool": "apply_refund", "order_id": "O1", "idempotency_key": "KEY-1",
    }]

    # 本轮模型重试同签名 → 重复拦截（禁止自动重试/生成新幂等键路径）
    retry = executor.execute(calls, state, None, SlowWriteManager(), budget=budget)
    assert "重复调用被拦截" in retry[0].result

    # Agent 收尾（新架构）：indeterminate 经 AgentTurnContext → TurnFinalizer
    # 强制转人工 + 可靠度压到 0.2 档 + 对账清单写入 agent
    from app.agent.chat import EcomAgent
    from app.agent.turn_context import AgentTurnContext as _TurnCtx
    from app.handoff.board import build_handoff_ticket
    from tests.unit.conftest import FakeChatClient

    agent = EcomAgent(session_path=str(tmp_path / "a.json"), client=FakeChatClient())
    ctx = _TurnCtx(user_input="退款查证", budget=TurnBudget.start(30))
    ctx.extra_indeterminate = list(state.indeterminate_writes)
    ctx.write_ops.refund.indeterminate = list(state.indeterminate_writes)
    finalizer = agent._finalizer
    result = finalizer._build_response(ctx)
    result.reply = "您的退款已成功办理。"
    result.confidence = 0.0
    finalizer._apply_write_op_state(ctx, result)
    assert result.requires_human is True
    # 无 committed 证据的成功宣称被改写（阶段C：禁止声称退款成功）
    assert "尚未确认退款完成" in result.reply
    assert ctx.indeterminate_writes == state.indeterminate_writes

    agent._indeterminate_writes = list(ctx.indeterminate_writes or state.indeterminate_writes)
    ticket = build_handoff_ticket(agent, result)
    on_rec = [a for a in ticket.suggested_actions if "KEY-1" in a and "O1" in a]
    assert on_rec and "退款对账" in on_rec[0] or any("对账" in a for a in ticket.suggested_actions)


def test_write_timeout_confirm_segment_key_from_internal_args(reset_settings):
    """confirm 段写超时：幂等键必须取 internal_args（服务端注入通道）。

    模型参数面不含保留字段（registry 拒收），arguments 兜底恒空 →
    对账记录幂等键为空=对账失效；合并 internal_args 后键=refund_id。
    """
    settings.tool_write_timeout_seconds = 0.2
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)

    class SlowWriteManager:
        def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
            if name == "apply_refund":
                time.sleep(0.5)  # 超过 0.2s 写超时
            return '{"ok": "refunded"}'

    state = ToolTurnState()
    state.refund_confirm = {
        "action": "confirm", "token": "tok-1",
        "refund_id": "REF-9", "order_id": "O1", "reason": "x",
    }
    budget = TurnBudget(deadline=time.monotonic() + 5.0)
    calls = [{
        "id": "c1", "name": "apply_refund",
        "arguments": '{"order_id": "O1", "reason": "x"}',
    }]
    outcomes = executor.execute(calls, state, None, SlowWriteManager(), budget=budget)
    payload = json.loads(outcomes[0].result)
    assert payload["status"] == "indeterminate"
    # 模型参数面保持干净，注入只在 internal_args
    assert outcomes[0].arguments == {"order_id": "O1", "reason": "x"}
    assert state.indeterminate_writes == [{
        "tool": "apply_refund", "order_id": "O1", "idempotency_key": "REF-9",
    }]


def test_trace_entry_merges_internal_args_into_write_ops():
    """结果回填必须合并 internal_args 再 observe：tokens_issued 正常回收、
    indeterminate 条目幂等键非空（跨轮 confirm 时 record 兜底也为空）。"""
    from app.agent.react_runner import ReactRunner
    from app.agent.tools.batch_executor import ToolOutcome
    from app.agent.turn_context import AgentTurnContext

    runner = ReactRunner(agent=None)
    ctx = AgentTurnContext(user_input="确认退款", budget=TurnBudget.start(30))
    outcome = ToolOutcome(
        call_id="c1", name="apply_refund",
        arguments={"order_id": "O1", "reason": "x"},
        result=json.dumps({"status": "indeterminate", "error": "REFUND_WRITE_TIMEOUT"}),
        sequence=1,
        internal_args={
            "confirmation_token": "tok-1",
            "idempotency_key": "REF-9",
            "refund_id": "REF-9",
        },
    )
    tracker = ctx.write_ops.refund
    tracker.tokens_issued.add("tok-1")  # 签发段在先：待回收凭证
    runner._trace_entry(ctx, outcome)
    # 合并后 observe 拿到 confirmation_token → 一次性凭证回收
    assert "tok-1" not in tracker.tokens_issued
    # 合并后 indeterminate 条目幂等键非空（不合并时 record 兜底为空）
    assert tracker.indeterminate[0]["idempotency_key"] == "REF-9"
    assert tracker.indeterminate[0]["order_id"] == "O1"


# ============================================================
# actor token：签发/校验/过期/伪造/scope 矩阵 + ToolManager 注入
# ============================================================
@pytest.fixture
def actor_secret(reset_settings):
    settings.mcp_actor_secret = "actor-secret-0123456789-abcdefghijklmnop"  # ≥32 字节
    return settings.mcp_actor_secret


def test_actor_token_roundtrip_and_scope_matrix(actor_secret):
    import jwt as pyjwt
    from app.mcp_client.actor import (
        SCOPE_ORDERS_READ,
        SCOPE_REFUND_WRITE,
        ActorTokenError,
        issue_actor_token,
        require_scopes,
        validate_actor_token,
    )

    token = issue_actor_token("u1", "s1", (SCOPE_ORDERS_READ,))
    claims = validate_actor_token(token)
    assert claims["sub"] == "u1" and claims["session_id"] == "s1"
    assert claims["aud"] == "ecom-mcp"
    require_scopes(claims, SCOPE_ORDERS_READ)  # 读单可用
    with pytest.raises(ActorTokenError):
        require_scopes(claims, SCOPE_REFUND_WRITE)  # 无退款 scope

    # 过期 token
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    expired = issue_actor_token(
        "u1", "", (SCOPE_ORDERS_READ,), ttl_seconds=1, now_utc=past,
    )
    with pytest.raises(ActorTokenError):
        validate_actor_token(expired)

    # 伪造（错误 audience）
    forged = pyjwt.encode(
        {"sub": "u9", "exp": (datetime.now(timezone.utc) + timedelta(seconds=60)),
         "iss": "ecom-agent", "aud": "other"},
        actor_secret, algorithm="HS256",
    )
    with pytest.raises(ActorTokenError):
        validate_actor_token(forged)

    # 缺密钥：签发/校验 fail-fast
    settings.mcp_actor_secret = ""
    from app.mcp_client.actor import McpSecurityConfigError

    with pytest.raises(McpSecurityConfigError):
        issue_actor_token("u1")


def test_validate_mcp_security_fail_fast(reset_settings):
    from app.mcp_client.actor import McpSecurityConfigError, validate_mcp_security

    settings.mcp_enabled = True
    settings.mcp_server_url = "http://127.0.0.1:9123/mcp"
    settings.auth_enabled = True
    settings.mcp_auth_token = ""
    settings.mcp_actor_secret = ""
    with pytest.raises(McpSecurityConfigError):
        validate_mcp_security()  # 双缺

    settings.mcp_auth_token = "service-token"
    with pytest.raises(McpSecurityConfigError):
        validate_mcp_security()  # 缺 actor 密钥

    settings.mcp_actor_secret = "a" * 40
    validate_mcp_security()  # 齐了

    settings.auth_enabled = False  # 开发：不做生产级强制
    settings.mcp_auth_token = ""
    settings.mcp_actor_secret = ""
    validate_mcp_security()


def test_tool_manager_injects_actor_for_sensitive_mcp(actor_secret):
    from app.mcp_client.actor import (
        SCOPE_ORDERS_READ,
        SCOPE_REFUND_WRITE,
        require_scopes,
        validate_actor_token,
    )
    from app.agent.tools.manager import ToolManager

    calls: list[tuple[str, str | None, bool]] = []

    def fake_call_tool(name, arguments, timeout=None, *, actor_token=None, write=False):
        calls.append((name, actor_token, write))
        return json.dumps({"success": True})

    fake_client = SimpleNamespace(
        connect=lambda: [
            {"type": "function", "function": {
                "name": n, "description": "d",
                "parameters": {"type": "object", "properties": {}},
            }}
            for n in ("query_order", "query_product", "query_logistics",
                      "apply_refund", "search_knowledge")
        ],
        call_tool=fake_call_tool,
        close=lambda: None,
    )
    tm = ToolManager(use_mcp=True, mcp_server_url="http://fake", mcp_client=fake_client)
    ctx = ToolContext(user_id="u1", session_id="s1")

    tm.execute_tool("query_order", {"order_id": "O1"}, ctx)
    name, token, write = calls[-1]
    claims = validate_actor_token(token)
    assert claims["sub"] == "u1" and claims["session_id"] == "s1"
    require_scopes(claims, SCOPE_ORDERS_READ)
    assert write is False  # 读

    tm.execute_tool("apply_refund", {"order_id": "O1", "reason": "x"}, ctx)
    name, token, write = calls[-1]
    claims = validate_actor_token(token)
    require_scopes(claims, SCOPE_REFUND_WRITE)
    assert write is True  # 写

    tm.execute_tool("query_product", {"keyword": "耳机"}, ctx)
    assert calls[-1][1] is None  # 非敏感工具不带 actor


# ============================================================
# 记忆保底与 bigram
# ============================================================
def test_token_set_no_cross_separator_bigrams():
    tokens = _token_set("苹果，香蕉")
    assert "苹果" in tokens and "香蕉" in tokens
    assert "果香" not in tokens  # 不跨标点造 bigram
    tokens2 = _token_set("苹果 香蕉")
    assert "果香" not in tokens2  # 不跨空格造 bigram
    tokens3 = _token_set("苹果iOS香蕉")
    assert "果i" not in tokens3 and "s香" not in tokens3  # 不跨 ASCII 段


def test_top8_all_identity_no_unconditional_injection():
    """阶段F：身份/偏好不再无条件保底注入无关请求（相关性阈值）。

    query 与全部事实无词面相关 → 不注入；query 与 preference 相关时
    该事实可入选（不再被 identity 挤占）。
    """
    from app.agent.memory.long_term import LongTermMemory

    now = datetime.now(timezone.utc)
    facts = [
        MemoryFact(content=f"订单咨询热点 {i}", category="identity",
                   created_at=(now - timedelta(days=1)).isoformat())
        for i in range(8)
    ]
    facts.append(MemoryFact(content="偏好空运发货", category="preference",
                            created_at=(now - timedelta(days=50)).isoformat()))
    ltm = LongTermMemory(user_id="u1", memory_dir=".")
    ltm.facts = facts

    # 无关请求：一律不注入（无保底）
    selected = ltm.select_facts_for_prompt(
        "完全无关的查询内容", max_facts=8, now_utc=now,
    )
    assert selected == []

    # 相关请求：相关事实注入，且 identity 挤不进无关名额
    selected2 = ltm.select_facts_for_prompt(
        "空运发货", max_facts=8, now_utc=now,
    )
    contents = {f.content for f in selected2}
    assert "偏好空运发货" in contents


# ============================================================
# 引用：前缀剥离 + verdict 缺失判 0
# ============================================================
def test_filename_discourse_prefix_stripped():
    assert extract_citations("请参考退货政策.md 和 详见 配送说明.txt") == [
        "退货政策.md", "配送说明.txt",
    ]
    assert extract_citations("来源 会员权益.md 规定…") == ["会员权益.md"]
    assert extract_citations("根据退换货政策.pdf 处理") == ["退换货政策.pdf"]
    # 书名号语境引用不受前缀剥离影响
    assert extract_citations("请参考《退货政策》。") == ["退货政策"]


def test_citation_check_verdict_missing_scores_zero():
    assert eval_metrics.citation_check(["退货政策"], True, None) == 0.0
    assert eval_metrics.citation_check([], True, None) == 0.0
    assert eval_metrics.citation_check([], False, None) is None  # 未配置仍跳过


# ============================================================
# 数据集门禁（citation 字段/覆盖）
# ============================================================
def test_dataset_gate_citation_rules(tmp_path, monkeypatch):
    from app.scripts import check_eval_dataset as gate

    monkeypatch.setattr(gate, "MIN_GOLDEN_CASES", 1)
    monkeypatch.setattr(gate, "MIN_RETRIEVAL_CASES", 0)

    def write(cases):
        p = tmp_path / "cases.json"
        p.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
        return p

    def minimal(cid, **kw):
        base = {"id": cid, "description": cid, "turns": ["问"]}
        base.update(kw)
        return base

    # 无引用场景覆盖 → 报错
    ev = write([minimal("c1"), minimal("c2")])
    problems = gate._problems(ev, tmp_path / "none.json")
    assert any("缺少 expected_citations 用例" in p for p in problems)
    assert any("缺少 forbid_unretrieved_citations" in p for p in problems)

    # 覆盖齐全
    ev = write([
        minimal("c1", expected_citations=["退货政策"], forbid_unretrieved_citations=True),
        minimal("c2", expected_citations=["配送说明"]),
    ])
    problems = gate._problems(ev, tmp_path / "none.json")
    assert not any("expected_citations" in p or "forbid" in p for p in problems)

    # 坏字段：非字符串/重复
    ev = write([
        minimal("c1", expected_citations=["", "x", "x"]),
        minimal("c2", forbid_unretrieved_citations="yes"),
    ])
    problems = gate._problems(ev, tmp_path / "none.json")
    assert any("含空/非字符串项" in p for p in problems)
    assert any("含重复项" in p for p in problems)
    assert any("必须是 bool" in p for p in problems)


# ============================================================
# MultiAgent 跨轮 step count 重置
# ============================================================
def test_multiagent_steps_reset_per_turn(tmp_path):
    from app.multi_agent.orchestrator import MultiAgentOrchestrator
    from tests.unit.conftest import sample_response

    agent = MultiAgentOrchestrator(
        session_path=str(tmp_path / "s.json"),
        memory_enabled=False, use_mcp=False, temperature=0.0,
    )
    first = list(agent.agents.keys())[0]
    agent.router.route = lambda user_input, messages: first
    for sub in agent.agents.values():
        sub.handle = lambda messages, ctx=None, max_steps=5, executor=None, state=None, budget=None: (
            "您的退款已处理。", [], 3,
        )
    agent._extract_structured_response = lambda text: sample_response(
        reply="您的退款已处理。"
    )
    agent.chat("第一轮")
    assert agent._react_steps_count == 3
    agent.chat("第二轮")
    assert agent._react_steps_count == 3  # 每轮重置：不是 6

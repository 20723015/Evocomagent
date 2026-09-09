"""Agent能力强化计划（v3.1）新增无网络单测。

覆盖：改造一（TurnBudget/预算耗尽/重复拦截/close 同受预算）、改造二
（Executor 分段并行峰值/回填保序/写屏障/SSE 新字段/MCP 并发前置）、
改造三（引用提取/归一/三态/分级/评估公式）、改造四（Dice/UTC/保底
≤8/摘要 3 条/召回排序）。全部本地 fake，无网络。
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.agent.citations import (
    apply_citation_policy,
    extract_citations,
    normalize_source,
    verify_citations,
)
from app.agent.context import ToolContext
from app.agent.memory.long_term import (
    LongTermMemory,
    MemoryFact,
    dice_score,
    score_fact,
    _token_set,
)
from app.agent.tools.batch_executor import (
    BUDGET_SKIP_ERROR,
    DUP_CALL_ERROR,
    TOOL_TIMEOUT_ERROR,
    ToolBatchExecutor,
    ToolTurnState,
    extract_sources_from_result,
)
from app.agent.turn_budget import TurnBudget, bind_budget, current_budget, reset_budget
from app.evaluation import metrics as eval_metrics
from app.config.settings import settings


# ============================================================
# 测试替身
# ============================================================
class RecordingToolManager:
    """记录并发峰值与每次调用时序的本地工具管理器替身。"""

    def __init__(self, delay: float = 0.12):
        self.delay = delay
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0
        self.calls: list[tuple[str, float, float]] = []  # (name, start, end)

    tool_definitions: list = []  # SubAgent.handle 会读取（fake 传空即可）

    def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        start = time.monotonic()
        if name == "apply_refund":
            time.sleep(self.delay * 1.5)  # 写工具更慢，便于观测屏障
        else:
            time.sleep(self.delay)
        with self._lock:
            self._active -= 1
            self.calls.append((name, start, time.monotonic()))
        return json.dumps({"ok": name}, ensure_ascii=False)


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeMsg:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _ScriptedLLMClient:
    """脚本化 OpenAI 替身：chat.completions.create 依序弹出消息。"""

    def __init__(self, script: list[_FakeMsg]):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self._script = list(script)

    def _create(self, **kwargs):
        assert self._script, "脚本已耗尽"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self._script.pop(0))]
        )


def _read_calls(tm: RecordingToolManager):
    return [name for name, _, _ in tm.calls]


# ============================================================
# 改造二：Executor
# ============================================================
def test_executor_max_active_two_and_stable_order():
    """单批次 parallel 段并发峰值 == tool_parallelism（双层限流约束）；回填保序。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)
    tm = RecordingToolManager(delay=0.15)
    calls = [
        {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
        {"id": "c2", "name": "query_product", "arguments": '{"sku": "B"}'},
        {"id": "c3", "name": "query_logistics", "arguments": '{"order_id": "A"}'},
        {"id": "c4", "name": "list_user_orders", "arguments": '{}'},
    ]
    outcomes = executor.execute(calls, ToolTurnState(), None, tm)

    assert tm.peak == 2  # 2 个 worker 并行，峰值不超 2
    assert [o.call_id for o in outcomes] == ["c1", "c2", "c3", "c4"]  # 回填 = 模型顺序
    assert [o.name for o in outcomes] == ["query_order", "query_product",
                                           "query_logistics", "list_user_orders"]
    assert all(o.result and not o.skipped for o in outcomes)


def test_executor_write_barrier_semantics():
    """写工具是串行屏障：屏障前段（read1/read2）完成先于写工具；其后新只读段再执行。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)
    tm = RecordingToolManager(delay=0.1)
    calls = [
        {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
        {"id": "c2", "name": "query_product", "arguments": '{"sku": "B"}'},
        {"id": "c3", "name": "apply_refund", "arguments": '{"order_id": "A", "reason": "x"}'},
        {"id": "c4", "name": "query_logistics", "arguments": '{"order_id": "A"}'},
    ]
    outcomes = executor.execute(calls, ToolTurnState(), None, tm)

    timing = {name: (s, e) for name, s, e in tm.calls}
    write_start, write_end = timing["apply_refund"]
    read1_end = timing["query_order"][1]
    read2_end = timing["query_product"][1]
    assert write_start >= max(read1_end, read2_end)  # 屏障：写工具晚于前段完成
    assert timing["query_logistics"][0] >= write_end  # 其后只读段晚于写工具
    assert [o.call_id for o in outcomes] == ["c1", "c2", "c3", "c4"]


def test_executor_dup_blocked_and_cross_turn_allowed():
    """同签名连续 ≥2 次 → 拦截（误差 JSON 回模型自愈）；新轮（新 state）合法重查。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)
    tm = RecordingToolManager(delay=0)
    state = ToolTurnState()

    calls = [
        {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
        {"id": "c2", "name": "query_order", "arguments": '{"order_id": "A"}'},  # 重复
    ]
    outcomes = executor.execute(calls, state, None, tm)
    assert outcomes[0].skipped is False
    assert outcomes[1].skipped is True
    assert DUP_CALL_ERROR in outcomes[1].result
    assert _read_calls(tm) == ["query_order"]  # 第二次未真正执行

    # 本轮内再查同签名仍被拦住（签名历史本轮有效）
    outcomes = executor.execute(calls[:1], state, None, tm)
    assert outcomes[0].skipped is True and DUP_CALL_ERROR in outcomes[0].result

    # 跨轮（新 state）：合法重查不受影响
    outcomes = executor.execute(calls[:1], ToolTurnState(), None, tm)
    assert outcomes[0].skipped is False


def test_executor_budget_expired_skips_all():
    """提交规则①：预算已耗尽 → 不发起新工具调用（错误 JSON，工具未执行）。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)
    tm = RecordingToolManager(delay=0)
    budget = TurnBudget(deadline=time.monotonic() - 1)
    outcomes = executor.execute(
        [
            {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
            {"id": "c2", "name": "query_product", "arguments": '{"sku": "B"}'},
        ],
        ToolTurnState(), None, tm, budget=budget,
    )
    assert all(o.skipped for o in outcomes)
    assert all(BUDGET_SKIP_ERROR in o.result for o in outcomes)
    assert _read_calls(tm) == []


def test_executor_readonly_timeout_discards_result():
    """提交规则③：到期停止等待只读任务（结果丢弃，错误 JSON 回填；不阻塞返回）。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)

    class SlowManager:
        def execute_tool(self, name, arguments, ctx=None, timeout=None, internal_args=None):
            time.sleep(0.5)  # 慢于 budget
            return '{"ok": "late"}'

    tm = SlowManager()
    budget = TurnBudget(deadline=time.monotonic() + 0.3)  # 提交后 ~0.3s 到期
    started = time.monotonic()
    outcomes = executor.execute(
        [
            {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
            {"id": "c2", "name": "query_product", "arguments": '{"sku": "B"}'},
        ],
        ToolTurnState(), None, tm, budget=budget,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 1.0  # 未等到慢任务（弃等）
    assert all(o.skipped and TOOL_TIMEOUT_ERROR in o.result for o in outcomes)


# ============================================================
# 2.5：守卫配置化（可开关、可量化）
# ============================================================
def test_executor_guard_disabled_allows_duplicates():
    """baseline 消融：tool_call_guard_enabled=False → 签名去重与次数上限均不拦截。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)
    executor.configure_guard(enabled=False)
    tm = RecordingToolManager(delay=0)
    calls = [
        {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
        {"id": "c2", "name": "query_order", "arguments": '{"order_id": "A"}'},  # 重复签名
        {"id": "c3", "name": "query_order", "arguments": '{"order_id": "B"}'},
        {"id": "c4", "name": "query_order", "arguments": '{"order_id": "C"}'},  # 超出 2 次上限
    ]
    outcomes = executor.execute(calls, ToolTurnState(), None, tm)
    assert [o.skipped for o in outcomes] == [False, False, False, False]
    assert len(_read_calls(tm)) == 4


def test_executor_guard_limit_configurable():
    """TOOL_MAX_CALLS_PER_NAME 配置生效；search_knowledge 独立上限。"""
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)
    executor.configure_guard(enabled=True, max_calls_per_name=1,
                             search_max_calls=3)
    tm = RecordingToolManager(delay=0)
    calls = [
        {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
        {"id": "c2", "name": "query_order", "arguments": '{"order_id": "B"}'},  # 超限
        {"id": "s1", "name": "search_knowledge", "arguments": '{"q": "a"}'},
        {"id": "s2", "name": "search_knowledge", "arguments": '{"q": "b"}'},
        {"id": "s3", "name": "search_knowledge", "arguments": '{"q": "c"}'},
        {"id": "s4", "name": "search_knowledge", "arguments": '{"q": "d"}'},  # 超限
    ]
    outcomes = executor.execute(calls, ToolTurnState(), None, tm)
    assert outcomes[0].skipped is False
    assert outcomes[1].skipped is True   # query_order 第 2 次超限（上限 1）
    assert outcomes[2].skipped is False
    assert outcomes[3].skipped is False
    assert outcomes[4].skipped is False  # search_knowledge 3 次内
    assert outcomes[5].skipped is True   # 第 4 次超限
    assert _read_calls(tm) == ["query_order", "search_knowledge",
                               "search_knowledge", "search_knowledge"]


def test_dist_stats_mean_median_percentiles():
    """2.5 分布统计：mean/median/P90/P95 正确计算。"""
    from app.evaluation.evaluator import _dist_stats

    stats = _dist_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert stats["mean"] == 5.5
    assert stats["median"] == 5.5
    assert stats["p90"] == 10  # nearest-rank：int(0.9*10)=9 → 第 9 项（0-based）=10
    assert stats["p95"] == 10
    assert stats["n"] == 10
    assert _dist_stats([]) == {"mean": 0.0, "median": 0, "p90": 0, "p95": 0, "n": 0}


def test_executor_sse_events_carry_ids_and_sequence():
    """SSE addititve 新字段：tool_call{tool_call_id,sequence} / tool_result 同。"""
    events: list[tuple[str, dict]] = []
    state = ToolTurnState(event_callback=lambda etype, data: events.append((etype, data)))
    executor = ToolBatchExecutor(parallelism=2, max_concurrent=16)
    tm = RecordingToolManager(delay=0.05)

    calls = [
        {"id": "c1", "name": "query_order", "arguments": '{"order_id": "A"}'},
        {"id": "c2", "name": "query_product", "arguments": '{"sku": "B"}'},
        # 写入错误参数：解析失败 → 错误 JSON 回模型自愈
        {"id": "c3", "name": "query_logistics", "arguments": "{bad json"},
    ]
    executor.execute(calls, state, None, tm)

    call_events = [d for e, d in events if e == "tool_call"]
    result_events = [d for e, d in events if e == "tool_result"]
    assert len(call_events) == 3 and len(result_events) == 3

    seqs = [d["sequence"] for d in call_events]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3  # 模型顺序单调
    for d in call_events:
        assert d["tool_call_id"] and isinstance(d["sequence"], int)
    for d in result_events:
        assert d["tool_call_id"] and d["tool_name"] and isinstance(d["sequence"], int)
    # 解析失败的 c3：错误 JSON 且 skipped
    bad = [d for d in result_events if d["tool_call_id"] == "c3"][0]
    assert "不是合法 JSON" in bad["result"]


def test_executor_extract_sources_excludes_tainted():
    """tainted=true 的检索块不进合法来源集合；doc/source_path 双路收。"""
    result = json.dumps({
        "results": [
            {"doc": "退换货政策.md", "source_path": "知识库/退回.md", "tainted": False},
            {"doc": "污染文档.pdf", "source_path": "x/坏.md", "tainted": True},
        ]
    }, ensure_ascii=False)
    sources = extract_sources_from_result(result)
    assert "退换货政策" in sources and "退回" in sources
    assert "污染文档" not in sources


# ============================================================
# 改造三：引用
# ============================================================
def test_extract_citations_context_and_filename():
    text = "根据《退货政策》，可以七天无理由。详见 退换货政策.md 与 docs/配送说明.txt。"
    assert extract_citations(text) == [
        "退货政策", "退换货政策.md", "docs/配送说明.txt",
    ]
    # 孤立书名号（商品名）不触发
    assert extract_citations("这个《星空玫瑰》香水很好闻。") == []
    # 语境词覆盖：参考/来源/政策
    assert extract_citations("请参考《会员权益》。") == ["会员权益"]
    assert extract_citations("来源《运输协议》规定…") == ["运输协议"]


def test_normalize_source():
    assert normalize_source("C:\\知识库\\退换货政策.MD") == "退换货政策"
    assert normalize_source("配送说明.md") == "配送说明"
    assert normalize_source("《退换货政策》") == "《退换货政策》"  # 书名号非文件名
    assert normalize_source(normalize_source("ＡＢＣ.pdf")) == "abc"  # NFKC+casefold


def test_verify_citations_three_states():
    sources = {"退货政策", "配送说明"}
    verdict = verify_citations(
        "根据《退货政策》与《配送说明》，并参考 未知文档.md。", sources,
    )
    assert set(verdict["cited"]) == {"退货政策", "配送说明", "未知文档.md"}
    assert set(verdict["matched"]) == {"退货政策", "配送说明"}
    assert verdict["missing"] == ["未知文档.md"]


def test_apply_citation_policy_grading(reset_settings):
    settings.citation_check_enabled = True
    from tests.unit.conftest import sample_response

    # 零检索却有引用 → requires_human + 置信撞低
    r = sample_response(reply="根据《退货政策》可以退。", confidence=0.9)
    verdict = apply_citation_policy(r, set())
    assert r.requires_human is True and r.confidence <= 0.5
    assert verdict["missing"] == ["退货政策"]

    # 有引用不匹配（部分 missing）→ 只压低置信度，不硬拦转人工
    r2 = sample_response(reply="根据《退货政策》可以退。", confidence=0.8,
                         requires_human=False)
    verdict2 = apply_citation_policy(r2, {"配送说明"})
    assert r2.requires_human is False
    assert r2.confidence == pytest.approx(0.8 * 0.6)
    assert verdict2["missing"] == ["退货政策"]

    # 无引用（纯闲聊/工具数据）→ 放行
    r3 = sample_response(reply="您的订单已发出。", confidence=0.7)
    apply_citation_policy(r3, set())
    assert r3.requires_human is False and r3.confidence == 0.7


def test_citation_check_formula():
    # 有 expected：规范化值比对得分
    verdict = {"cited": ["《退货政策》"], "matched": ["退货政策"], "missing": []}
    assert eval_metrics.citation_check(["退货政策"], False, verdict) == pytest.approx(1.0)
    assert eval_metrics.citation_check(["退货政策.md"], False, verdict) == 1.0  # 文件名 vs 书名语义等价（规范化后一致）
    assert eval_metrics.citation_check(["退货政策", "配送说明"], False, verdict) == 0.5

    # forbid + missing → 直接 0
    bad = {"cited": ["退货政策"], "matched": [], "missing": ["退货政策"]}
    assert eval_metrics.citation_check([], True, bad) == 0.0

    # 两项均未配置 → None（不计入通过判定）
    assert eval_metrics.citation_check([], False, verdict) is None
    # 修复计划：配置了 expected/forbid 而 verdict 缺失 → 判 0（评估不得静默跳过）
    assert eval_metrics.citation_check(["退货政策"], True, None) == 0.0


# ============================================================
# 改造四：记忆 naive 排序
# ============================================================
def test_dice_empty_set_returns_zero():
    assert dice_score(set(), {"a"}) == 0.0
    assert dice_score({"a"}, set()) == 0.0
    assert dice_score({"苹果"}, {"苹果"}) == pytest.approx(2 * 1 / 2)
    # 无交集 → 0
    assert dice_score({"苹果"}, {"香蕉"}) == 0.0


def test_score_fact_utc_age_and_category_weight():
    now = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)
    q_tokens = _token_set("喜欢苹果")
    # 未来时间 clamp → age=0 → 新近度最大
    future = now - timedelta(days=0)
    s1 = score_fact("喜欢苹果", "other", (now + timedelta(days=5)).isoformat(),
                    q_tokens, now)
    s2 = score_fact("喜欢苹果", "identity", (now + timedelta(days=5)).isoformat(),
                    q_tokens, now)
    assert s2 - s1 == pytest.approx(0.15)  # 类别权重差
    # 新近度衰减：年龄越大分越低
    s_old = score_fact("喜欢苹果", "other", (now - timedelta(days=365)).isoformat(),
                       q_tokens, now)
    assert s2 > s_old
    # 缺失 created_at 按 90 天
    s_missing = score_fact("喜欢苹果", "other", "", q_tokens, now)
    # s2 是 identity（权重差 0.15）且 age=0（新近度 0.1）；缺失按 90 天
    assert s_missing == pytest.approx(
        s2 - 0.15 - 0.1 + 0.1 * math.exp(-90 / 180.0)
    )


def _ltm_with_facts(facts: list[MemoryFact]) -> LongTermMemory:
    ltm = LongTermMemory(user_id="u1", memory_dir=".")  # 只当容器，不落盘
    ltm.facts = facts
    return ltm


def test_select_facts_relevance_threshold_and_cap():
    """阶段F：注入有相关性阈值与上限，身份/偏好不再无条件保底。"""
    now = datetime.now(timezone.utc)
    facts = [
        MemoryFact(content=f"订单咨询信息 {i}", category="other",
                   created_at=(now - timedelta(days=1)).isoformat())
        for i in range(12)
    ]
    facts.append(MemoryFact(content="用户是 VIP 客户", category="identity",
                            created_at=(now - timedelta(days=30)).isoformat()))
    facts.append(MemoryFact(content="偏好顺丰快递", category="preference",
                            created_at=(now - timedelta(days=30)).isoformat()))
    ltm = _ltm_with_facts(facts)

    selected = ltm.select_facts_for_prompt("订单咨询信息", max_facts=8, now_utc=now)
    assert len(selected) == 8  # 严格 ≤8
    assert all("订单咨询信息" in f.content for f in selected)  # 只注入相关事实

    # 无关 query：身份/偏好不再保底注入（空集合）
    selected_none = ltm.select_facts_for_prompt(
        "毫不相干的查询内容", max_facts=8, now_utc=now,
    )
    assert selected_none == []

    # 相关的 preference 可以入选
    selected_pref = ltm.select_facts_for_prompt(
        "顺丰快递", max_facts=8, now_utc=now,
    )
    assert any(f.content == "偏好顺丰快递" for f in selected_pref)


def test_build_prompt_section_summaries_still_three():
    now = datetime.now(timezone.utc)
    ltm = _ltm_with_facts([
        MemoryFact(content="喜欢苹果", category="preference",
                   created_at=(now - timedelta(days=1)).isoformat()),
    ])
    ltm.interaction_summaries = [
        {"summary": f"摘要{i}", "timestamp": (now - timedelta(days=i)).isoformat()}
        for i in range(5)
    ]
    section = ltm.build_prompt_section("苹果")
    assert section is not None
    # 保持最近 3 条（最末 3 条），不是最早 3 条
    assert "摘要2" in section and "摘要3" in section and "摘要4" in section
    assert "摘要0" not in section and "摘要1" not in section
    assert "已按与当前问题的相关性筛选" in section

    # 无 query：不带"已筛选"说明
    assert "已按与当前问题的相关性筛选" not in ltm.build_prompt_section("")


def test_recall_user_memory_top10_when_query_given():
    from app.agent.memory.manager import MemoryManager
    from app.agent.tools.memory_tool import recall_user_memory

    manager = MemoryManager(
        client=None, model="fake", user_id="u1", memory_dir=".",
        memory_enabled=True,
    )
    now = datetime.now(timezone.utc)
    manager.ltm.facts = [
        MemoryFact(content=f"售后咨询事项 {i}", category="other",
                   created_at=(now - timedelta(days=1)).isoformat())
        for i in range(12)
    ]
    ctx = ToolContext(user_id="u1", session_id="s1", memory=manager)

    result = recall_user_memory("售后咨询事项", ctx=ctx)
    assert result["success"] is True
    assert result["query"] == "售后咨询事项"
    assert len(result["long_term_facts"]) == 10  # top-10

    # 空 query → 全量（兼容旧行为）
    full = recall_user_memory("", ctx=ctx)
    assert len(full["long_term_facts"]) == 12


# ============================================================
# 改造一：TurnBudget / 预算耗尽 fallback / close 同受预算
# ============================================================
def test_chat_budget_exhausted_returns_deterministic_fallback(reset_settings, tmp_path):
    from app.agent.chat import EcomAgent
    from tests.unit.conftest import FakeChatClient

    settings.turn_budget_seconds = -5  # 立即耗尽
    client = FakeChatClient()
    agent = EcomAgent(session_path=str(tmp_path / "s.json"), client=client)
    agent.memory_manager.memory_enabled = False
    result = agent.chat("查询订单")

    assert result.requires_human is True
    assert result.intent.value == "other"
    assert result.confidence == 0.0  # 预算耗尽 → 可靠度 0.0
    assert client.calls == []  # 全程零 LLM 调用（ReAct 前预算检查 + 零 LLM fallback）
    # 会话已保存（含 fallback 回复；单条 assistant 消息折叠格式）
    saved = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert saved["messages"][-1]["role"] == "assistant"
    assert saved["messages"][-1].get("metadata", {}).get("schema") == 2


def test_close_makes_no_llm_calls_and_releases_resources(reset_settings, tmp_path):
    """阶段F：close() 只释放本地资源，不再调用 LLM（巩固交异步 memory job）。"""
    from app.agent.chat import EcomAgent
    from tests.unit.conftest import FakeChatClient

    client = FakeChatClient()
    agent = EcomAgent(session_path=str(tmp_path / "s.json"), client=client)
    agent.memory_manager.consolidate_to_long_term = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("close 不得调 LLM 巩固"))
    )
    agent.close()  # 不抛异常、零 LLM
    assert client.calls == []

    # 自建执行器关闭 + 幂等
    agent.close()
    assert agent.tool_executor.closed is True


# ============================================================
# 改造一/二：SubAgent 与 orchestrator
# ============================================================
def test_subagent_handle_returns_steps_and_no_hardcode(reset_settings):
    from app.multi_agent.agents import SubAgent

    settings.max_react_steps = 3
    script = [
        _FakeMsg("先查订单", tool_calls=[_FakeToolCall("call_1", "query_order",
                                                       '{"order_id": "A"}')]),
        _FakeMsg("最终答复"),
    ]
    sub = SubAgent(name="售后", system_prompt="p",
                   tool_manager=RecordingToolManager(delay=0),
                   client=_ScriptedLLMClient(script), model="fake", temperature=0.0)

    content, new_messages, steps = sub.handle(
        [{"role": "user", "content": "查我的单"}], ctx=None,
        executor=ToolBatchExecutor(parallelism=2, max_concurrent=16),
        state=ToolTurnState(),
    )
    assert content == "最终答复"
    assert steps == 2  # 实际步数：第一次带工具、第二次终结
    tool_msgs = [m for m in new_messages if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[-1]["tool_call_id"] == "call_1"
    assert new_messages[-1]["role"] == "assistant"
    assert _read_calls(sub.tool_manager) == ["query_order"]


def test_subagent_budget_expired_raises():
    from app.multi_agent.agents import SubAgent
    from app.agent.turn_budget import LLMBudgetExhausted

    sub = SubAgent(name="售后", system_prompt="p", tool_manager=object(),
                   client=_ScriptedLLMClient([_FakeMsg("x")]), model="fake",
                   temperature=0.0)
    with pytest.raises(LLMBudgetExhausted):
        sub.handle([{"role": "user", "content": "hi"}],
                   budget=TurnBudget(deadline=time.monotonic() - 1))


def test_orchestrator_accumulates_react_steps(tmp_path):
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
    result = agent.chat("我要退款")

    assert agent._react_steps_count == 3  # 不再恒为 1
    assert result.reply == "您的退款已处理。"
    assert agent._last_citation_verdict is not None  # 引用校验已接入


# ============================================================
# MCP 并发生命周期前置（改造二）
# ============================================================
class _FakeMCPResult:
    isError = False

    def __init__(self, ok=True):
        self.content = [SimpleNamespace(text='{"ok": true}' if ok else '{"error": "x"}')]


class _FakeMCPSession:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        self.meta_seen: list[dict] = []

    async def call_tool(self, name, arguments, meta=None):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if meta is not None:
                self.meta_seen.append(dict(meta))
            await asyncio.sleep(self.delay)
            return _FakeMCPResult()
        finally:
            with self._lock:
                self.active -= 1


def _spin_mcp_client(session):
    """后台跑一个事件循环（idle 事件等待），返回 (client, stop_fn)。

    client._session/_loop/_close_event 以测试替身就位——不触碰真实网络连接；
    stop() 置位 idle 事件使循环线程退出，返回线程是否仍存活。
    """
    from app.mcp_client.client import MCPClient

    loop = asyncio.new_event_loop()
    client = MCPClient("http://fake")
    client._session = session
    holder: dict = {}

    def runner():
        asyncio.set_event_loop(loop)
        holder["idle"] = asyncio.Event()  # 必须在 loop 线程内创建
        loop.run_until_complete(holder["idle"].wait())
        # close 竞争后可能遗留 pending 任务：取消并回收（避免破坏性告警）
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    wait = 0.0
    while "idle" not in holder and wait < 2:
        time.sleep(0.005)
        wait += 0.005
    assert "idle" in holder  # 循环线程已启动
    client._loop = loop
    client._close_event = holder["idle"]

    def stop():
        loop.call_soon_threadsafe(holder["idle"].set)
        t.join(timeout=5)
        return t.is_alive()

    return client, stop


def test_mcp_client_concurrent_requests():
    """并发请求测试：run_coroutine_threadsafe 提交线程安全，多路并发全部返回。"""
    from app.mcp_client.client import MCPClient

    session = _FakeMCPSession(delay=0.02)
    client, stop = _spin_mcp_client(session)

    results: list[str] = []
    errors: list[BaseException] = []

    def worker():
        try:
            for _ in range(3):
                results.append(client.call_tool("t", {"a": 1}, timeout=5))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors
    assert len(results) == 18
    assert all('"ok": true' in r for r in results)
    assert session.peak >= 2  # 确有并发重叠
    assert stop() is False


def test_mcp_client_close_race_with_inflight():
    """close 与在途调用竞争：不崩溃、不挂死，结果要么成功要么错误 JSON。"""
    from app.mcp_client.client import MCPClient

    session = _FakeMCPSession(delay=0.4)
    client, stop = _spin_mcp_client(session)

    holder: dict = {}

    def caller():
        try:
            holder["result"] = client.call_tool("slow", {}, timeout=3)
        except BaseException as e:  # noqa: BLE001
            holder["error"] = repr(e)

    t = threading.Thread(target=caller)
    t.start()
    time.sleep(0.1)  # 让在途调用进入 loop
    client.close()  # 竞争关闭（join 5s）

    t.join(timeout=10)
    assert not t.is_alive()
    assert "error" not in holder  # 无异常外溢（call_tool 自身兜底错误 JSON）
    payload = json.loads(holder["result"])
    assert set(payload) == {"ok"} or "error" in payload  # 成或败，二者皆优雅
    # close 后任一新调用 → 未连接错误（确定性）
    assert "未连接" in client.call_tool("t", {}, timeout=1)
    assert stop() is False


# ============================================================
# 数据集新字段（改造三）
# ============================================================
def test_dataset_loads_citation_fields(tmp_path):
    from app.evaluation.dataset import load_dataset

    p = tmp_path / "cases.json"
    p.write_text(json.dumps({
        "cases": [
            {
                "id": "c1", "description": "引用校验",
                "turns": ["按退货政策处理"],
                "expected_citations": ["退货政策"],
                "forbid_unretrieved_citations": True,
            }
        ]
    }, ensure_ascii=False), encoding="utf-8")
    cases = load_dataset(p)
    assert cases[0].expected_citations == ["退货政策"]
    assert cases[0].forbid_unretrieved_citations is True

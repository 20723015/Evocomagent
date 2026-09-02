"""Prometheus 指标（阶段四 4.3）。

看板（4.6）依赖的指标：
- chat_latency_seconds（直方图）
- llm_tokens_total{direction,purpose}
- llm_cost_estimated{model}
- tool_calls_total{tool,result}
- react_steps_bucket / react_steps_count
- handoff_total
- session_conflict_total
"""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
)

CHAT_LATENCY = Histogram(
    "chat_latency_seconds", "单轮对话完成延迟（含 ReAct+tools）",
    buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 40, 60),
)
LLM_TOKENS = Counter(
    "llm_tokens_total", "LLM token 用量",
    labelnames=("direction", "purpose"),
)
LLM_COST = Counter(
    "llm_cost_estimated", "估算 LLM 成本（按 token 单价），单位: 分",
    labelnames=("model",),
)
LLM_RETRIES = Counter(
    "llm_retries_total", "LLM 调用重试次数",
    labelnames=("model", "kind"),
)
TOOL_CALLS = Counter(
    "tool_calls_total", "工具调用结果",
    labelnames=("tool", "result"),
)
REACT_STEPS = Histogram(
    "react_steps", "ReAct 循环步数", buckets=(1, 2, 3, 4, 5, 8, 12, 20),
)
HANDOFF = Counter(
    "handoff_total", "转人工次数", labelnames=("reason",),
)
SESSION_CONFLICT = Counter(
    "session_conflict_total", "会话锁冲突（409）",
)
RATE_LIMITED = Counter(
    "rate_limited_total", "限流拒绝（429）", labelnames=("kind",),
)
LLM_BUDGET_REJECTED = Counter(
    "llm_budget_rejected_total", "单轮/日预算拦截",
)
TURN_BUDGET_EXHAUSTED = Counter(
    "turn_budget_exhausted_total", "轮次预算耗尽（按阶段）",
    labelnames=("phase",),  # react|extract|stm|summary|ltm|semaphore
)
DUP_BLOCKED = Counter(
    "react_duplicate_blocked_total", "重复工具调用被拦截",
    labelnames=("tool",),
)
TOOL_LIMIT_REACHED = Counter(
    "tool_per_name_limit_reached_total", "达到单工具轮内次数上限被拦截",
    labelnames=("tool",),
)
TOOL_CALLS_PER_TURN = Histogram(
    "tool_calls_per_turn", "单轮（一次 chat）实际工具调用数（2.5 分布观测）",
    buckets=(1, 2, 3, 4, 5, 8, 12, 16, 24),
)
TOOL_TIMEOUT = Counter(
    "tool_timeout_total", "工具执行超时",
    labelnames=("kind",),  # readonly|permit|write
)
MCP_WRITE_INDETERMINATE = Counter(
    "mcp_write_indeterminate_total", "远端写操作结果未知（超时）",
    labelnames=("tool",),
)
IN_PROGRESS = Gauge(
    "agent_inflight", "同时进行的 Agent 轮数",
)

# ---------- 5.x 生产可观测性（依赖/演进/对象存储/精排降级/outbox）----------
DEPENDENCY_READINESS = Gauge(
    "dependency_readiness", "依赖就绪状态（1=ok 0=degraded）",
    labelnames=("component",),  # redis | mysql_schema | es_kb_alias | object_store | reranker
)
EVOLUTION_PHASE = Gauge(
    "evolution_phase", "evolution 发布事务当前阶段（1=在该阶段）",
    labelnames=("phase",),  # PREPARED | INDEX_BUILT | ALIAS_ACTIVATED | POINTER_UPDATED | LEDGER_COMMITTED
)
EVOLUTION_RECOVERY = Counter(
    "evolution_recovery_total", "发布事务恢复动作次数",
    labelnames=("action",),  # rollback | forward | ledger_only | blocked
)
EVOLUTION_BLOCKED = Gauge(
    "evolution_kb_write_blocked", "KB 写入全局阻塞标记（1=阻塞待人工 reconcile）",
)
ALIAS_POINTER_MISMATCH = Gauge(
    "alias_pointer_mismatch", "ES alias 与 generation 指针不一致（1=不一致）",
)
OBJECT_STORE_FAILURES = Counter(
    "object_store_failures_total", "对象存储操作失败次数",
    labelnames=("op",),  # put | get | list | delete | delete_prefix
)
RERANKER_FALLBACK = Counter(
    "reranker_fallback_total", "精排器失败退回原序次数",
)
OUTBOX_BACKLOG = Gauge(
    "outbox_backlog", "待同步 outbox 消息数（ES 同步滞后）",
)

# 估算单价（美元/1M token，粗糙常量——真实账单以 4.6 成本看板校准误差 <10%）
_MODEL_PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def record_chat_latency(seconds: float) -> None:
    CHAT_LATENCY.observe(seconds)


def record_tokens(direction: str, purpose: str, tokens: int, model: str = "") -> None:
    if not tokens:
        return
    LLM_TOKENS.labels(direction=direction, purpose=purpose).inc(tokens)
    if model:
        price = _MODEL_PRICES.get(model)
        if price:
            instr_per_m = price[0]
            out_per_m = price[1]
            # 粗略按输入 70% / 输出 30% 估算（无细粒度时）；单位：分
            cents = (tokens * 70 / 100) / 1_000_000 * instr_per_m * 100
            LLM_COST.labels(model=model).inc(cents)


def record_tool_call(tool: str, result: str) -> None:
    TOOL_CALLS.labels(tool=tool, result=result).inc()


def record_react_steps(steps: int) -> None:
    REACT_STEPS.observe(steps)


def record_handoff(reason: str = "requires_human") -> None:
    HANDOFF.labels(reason=reason).inc()


def record_conflict() -> None:
    SESSION_CONFLICT.inc()


def record_budget_exhausted(phase: str) -> None:
    """轮次预算耗尽打点（phase: react|extract|stm|summary|ltm|semaphore）。"""
    TURN_BUDGET_EXHAUSTED.labels(phase=phase).inc()


def record_dup_blocked(tool: str) -> None:
    DUP_BLOCKED.labels(tool=tool).inc()


def record_tool_limit_reached(tool: str) -> None:
    """单工具轮内次数上限命中（2.5 观测：评估与生产共用）。"""
    TOOL_LIMIT_REACHED.labels(tool=tool).inc()


def record_tool_calls_per_turn(count: int) -> None:
    """每轮工具调用数直方图（2.5：评估报告出 mean/median/P90/P95 分布）。"""
    if count > 0:
        TOOL_CALLS_PER_TURN.observe(count)


# ---------- 5.x 生产可观测性 ----------
def set_dependency_readiness(component: str, ready: bool) -> None:
    """依赖就绪状态（/readyz 探针与启动时上报）。"""
    DEPENDENCY_READINESS.labels(component=component).set(1 if ready else 0)


def set_evolution_phase(phase: str) -> None:
    """发布事务当前阶段（同名阶段置 1，其余 0）。"""
    for name in ("PREPARED", "INDEX_BUILT", "ALIAS_ACTIVATED",
                 "POINTER_UPDATED", "LEDGER_COMMITTED"):
        EVOLUTION_PHASE.labels(phase=name).set(1 if name == phase else 0)


def record_evolution_recovery(action: str) -> None:
    """恢复动作计数（rollback/forward/ledger_only/blocked）。"""
    EVOLUTION_RECOVERY.labels(action=action).inc()


def set_kb_write_blocked(blocked: bool) -> None:
    EVOLUTION_BLOCKED.set(1 if blocked else 0)


def set_alias_pointer_mismatch(mismatched: bool) -> None:
    ALIAS_POINTER_MISMATCH.set(1 if mismatched else 0)


def record_object_store_failure(op: str) -> None:
    OBJECT_STORE_FAILURES.labels(op=op).inc()


def record_reranker_fallback() -> None:
    RERANKER_FALLBACK.inc()


def set_outbox_backlog(count: int) -> None:
    OUTBOX_BACKLOG.set(count)


def record_tool_timeout(kind: str) -> None:
    TOOL_TIMEOUT.labels(kind=kind).inc()


def record_mcp_write_indeterminate(tool: str) -> None:
    MCP_WRITE_INDETERMINATE.labels(tool=tool).inc()

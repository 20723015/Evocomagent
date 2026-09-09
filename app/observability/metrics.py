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
    Gauge,
    Histogram,
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

# ---------- 单 Agent 全量优化计划（阶段G）----------
LLM_CALLS = Counter(
    "llm_calls_total", "LLM 调用次数（按用途）", labelnames=("purpose",),
)
TOOL_RESULT_CODES = Counter(
    "tool_result_codes_total", "工具稳定结果码分布", labelnames=("tool", "code"),
)
TOOL_SIDE_EFFECT = Counter(
    "tool_side_effect_total", "工具副作用状态", labelnames=("tool", "status"),
)
CONTEXT_TRUNCATED = Counter(
    "context_truncated_total", "上下文截断/压缩发生（按位置）",
    labelnames=("stage",),  # history | memory
)
CITATION_COVERAGE = Gauge(
    "citation_coverage_ratio", "引用覆盖率（matched/cited，最近一轮）",
)
FACT_GUARD_FAILURES = Counter(
    "fact_guard_failures_total", "声明级事实接地失败（删除无证据数字）",
)
WRITE_CLAIM_BLOCKED = Counter(
    "write_claim_blocked_total", "无 committed 证据的退款成功宣称被改写",
)
WRITE_BLOCKED = Counter(
    "write_blocked_total", "写操作被状态机拦截（按工具）", labelnames=("tool",),
)
# 记忆任务（阶段F）
MEMORY_JOB_BACKLOG = Gauge(
    "memory_job_backlog", "待处理 memory job 数（pending+租约过期）",
)
MEMORY_JOB_LATENCY = Histogram(
    "memory_job_latency_seconds", "memory job 从入队到完成延迟",
    buckets=(1, 5, 15, 60, 300, 900, 3600),
)
MEMORY_JOB_RETRIES = Counter(
    "memory_job_retries_total", "memory job 重试次数",
)
MEMORY_JOB_DEAD = Counter(
    "memory_job_dead_total", "memory job 死信（超过重试上限）",
)
LLM_CONSOLIDATION_CALLS = Counter(
    "llm_consolidation_calls_total", "记忆巩固期间的 LLM 调用（应为 0：close/轮内已去 LLM）",
)
# ---------- Review 修复（Review 问题全量修复计划）----------
FINAL_PROTOCOL_CORRECTIONS = Counter(
    "final_protocol_corrections_total", "终答协议纠错（纯文本输出要求改走 final_response）",
)
TURNS_MISSING_FINAL = Counter(
    "turns_missing_final_total", "无最终回复轮次（强制终答失败/fallback 收尾）",
)
REFUND_CONFIRMATION_BLOCKED = Counter(
    "refund_confirmation_blocked_total", "退款确认拦截（按原因）",
    labelnames=("reason",),  # ambiguous | reserved_args | multiple_pending | cancel
)
MEMORY_JOB_OBSOLETE = Counter(
    "memory_job_obsolete_total", "reset 后按 obsolete 处理的旧 memory job",
)
WATERMARK_REGRESSION_BLOCKED = Counter(
    "watermark_regression_blocked_total", "记忆水位回退被数据库单调 clamp 阻断",
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


# 单价表（美元/1M token）：prompt 与 completion 分开计价——
# 修复历史「70/30 猜测」成本口径（真实 usage 由 ResilientLLM 归集后传入）
def record_llm_usage(model: str, purpose: str,
                     prompt_tokens: int, completion_tokens: int) -> None:
    """按真实 usage 拆分方向计费（分）；无 usage 时只记 token 不记成本。"""
    if prompt_tokens:
        LLM_TOKENS.labels(direction="prompt", purpose=purpose).inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(direction="completion", purpose=purpose).inc(completion_tokens)
    if not (prompt_tokens or completion_tokens) or not model:
        return
    price = _MODEL_PRICES.get(model)
    if price:
        instr_per_m, out_per_m = price
        cents = (
            prompt_tokens / 1_000_000 * instr_per_m * 100
            + completion_tokens / 1_000_000 * out_per_m * 100
        )
        LLM_COST.labels(model=model).inc(cents)


def record_llm_call_count(purpose: str) -> None:
    LLM_CALLS.labels(purpose=purpose).inc()


def record_tool_result_code(tool: str, code: str) -> None:
    TOOL_RESULT_CODES.labels(tool=tool, code=code or "UNKNOWN").inc()


def record_tool_side_effect(tool: str, status: str) -> None:
    TOOL_SIDE_EFFECT.labels(tool=tool, status=status).inc()


def record_context_truncated(stage: str) -> None:
    CONTEXT_TRUNCATED.labels(stage=stage).inc()


def record_citation_coverage(covered: int, total: int) -> None:
    if total > 0:
        CITATION_COVERAGE.set(covered / total)


def record_fact_guard_failure() -> None:
    FACT_GUARD_FAILURES.inc()


def record_write_claim_blocked() -> None:
    WRITE_CLAIM_BLOCKED.inc()


def record_write_blocked(tool: str) -> None:
    WRITE_BLOCKED.labels(tool=tool).inc()

def record_final_protocol_correction() -> None:
    FINAL_PROTOCOL_CORRECTIONS.inc()


def record_turn_missing_final() -> None:
    TURNS_MISSING_FINAL.inc()


def record_refund_confirmation_blocked(reason: str) -> None:
    REFUND_CONFIRMATION_BLOCKED.labels(reason=reason).inc()


def record_memory_job_obsolete() -> None:
    MEMORY_JOB_OBSOLETE.inc()


def record_watermark_regression_blocked() -> None:
    WATERMARK_REGRESSION_BLOCKED.inc()


def record_memory_job_retry() -> None:
    MEMORY_JOB_RETRIES.inc()


def record_memory_job_dead() -> None:
    MEMORY_JOB_DEAD.inc()


def record_memory_job_latency(seconds: float) -> None:
    MEMORY_JOB_LATENCY.observe(max(seconds, 0.0))


def set_memory_job_backlog(count: int) -> None:
    MEMORY_JOB_BACKLOG.set(max(count, 0))


RAG_RETRIEVE_EMPTY = Counter(
    "rag_retrieve_empty_total", "检索空结果（拒答校准观测）",
)


# ---------- KB 异步建库任务（多实例异步改造）----------
KB_JOB_BACKLOG = Gauge(
    "kb_job_backlog", "待处理 KB 建库任务数（queued+retry_wait+running+blocked）",
    labelnames=("operation",),  # upload | delete
)
KB_JOB_OLDEST_AGE = Gauge(
    "kb_job_oldest_age_seconds", "最老未完成任务的年龄（积压告警依据）",
    labelnames=("operation",),
)
KB_JOB_BLOCKED = Gauge(
    "kb_job_blocked", "blocked 任务数（alias 指向未知代，待人工 reconcile）",
)
KB_JOB_STAGE_LATENCY = Histogram(
    "kb_job_stage_seconds", "KB 任务阶段耗时",
    labelnames=("stage",),  # validating|parsing|waiting_for_lock|chunking|embedding|writing_index|activating|finalizing
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
KB_JOB_E2E_LATENCY = Histogram(
    "kb_job_e2e_seconds", "KB 任务端到端延迟（入队→终态）",
    labelnames=("operation",),
    buckets=(1, 5, 15, 60, 300, 900, 3600),
)
KB_JOB_RETRIES = Counter(
    "kb_job_retries_total", "KB 任务自动重试次数", labelnames=("operation",),
)
KB_JOB_DEAD = Counter(
    "kb_job_dead_total", "KB 任务死信（重试耗尽/永久错误）", labelnames=("operation",),
)
KB_JOB_TAKEOVER = Counter(
    "kb_job_takeover_total", "KB 任务租约过期接管次数（Worker 崩溃恢复）",
    labelnames=("operation",),
)
KB_JOB_LEASE_LOST = Counter(
    "kb_job_lease_lost_total", "KB 任务租约/所有权丢失次数",
    labelnames=("kind",),  # heartbeat | ownership
)
KB_JOB_LOCK_WAIT = Counter(
    "kb_job_lock_wait_total", "KB 任务等待全局写锁重回队列次数（不计失败）",
    labelnames=("operation",),
)


# ---------- 人工客服问答沉淀（人工知识支路）----------
HUMAN_QA_QUEUE_DEPTH = Gauge(
    "human_qa_queue_depth", "待导入的人工候选事件数（handoff:evolution:pending）",
)
HUMAN_QA_EVENT_AGE = Gauge(
    "human_qa_event_age_seconds", "最老待导入事件的年龄（导入延迟告警）",
)
HUMAN_QA_IMPORT = Counter(
    "human_qa_import_total", "人工候选导入结果",
    labelnames=("result",),  # imported | rejected | skipped_duplicate | dropped
)
HUMAN_QA_REVIEW = Counter(
    "human_qa_review_total", "人工候选夜间 LLM 评审结论（建议性，不自动发布）",
    labelnames=("result",),  # recommended | duplicate | low_value | judge_failed
)
REVIEW_CONFLICTS = Counter(
    "review_revision_conflicts_total", "审核编辑/批准的 revision 乐观锁冲突",
)


HUMAN_KNOWLEDGE_CLASSIFICATION = Counter(
    "human_knowledge_classification_total", "人工候选评审分类结果",
    labelnames=("classification",),  # new|duplicate|update|low_value|authoritative_conflict|...
)
HUMAN_EVAL_QUEUE_DEPTH = Gauge(
    "human_eval_queue_depth", "人工评审队列深度（queued+retry_wait）",
)
HUMAN_EVAL_OLDEST_AGE = Gauge(
    "human_eval_oldest_age_seconds", "最老评审任务年龄（>26h 告警）",
)
HUMAN_EVAL_BLOCKED = Gauge(
    "human_eval_blocked", "blocked 评审任务数（人工重试入口）",
)
# 人工知识生命周期观测（010：语义去重/审批快照/补偿下架）
HUMAN_DEDUP_DECISION = Counter(
    "human_dedup_decision_total", "人工链路语义去重决策（评审+发布两端）",
    labelnames=("decision",),  # kept|duplicate|replaced|target_changed|answer_side
)
HUMAN_REPLACE_CROSS_CALIBER = Counter(
    "human_replace_cross_caliber_total",
    "替换目标为自动沉淀文档（value_score 与旧 quality_score 跨口径）次数",
)
HUMAN_APPROVAL_STALE = Counter(
    "human_approval_stale_total", "审批快照过期终止（revision 漂移/摘要不一致/状态漂移）",
    labelnames=("reason",),  # revision_drift|digest_mismatch|candidate_status
)
HUMAN_COMPENSATION_RETIRE = Counter(
    "human_compensation_retire_total", "补偿下架任务入队次数（激活后漂移/更高来源版本）",
)
HUMAN_LIFECYCLE_INCONSISTENCY = Counter(
    "human_lifecycle_inconsistency_total", "生命周期不一致观测（CAS miss/文档与行状态背离）",
    labelnames=("kind",),  # settle_cas_miss|document_missing|row_status
)
HUMAN_PUBLISH_PROBE_FAIL = Counter(
    "human_publish_probe_fail_total", "发布存在性探针失败次数（批次回退重试）",
)


def record_human_dedup_decision(decision: str) -> None:
    HUMAN_DEDUP_DECISION.labels(decision=decision or "unknown").inc()


def record_human_replace_cross_caliber() -> None:
    HUMAN_REPLACE_CROSS_CALIBER.inc()


def record_human_approval_stale(reason: str) -> None:
    HUMAN_APPROVAL_STALE.labels(reason=reason or "unknown").inc()


def record_human_compensation_retire() -> None:
    HUMAN_COMPENSATION_RETIRE.inc()


def record_human_lifecycle_inconsistency(kind: str) -> None:
    HUMAN_LIFECYCLE_INCONSISTENCY.labels(kind=kind or "unknown").inc()


def record_human_publish_probe_fail() -> None:
    HUMAN_PUBLISH_PROBE_FAIL.inc()


def record_human_knowledge_classification(classification: str) -> None:
    HUMAN_KNOWLEDGE_CLASSIFICATION.labels(
        classification=classification or "unknown",
    ).inc()


def set_human_eval_stats(stats: dict) -> None:
    try:
        HUMAN_EVAL_QUEUE_DEPTH.set(max(int(stats.get("queue_depth") or 0), 0))
        HUMAN_EVAL_OLDEST_AGE.set(
            max(float(stats.get("oldest_age_seconds") or 0.0), 0.0))
        HUMAN_EVAL_BLOCKED.set(max(int(stats.get("blocked") or 0), 0))
    except Exception:
        pass


def record_human_qa_import(result: str) -> None:
    HUMAN_QA_IMPORT.labels(result=result or "unknown").inc()


def record_human_qa_review(result: str) -> None:
    HUMAN_QA_REVIEW.labels(result=result or "unknown").inc()


def set_human_qa_queue(depth: int, oldest_age_seconds: float = 0.0) -> None:
    HUMAN_QA_QUEUE_DEPTH.set(max(int(depth), 0))
    HUMAN_QA_EVENT_AGE.set(max(float(oldest_age_seconds), 0.0))


def record_review_conflict() -> None:
    REVIEW_CONFLICTS.inc()


def record_kb_job_retry(operation: str) -> None:
    KB_JOB_RETRIES.labels(operation=operation or "unknown").inc()


def record_kb_job_dead(operation: str) -> None:
    KB_JOB_DEAD.labels(operation=operation or "unknown").inc()


def record_kb_job_takeover(operation: str) -> None:
    KB_JOB_TAKEOVER.labels(operation=operation or "unknown").inc()


def record_kb_job_lease_lost(kind: str) -> None:
    KB_JOB_LEASE_LOST.labels(kind=kind).inc()


def record_kb_job_lock_wait(operation: str) -> None:
    KB_JOB_LOCK_WAIT.labels(operation=operation or "unknown").inc()


def record_kb_job_e2e(operation: str, seconds: float) -> None:
    KB_JOB_E2E_LATENCY.labels(operation=operation or "unknown").observe(max(seconds, 0.0))


def record_kb_job_stage(stage: str, seconds: float) -> None:
    KB_JOB_STAGE_LATENCY.labels(stage=stage or "unknown").observe(max(seconds, 0.0))


def set_kb_job_stats(stats: dict) -> None:
    """由 job_store.stats() 驱动的积压/年龄/blocked 仪表盘。"""
    try:
        for op, n in (stats.get("backlog") or {}).items():
            KB_JOB_BACKLOG.labels(operation=op).set(max(n, 0))
        for op, age in (stats.get("oldest_age_seconds") or {}).items():
            KB_JOB_OLDEST_AGE.labels(operation=op).set(max(age, 0.0))
        KB_JOB_BLOCKED.set(max(int(stats.get("blocked") or 0), 0))
    except Exception:
        pass


def record_rag_retrieve(diag: dict) -> None:
    """检索证据诊断指标（阶段D）：空结果计数。"""
    try:
        if not diag or not diag.get("n_items"):
            RAG_RETRIEVE_EMPTY.inc()
    except Exception:
        pass


def record_tokens(direction: str, purpose: str, tokens: int, model: str = "") -> None:
    """兼容保留：无方向拆分时的记账（不再用 70/30 估算成本）。"""
    if not tokens:
        return
    LLM_TOKENS.labels(direction=direction, purpose=purpose).inc(tokens)


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


def record_turn_usage(ctx, state) -> None:
    """轮次用量指标（阶段G）：工具结果码/副作用、上下文截断位置。

    ctx: AgentTurnContext；state: ToolTurnState（读取每工具调用计数）。
    """
    try:
        for entry in ctx.tool_trace:
            record_tool_result_code(entry.name, entry.code)
            if entry.side_effect_status != "none":
                record_tool_side_effect(entry.name, entry.side_effect_status)
        if ctx.context_truncated and ctx.truncation_stage:
            record_context_truncated(ctx.truncation_stage)
    except Exception:
        pass


def record_mcp_write_indeterminate(tool: str) -> None:
    MCP_WRITE_INDETERMINATE.labels(tool=tool).inc()

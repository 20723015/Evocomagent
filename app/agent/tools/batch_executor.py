"""工具批次执行器（Agent能力强化计划·改造二 + 修复计划）：原序分段并行 + 双层限流。

收敛主 Agent（chat.py）与 SubAgent（agents.py）的循环内机制，不再各复制一套：

ToolBatchExecutor（pod 级，无状态；app.state 持有；CLI/评估自建并负责 close）
- 参数解析（json.loads 失败 → 错误 JSON，错误回模型自愈）
- 重复调用检测（同签名本轮内不要求连续 → 拦截；同工具轮内次数上限，默认 2）
- 并行安全分类（PARALLEL_SAFE 白名单；写工具/未入白名单者串行）
- 双层限流：tool_parallelism（单批次提交波次）+ tool_max_concurrent
  （pod 固定池 max_workers = 提交许可总数：先拿 permit 再提交，线程总数有界）

修复计划（对 2a7f2eb 的资源/写语义修正）：
- 每段临时 ThreadPoolExecutor → pod 级固定池（无界排队线程删除）；
- 只读任务到期：尝试 cancel，已运行任务允许跑完并释放 permit；
- 每个只读段 / 串行屏障 / 写工具启动前重新检查预算（耗尽则写工具绝不动）；
- tool_write_timeout_seconds：远端写超时 → status=indeterminate（结果未知，
  禁止自动重试），记录 ToolTurnState.indeterminate_writes 供 Agent 强制
  转人工并写 handoff ticket 对账。

ToolTurnState（请求级，每轮 chat() 新建——pod 单例绝不持有任何请求状态）
- 签名历史、来源集合（改造三）、sequence 计数器、事件回调句柄、
  indeterminate_writes（写结果未知清单）
- on_outcomes：按稳定顺序一次性消费结果（RunTrace/评估）

预算规则（评审·三轮2——ContextVar 不能中断已运行的调用）：
- 提交前检查剩余预算，不足则不发起新工具调用（错误 JSON 回模型）；
- MCP/HTTP 工具调用显式传递 remaining 作为超时；
- 到期后停止等待**只读**任务（结果丢弃）；
- **写工具必须同步等待其幂等结果**——绝不允许响应 fallback 后后台遗留副作用；
  等待上限 = tool_write_timeout_seconds，超时即「结果未知」而非安全失败。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.agent.tools.executor")

# 并行白名单（首批）：纯只读。load_skill 暂不入（loader 延迟缓存写入无锁），
# apply_refund 是写工具（串行屏障）。
PARALLEL_SAFE: frozenset[str] = frozenset({
    "query_order",
    "query_product",
    "query_logistics",
    "list_user_orders",
    "search_knowledge",
    "recall_user_memory",
})

WRITE_TOOLS: frozenset[str] = frozenset({"apply_refund"})

# 阶段C：注册了状态机的写工具集合（新写工具必须注册，否则执行前被拦截）
_WRITE_STATE_MACHINE_TOOLS: frozenset[str] = frozenset({
    "apply_refund",
})

DUP_CALL_ERROR = "重复调用被拦截：请换参数或基于已有结果回答（同一调用本轮已执行过）"
BUDGET_SKIP_ERROR = "本轮预算已耗尽，工具调用被跳过"
TOOL_TIMEOUT_ERROR = "工具执行超时（本轮预算耗尽），结果丢弃"
PERMIT_TIMEOUT_ERROR = "工具调度超时：并发许可等待超时，调用被跳过"
WRITE_INDETERMINATE_ERROR = (
    "远端写操作执行超时，结果未知（indeterminate）：禁止自动重试，"
    "已转人工按幂等键对账"
)
PARSE_ERROR = "工具参数不是合法 JSON，请检查后重试"

# 轮内同工具调用次数上限（2.5 起可配置）：默认 2 次，search_knowledge 放宽到
# 3 次（换关键词重检索是合法行为）。守卫有总开关（tool_call_guard_enabled），
# 供消融基线关闭对比；上限值读 settings.tool_max_calls_per_name /
# tool_search_max_calls（单测可 configure_guard 覆盖）。


@dataclass
class ToolOutcome:
    """一次工具调用的稳定结果（回填顺序 = 模型给定顺序）。"""

    call_id: str
    name: str
    arguments: dict
    result: str  # JSON 字符串（含错误 JSON）
    sequence: int
    skipped: bool = False  # True = 未真正执行（解析失败/重复拦截/预算跳过/超时弃等/indeterminate）
    internal_args: dict | None = None  # Review 修复：仅执行器可写的内部参数通道（确认凭证注入）


@dataclass
class ToolTurnState:
    """请求级轮次状态：每轮 chat() 新建，绝不进 pod 单例。"""

    event_callback: Callable[[str, dict], None] | None = None
    on_outcomes: Callable[[list[ToolOutcome]], None] | None = None
    sources: set[str] = field(default_factory=set)  # 改造三：本轮检索来源（规范化值）
    indeterminate_writes: list[dict] = field(default_factory=list)  # 写结果未知清单
    seen_signatures: set[str] = field(default_factory=set)  # 本轮内出现过的签名（去重用）
    per_name_counts: dict[str, int] = field(default_factory=dict)  # 本轮内同工具调用次数
    write_ops: object | None = None  # 阶段C：写操作状态机（WriteOpTracker；惰性建）
    refund_confirm: dict | None = None  # Review 修复：服务端确认判定（confirm 时含注入载荷）
    _sequence: int = 0

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def emit(self, event_type: str, data: dict) -> None:
        if self.event_callback is not None:
            try:
                self.event_callback(event_type, data)
            except Exception:
                log.warning("tool 事件回调失败", exc_info=True)

    def write_tracker(self):
        """写操作状态机（惰性构建，避免无写轮次的构造开销）。"""
        if self.write_ops is None:
            from app.agent.write_ops import WriteOpTracker

            self.write_ops = WriteOpTracker()
        return self.write_ops


def parse_arguments(raw) -> tuple[dict | None, str | None]:
    """解析工具参数；失败返回 (None, 错误 JSON)——错误回模型自愈。"""
    if isinstance(raw, dict):
        return dict(raw), None
    try:
        parsed = json.loads(raw if isinstance(raw, str) else json.dumps(raw))
        return (parsed if isinstance(parsed, dict) else {}), None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, json.dumps(
            {"error": PARSE_ERROR}, ensure_ascii=False,
        )


def signature_of(name: str, arguments: dict) -> str:
    """重复检测签名：name + 参数 canonical JSON（仅本轮内比较）。"""
    return f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def extract_sources_from_result(result: str) -> list[str]:
    """从 search_knowledge 结果 JSON 提取来源别名（排除 tainted=true 的块）。

    被污染的召回不能为引用背书（改造三）；来源同时收 doc 与 source_path 两路。
    返回规范化值（citations.normalize_source）。
    """
    from app.agent.citations import normalize_source

    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict) or item.get("tainted") is True:
            continue  # tainted 字段（kb_chunk_tainted 是函数名不是字段名）
        for key in ("doc", "source_path"):
            value = item.get(key)
            if value:
                out.append(normalize_source(str(value)))
    return out


class ToolBatchExecutor:
    """pod 级无状态批次执行器：固定线程池 + 提交许可（双层限流）。

    2.5：调用守卫（签名去重 + 单工具次数上限）可配置——
    tool_call_guard_enabled=False 时仅保留预算检查（消融 baseline 用）。
    """

    def __init__(self, *, parallelism: int | None = None,
                 max_concurrent: int | None = None):
        self._parallelism = (
            settings.tool_parallelism if parallelism is None else max(parallelism, 1)
        )
        concurrent = (
            settings.tool_max_concurrent
            if max_concurrent is None else max(max_concurrent, 1)
        )
        # pod 级固定池：线程总数 = 全局并发上限（不再逐段建临时池）
        self._pool = ThreadPoolExecutor(
            max_workers=concurrent, thread_name_prefix="tool-pod",
        )
        # 提交许可：先拿 permit 再 submit——固定池队列不会出现无界排队线程
        self._permit = threading.BoundedSemaphore(concurrent)
        self._closed = False
        self._active = 0
        self._active_lock = threading.Lock()
        self._active_peak = 0
        # 2.5：守卫开关 + 次数上限（配置化；单测可显式覆盖）
        self._guard_enabled = settings.tool_call_guard_enabled
        self._max_calls_per_name = max(settings.tool_max_calls_per_name, 1)
        self._search_max_calls = max(settings.tool_search_max_calls, 1)

    def configure_guard(self, *, enabled: bool | None = None,
                        max_calls_per_name: int | None = None,
                        search_max_calls: int | None = None) -> None:
        """测试/评测按配置覆盖守卫参数（默认跟随 settings）。"""
        if enabled is not None:
            self._guard_enabled = bool(enabled)
        if max_calls_per_name is not None:
            self._max_calls_per_name = max(max_calls_per_name, 1)
        if search_max_calls is not None:
            self._search_max_calls = max(search_max_calls, 1)

    # ---------- 观测（测试用） ----------
    @property
    def active_peak(self) -> int:
        with self._active_lock:
            return self._active_peak

    @property
    def closed(self) -> bool:
        return self._closed

    def reset_metrics(self) -> None:
        with self._active_lock:
            self._active = 0
            self._active_peak = 0

    def close(self) -> None:
        """幂等关闭固定池（FastAPI lifespan / Sandbox / 自建 Agent 所有者负责）。"""
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=False)

    # ---------- 入口 ----------
    def execute(
        self,
        tool_calls: list[dict],  # [{id, name, arguments(raw str|dict)}] 模型给定顺序
        state: ToolTurnState,
        ctx,
        tool_manager,
        budget=None,  # Optional[TurnBudget]
    ) -> list[ToolOutcome]:
        """执行一批工具调用；返回与输入等长、按模型顺序排列的 ToolOutcome。"""
        if self._closed:
            raise RuntimeError("ToolBatchExecutor 已关闭，拒绝新批次")
        outcomes: list[ToolOutcome | None] = [None] * len(tool_calls)
        plan: list[tuple[int, str, dict]] = []  # (index, name, args)

        for i, call in enumerate(tool_calls):
            call_id = str(call.get("id", ""))
            name = str(call.get("name", ""))
            arguments, parse_error = parse_arguments(call.get("arguments"))
            if parse_error is not None:
                # 模型确实请求了该调用：tool_call 事件与序号照发（客户端可
                # 凭 tool_call_id 对回 tool_result 错误 JSON）
                outcomes[i] = self._make_outcome(
                    state, call_id, name, {}, parse_error, skipped=True,
                    emit_call=True,
                )
                continue
            signature = signature_of(name, arguments)
            # 重复检测（2026-08 评测后收紧；2.5 可配置）：
            # 1) 同签名本轮内出现过（不要求连续）→ 拦截；
            # 2) 同工具轮内调用超过上限（默认 2 次，search_knowledge 3 次）→ 拦截。
            # 跨轮 state 每轮新建，多轮合法重查不受影响。
            if self._guard_enabled and signature in state.seen_signatures:
                log.info("tool.dup_blocked", name=name)
                record_dup_blocked(name)
                outcomes[i] = self._make_outcome(
                    state, call_id, name, arguments,
                    json.dumps({"error": DUP_CALL_ERROR}, ensure_ascii=False),
                    skipped=True, emit_call=True,
                )
                continue
            name_count = state.per_name_counts.get(name, 0) + 1
            limit = (
                self._search_max_calls if name == "search_knowledge"
                else self._max_calls_per_name
            )
            if self._guard_enabled and name_count > limit:
                log.info("tool.burst_blocked", name=name, count=name_count)
                record_tool_limit_reached(name)
                outcomes[i] = self._make_outcome(
                    state, call_id, name, arguments,
                    json.dumps(
                        {"error": f"同一工具调用次数过多（本轮已 {name_count - 1} 次），请基于已有结果回答"},
                        ensure_ascii=False,
                    ),
                    skipped=True, emit_call=True,
                )
                continue
            state.seen_signatures.add(signature)
            state.per_name_counts[name] = name_count
            if budget is not None and budget.expired():
                # 提交规则①：剩余预算不足 → 不发起新工具调用（写工具同规，
                # 预算耗尽后绝不动）
                outcomes[i] = self._make_outcome(
                    state, call_id, name, arguments,
                    json.dumps({"error": BUDGET_SKIP_ERROR}, ensure_ascii=False),
                    skipped=True, emit_call=True,
                )
                continue
            # 阶段C：写操作状态机拦截（缺参/越权重试/未注册状态机的一律不执行
            # ——不依赖模型自觉遵守流程；预算检查在前，耗尽后根本到不了这里）
            if name in WRITE_TOOLS or name in _WRITE_STATE_MACHINE_TOOLS:
                # Review 修复：本轮确认判定为 ambiguous/cancel → 禁止一切写
                # 工具。cancel 若不在这里 fail-closed，模型可在取消后重新
                # 调用第一段 apply_refund，重新签发待确认请求。
                decision = state.refund_confirm or {}
                if decision.get("action") in ("ambiguous", "cancel"):
                    from app.observability.metrics import (
                        record_refund_confirmation_blocked as _rcb,
                    )

                    _rcb(str(decision.get("action")))
                    outcomes[i] = self._make_outcome(
                        state, call_id, name, arguments,
                        json.dumps({
                            "error": "REFUND_CONFIRMATION_REQUIRED",
                            "message": (
                                "用户已取消待确认的退款操作，本轮已禁止重新发起；"
                                "如需退款请等待下一轮重新明确提出申请。"
                                if decision.get("action") == "cancel" else
                                "存在待用户确认的退款操作，本轮已禁止执行写操作；"
                                "请先回应用户的确认或取消"
                            ),
                        }, ensure_ascii=False),
                        skipped=True, emit_call=True,
                    )
                    continue
                if decision.get("action") == "confirm":
                    # 服务端判定的确认只适用于精确的待确认订单+原因。若模型
                    # 改写了任一字段，禁止把它当成第一段新退款请求，避免借
                    # 着用户对 O1 的确认发起 O2/另一原因退款。
                    target_order = str(decision.get("order_id", "") or "").strip()
                    target_reason = str(decision.get("reason", "") or "").strip()
                    call_order = str(arguments.get("order_id", "") or "").strip()
                    call_reason = str(arguments.get("reason", "") or "").strip()
                    if (call_order != target_order
                            or call_reason != target_reason):
                        from app.observability.metrics import (
                            record_refund_confirmation_blocked as _rcb,
                        )

                        _rcb("target_mismatch")
                        outcomes[i] = self._make_outcome(
                            state, call_id, name, arguments,
                            json.dumps({
                                "error": "REFUND_CONFIRMATION_TARGET_MISMATCH",
                                "message": (
                                    "本轮确认仅适用于用户明确确认的同一订单和退款原因；"
                                    "参数不匹配，已禁止执行。"
                                ),
                            }, ensure_ascii=False),
                            skipped=True, emit_call=True,
                        )
                        continue
                verdict = state.write_tracker().check(name, arguments)
                if verdict is not None:
                    log.info("tool.write_blocked name=%s", name)
                    record_write_blocked(name)
                    state.write_tracker().blocked_calls.append(
                        {"tool": name, "arguments": arguments}
                    )
                    outcomes[i] = self._make_outcome(
                        state, call_id, name, arguments, verdict,
                        skipped=True, emit_call=True,
                    )
                    continue
            internal_args = None
            if name == "apply_refund":
                decision = state.refund_confirm or {}
                if decision.get("action") == "confirm" and str(
                    arguments.get("order_id", "")
                ) == str(decision.get("order_id", "")):
                    # Review 修复：仅服务端判定 confirm 时经内部通道注入
                    # token/幂等键——模型参数面永远不含保留字段
                    internal_args = {
                        "confirmation_token": decision.get("token", ""),
                        "idempotency_key": decision.get("refund_id", ""),
                        "refund_id": decision.get("refund_id", ""),
                    }
            plan.append((i, name, arguments))
            outcomes[i] = ToolOutcome(
                call_id=call_id, name=name, arguments=arguments,
                result="", sequence=state.next_sequence(),
                internal_args=internal_args,
            )
            state.emit("tool_call", {
                "name": name, "arguments": arguments,
                "tool_call_id": call_id, "sequence": outcomes[i].sequence,
            })

        # 原序分段：连续 PARALLEL_SAFE 段并行；其余（含写工具）串行屏障
        segments: list[tuple[str, list[int]]] = []  # ("read", indices) | ("write"/"barrier", [idx])
        read_seg: list[int] = []

        def flush_reads():
            if read_seg:
                segments.append(("read", list(read_seg)))
                read_seg.clear()

        for idx, name, _args in plan:
            if name in PARALLEL_SAFE and name not in WRITE_TOOLS:
                read_seg.append(idx)
            else:
                flush_reads()
                segments.append(("write" if name in WRITE_TOOLS else "barrier", [idx]))
        flush_reads()

        budget_exhausted = False
        for seg_kind, indices in segments:
            if budget_exhausted or (budget is not None and budget.expired()):
                # 段前预算重查：写工具在预算耗尽后绝不动
                for idx in indices:
                    self._mark_skipped(outcomes[idx], state, BUDGET_SKIP_ERROR)
                continue
            if seg_kind == "read":
                budget_exhausted = self._run_parallel_segment(
                    indices, outcomes, state, ctx, tool_manager, budget,
                )
            else:
                self._run_barrier(
                    outcomes, indices[0], state, ctx, tool_manager,
                    budget, write=(seg_kind == "write"),
                )

        final = [o for o in outcomes if o is not None]
        # 稳定回填：on_outcomes 按模型给定顺序一次性交付（评估轨迹保序）
        if state.on_outcomes is not None:
            try:
                state.on_outcomes(final)
            except Exception:
                log.warning("on_outcomes 轨迹回调失败", exc_info=True)
        return final

    # ---------- 内部 ----------
    def _make_outcome(self, state, call_id, name, arguments, result, *,
                      skipped: bool, emit_call: bool) -> ToolOutcome:
        sequence = state.next_sequence()
        if emit_call:
            state.emit("tool_call", {
                "name": name, "arguments": arguments,
                "tool_call_id": call_id, "sequence": sequence,
            })
        outcome = ToolOutcome(
            call_id=call_id, name=name, arguments=arguments,
            result=result, sequence=sequence, skipped=skipped,
        )
        self._emit_result(state, outcome)
        return outcome

    def _emit_result(self, state, outcome: ToolOutcome) -> None:
        state.emit("tool_result", {
            "tool_call_id": outcome.call_id, "tool_name": outcome.name,
            "sequence": outcome.sequence, "result": outcome.result,
        })

    def _mark_skipped(self, outcome: ToolOutcome, state, error: str) -> None:
        """段前/许可失败：未真正执行——错误 JSON + tool_result 事件。"""
        outcome.result = json.dumps({"error": error}, ensure_ascii=False)
        outcome.skipped = True
        self._emit_result(state, outcome)

    def _acquire_permit(self, budget) -> bool:
        """提交前获取全局许可；有轮次预算时等待上限 = remaining（不穿越 deadline）。"""
        if budget is None:
            self._permit.acquire()
            return True
        remaining = budget.remaining()
        if remaining <= 0:
            return False
        return self._permit.acquire(timeout=remaining)

    def _submit(self, name, arguments, ctx, tool_manager, remaining,
                budget, state, outcome) -> Future | None:
        """permit + 固定池提交；失败（许可超时）→ 标记跳过并返回 None。"""
        if not self._acquire_permit(budget):
            record_tool_timeout("permit")
            self._mark_skipped(outcome, state, PERMIT_TIMEOUT_ERROR)
            return None
        # 获取许可本身可能正好耗尽最后一点预算。提交前再做一次 fail-closed
        # 检查，尤其保证排在慢只读段后的写操作不会越过 deadline 启动。
        if budget is not None and budget.expired():
            self._permit.release()
            record_tool_timeout("permit")
            self._mark_skipped(outcome, state, BUDGET_SKIP_ERROR)
            return None
        future = self._pool.submit(
            self._run_one, name, arguments, ctx, tool_manager, remaining,
            outcome.internal_args,
        )
        future.add_done_callback(lambda f: self._permit.release())
        return future

    def _run_one(self, name: str, arguments: dict, ctx, tool_manager,
                 remaining: float | None,
                 internal_args: dict | None = None) -> str:
        """单工具执行（固定池 worker 内）；MCP/HTTP 传递 remaining 超时。

        internal_args：仅执行器可写的内部参数通道（Review 修复）——
        确认凭证/幂等键在本通道注入，模型参数面不含保留字段。
        """
        with self._active_lock:
            self._active += 1
            self._active_peak = max(self._active_peak, self._active)
        try:
            # Review 修复：internal_args 走独立通道（校验后合并），
            # 模型参数面与工具校验面均不含保留字段
            return tool_manager.execute_tool(
                name, arguments, ctx, timeout=remaining,
                internal_args=internal_args,
            )
        finally:
            with self._active_lock:
                self._active -= 1

    def _run_parallel_segment(self, indices: list[int], outcomes, state, ctx,
                              tool_manager, budget) -> bool:
        """连续只读段：波次提交（每次 ≤ parallelism）；到期 cancel + 弃等。

        已运行任务允许完成并释放 permit——线程总数由固定池保证有界。
        """
        remaining = budget.remaining() if budget is not None else None
        submit_remaining = (
            remaining if remaining is not None and remaining > 0 else None
        )
        # 段级 deadline：逐个 future 的等待被夹到同一时刻，总等待不随段长放大
        deadline = (
            None if remaining is None else time.monotonic() + max(remaining, 0.0)
        )
        queue = list(indices)
        deadline_reached = False
        while queue:
            if budget is not None and budget.expired():
                for idx in queue:
                    self._mark_skipped(outcomes[idx], state, BUDGET_SKIP_ERROR)
                return True
            wave = queue[: self._parallelism]
            queue = queue[self._parallelism :]

            futures: dict[int, Future] = {}
            for idx in wave:
                outcome = outcomes[idx]
                future = self._submit(
                    outcome.name, outcome.arguments, ctx, tool_manager,
                    submit_remaining, budget, state, outcome,
                )
                if future is not None:
                    futures[idx] = future
                elif budget is not None:
                    # 有预算时 _submit 返回 None 表示许可等待或提交前检查
                    # 已触达 deadline；后续段（特别是写段）必须熔断。
                    deadline_reached = True

            for idx in wave:  # 按模型原序等待/回填
                future = futures.get(idx)
                if future is None:
                    continue  # permit 超时已标记跳过
                outcome = outcomes[idx]
                wait = (
                    None if deadline is None
                    else max(deadline - time.monotonic(), 0.0)
                )
                try:
                    result = future.result(timeout=wait)
                except FutureTimeout:
                    # 提交规则③：到期停止等待只读任务（尝试 cancel；
                    # 已运行任务自身跑完自行丢结果并释放 permit）
                    log.warning("tool.readonly_timeout name=%s", outcome.name)
                    record_tool_timeout("readonly")
                    future.cancel()
                    self._finish(outcome, json.dumps(
                        {"error": TOOL_TIMEOUT_ERROR}, ensure_ascii=False,
                    ), state, skipped=True)
                    deadline_reached = True
                    continue
                except Exception as e:  # noqa: BLE001
                    self._finish(outcome, json.dumps(
                        {"error": f"工具执行出错: {e}"}, ensure_ascii=False,
                    ), state)
                    continue
                self._finish(outcome, result, state)
        return deadline_reached

    def _run_barrier(self, outcomes, idx: int, state, ctx, tool_manager,
                     budget, write: bool) -> None:
        """串行屏障：写工具（同步等幂等结果）或白名单外工具（保守串行）。"""
        outcome = outcomes[idx]
        future = self._submit(
            outcome.name, outcome.arguments, ctx, tool_manager, None,
            budget, state, outcome,
        )
        if future is None:
            return  # permit 超时已标记跳过
        if write:
            # 写超时 ≠ 安全失败：远端把「结果未知」当 indeterminate 处理
            wait = max(settings.tool_write_timeout_seconds, 0.0)
            try:
                result = future.result(timeout=wait)
            except FutureTimeout:
                log.warning("tool.write_timeout name=%s", outcome.name)
                record_tool_timeout("write")
                record_mcp_write_indeterminate(outcome.name)
                future.cancel()
                # 幂等键在 internal_args（服务端注入通道，模型参数面
                # 不含保留字段）；arguments 只兜底开发直退路径
                internal = outcome.internal_args or {}
                state.indeterminate_writes.append({
                    "tool": outcome.name,
                    "order_id": str(outcome.arguments.get("order_id", "")),
                    "idempotency_key": str(
                        internal.get("idempotency_key")
                        or internal.get("refund_id")
                        or outcome.arguments.get("idempotency_key") or ""
                    ),
                })
                self._finish(outcome, json.dumps({
                    "status": "indeterminate",
                    "tool": outcome.name,
                    "error": WRITE_INDETERMINATE_ERROR,
                }, ensure_ascii=False), state, skipped=True)
                return
            except Exception as e:
                log.warning("tool.execute_failed name=%s", outcome.name, exc_info=True)
                self._finish(outcome, json.dumps(
                    {"error": f"工具执行出错: {e}"}, ensure_ascii=False,
                ), state)
                return
            self._finish(outcome, result, state)
            return
        # 非写屏障：无等待上限不合理——按轮次剩余夹逼（本地工具即时）
        deadline = None if budget is None else time.monotonic() + max(budget.remaining(), 0.0)
        try:
            if deadline is None:
                result = future.result()
            else:
                result = future.result(timeout=max(deadline - time.monotonic(), 0.0))
        except FutureTimeout:
            log.warning("tool.barrier_timeout name=%s", outcome.name)
            record_tool_timeout("readonly")
            future.cancel()
            self._finish(outcome, json.dumps(
                {"error": TOOL_TIMEOUT_ERROR}, ensure_ascii=False,
            ), state, skipped=True)
            return
        except Exception as e:
            log.warning("tool.execute_failed name=%s", outcome.name, exc_info=True)
            self._finish(outcome, json.dumps(
                {"error": f"工具执行出错: {e}"}, ensure_ascii=False,
            ), state)
            return
        self._finish(outcome, result, state)

    def _finish(self, outcome: ToolOutcome, result: str, state,
                skipped: bool = False) -> ToolOutcome:
        outcome.result = result
        outcome.skipped = skipped
        self._emit_result(state, outcome)
        # 改造三：search_knowledge 来源入本轮来源集合（tainted 已在提取时排除）
        if outcome.name == "search_knowledge":
            state.sources.update(extract_sources_from_result(result))
        return outcome


def record_dup_blocked(tool: str) -> None:
    from app.observability.metrics import record_dup_blocked as _r
    _r(tool)


def record_tool_limit_reached(tool: str) -> None:
    from app.observability.metrics import record_tool_limit_reached as _r
    _r(tool)


def record_tool_timeout(kind: str) -> None:
    from app.observability.metrics import record_tool_timeout as _r
    _r(kind)


def record_mcp_write_indeterminate(tool: str) -> None:
    from app.observability.metrics import record_mcp_write_indeterminate as _r
    _r(tool)


def record_write_blocked(tool: str) -> None:
    from app.observability.metrics import record_write_blocked as _r
    _r(tool)

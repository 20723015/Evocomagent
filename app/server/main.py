"""FastAPI 服务骨架（阶段一 1.1 / 阶段二 2.2/2.6/2.7）：
POST /v1/chat、POST /v1/sessions/reset、GET /healthz、GET /readyz。

- 执行模型（1.7）：路由层 async def 只做编排，同步 Agent 一律经
  runtime.run_agent_turn 丢线程池，事件循环内不直接调同步 LLM/工具。
- 会话锁（2.2）：同一 session 同一时刻一个写入者，冲突返回 409。
- idle 兜底巩固（2.3）：后台任务扫描静默会话，防 pod 重启丢 LTM。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import anyio
from anyio import to_thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import settings
from app.handoff.board import HandoffConflict, HandoffNotFound, HandoffTicket
from app.security.identifiers import InvalidIdentifier, validate_identifier
from app.security.principal import SCOPE_HUMAN_INGEST, SCOPE_OPS, authorize_scopes
from app.security.ratelimit import BudgetStoreUnavailable
from app.server import runtime
from app.server.channels import (
    InvalidChannel,
    build_channel_outbox,
    channel_user_id,
    normalize_inbound,
    outbound_payload,
    validate_channel_id,
)
from app.server.deps import (
    PodComponents,
    authenticate_user,
    build_agent,
    build_pod_components,
)
from app.server.schema import (
    ChannelMessageRequest,
    ChannelMessageResponse,
    ChannelOutboundResponse,
    ChatRequest,
    ChatResponse,
    HandoffNoteRequest,
    HandoffResolveRequest,
    HealthResponse,
    SessionResetRequest,
    SessionResetResponse,
)
from app.stores.base import (
    SessionConflictError,
    SessionOwnershipError,
    StorageUnavailableError,
)
from app.stores.locks import (
    SessionLease,
    SessionLockBackendUnavailable,
    SessionLockLost,
)

logger = logging.getLogger("app.server")

PROJECT = "并夕夕 · 智能客服「小夕」API"

# 3.5：命中 guardrail 的降级话术（转人工）
SAFE_FALLBACK_REPLY = "抱歉，我无法处理您这条请求，已为您转接人工客服，请稍候。"
# 阶段A：InputPolicy 的输入拦截话术（Agent 内同样使用，语义一致）
GUARDRAIL_BLOCK_REPLY = (
    "抱歉，您的消息包含疑似指令注入/敏感内容，出于安全考虑已被拦截，"
    "本条消息已转接人工客服处理，请稍候。"
)

# GET /v1/chat/stream 弃用标记（0.3.0 弃用一个版本，0.4.0 删除）。
# Sunset 取 0.4.0 计划发布窗口；deprecation 事件为老客户端未知的 SSE 事件
# 类型，按规范被安全忽略（评审·坑5 验收）。
SSE_DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Sunset": "Tue, 01 Sep 2026 00:00:00 GMT",
    "Link": '</v1/chat/stream>; rel="successor-version"',
}
SSE_DEPRECATION_EVENT = {
    "message": "GET /v1/chat/stream 已弃用，0.4.0 起移除，请改用 POST /v1/chat/stream",
    "sunset": "2026-09-01",
}

# P2-3：SLA 超时扫描间隔（秒）。模块级常量（不新增 settings 开关）——
# 后台兜底扫描保证即使无人轮询看板，超时工单也会被首次观测并计入指标。
HANDOFF_SLA_SCAN_SECONDS = 60


def _sse(event_type: str, data: dict) -> str:
    """SSE 帧：`event: <type>\ndata: <json>\n\n`。"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _handoff_meta(ticket: HandoffTicket | None) -> dict | None:
    """P2-3：用户可见的转人工状态（会话 meta / 响应字段 / 渠道出站共用）。

    「已转人工，工单号 X」随 ChatResponse.handoff、SSE handoff 事件与渠道
    出站消息透出，用户可凭工单号与坐席对账。
    """
    if ticket is None:
        return None
    return {
        "ticket_id": ticket.ticket_id,
        "status": ticket.status,
        "message": f"已转人工，工单号 {ticket.ticket_id}，请等待坐席接入。",
    }


def _safe_fallback(session_id: str, handoff: dict | None = None) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        reply=SAFE_FALLBACK_REPLY,
        intent="other",
        confidence=0.0,
        requires_human=True,
        follow_up_question=None,
        handoff=handoff,
    )


def _request_credentials(user_id: str) -> dict:
    """请求级外部凭证（2.3）：JWT 主体 + 商家服务级 Bearer。

    凭证随 ToolContext 注入，永不进 prompt/轨迹/日志；只会被
    HTTPCommerceGateway 消费为 Authorization 头。
    """
    from app.config.settings import settings

    creds = {"sub": user_id, "scope": "write"}
    if settings.commerce_api_key:
        creds["commerce_token"] = settings.commerce_api_key
    return creds


def _input_prefilter(message: str):
    """服务端快速输入预检（阶段A：经统一 InputPolicy，guardrail-only）。

    返回 (message, blocked)。注入 → blocked（调用方返回降级话术）；
    PII → 脱敏后的 message。范围闸门等完整评估由 Agent 内 InputPolicy
    执行（需要 client/model）——两处共用同一核心实现。
    """
    from app.agent.input_policy import evaluate_input

    decision = evaluate_input(message)
    return decision.text, decision.action == "guardrail_block"


async def _complete_handoff(
    components: PodComponents, agent, result, reply: str, requires_human: bool
) -> HandoffTicket | None:
    """共享响应完成器：普通与 SSE 接口统一的 Handoff 工单创建（阶段A）。

    requires_human 时生成 handoff 包推坐席；工单失败不阻断已完成的对话。
    返回工单（成功时）供用户侧状态（工单号）透出；失败/无需转人工 → None。
    """
    if not requires_human:
        return None
    from app.observability.metrics import record_handoff

    record_handoff()
    try:
        if getattr(result, "reply", "") != reply:
            result.reply = reply  # 输出侧纵深防御替换同步进工单内容
        from app.handoff.board import build_handoff_ticket

        ticket = build_handoff_ticket(agent, result)
        # board.create 是同步 Redis 调用（C1：不阻塞事件循环）
        await asyncio.to_thread(components.handoff_board.create, ticket)
        return ticket
    except Exception:
        logger.warning("handoff ticket 创建失败", exc_info=True)
        return None


async def _create_guardrail_handoff(
    components: PodComponents, user_id: str, session_id: str, message: str
) -> HandoffTicket | None:
    """输入护栏命中 → 建转人工工单（reply 已是安全话术，仍需坐席复核原始输入）。

    blocked 发生在 build_agent 之前（无 agent/result 上下文），直构最小工单；
    建单失败不阻断降级响应（与 _complete_handoff 同口径）。返回工单供
    用户侧状态（工单号）透出。
    """
    from app.observability.metrics import record_handoff

    record_handoff("guardrail_blocked")
    try:
        from app.handoff.board import HandoffTicket, _now

        ticket = HandoffTicket(
            ticket_id=uuid.uuid4().hex,
            user_id=user_id,
            session_id=session_id,
            intent="other",
            question=message,
            reply=SAFE_FALLBACK_REPLY,
            suggested_actions=["输入命中注入护栏，请人工确认原始输入"],
            created_at=_now(),
        )
        await asyncio.to_thread(components.handoff_board.create, ticket)
        return ticket
    except Exception:
        logger.warning("guardrail handoff ticket 创建失败", exc_info=True)
        return None


async def _run_chat_turn(
    components: PodComponents,
    user_id: str,
    session_id: str,
    message: str,
    *,
    credentials: dict | None = None,
):
    """共享 chat 管线（/v1/chat 与渠道 webhook 复用，P2-2）。

    限流（RPS/日预算）→ 会话租约（同 session 单写入者）→ build_agent →
    runtime.run_agent_turn → close；异常映射与历史 /v1/chat 完全一致：
    429（限流）/ 409（锁冲突、CAS 冲突）/ 403（会话归属）/ 503（锁后端、
    租约失效、预算存储、外置存储故障）。返回 (agent, result)。

    写确认协议（pending_write / 工具状态机）全部在 Agent 内，本函数不复制。
    """
    from app.observability.metrics import (
        RATE_LIMITED,
        REACT_STEPS,
        record_chat_latency,
        record_conflict,
    )

    limiter = components.limiter
    limiter.bind_user(user_id)  # 修复计划·四：LLM 包装层据此预留/结算
    if not limiter.allow_rps(user_id):
        RATE_LIMITED.labels(kind="rps").inc()
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    try:
        if not limiter.allow_budget(user_id):
            RATE_LIMITED.labels(kind="budget").inc()
            raise HTTPException(
                status_code=429, detail="今日用量已达上限，请明日再试或联系人工客服"
            )
    except BudgetStoreUnavailable as e:
        # 修复计划·四：预算存储故障（redis_required）→ 503，不降级本地计数
        raise HTTPException(status_code=503, detail=f"预算存储暂不可用: {e}") from e

    # 评审二轮 B2：用量按 request 归集（ContextVar 随 anyio 线程池传播，
    # turn 线程与 close 线程的 LLM 调用都记到本请求名下）
    request_id = uuid.uuid4().hex
    components.usage_tracker.begin_request(request_id)
    try:
        # 2.2：同 session 只有一个写入者（连点/双 pod 竞态）→ 冲突 409；
        # 修复计划·一：锁后端不可用（生产 redis_required）→ 503 而非降级
        try:
            lease_ctx = SessionLease(
                components.locks, user_id, session_id or "session"
            )
            lease = lease_ctx.__enter__()
        except SessionLockBackendUnavailable as e:
            raise HTTPException(
                status_code=503, detail=f"会话锁后端暂不可用: {e}"
            ) from e
        try:
            if lease.token is None:
                record_conflict()
                raise HTTPException(
                    status_code=409,
                    detail="该会话正在处理中，请稍候再试",
                )
            try:
                agent = build_agent(
                    user_id,
                    session_id,
                    components,
                    credentials=credentials,
                )
            except SessionOwnershipError as e:
                # 3.1：session 归属不属于当前用户 → 403
                raise HTTPException(status_code=403, detail=str(e)) from e
            # 修复计划·一：绑定租约校验回调（写工具/保存/重置前校验）
            if hasattr(agent, "bind_lease_guard"):
                agent.bind_lease_guard(lease.assert_owned)
            try:
                started = time.monotonic()
                result = await runtime.run_agent_turn(agent, message)
                record_chat_latency(time.monotonic() - started)
                REACT_STEPS.observe(getattr(agent, "_react_steps_count", 1) or 1)
            except SessionConflictError as e:
                # CAS 冲突（本轮期间被其他写入者改过）→ 客户端应重读重试
                record_conflict()
                raise HTTPException(status_code=409, detail=f"会话版本冲突: {e}") from e
            except SessionLockLost as e:
                # 租约中途失效：禁止提交副作用/会话状态 → 503（可重试）
                raise HTTPException(status_code=503, detail=f"会话租约已失效: {e}") from e
            except StorageUnavailableError as e:
                # 5.2 故障注入：外置存储断连 → 503（明确不静默降级写文件，避免状态分裂）
                raise HTTPException(status_code=503, detail=f"存储暂不可用: {e}") from e
            finally:
                # close 的 LLM 巩固调用同样记到本请求（在 end_request 之前）
                await runtime.run_agent_close(agent)
        except BudgetStoreUnavailable as e:
            # 修复计划·二轮 7：reserve/settle/close 阶段预算存储故障统一 503
            raise HTTPException(
                status_code=503, detail=f"预算存储暂不可用: {e}"
            ) from e
        finally:
            lease.release()
    finally:
        # 修复计划·四：清理/观测统一 finally——不再二次扣费（LLM 包装层已按
        # usage 原子结算）；锁冲突/构建失败/异常路径也保证清理
        components.usage_tracker.end_request(request_id)
    return agent, result


def _es_of(components):
    """取组件当前可用的 ES 客户端（修复计划·二轮 5：单一可恢复 provider）。

    provider 每次调用反映当前可用性；兼容仅有 es() 方法的轻量替身。
    """
    provider = getattr(components, "es_provider", None)
    if provider is not None:
        return provider()
    fn = getattr(components, "es", None)
    if callable(fn):
        return fn()
    return None


def _channel_outbox(request: Request):
    """渠道出站队列（P2-2）：lifespan 装配；缺失时按 Redis 可用性惰性补建。

    与 handoff board 同模式：Redis 可用 → 持久队列（多 Pod 共享），
    否则进程内队列（单机开发/测试）。
    """
    outbox = getattr(request.app.state, "channel_outbox", None)
    if outbox is None:
        components = getattr(request.app.state, "components", None)
        outbox = build_channel_outbox(
            getattr(components, "redis", None) if components is not None else None
        )
        request.app.state.channel_outbox = outbox
    return outbox


def _mark_sla_breaches_sync(board) -> int:
    """同步标记首次超时的工单（线程池执行）；返回新增数量。"""
    mark = getattr(board, "mark_sla_breaches", None)
    if not callable(mark):
        return 0
    newly = mark()
    return len(newly or [])


async def _handoff_sla_loop(components: PodComponents) -> None:
    """P2-3：SLA 超时后台兜底扫描（无人轮询看板也能观测到超时）。

    首次观测才计数（去重由工单板负责），异常静默下轮再试；关闭时取消。
    """
    board = getattr(components, "handoff_board", None)
    if board is None or not callable(getattr(board, "mark_sla_breaches", None)):
        return
    from app.observability.metrics import record_handoff_sla_breach

    while True:
        try:
            await anyio.sleep(HANDOFF_SLA_SCAN_SECONDS)
            newly = await to_thread.run_sync(board.mark_sla_breaches)
            if newly:
                record_handoff_sla_breach(len(newly))
                logger.info("SLA 超时工单 %s 张", len(newly))
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning("SLA 超时扫描异常", exc_info=True)


def _validate_request_ids(user_id: str, session_id: str) -> None:
    """user_id/session_id 字符集白名单（安全修复 P2：路径穿越）。

    标识符会拼进会话/记忆文件路径与 Redis key；`../` 等一律 422 拒绝。
    """
    try:
        validate_identifier(user_id, "user_id")
        validate_identifier(session_id, "session_id", allow_empty=True)
    except InvalidIdentifier as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _disconnect_wait_bound_seconds() -> float:
    """断连后等待不可中断 Agent 工作的上限（评审·坑1：防 slowloris 变体）。

    Agent能力强化计划·改造一：由 turn_budget_seconds 推导（覆盖 LLM 全链路
    deadline + 工具提交规则：只读弃等 / 写工具同步等幂等结果可能略超预算），
    不再按 max_react_steps × 单次超时估算——步数扩到 8 后该估算与真实墙钟
    脱钩。×1.5 留收尾余量；超时后收尾交线程自行完成，不占用请求。
    """
    return max(settings.turn_budget_seconds, 30.0) * 1.5


def _current_trace_id() -> str:
    """评审二轮 B5：错误响应携带关联 id。

    优先取当前 OTel span 的 trace_id（阶段四埋点），无有效 span 时生成
    短随机 id——两者都能把客户端拿到的错误与服务端日志对上。
    """
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and ctx.is_valid:
            return format(
                otel_trace.get_current_span().get_span_context().trace_id, "032x"
            )[:16]
    except Exception:
        pass
    return uuid.uuid4().hex[:16]


def _schedule_disconnected_finalize(
    task, agent, components, user_id, limiter, request_id: str = "",
    lease=None, registry: dict | None = None,
) -> None:
    """断连收尾（评审·坑3 / 修复计划·一/二轮 6：租约随任务移交后台）。

    收尾（等 Agent 工作真正结束 → close → 用量记账 → 释放会话租约）放请求
    cancel scope 之外的独立 asyncio task——生成器 finally 在 GeneratorExit
    上下文里不允许再 await 可能让出事件循环的操作。

    等待上限（_disconnect_wait_bound_seconds）只用于记录 overdue 告警，
    不再代表放弃/释放锁：Agent 线程不可中断，必须等它真正结束才 close、
    入账并释放租约（否则锁提前释放会让第二个请求并发写同一会话）。

    registry：应用级 {task: lease} 映射——持强引用防 GC，关闭时可对超时任务
    的租约执行「只停续租、不删锁」（TTL 回收），绝不 cancel 触发 finally 放锁。
    """
    bound = _disconnect_wait_bound_seconds()

    def _settle() -> None:
        """用量清理/观测（request 粒度，评审二轮 B2；幂等——end 后重复调用为 0）。

        修复计划·四：只清理，不再二次扣费（LLM 包装层已按 usage 原子结算）。
        """
        components.usage_tracker.end_request(request_id)

    async def _finalize():
        try:
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=bound)
                except asyncio.TimeoutError:
                    logger.warning(
                        "客户端断连后 Agent 工作超过 %.0fs 未结束（overdue 告警）；"
                        "继续等待其真正结束，期间不释放会话锁",
                        bound,
                    )
                    # 只告警，不放弃：继续等待真正的结束条件（Agent 真正结束）
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning("断连后的 Agent 轮次以异常结束", exc_info=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("断连后的 Agent 轮次以异常结束", exc_info=True)
            try:
                await runtime.run_agent_close(agent)
            except Exception:
                logger.warning("断连收尾 agent.close() 失败", exc_info=True)
            _settle()
        except asyncio.CancelledError:
            # 被取消（关闭超时等）：只停续租，不删锁——底层 Agent 线程可能仍在跑
            if lease is not None:
                try:
                    lease.stop_renew()
                except Exception:
                    logger.warning("断连收尾停止续租失败", exc_info=True)
            raise
        finally:
            # 租约最后释放：Agent 真正结束后才放锁；abandoned 状态不主动删除
            if lease is not None and not lease.abandoned:
                try:
                    lease.release()
                except Exception:
                    logger.warning("断连收尾释放会话租约失败", exc_info=True)

    finalize_task = asyncio.create_task(_finalize())
    if registry is not None:
        registry[finalize_task] = lease

    def _log_crash(done: asyncio.Task) -> None:
        if registry is not None:
            registry.pop(done, None)
        if not done.cancelled() and done.exception() is not None:
            logger.warning("断连收尾任务异常", exc_info=done.exception())

    finalize_task.add_done_callback(_log_crash)


async def _idle_consolidator_loop(components: PodComponents) -> None:
    """后台兜底巩固：每 scan 间隔扫描一次静默会话（2.3），出错静默下轮再试。"""
    if components.redis is None:
        return
    # SQL 模式由持久化 memory-job worker 独占巩固水位；idle consolidator
    # 使用折叠模型消息长度，和 chat_messages.seq 不同，不能并行运行。
    # memory-job worker（SQL 或文件队列）启用后由它独占记忆巩固；idle
    # consolidator 只作为显式关闭 worker 时的兼容兜底。文件模式也不能
    # 并行：文件队列的水位不是 SessionState.consolidated_len，双跑会重复抽取。
    if components.db_engine is not None or settings.memory_job_worker_enabled:
        return
    if settings.memory_consolidate_idle_minutes <= 0:
        return
    from app.stores.idle_consolidator import run_idle_consolidation

    while True:
        try:
            await anyio.sleep(
                settings.memory_consolidate_scan_minutes * 60 + random.random() * 30
            )
            handled = await to_thread.run_sync(
                partial(
                    run_idle_consolidation,
                    components.session_store,
                    components.ltm_store,
                    components.client,
                    settings.model_name,
                    idle_minutes=settings.memory_consolidate_idle_minutes,
                    memory_dir=settings.memory_dir,
                    max_ltm_facts=settings.max_ltm_facts,
                )
            )
            if handled:
                logger.info("idle consolidate 完成: %s", handled)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning("idle consolidate 扫描异常", exc_info=True)


async def _kb_gc_loop(components: PodComponents) -> None:
    """KB 上传保留期 GC：定时执行（须持 kb_write 锁；上限处理，失败静默下轮再试）。"""
    svc = getattr(components, "upload_service", None)
    if svc is None:
        return
    while True:
        try:
            await anyio.sleep(settings.kb_gc_interval_seconds + random.random() * 30)
        except asyncio.CancelledError:
            return

        def _gc_once():
            lock = None
            try:
                import logging as _log

                lock = svc._make_lock()
                lock.acquire(phase="gc")
                return svc.gc(limit=settings.kb_gc_max_items)
            except Exception as e:
                _log.getLogger("app.server").warning("kb gc 失败: %s", e)
                return 0
            finally:
                if lock is not None:
                    try:
                        lock.release()
                    except Exception:
                        pass

        try:
            handled = await to_thread.run_sync(_gc_once)
            if handled:
                logger.info("kb gc 完成 %s 项", handled)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning("kb gc 异常", exc_info=True)


async def _es_outbox_loop(components: PodComponents) -> None:
    """阶段八：MySQL→ES 消息同步 + reset 删除事件（outbox）。

    ES 挂了只跳过重试，不影响主流程；删除事件未完成期间由运营搜索 tombstone
    过滤保证「重置后立即不可搜索」。
    """
    if components.db_engine is None:
        return
    from app.stores.sql.outbox import ensure_message_index

    index = components.message_index
    while True:
        try:
            await anyio.sleep(30)
            es_client = _es_of(components)  # 修复计划·三：每轮取可恢复 provider
            if es_client is None:
                continue  # ES 未配置/暂不可用：下轮再试（恢复后自动继续）
            try:
                ensure_message_index(es_client, index)
            except Exception:
                logger.warning("message_search 索引初始化失败", exc_info=True)
                continue
            # 5.x：outbox 积压指标 + 删除滞后（修复计划·二）
            from app.observability.metrics import (
                set_outbox_backlog,
                set_outbox_delete_backlog,
            )
            from app.stores.sql.outbox import (
                count_pending_delete_events,
                count_pending_outbox,
                run_outbox_once,
            )

            try:
                backlog = await to_thread.run_sync(
                    partial(count_pending_outbox, components.db_engine)
                )
                set_outbox_backlog(backlog)
                del_count, del_lag = await to_thread.run_sync(
                    partial(count_pending_delete_events, components.db_engine)
                )
                set_outbox_delete_backlog(del_count, del_lag)
            except Exception:
                pass
            synced = await to_thread.run_sync(
                partial(
                    run_outbox_once,
                    components.db_engine, es_client, index,
                )
            )
            if synced:
                logger.info("outbox→ES 同步 %s 条", synced)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning("outbox→ES 同步异常", exc_info=True)


async def _memory_job_worker_loop(components: PodComponents) -> None:
    """阶段F：异步记忆 worker 循环（SQL/文件队列；LLM 巩固不再占用对话轮）。

    每 5 秒领取一批 memory job（含崩溃租约接管与水位幂等）；空转/异常静默
    下轮再试，绝不影响主流程。关闭时由 lifespan 取消。
    """
    if not settings.memory_job_worker_enabled:
        return

    def _tick() -> bool:
        from app.agent.memory.jobs import build_memory_job_worker

        worker = getattr(components, "memory_job_worker", None)
        if worker is None:
            worker = build_memory_job_worker(components)
            components.memory_job_worker = worker
        try:
            return worker.process_once() > 0
        except Exception as e:
            logger.warning("memory job worker 异常: %s", type(e).__name__)
            return False

    while True:
        try:
            await anyio.sleep(5)
            processed = await to_thread.run_sync(_tick)
            if processed:
                logger.info("memory job worker 处理 %s 批", processed)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning("memory job worker 循环异常", exc_info=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # 阶段二 2.7：redis_required=true 时 build_pod_components 内 fast-fail
    app.state.components: PodComponents = build_pod_components()
    # 阶段四：结构日志 / 追踪初始化
    from app.observability.logging import configure_logging
    from app.observability.tracing import init_tracing

    configure_logging(json_output=settings.log_format != "console")
    init_tracing(settings.otel_service_name)
    # 3.1：auth_enabled 时空/短 JWT_SECRET → 启动即失败（fail-fast，不在运行时静默兜底）
    from app.security.jwt import validate_jwt_secret

    validate_jwt_secret()
    # 修复计划·一/二轮 6：断连收尾任务应用级 {task: lease} 映射
    app.state.finalize_tasks: dict = {}
    # P2-2：渠道出站队列（Redis 可用 → 持久；否则进程内）
    app.state.channel_outbox = build_channel_outbox(
        getattr(app.state.components, "redis", None)
    )
    app.state.consolidator_task = asyncio.create_task(
        _idle_consolidator_loop(app.state.components)
    )
    app.state.es_outbox_task = asyncio.create_task(
        _es_outbox_loop(app.state.components)
    )
    app.state.kb_gc_task = asyncio.create_task(_kb_gc_loop(app.state.components))
    # 阶段F：异步记忆 worker（SQL memory_jobs / 文件轻量队列）
    app.state.memory_job_task = asyncio.create_task(
        _memory_job_worker_loop(app.state.components)
    )
    # P2-3：SLA 超时后台扫描
    app.state.handoff_sla_task = asyncio.create_task(
        _handoff_sla_loop(app.state.components)
    )
    yield
    # 修复计划·一/二轮 6：先停止接流量（uvicorn 已在 shutdown 前停止 accept），
    # 再等待断连收尾任务完成——它们持有会话租约。
    # 等待窗口 = 应用 drain 上限（断连任务上限）；K8s terminationGracePeriodSeconds
    # 在此基础上再加 30 秒强制退出缓冲（Helm 模板断言）。
    pending = list(getattr(app.state, "finalize_tasks", {}).items())
    if pending:
        grace = _disconnect_wait_bound_seconds()
        logger.info("关闭：等待 %d 个断连收尾任务（上限 %.0fs）", len(pending), grace)
        _, still = await asyncio.wait([t for t, _ in pending], timeout=grace)
        for task in still:
            lease = getattr(app.state, "finalize_tasks", {}).get(task)
            # 绝不 cancel（会触发 finally 释放锁）——只停续租，剩余锁由 TTL 回收
            if lease is not None:
                try:
                    lease.stop_renew()
                except Exception:
                    logger.warning("关闭：停止续租失败", exc_info=True)
        if still:
            logger.warning(
                "关闭：仍有 %d 个断连收尾任务未完成；已停止续租、不释放会话锁"
                "（TTL 回收，Agent 线程继续收尾）",
                len(still),
            )
    app.state.kb_gc_task.cancel()
    try:
        await app.state.kb_gc_task
    except asyncio.CancelledError:
        pass
    app.state.es_outbox_task.cancel()
    try:
        await app.state.es_outbox_task
    except asyncio.CancelledError:
        pass
    app.state.consolidator_task.cancel()
    try:
        await app.state.consolidator_task
    except asyncio.CancelledError:
        pass
    app.state.memory_job_task.cancel()
    try:
        await app.state.memory_job_task
    except asyncio.CancelledError:
        pass
    app.state.handoff_sla_task.cancel()
    try:
        await app.state.handoff_sla_task
    except asyncio.CancelledError:
        pass
    mcp = app.state.components.mcp_client
    if mcp is not None:
        try:
            mcp.close()
        except Exception:
            logger.warning("MCP client 关闭失败", exc_info=True)
    executor = app.state.components.tool_executor
    if executor is not None:
        try:
            executor.close()
        except Exception:
            logger.warning("工具执行器关闭失败", exc_info=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title=PROJECT,
        version="0.3.0",
        description="电商客服 Agent 多用户 API（0.3.0：安全修复基线；GET SSE 已弃用）",
        lifespan=_lifespan,
    )

    from app.server.human_knowledge import router as human_knowledge_router
    from app.server.uploads import router as kb_uploads_router

    app.include_router(kb_uploads_router)
    app.include_router(human_knowledge_router)

    @app.get("/metrics", include_in_schema=False, tags=["ops"])
    async def metrics():
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        """兜底：未处理异常返回 JSON 500 并记录日志。

        安全修复 P2：响应体不回显异常文本（历史 `内部错误: {exc}` 会把内部
        路径/SQL/密钥细节泄给客户端）。评审二轮 B5：携带 trace_id 供客户端
        与服务端日志对账（OTel span 有效时取其 trace_id，否则短随机 id）。
        """
        trace_id = _current_trace_id()
        logger.error(
            "unhandled error path=%s trace_id=%s",
            request.url.path,
            trace_id,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "内部错误，请稍后重试", "trace_id": trace_id},
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz():
        """进程存活探针：只表示 uvicorn 活着（liveness）。"""
        return {"status": "ok", "components": {}}

    @app.get("/readyz", response_model=HealthResponse, tags=["ops"])
    async def readyz(request: Request):
        """就绪探针：4.1 起检查 Redis/MySQL schema/ES alias/对象存储。

        兼容保留 status 字段；新增 components 明细。
        各依赖在 settings 对应开关开启时才强校验（未配置=不依赖=视为就绪）；
        Redis 在 redis_required=true 时强校验（2.7 语义不变）。
        """
        components = getattr(request.app.state, "components", None)
        if components is None:
            raise HTTPException(status_code=503, detail="组件未就绪")

        # ---------- 依赖状态：not_configured / ok / unavailable ----------
        deps: dict[str, str] = {}

        redis = getattr(components, "redis", None)
        if redis is None:
            deps["redis"] = "unavailable" if settings.redis_required else "not_configured"
        else:
            try:
                redis.ping()
                deps["redis"] = "ok"
            except Exception:
                deps["redis"] = "unavailable"

        if getattr(components, "db_engine", None) is None:
            deps["mysql_schema"] = "not_configured"
        else:
            try:
                from app.stores.sql.engine import _verify_schema_version

                if settings.app_env.lower() == "prod":
                    _verify_schema_version(components.db_engine)
                deps["mysql_schema"] = "ok"
            except Exception:
                deps["mysql_schema"] = "unavailable"

        # ES：可恢复 provider（未配置 → not_configured；配置但连不上 → unavailable）
        from app.agent.rag.es_util import es_dependency_state, get_es_client

        deps["es"] = (
            "ok" if _es_of(components) is not None else es_dependency_state()
        )

        es_alias = "not_configured"
        if settings.rag_backend == "es":
            if deps["es"] != "ok":
                es_alias = "unavailable"
            else:
                try:
                    hits = get_es_client().indices.get_alias(
                        name=f"{settings.es_index_prefix}-kb-active"
                    )
                    es_alias = "ok" if hits else "unavailable"
                except Exception:
                    es_alias = "unavailable"
        deps["es_kb_alias"] = es_alias

        object_store = getattr(components, "object_store", None)
        needs_s3 = (
            settings.kb_upload_storage == "s3"
            or settings.turns_archive_backend == "s3"
        )
        if object_store is None:
            deps["object_store"] = "unavailable" if needs_s3 else "not_configured"
        else:
            try:
                deps["object_store"] = (
                    "ok" if object_store.healthcheck() else "unavailable"
                )
            except Exception:
                deps["object_store"] = "unavailable"

        # RAG 修复计划·3：检索配置健康校验（ES/generation/embedding 一致/
        # reranker/阈值打分器）；rag_backend=es 时任一不可用 → 聊天核心 fail-closed
        from app.agent.rag.health import check_rag_configuration

        try:
            rag_health = await asyncio.to_thread(check_rag_configuration)
        except Exception as e:
            rag_health = {
                "status": "unavailable",
                "checks": {"rag_health": "unavailable"},
                "errors": [f"RAG 健康校验异常: {type(e).__name__}"],
            }
        deps["rag"] = rag_health["status"]
        if rag_health["status"] == "unavailable":
            logger.warning("readyz RAG 校验未通过: %s", rag_health["errors"])

        # ---------- 能力分级 ----------
        def _worst(states: list[str]) -> str:
            if "unavailable" in states:
                return "unavailable"
            if states and all(s == "not_configured" for s in states):
                return "not_configured"
            return "ok"

        # chat：Redis（required 时）、MySQL Schema、rag_backend=es 时的 ES alias
        # 与 RAG 配置校验；任一不可用 → 聊天核心 fail-closed（503 not_ready）
        chat_deps: list[str] = []
        if settings.redis_required:
            chat_deps.append(deps["redis"])
        if getattr(components, "db_engine", None) is not None:
            chat_deps.append(deps["mysql_schema"])
        if settings.rag_backend == "es":
            chat_deps.append(
                "ok" if deps["es_kb_alias"] == "ok" else "unavailable"
            )
            chat_deps.append(rag_health["status"])

        # kb_upload：DB、Redis（required 时）、配置为 S3 时的对象存储
        if not settings.kb_upload_enabled:
            kb_upload = "not_configured"
        else:
            kb_states: list[str] = []
            if getattr(components, "db_engine", None) is None:
                kb_states.append("unavailable")
            else:
                kb_states.append(deps["mysql_schema"])
            if settings.redis_required:
                kb_states.append(deps["redis"])
            if settings.kb_upload_storage == "s3":
                kb_states.append(deps["object_store"])
            kb_upload = _worst(kb_states)

        capabilities = {
            "chat": _worst(chat_deps),
            "message_search": _worst([deps["es"]]),
            "kb_upload": kb_upload,
            "turn_archive": (
                _worst([deps["object_store"]])
                if settings.turns_archive_backend == "s3"
                else "not_configured"
            ),
            "rag": rag_health["status"],
        }

        if capabilities["chat"] == "unavailable":
            status = "not_ready"
        elif any(v == "unavailable" for v in capabilities.values()):
            status = "degraded"
        else:
            status = "ready"

        # 5.x：依赖就绪指标（探针同源，供告警与看板）
        from app.observability.metrics import set_dependency_readiness

        for component, value in deps.items():
            set_dependency_readiness(component, value == "ok")

        # 5.x：alias/pointer 一致性指标（es 后端时）
        if deps["es"] == "ok":
            try:
                from app.evolution.generation import GenerationStore
                from app.observability.metrics import set_alias_pointer_mismatch

                es = get_es_client()
                gen_store = GenerationStore(Path(settings.kb_generation_path))
                pointer = gen_store.active("es")
                alias_target = ""
                if es is not None and pointer is not None:
                    try:
                        hits = es.indices.get_alias(
                            name=f"{settings.es_index_prefix}-kb-active"
                        )
                        alias_target = str(list(hits.keys())[0]) if hits else ""
                    except Exception:
                        alias_target = ""
                    set_alias_pointer_mismatch(
                        bool(alias_target and alias_target != pointer.target)
                    )
            except Exception:
                pass

        payload = {"status": status, "components": deps, "capabilities": capabilities}
        if status == "not_ready":
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(body: ChatRequest, request: Request):
        components: PodComponents = request.app.state.components
        # 3.1：auth_enabled 时 user_id 从 Bearer JWT 解出（请求体不再接受）
        user_id = authenticate_user(request, body.user_id)

        # 3.5/阶段A 输入侧：统一 InputPolicy 快速预检（注入 → 降级话术转人工；
        # PII → 脱敏后继续）；完整评估（含范围闸门）在 Agent 内执行
        message, blocked = _input_prefilter(body.message)
        if blocked:
            logger.warning("guardrail block input user=%s", user_id)
            ticket = await _create_guardrail_handoff(
                components, user_id, body.session_id, body.message
            )
            return _safe_fallback(body.session_id, _handoff_meta(ticket))

        # 3.7 限流 + 2.2 会话租约 + Agent 轮次（渠道入口复用同一管线，P2-2）
        agent, result = await _run_chat_turn(
            components,
            user_id,
            body.session_id,
            message,
            credentials=_request_credentials(user_id),
        )

        reply = result.reply
        requires_human = result.requires_human
        # 3.5 输出侧：敏感词 → 降级安全话术并转人工（Agent 内 TurnFinalizer
        # 已做过同一检查——此处为纵深防御，命中时结果一致）
        if settings.guardrails_enabled:
            from app.security.guardrails import check_output

            out_verdict = check_output(reply)
            if out_verdict.blocked:
                logger.warning("guardrail block output user=%s", user_id)
                reply = SAFE_FALLBACK_REPLY
                requires_human = True
        # P2-3：工单号随响应透出（用户可见「已转人工，工单号 X」）
        ticket = await _complete_handoff(
            components, agent, result, reply, requires_human
        )

        return ChatResponse(
            session_id=agent.session_id,
            reply=reply,
            intent=result.intent.value
            if hasattr(result.intent, "value")
            else str(result.intent),
            confidence=result.confidence,
            requires_human=requires_human,
            follow_up_question=result.follow_up_question,
            pending_turn=getattr(agent, "pending_turn", None),
            handoff=_handoff_meta(ticket),
        )

    @app.get("/v1/chat/stream", tags=["chat"])
    async def chat_stream(
        request: Request,
        user_id: str = "",
        session_id: str = "",
        message: str = "",
    ):
        """（已弃用）4.5 SSE 流式：meta → thought/tool_call/... → reply → end。

        0.3.0 起弃用一个版本：响应头 Deprecation/Sunset + 流首 deprecation
        事件（老客户端未知的 SSE 事件类型，按规范安全忽略）；0.4.0 删除，
        继任端点 POST /v1/chat/stream。
        """
        return await _chat_stream_response(
            request,
            user_id=user_id,
            session_id=session_id,
            message=message,
            deprecate=True,
        )

    @app.post("/v1/chat/stream", response_class=StreamingResponse, tags=["chat"])
    async def chat_stream_post(request: Request, body: ChatRequest):
        """4.5 SSE 流式（继任端点，0.3.0 新增）：事件序列同 GET 版，参数走
        JSON body（与 POST /v1/chat 的请求模型一致）。

        事件序列：meta → thought* → tool_call* /
        tool_result* ×（Agent能力强化计划·改造二） → reply → end。

        改造二（additive 字段，老客户端忽略）：
        - tool_call：新增 `tool_call_id` + `sequence`（模型给定顺序的序号）；
        - tool_result：新增 `tool_call_id` + `tool_name` + `sequence`。
        其余事件字段（thought{text} / reply{reply,intent,confidence,
        requires_human,follow_up_question} / meta{session_id} / end{ok} /
        error{detail,trace_id} / deprecation）不变。

        推理模型适配 T6（additive 事件，老客户端按 SSE 规范忽略未知类型；
        与上方 `:938-941` 弃用事件的既有兼容做法一致）：
        - 开关 `sse_reasoning_enabled`（默认 **false**）打开时事件序列变为
          `thought* → reasoning* thought*`：`reasoning{text,step}` 承载模型推理
          原文（透出侧已过输出 guardrails，长度有界，见 chat.emit_reasoning）；
        - 关闭（默认）时序列与字段和改造二逐字节一致——「透出受控状态说明、
          不透出模型原始思考」仍是有意设计，开启属产品/合规决策；
        - 两种情况下 reasoning 都进审计存储（turns archive，T3），
          "可观测可审计"不依赖透出。
        """
        return await _chat_stream_response(
            request,
            user_id=body.user_id,
            session_id=body.session_id,
            message=body.message,
            deprecate=False,
        )

    async def _chat_stream_response(
        request: Request,
        *,
        user_id: str,
        session_id: str,
        message: str,
        deprecate: bool,
    ) -> StreamingResponse:
        """SSE 流式共用实现（安全修复 P2 的核心重构）。

        历史缺陷：SessionLease 在路由 `return StreamingResponse(...)` 时即
        退出——生成器实际消费在锁释放之后，流式期间会话无保护。现在：
        - 租约在生成器体内获取/释放（finally），覆盖整个流式生命周期；
        - Agent 任务在生成器内启动（客户端不消费就不开跑，天然缓解慢连接
          占池）；断连（GeneratorExit/CancelledError）把租约随 Agent task 一并
          移交独立收尾任务：等待上限只记 overdue 告警，Agent 真正结束后才
          close/入账/释放锁（生成器 finally 内不再 await，评审·坑1/坑3）。
        """
        components: PodComponents = request.app.state.components
        user_id = authenticate_user(request, user_id)
        _validate_request_ids(user_id, session_id)
        headers = dict(SSE_DEPRECATION_HEADERS) if deprecate else None

        if settings.guardrails_enabled:
            from app.security.guardrails import check_input

            verdict = check_input(message)
            if verdict.blocked:
                logger.warning("guardrail block input user=%s", user_id)
                ticket = await _create_guardrail_handoff(
                    components, user_id, session_id, message
                )
                handoff = _handoff_meta(ticket)

                async def _blocked():
                    if deprecate:
                        yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                    yield _sse(
                        "reply",
                        {
                            "reply": SAFE_FALLBACK_REPLY,
                            "requires_human": True,
                            "handoff": handoff,
                        },
                    )
                    if handoff is not None:
                        yield _sse("handoff", handoff)
                    yield _sse("end", {"ok": True})

                return StreamingResponse(
                    _blocked(),
                    media_type="text/event-stream",
                    headers=headers,
                )
            message = verdict.text

        limiter = components.limiter
        limiter.bind_user(user_id)  # 修复计划·四：LLM 包装层据此预留/结算
        if not limiter.allow_rps(user_id):

            async def _limited():
                if deprecate:
                    yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                yield _sse("error", {"detail": "请求过于频繁，请稍后再试"})
                yield _sse("end", {"ok": False})

            return StreamingResponse(
                _limited(),
                media_type="text/event-stream",
                headers=headers,
            )

        # 与同步端点 /v1/chat 同口径：流式路径不得绕过每日 token 预算
        try:
            budget_ok = limiter.allow_budget(user_id)
        except BudgetStoreUnavailable as e:
            # 修复计划·四：预算存储故障 → 503（响应尚未开始，可返回 HTTP 状态）
            raise HTTPException(status_code=503, detail=f"预算存储暂不可用: {e}") from e
        if not budget_ok:
            from app.observability.metrics import RATE_LIMITED

            RATE_LIMITED.labels(kind="budget").inc()

            async def _budget_limited():
                if deprecate:
                    yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                yield _sse(
                    "error",
                    {"detail": "今日用量已达上限，请明日再试或联系人工客服"},
                )
                yield _sse("end", {"ok": False})

            return StreamingResponse(
                _budget_limited(),
                media_type="text/event-stream",
                headers=headers,
            )

        # 归属校验在生成器外完成：403 语义保留（不降级为流内 error 事件）；
        # 它只读 session 文档，不需要会话锁
        try:
            agent = build_agent(
                user_id,
                session_id,
                components,
                credentials=_request_credentials(user_id),
            )
        except SessionOwnershipError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e

        async def _stream():
            # 评审二轮 B2：request 粒度用量归集——在生成器体内设定
            # ContextVar，finalize 独立任务经创建时上下文继承；
            # anyio 线程池拷贝上下文，close 阶段的巩固调用同样入账
            request_id = uuid.uuid4().hex
            components.usage_tracker.begin_request(request_id)
            lease = None
            handed_off = False
            task = None
            interrupted = False
            agent_closed = False
            try:
                try:
                    lease = SessionLease(
                        components.locks,
                        user_id,
                        session_id or "session",
                    ).__enter__()
                except SessionLockBackendUnavailable as e:
                    # 修复计划·一：锁后端不可用 → 稳定错误码（响应头已发出，
                    # 只能走流内 error 事件 + end.ok=false）
                    if deprecate:
                        yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                    yield _sse("error", {
                        "code": "session_lock_unavailable",
                        "detail": f"会话锁后端暂不可用: {e}",
                    })
                    yield _sse("end", {"ok": False})
                    return
                if lease.token is None:
                    if deprecate:
                        yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                    yield _sse("error", {
                        "code": "session_lock_conflict",
                        "detail": "该会话正在处理中，请稍候再试",
                    })
                    yield _sse("end", {"ok": False})
                    return
                # 修复计划·一：绑定租约校验（写工具/保存前校验）
                if hasattr(agent, "bind_lease_guard"):
                    agent.bind_lease_guard(lease.assert_owned)
                if deprecate:
                    yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                yield _sse("meta", {
                    "session_id": agent.session_id,
                    # 掉线恢复：非空 = 上次回复未完成（客户端可提示重发）
                    "pending_turn": getattr(agent, "pending_turn", None),
                })

                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                agent.event_callback = lambda etype, data: loop.call_soon_threadsafe(
                    queue.put_nowait, (etype, data)
                )
                task = asyncio.create_task(runtime.run_agent_turn(agent, message))
                # 完成哨兵也走队列：事件与哨兵 FIFO 有序，避免
                # 「task 完成但回调尚未入队」的丢失竞态
                task.add_done_callback(
                    lambda _t: loop.call_soon_threadsafe(
                        queue.put_nowait, ("__done__", None)
                    )
                )

                while True:
                    etype, data = await queue.get()
                    if etype == "__done__":
                        break
                    yield _sse(etype, data)
                result = task.result()
                reply = result.reply
                requires_human = result.requires_human
                if settings.guardrails_enabled:
                    from app.security.guardrails import check_output

                    if check_output(reply).blocked:
                        reply = SAFE_FALLBACK_REPLY
                        requires_human = True
                yield _sse(
                    "reply",
                    {
                        "reply": reply,
                        "intent": result.intent.value
                        if hasattr(result.intent, "value")
                        else str(result.intent),
                        "confidence": result.confidence,
                        "requires_human": requires_human,
                        "follow_up_question": result.follow_up_question,
                    },
                )
                # 阶段A：普通与 SSE 复用同一 Handoff 完成器（SSE 之前漏建工单）
                # P2-3：工单号以 handoff 事件（+reply.handoff 字段）透出给用户
                ticket = await _complete_handoff(
                    components, agent, result, reply, requires_human
                )
                handoff = _handoff_meta(ticket)
                if handoff is not None:
                    yield _sse("handoff", handoff)
                # 修复计划·二轮 7：先完成需要计费的 close（可能触发预算故障），
                # 再报告成功——避免先 end.ok=true 之后才暴露预算故障。
                try:
                    await runtime.run_agent_close(agent)
                    agent_closed = True
                except BudgetStoreUnavailable as e:
                    if deprecate:
                        yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                    yield _sse("error", {
                        "code": "budget_store_unavailable",
                        "detail": f"预算存储暂不可用: {e}",
                    })
                    yield _sse("end", {"ok": False})
                    return
                yield _sse("end", {"ok": True})
            except BudgetStoreUnavailable as e:
                # reserve/settle 阶段预算存储故障 → 稳定错误码 + end.ok=false
                if deprecate:
                    yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                yield _sse("error", {
                    "code": "budget_store_unavailable",
                    "detail": f"预算存储暂不可用: {e}",
                })
                yield _sse("end", {"ok": False})
            except (GeneratorExit, asyncio.CancelledError):
                # 客户端断连：Agent 线程不可中断——租约随 Agent task 一并移交
                # 后台收尾任务（本生成器不再释放；续期继续，直到 Agent 真正
                # 结束才 close/入账/放锁）。本 finally 内不做任何 await。
                interrupted = True
                if lease is not None and not lease.handed_off:
                    lease.handover()
                    handed_off = True
                _schedule_disconnected_finalize(
                    task,
                    agent,
                    components,
                    user_id,
                    limiter,
                    request_id,
                    lease=lease,
                    registry=getattr(request.app.state, "finalize_tasks", None),
                )
                raise
            except SessionLockLost as e:
                # 修复计划·一：租约中途失效 → 稳定错误码 + end.ok=false
                trace_id = _current_trace_id()
                logger.warning(
                    "stream session lock lost session=%s/%s trace_id=%s",
                    user_id, session_id, trace_id,
                )
                yield _sse("error", {
                    "code": "session_lock_lost",
                    "detail": "会话租约已失效，本轮结果未提交，请重试",
                    "trace_id": trace_id,
                })
                yield _sse("end", {"ok": False})
            except Exception as e:
                # 安全修复 P2：不回显异常细节（与 500 脱敏同一原则）
                trace_id = _current_trace_id()
                logger.error(
                    "stream turn failed session=%s/%s trace_id=%s",
                    user_id,
                    session_id,
                    trace_id,
                    exc_info=e,
                )
                yield _sse(
                    "error",
                    {
                        "detail": "本轮对话处理失败，请稍后重试",
                        "trace_id": trace_id,
                    },
                )
                yield _sse("end", {"ok": False})
            finally:
                if not interrupted:
                    if not agent_closed:
                        # 正常/业务错误收尾：不在 GeneratorExit 上下文，可安全
                        # await——close（巩固调用用量记到本请求）；成功路径已在
                        # end 之前 close 过，避免重复
                        try:
                            await runtime.run_agent_close(agent)
                        except Exception:
                            logger.warning("stream agent.close() 失败", exc_info=True)
                    components.usage_tracker.end_request(request_id)
                # 修复计划·一：已移交后台的租约由收尾任务释放（不在此重复释放）
                if lease is not None and not handed_off:
                    lease.release()

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    # ============================================================
    # P2-2 渠道适配层（generic webhook；渠道就绪性演示，不接真实第三方渠道）
    # ============================================================
    @app.post(
        "/v1/channels/{channel}/messages",
        response_model=ChannelMessageResponse,
        tags=["channel"],
    )
    async def channel_message(
        channel: str, body: ChannelMessageRequest, request: Request
    ):
        """通用渠道 webhook：归一 → 会话路由（user_id 映射）→ 复用 chat 管线。

        鉴权：外部系统接入面 RBAC（human_chat_ingest scope，与
        /v1/human-conversations/batch 同口径）；auth 关闭且未强制 RBAC 时
        开发直通。限流/会话租约与 /v1/chat 完全一致（同一 _run_chat_turn）。

        出站：**轮询**——回复进入渠道出站队列，适配器用
        GET /v1/channels/{channel}/outbound?cursor=N 拉取（选型理由见
        app/server/channels.py 模块 docstring）。webhook 响应只给受理元数据
        （message_id / outbound_seq / requires_human / 工单号），不回传回复正文。

        写确认协议（pending_write）由 Agent 内部处理，本入口不复制该逻辑。
        """
        principal = authorize_scopes(request, SCOPE_HUMAN_INGEST)
        components: PodComponents = request.app.state.components
        try:
            validate_channel_id(channel)
            inbound = normalize_inbound(channel, body)
        except (InvalidChannel, InvalidIdentifier) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        user_id = channel_user_id(inbound.channel, inbound.external_user_id)
        _validate_request_ids(user_id, inbound.session_id)

        # 输入预检与 /v1/chat 同口径（注入 → 安全话术 + 工单；PII → 脱敏继续）
        message, blocked = _input_prefilter(inbound.message)
        handoff_ticket = None
        pending_turn = None
        if blocked:
            logger.warning(
                "channel guardrail block channel=%s user=%s operator=%s",
                inbound.channel, user_id, principal.sub or principal.via,
            )
            handoff_ticket = await _create_guardrail_handoff(
                components, user_id, inbound.session_id, inbound.message
            )
            reply, intent, confidence, requires_human = (
                SAFE_FALLBACK_REPLY, "other", 0.0, True,
            )
        else:
            agent, result = await _run_chat_turn(
                components,
                user_id,
                inbound.session_id,
                message,
                credentials=_request_credentials(user_id),
            )
            reply = result.reply
            requires_human = result.requires_human
            intent = (
                result.intent.value
                if hasattr(result.intent, "value")
                else str(result.intent)
            )
            confidence = result.confidence
            pending_turn = getattr(agent, "pending_turn", None)
            if settings.guardrails_enabled:
                from app.security.guardrails import check_output

                if check_output(reply).blocked:
                    logger.warning(
                        "channel guardrail block output channel=%s user=%s",
                        inbound.channel, user_id,
                    )
                    reply = SAFE_FALLBACK_REPLY
                    requires_human = True
            handoff_ticket = await _complete_handoff(
                components, agent, result, reply, requires_human
            )

        handoff = _handoff_meta(handoff_ticket)
        payload = outbound_payload(
            inbound=inbound,
            user_id=user_id,
            reply=reply,
            intent=intent,
            confidence=confidence,
            requires_human=requires_human,
            handoff=handoff,
            pending_turn=pending_turn,
        )
        outbox = _channel_outbox(request)
        try:
            # 同步 Redis/内存写入（C1：不阻塞事件循环）
            seq = await asyncio.to_thread(outbox.append, inbound.channel, payload)
        except Exception as e:
            # 出站入队失败：回复会丢——明确 503 让渠道侧稍后重试（不静默丢消息）
            logger.error(
                "channel outbound append failed channel=%s user=%s",
                inbound.channel, user_id, exc_info=e,
            )
            raise HTTPException(
                status_code=503, detail="渠道出站队列暂不可用，请稍后重试"
            ) from e
        logger.info(
            "channel inbound channel=%s external=%s user=%s session=%s seq=%s",
            inbound.channel, inbound.external_user_id, user_id,
            inbound.session_id, seq,
        )
        return ChannelMessageResponse(
            channel=inbound.channel,
            message_id=inbound.message_id,
            user_id=user_id,
            session_id=inbound.session_id,
            status="replied",
            outbound_seq=seq,
            requires_human=requires_human,
            handoff=handoff,
        )

    @app.get(
        "/v1/channels/{channel}/outbound",
        response_model=ChannelOutboundResponse,
        tags=["channel"],
    )
    async def channel_outbound(
        channel: str, request: Request, cursor: int = 0, limit: int = 50
    ):
        """渠道出站轮询：返回 cursor 之后的出站消息（至少一次投递）。

        cursor = 上次响应的 next_cursor（0 = 从最早保留的消息开始）。
        消息保留上限 MAX_OUTBOUND_RETAINED（超出裁剪最旧），适配器应持续消费。
        """
        authorize_scopes(request, SCOPE_HUMAN_INGEST)
        try:
            validate_channel_id(channel)
        except InvalidChannel as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        outbox = _channel_outbox(request)
        result = await asyncio.to_thread(
            outbox.list_since,
            channel,
            max(int(cursor), 0),
            min(max(int(limit), 1), 200),
        )
        return {"channel": channel, **result}

    @app.post("/v1/handoffs", tags=["handoff"])
    async def create_handoff(request: Request, body: dict):
        """6.1：手动为 (user, session) 创建转人工工单（一般由 requires_human 自动触发）。

        安全修复 P1：运营面端点，要求 ops scope（auth 关闭且未强制时开发直通）。
        ops 端点：为指定用户建单——user_id 取请求体（坐席代客建单），
        认证主体只用于鉴权，不参与定位目标用户。
        """
        authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        user_id = str(body.get("user_id", ""))
        _validate_request_ids(user_id, str(body.get("session_id", "")))
        agent = build_agent(user_id, body.get("session_id", ""), components)
        try:
            await runtime.run_agent_close(agent)
        finally:
            pass
        from app.handoff.board import HandoffTicket, _now

        ticket = HandoffTicket(
            ticket_id=uuid.uuid4().hex,
            user_id=user_id,
            session_id=body.get("session_id", ""),
            intent="manual",
            summary=getattr(agent, "summary", "") or "",
            created_at=_now(),
        )
        await asyncio.to_thread(components.handoff_board.create, ticket)
        return {"ticket_id": ticket.ticket_id, "status": "pending"}

    @app.get("/v1/handoffs", tags=["handoff"])
    async def list_handoffs(
        request: Request, status: str = "pending", mine: bool = False
    ):
        """工单看板（运营面，ops scope）：支持 pending/resolved/all。

        P2-3 坐席工作台：
        - mine=true → 只返回当前认证主体（principal.sub）领取的工单（「我的」列表）；
        - 返回 to_ops_dict（附 SLA 截止时间/剩余秒数/是否超时），UI 超时标红；
        - 每次看板轮询顺带做 SLA 超时「首次观测」标记并累加
          handoff_sla_breach_total（去重由工单板负责，重复轮询不重复计数）。

        capabilities.human_qa_evolution：UI 据此显示或禁用「沉淀为知识」表单
        ——仅当功能开关开启且工单板为 Redis 持久实现（进程内队列不可恢复）。
        """
        principal = authorize_scopes(request, SCOPE_OPS)
        if status not in ("pending", "resolved", "all"):
            raise HTTPException(
                status_code=422,
                detail=f"未知工单状态: {status}（支持 pending/resolved/all）",
            )
        components: PodComponents = request.app.state.components
        board = components.handoff_board
        # SLA 超时首次观测计数（P2-3；board 缺失该方法时跳过）
        newly = await asyncio.to_thread(_mark_sla_breaches_sync, board)
        if newly:
            from app.observability.metrics import record_handoff_sla_breach

            record_handoff_sla_breach(newly)
        # board.list 内部是 smembers + 逐 id get（N+1 Redis 往返）→ 线程池执行
        tickets = await asyncio.to_thread(
            board.list, status, assignee=principal.sub if mine else ""
        )
        return {
            "tickets": [t.to_ops_dict() for t in tickets],
            "assignee": principal.sub if mine else "",
            "capabilities": {
                "human_qa_evolution": (
                    settings.human_qa_evolution_enabled
                    and bool(getattr(board, "durable", False))
                ),
            },
        }

    @app.post("/v1/handoffs/{ticket_id}/claim", tags=["handoff"])
    async def claim_handoff(ticket_id: str, request: Request):
        """P2-3：坐席领取工单（工单进「我的」列表）。

        并发安全：Redis 侧 Lua「读-比较-写」原子执行，两方同时领取只有一个
        成功；进程内实现用 RLock 保护同一临界区。他人已领取/已解决 → 409，
        同一坐席重复领取幂等（already=true，不重复追加留痕）。
        领取人只取认证主体 sub（不接受请求体伪造）。
        """
        principal = authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        try:
            ticket, already = await asyncio.to_thread(
                components.handoff_board.claim, ticket_id, principal.sub
            )
        except HandoffNotFound as e:
            raise HTTPException(status_code=404, detail="工单不存在") from e
        except HandoffConflict as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {
            "ticket_id": ticket_id,
            "status": "claimed",
            "assignee": ticket.assignee,
            "claimed_at": ticket.claimed_at,
            "already": already,
            "ticket": ticket.to_ops_dict(),
        }

    @app.post("/v1/handoffs/{ticket_id}/notes", tags=["handoff"])
    async def add_handoff_note(
        ticket_id: str, request: Request, body: HandoffNoteRequest
    ):
        """P2-3：追加坐席处理备注（append-only 留痕，不改变工单状态）。

        备注人只取认证主体 sub；备注进 events（与领取/解决同一事件流）。
        """
        principal = authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        try:
            ticket = await asyncio.to_thread(
                components.handoff_board.add_note,
                ticket_id,
                principal.sub,
                body.note,
            )
        except HandoffNotFound as e:
            raise HTTPException(status_code=404, detail="工单不存在") from e
        events = ticket.events
        return {
            "ticket_id": ticket_id,
            "event": events[-1] if events else None,
            "events": events,
            "ticket": ticket.to_ops_dict(),
        }

    @app.post("/v1/handoffs/{ticket_id}/resolve", tags=["handoff"])
    async def resolve_handoff(
        ticket_id: str,
        request: Request,
        body: HandoffResolveRequest,
    ):
        """6.1：坐席回写结论；reclaim=true 表示会话可继续（下一轮同一 session 直接续上）。

        人工客服问答沉淀（human_qa_evolution_enabled）：resolution 勾选
        knowledge_candidate 时规范问题/答案/依据必填，Redis Lua 原子完成
        工单转 resolved + 事件保存 + evolution pending 登记；幂等键 =
        SHA-256("human-handoff:v1:" + ticket_id + ":" + resolution_version)，
        相同内容（无论 version）重放返回同一事件，不同内容返回 409。
        resolved_by 只取 authorize_scopes 的认证主体 sub（不接受请求体伪造）；
        功能开启但工单板非持久（无 Redis）时勾选沉淀 503，普通工单照常解决。

        安全修复 P1：运营面端点，要求 ops scope（auth 关闭且未强制时开发直通）。
        """
        principal = authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        resolution = dict(body.resolution)
        if resolution.get("knowledge_candidate"):
            # 人工知识沉淀链路已切换为「外部会话批量接入 → MySQL 评审 →
            # 人工批量发布」；工单勾选沉淀已弃用——明确报错避免静默丢失。
            raise HTTPException(
                status_code=410,
                detail="knowledge_candidate 已弃用：人工知识改由 "
                "POST /v1/human-conversations/batch 批量推送会话沉淀",
            )
        ticket = await asyncio.to_thread(components.handoff_board.get, ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        try:
            event, duplicate = await asyncio.to_thread(
                components.handoff_board.resolve_atomic,
                ticket_id,
                resolution,
                principal.sub,
                resolution_version=body.resolution_version,
            )
        except HandoffConflict as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except HandoffNotFound as e:  # get 与 resolve 之间的竞态删除
            raise HTTPException(status_code=404, detail="工单不存在") from e
        # P2-3：回读工单（含 assignee/resolved_at/events 留痕与 SLA 判定）
        resolved = await asyncio.to_thread(components.handoff_board.get, ticket_id)
        return {
            "ticket_id": ticket_id,
            "status": "resolved",
            "reclaim_session": body.reclaim,
            "resolution_event": event,
            "duplicate": duplicate,
            "assignee": getattr(resolved, "assignee", "") if resolved else "",
            "ticket": resolved.to_ops_dict() if resolved is not None else None,
        }

    @app.get("/v1/messages/search", tags=["handoff"])
    async def message_search(
        request: Request,
        user_id: str = "",
        q: str = "",
        session_id: str = "",
        limit: int = 50,
    ):
        """阶段八：对话全文检索（坐席接手/质检）。

        修复计划·五：运营面端点，要求 ops scope（无 scope → 403）。
        - 操作者取 authorize_scopes 的认证主体（不用于定位目标用户）；
        - user_id 是**必填目标客户**参数（ops 可搜索任意明确指定的用户）；
          缺失 → 422；
        - 审计记录操作者/目标用户/session/查询摘要哈希/命中数/trace_id，
          不记录原始查询正文。
        修复计划·二：返回前按删除 tombstone 过滤旧 UUID（及 legacy 空 UUID），
        保证 Reset 后立即不可搜索（即使 ES 删除事件尚未执行）。
        修复计划·三：ES 不可用 → 503（不再返回易被误读为「确实没有结果」的空数组）。
        """
        principal = authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        if not user_id:
            raise HTTPException(status_code=422, detail="缺少目标用户参数 user_id")
        _validate_request_ids(user_id, session_id)

        # 修复计划·二轮 4：tombstone 先读且 fail-closed——DB 缺失/查询失败
        # 一律 503，绝不返回可能含旧会话数据的空集合假象。
        from app.stores.sql.outbox import (
            MessageTombstoneUnavailable,
            load_message_tombstones,
        )

        try:
            tombstones = await asyncio.to_thread(
                load_message_tombstones, components.db_engine, user_id
            )
        except MessageTombstoneUnavailable as e:
            logger.error("message_search tombstone 不可用: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "消息检索暂不可用（删除标记不可读）",
                    "code": "message_tombstone_unavailable",
                },
            )

        es_client = _es_of(components)
        if es_client is None or not components.message_index:
            raise HTTPException(status_code=503, detail="消息检索暂不可用（ES 未配置/不可达）")
        if not q.strip():
            return {"hits": [], "degraded": False}

        must = [{"match": {"content": q}}]
        filters = [{"term": {"user_id": user_id}}]
        if session_id:
            filters.append({"term": {"session_key": f"{user_id}/{session_id}"}})
        try:
            # 同步 ES 客户端调用（C1：不阻塞事件循环）
            resp = await asyncio.to_thread(
                es_client.search,
                index=components.message_index,
                query={"bool": {"must": must, "filter": filters}},
                sort=[{"ts": "desc"}],
                size=min(max(limit, 1), 200),
                source=["session_key", "session_uuid", "seq", "role", "content", "ts"],
            )
        except Exception as e:
            logger.warning("message_search ES 查询失败: %s", type(e).__name__)
            from app.agent.rag.es_util import invalidate_es_client

            invalidate_es_client(f"search_failed:{type(e).__name__}")
            raise HTTPException(status_code=503, detail="消息检索暂不可用（ES 查询失败）") from e

        # 修复计划·三轮 P1-1：ES 查询期间可能发生 Reset（删除事件同事务提交），
        # 查询后重读 tombstone 与快照取并集再过滤；二次读取失败仍 fail-closed
        # （503，绝不返回可能含旧会话数据的 ES 结果）。
        try:
            post_tombstones = await asyncio.to_thread(
                load_message_tombstones, components.db_engine, user_id
            )
        except MessageTombstoneUnavailable as e:
            logger.error("message_search tombstone 二次读取不可用: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "消息检索暂不可用（删除标记不可读）",
                    "code": "message_tombstone_unavailable",
                },
            )
        for key, uuids in post_tombstones.items():
            tombstones.setdefault(key, set()).update(uuids)

        # 修复计划·二轮 3/4：过滤 tombstone（旧 UUID 与 legacy 空 UUID），
        # 并按 (session_key, seq) 去重——兼容新旧文档 ID（legacy _id 不含 UUID）。
        hits: list[dict] = []
        seen: set[tuple] = set()
        for h in resp.get("hits", {}).get("hits", []):
            src = h.get("_source", {}) or {}
            session_key = src.get("session_key", "")
            stale = src.get("session_uuid") or ""
            blocked = tombstones.get(session_key)
            if blocked is not None and (stale in blocked or not stale):
                continue  # 旧会话实例（或 legacy 无 UUID）→ 已重置，不得返回
            dedup_key = (session_key, src.get("seq"))
            if dedup_key in seen:
                continue  # 同 (session_key, seq) 的新旧文档重复：保留首条（ts 倒序）
            seen.add(dedup_key)
            hits.append({
                "session_id": session_key.split("/", 1)[-1],
                "seq": src.get("seq"),
                "role": src.get("role", ""),
                "content": src.get("content", ""),
                "ts": src.get("ts"),
            })

        # 审计（不记录原始查询正文）：操作者/目标/session/查询摘要/命中数/trace
        import hashlib

        logger.info(
            "ops.message_search operator=%s target=%s session=%s query_sha=%s hits=%s trace_id=%s",
            principal.sub or principal.via,
            user_id,
            session_id or "-",
            hashlib.sha256(q.encode("utf-8")).hexdigest()[:16],
            len(hits),
            _current_trace_id(),
        )
        return {"hits": hits, "degraded": False}

    @app.post("/v1/sessions/reset", response_model=SessionResetResponse, tags=["chat"])
    async def reset_session(body: SessionResetRequest, request: Request):
        """评审二轮 B1：reset 是写操作，与 chat/stream 一样必须持有会话锁
        （历史实现不抢锁，可在流式/对话进行中并发删除会话文档）。
        """
        user_id = authenticate_user(request, body.user_id)
        _validate_request_ids(user_id, body.session_id)
        try:
            agent = build_agent(user_id, body.session_id, request.app.state.components)
        except SessionOwnershipError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        components = request.app.state.components
        # 修复计划·一：锁后端不可用（生产 redis_required）→ 503，不降级
        try:
            lease = SessionLease(
                components.locks,
                user_id,
                body.session_id or "session",
            ).__enter__()
        except SessionLockBackendUnavailable as e:
            raise HTTPException(
                status_code=503, detail=f"会话锁后端暂不可用: {e}"
            ) from e
        try:
            if lease.token is None:
                raise HTTPException(
                    status_code=409,
                    detail="该会话正在处理中，请稍候再试",
                )
            # 修复计划·一：reset 删除会话前校验租约（无锁不得删）
            if hasattr(agent, "bind_lease_guard"):
                agent.bind_lease_guard(lease.assert_owned)
            try:
                await runtime.run_agent_reset(agent)
            except SessionLockLost as e:
                raise HTTPException(status_code=503, detail=f"会话租约已失效: {e}") from e
            finally:
                await runtime.run_agent_close(agent)
        finally:
            lease.release()
        return SessionResetResponse(session_id=agent.session_id)

    # Web 聊天界面（静态单页，无需前端构建链；对话走 POST /v1/chat）。
    # 必须在所有 API 路由注册之后 mount：Mount("/") 全捕获，先注册会遮蔽 API。
    app.mount(
        "/",
        StaticFiles(directory=Path(__file__).parent / "static", html=True),
        name="web",
    )

    return app


app = create_app()

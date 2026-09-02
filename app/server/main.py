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
from app.security.identifiers import InvalidIdentifier, validate_identifier
from app.security.principal import SCOPE_OPS, authorize_scopes
from app.server import runtime
from app.server.deps import (
    PodComponents,
    authenticate_user,
    build_agent,
    build_pod_components,
)
from app.server.schema import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionResetRequest,
    SessionResetResponse,
)
from app.stores.base import (
    SessionConflictError,
    SessionOwnershipError,
    StorageUnavailableError,
)
from app.stores.locks import SessionLease

logger = logging.getLogger("app.server")

PROJECT = "并夕夕 · 智能客服「小夕」API"

# 3.5：命中 guardrail 的降级话术（转人工）
SAFE_FALLBACK_REPLY = "抱歉，我无法处理您这条请求，已为您转接人工客服，请稍候。"

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


def _sse(event_type: str, data: dict) -> str:
    """SSE 帧：`event: <type>\ndata: <json>\n\n`。"""
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def _safe_fallback(session_id: str) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        reply=SAFE_FALLBACK_REPLY,
        intent="other",
        confidence=0.0,
        requires_human=True,
        follow_up_question=None,
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
            return format(otel_trace.get_current_span().get_span_context().trace_id, "032x")[:16]
    except Exception:  # noqa: BLE001 —— 观测不可用不影响错误响应
        pass
    return uuid.uuid4().hex[:16]


def _schedule_disconnected_finalize(task, agent, components, user_id, limiter,
                                   request_id: str = "") -> None:
    """断连收尾（评审·坑3：GeneratorExit 清理雷区）。

    收尾（等 Agent 工作结束 → close → 用量记账）放请求 cancel scope 之外的
    独立 asyncio task——生成器 finally 在 GeneratorExit 上下文里不允许再
    await 可能让出事件循环的操作。等待有上限（_disconnect_wait_bound_seconds），
    超时记日志、不再 close（线程仍持有 agent，由其自行完成）。
    """
    bound = _disconnect_wait_bound_seconds()

    def _settle() -> None:
        """用量入账（request 粒度，评审二轮 B2；幂等——end 后重复调用为 0）。"""
        tokens = components.usage_tracker.end_request(request_id)
        limiter.consume_tokens(user_id, tokens)

    async def _finalize():
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=bound)
            except asyncio.TimeoutError:
                logger.warning(
                    "客户端断连后 Agent 工作未在 %.0fs 内结束，交由工作线程"
                    "自行完成（本轮 close 跳过，已发生用量照常入账）", bound,
                )
                _settle()
                return
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 —— 断连后的轮次结果只留日志
                logger.warning("断连后的 Agent 轮次以异常结束", exc_info=True)
        try:
            await runtime.run_agent_close(agent)
        except Exception:  # noqa: BLE001
            logger.warning("断连收尾 agent.close() 失败", exc_info=True)
        _settle()

    finalize_task = asyncio.create_task(_finalize())

    def _log_crash(done: asyncio.Task) -> None:
        if not done.cancelled() and done.exception() is not None:
            logger.warning("断连收尾任务异常", exc_info=done.exception())

    finalize_task.add_done_callback(_log_crash)


async def _idle_consolidator_loop(components: PodComponents) -> None:
    """后台兜底巩固：每 scan 间隔扫描一次静默会话（2.3），出错静默下轮再试。"""
    if components.redis is None:
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
        except Exception:  # noqa: BLE001 —— 后台任务自身崩溃不应拖垮 pod
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
            except Exception as e:  # noqa: BLE001 —— GC 失败不影响主流程
                _log.getLogger("app.server").warning("kb gc 失败: %s", e)
                return 0
            finally:
                if lock is not None:
                    try:
                        lock.release()
                    except Exception:  # noqa: BLE001
                        pass

        try:
            handled = await to_thread.run_sync(_gc_once)
            if handled:
                logger.info("kb gc 完成 %s 项", handled)
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.warning("kb gc 异常", exc_info=True)


async def _es_outbox_loop(components: PodComponents) -> None:
    """阶段八：MySQL→ES 消息同步（outbox）。ES 挂了只跳过重试，不影响主流程。"""
    if components.db_engine is None or components.es_client is None:
        return
    from app.stores.sql.outbox import ensure_message_index, sync_outbox_to_es

    index = components.message_index
    try:
        ensure_message_index(components.es_client, index)
    except Exception:  # noqa: BLE001 —— 索引建失败下轮再试
        logger.warning("message_search 索引初始化失败", exc_info=True)
    while True:
        try:
            await anyio.sleep(30)
            from app.stores.sql.outbox import count_pending_outbox

            # 5.x：outbox 积压指标（ES 同步滞后告警依据）
            from app.observability.metrics import set_outbox_backlog

            try:
                backlog = await to_thread.run_sync(
                    partial(count_pending_outbox, components.db_engine)
                )
                set_outbox_backlog(backlog)
            except Exception:  # noqa: BLE001 —— 积压统计失败不影响同步
                pass
            synced = await to_thread.run_sync(
                partial(sync_outbox_to_es, components.db_engine,
                        components.es_client, index)
            )
            if synced:
                logger.info("outbox→ES 同步 %s 条", synced)
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 —— 单轮同步异常不影响循环
            logger.warning("outbox→ES 同步异常", exc_info=True)


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
    app.state.consolidator_task = asyncio.create_task(
        _idle_consolidator_loop(app.state.components)
    )
    app.state.es_outbox_task = asyncio.create_task(
        _es_outbox_loop(app.state.components)
    )
    app.state.kb_gc_task = asyncio.create_task(
        _kb_gc_loop(app.state.components)
    )
    yield
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
    mcp = app.state.components.mcp_client
    if mcp is not None:
        try:
            mcp.close()
        except Exception:  # noqa: BLE001
            logger.warning("MCP client 关闭失败", exc_info=True)
    executor = app.state.components.tool_executor
    if executor is not None:
        try:
            executor.close()
        except Exception:  # noqa: BLE001
            logger.warning("工具执行器关闭失败", exc_info=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title=PROJECT,
        version="0.3.0",
        description="电商客服 Agent 多用户 API（0.3.0：安全修复基线；GET SSE 已弃用）",
        lifespan=_lifespan,
    )

    from app.server.uploads import router as kb_uploads_router

    app.include_router(kb_uploads_router)

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
            "unhandled error path=%s trace_id=%s", request.url.path, trace_id,
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
        checks: dict[str, str] = {}

        def _check(name: str, fn) -> None:
            try:
                checks[name] = "ok" if fn() else "degraded"
            except Exception as e:  # noqa: BLE001 —— 探针单项失败不整体崩
                checks[name] = f"error: {type(e).__name__}"

        # 2.7：生产要求 Redis 时，未就绪 → 不接流量（探针 fail）；
        # 4.3：实际 ping 探活——对象存在不代表连接可用（kind 验收暴露）
        if settings.redis_required and components.redis is not None:
            try:
                components.redis.ping()
            except Exception:  # noqa: BLE001 —— 连接不可用即未就绪
                raise HTTPException(status_code=503, detail="Redis 未就绪（ping 失败）")
        if settings.redis_required and components.redis is None:
            raise HTTPException(status_code=503, detail="Redis 未就绪（redis_required=true）")
        checks["redis"] = "ok" if components.redis is not None else "not_configured"

        # 4.1：MySQL schema 版本校验（只读，不自动 DDL）
        if components.db_engine is not None:
            def _db_ok():
                from app.stores.sql.engine import _verify_schema_version
                if settings.app_env.lower() == "prod":
                    _verify_schema_version(components.db_engine)
                    return True
                return True
            _check("mysql_schema", _db_ok)
        else:
            checks["mysql_schema"] = "not_configured"

        # 4.1：ES alias 指向（KB 后端为 es 时校验；否则视为不依赖）
        if getattr(components, "es_client", None) is not None:
            def _es_alias_ok():
                from app.agent.rag.es_util import get_es_client
                es = get_es_client()
                if es is None:
                    return False
                alias = f"{settings.es_index_prefix}-kb-active"
                try:
                    hits = es.indices.get_alias(name=alias)
                    return bool(hits)
                except Exception:  # noqa: BLE001 —— alias 不存在
                    return False
            _check("es_kb_alias", _es_alias_ok)
        else:
            checks["es_kb_alias"] = "not_configured"

        # 4.1：对象存储探活（配置了 S3 才校验）
        object_store = getattr(components, "object_store", None)
        if object_store is not None:
            _check("object_store", object_store.healthcheck)
        else:
            checks["object_store"] = "not_configured"

        degraded = [k for k, v in checks.items() if v == "degraded" or v.startswith("error")]
        ready = "ready" if not degraded else "degraded"
        # 5.x：依赖就绪指标（探针同源，供告警与看板）
        from app.observability.metrics import set_dependency_readiness

        for component, value in checks.items():
            set_dependency_readiness(component, value == "ok")
        # 5.x：alias/pointer 一致性指标（es 后端时）
        if getattr(components, "es_client", None) is not None:
            try:
                from app.agent.rag.es_util import get_es_client
                from app.evolution.generation import GenerationStore
                from app.evolution.index_service import IndexBuildService
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
                    except Exception:  # noqa: BLE001
                        alias_target = ""
                    set_alias_pointer_mismatch(
                        bool(alias_target and alias_target != pointer.target)
                    )
            except Exception:  # noqa: BLE001 —— 指标采集失败不影响探针
                pass
        return {"status": ready, "components": checks}

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(body: ChatRequest, request: Request):
        components: PodComponents = request.app.state.components
        # 3.1：auth_enabled 时 user_id 从 Bearer JWT 解出（请求体不再接受）
        user_id = authenticate_user(request, body.user_id)

        # 3.5 输入侧：注入 → 降级话术转人工；PII → 脱敏后继续
        message = body.message
        if settings.guardrails_enabled:
            from app.security.guardrails import check_input

            verdict = check_input(message)
            if verdict.blocked:
                logger.warning("guardrail block input user=%s", user_id)
                return _safe_fallback(body.session_id)
            message = verdict.text

        # 3.7：per-user RPS 与日预算（超限 429，客户端展示并升级人工）
        from app.observability.metrics import (
            RATE_LIMITED,
            record_chat_latency,
            record_conflict,
            record_handoff,
            REACT_STEPS,
        )

        limiter = components.limiter
        if not limiter.allow_rps(user_id):
            RATE_LIMITED.labels(kind="rps").inc()
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        if not limiter.allow_budget(user_id):
            RATE_LIMITED.labels(kind="budget").inc()
            raise HTTPException(status_code=429, detail="今日用量已达上限，请明日再试或联系人工客服")

        # 评审二轮 B2：用量按 request 归集（ContextVar 随 anyio 线程池传播，
        # turn 线程与 close 线程的 LLM 调用都记到本请求名下）
        request_id = uuid.uuid4().hex
        components.usage_tracker.begin_request(request_id)

        # 2.2：同 session 只有一个写入者（连点/双 pod 竞态）→ 冲突 409
        with SessionLease(
            components.locks, user_id, body.session_id or "session"
        ) as lease:
            if lease.token is None:
                record_conflict()
                raise HTTPException(
                    status_code=409,
                    detail="该会话正在处理中，请稍候再试",
                )
            try:
                agent = build_agent(
                    user_id,
                    body.session_id,
                    components,
                    credentials=_request_credentials(user_id),
                )
            except SessionOwnershipError as e:
                # 3.1：session 归属不属于当前用户 → 403
                raise HTTPException(status_code=403, detail=str(e)) from e
            try:
                started = time.monotonic()
                result = await runtime.run_agent_turn(agent, message)
                record_chat_latency(time.monotonic() - started)
                REACT_STEPS.observe(getattr(agent, "_react_steps_count", 1) or 1)
            except SessionConflictError as e:
                # CAS 冲突（本轮期间被其他写入者改过）→ 客户端应重读重试
                record_conflict()
                raise HTTPException(status_code=409, detail=f"会话版本冲突: {e}") from e
            except StorageUnavailableError as e:
                # 5.2 故障注入：外置存储断连 → 503（明确不静默降级写文件，避免状态分裂）
                raise HTTPException(status_code=503, detail=f"存储暂不可用: {e}") from e
            finally:
                # close 的 LLM 巩固调用同样记到本请求（在 end_request 之前）
                await runtime.run_agent_close(agent)

        # 3.7：实际用量入账（成本护栏；评审二轮 B2：request 粒度）
        tokens = components.usage_tracker.end_request(request_id)
        limiter.consume_tokens(user_id, tokens)

        reply = result.reply
        requires_human = result.requires_human
        # 3.5 输出侧：敏感词 → 降级安全话术并转人工
        if settings.guardrails_enabled:
            from app.security.guardrails import check_output

            out_verdict = check_output(reply)
            if out_verdict.blocked:
                logger.warning("guardrail block output user=%s", user_id)
                reply = SAFE_FALLBACK_REPLY
                requires_human = True
        if requires_human:
            record_handoff()
            # 6.1：转人工 → 生成 handoff 包推坐席（ticket 可查询/回写/续会话）
            try:
                from app.handoff.board import build_handoff_ticket

                components.handoff_board.create(build_handoff_ticket(agent, result))
            except Exception:  # noqa: BLE001 —— 工单失败不阻断已完成的对话
                logger.warning("handoff ticket 创建失败", exc_info=True)

        return ChatResponse(
            session_id=agent.session_id,
            reply=reply,
            intent=result.intent.value if hasattr(result.intent, "value") else str(result.intent),
            confidence=result.confidence,
            requires_human=requires_human,
            follow_up_question=result.follow_up_question,
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
        return _chat_stream_response(
            request, user_id=user_id, session_id=session_id,
            message=message, deprecate=True,
        )

    @app.post("/v1/chat/stream", response_class=StreamingResponse, tags=["chat"])
    async def chat_stream_post(request: Request, body: ChatRequest):
        """4.5 SSE 流式（继任端点，0.3.0 新增）：事件序列同 GET 版，参数走
        JSON body（与 POST /v1/chat 的请求模型一致）。

        事件序列：meta → route?（多 Agent）→ thought* → tool_call* /
        tool_result* ×（Agent能力强化计划·改造二） → reply → end。

        改造二（additive 字段，老客户端忽略）：
        - tool_call：新增 `tool_call_id` + `sequence`（模型给定顺序的序号）；
        - tool_result：新增 `tool_call_id` + `tool_name` + `sequence`。
        其余事件字段（thought{text} / reply{reply,intent,confidence,
        requires_human,follow_up_question} / meta{session_id} / end{ok} /
        error{detail,trace_id} / deprecation）不变。
        """
        return _chat_stream_response(
            request, user_id=body.user_id, session_id=body.session_id,
            message=body.message, deprecate=False,
        )

    def _chat_stream_response(
        request: Request, *, user_id: str, session_id: str,
        message: str, deprecate: bool,
    ) -> StreamingResponse:
        """SSE 流式共用实现（安全修复 P2 的核心重构）。

        历史缺陷：SessionLease 在路由 `return StreamingResponse(...)` 时即
        退出——生成器实际消费在锁释放之后，流式期间会话无保护。现在：
        - 租约在生成器体内获取/释放（finally），覆盖整个流式生命周期；
        - Agent 任务在生成器内启动（客户端不消费就不开跑，天然缓解慢连接
          占池）；断连（GeneratorExit/CancelledError）走独立收尾任务：
          有界等待 + shield，生成器 finally 内不再 await（评审·坑1/坑3）。
        """
        components: PodComponents = request.app.state.components
        user_id = authenticate_user(request, user_id)
        _validate_request_ids(user_id, session_id)
        headers = dict(SSE_DEPRECATION_HEADERS) if deprecate else None

        if settings.guardrails_enabled:
            from app.security.guardrails import check_input

            verdict = check_input(message)
            if verdict.blocked:
                async def _blocked():
                    if deprecate:
                        yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                    yield _sse("reply", {"reply": SAFE_FALLBACK_REPLY,
                                         "requires_human": True})
                    yield _sse("end", {"ok": True})
                return StreamingResponse(
                    _blocked(), media_type="text/event-stream", headers=headers,
                )
            message = verdict.text

        limiter = components.limiter
        if not limiter.allow_rps(user_id):
            async def _limited():
                if deprecate:
                    yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                yield _sse("error", {"detail": "请求过于频繁，请稍后再试"})
                yield _sse("end", {"ok": False})
            return StreamingResponse(
                _limited(), media_type="text/event-stream", headers=headers,
            )

        # 归属校验在生成器外完成：403 语义保留（不降级为流内 error 事件）；
        # 它只读 session 文档，不需要会话锁
        try:
            agent = build_agent(
                user_id, session_id, components,
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
            lease = SessionLease(
                components.locks, user_id, session_id or "session",
            )
            lease.__enter__()
            task = None
            interrupted = False
            try:
                if lease.token is None:
                    if deprecate:
                        yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                    yield _sse("error", {"detail": "该会话正在处理中，请稍候再试"})
                    yield _sse("end", {"ok": False})
                    return
                if deprecate:
                    yield _sse("deprecation", SSE_DEPRECATION_EVENT)
                yield _sse("meta", {"session_id": agent.session_id})

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
                yield _sse("reply", {
                    "reply": reply,
                    "intent": result.intent.value
                    if hasattr(result.intent, "value") else str(result.intent),
                    "confidence": result.confidence,
                    "requires_human": requires_human,
                    "follow_up_question": result.follow_up_question,
                })
                yield _sse("end", {"ok": True})
            except (GeneratorExit, asyncio.CancelledError):
                # 客户端断连：Agent 线程不可中断——有界等待其自然结束后
                # 在请求 scope 之外收尾（本 finally 内不做任何 await）
                interrupted = True
                _schedule_disconnected_finalize(
                    task, agent, components, user_id, limiter, request_id,
                )
                raise
            except Exception as e:  # noqa: BLE001 —— 流式期间异常推给客户端
                # 安全修复 P2：不回显异常细节（与 500 脱敏同一原则）
                trace_id = _current_trace_id()
                logger.error("stream turn failed session=%s/%s trace_id=%s",
                             user_id, session_id, trace_id, exc_info=e)
                yield _sse("error", {
                    "detail": "本轮对话处理失败，请稍后重试",
                    "trace_id": trace_id,
                })
                yield _sse("end", {"ok": False})
            finally:
                if not interrupted:
                    # 正常/业务错误收尾：不在 GeneratorExit 上下文，可安全
                    # await——先 close（巩固调用的用量记到本请求）再入账
                    await runtime.run_agent_close(agent)
                    tokens = components.usage_tracker.end_request(request_id)
                    limiter.consume_tokens(user_id, tokens)
                lease.__exit__(None, None, None)

        return StreamingResponse(
            _stream(), media_type="text/event-stream", headers=headers,
        )

    @app.post("/v1/handoffs", tags=["handoff"])
    async def create_handoff(request: Request, body: dict):
        """6.1：手动为 (user, session) 创建转人工工单（一般由 requires_human 自动触发）。

        安全修复 P1：运营面端点，要求 ops scope（auth 关闭且未强制时开发直通）。
        """
        authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        user_id = authenticate_user(request, str(body.get("user_id", "")))
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
        components.handoff_board.create(ticket)
        return {"ticket_id": ticket.ticket_id, "status": "pending"}

    @app.get("/v1/handoffs", tags=["handoff"])
    async def list_handoffs(request: Request, status: str = "pending"):
        """安全修复 P1：运营面端点，要求 ops scope。"""
        authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        tickets = components.handoff_board.list(status)
        return {"tickets": [t.to_dict() for t in tickets]}

    @app.post("/v1/handoffs/{ticket_id}/resolve", tags=["handoff"])
    async def resolve_handoff(ticket_id: str, request: Request, body: dict):
        """6.1：坐席回写结论；reclaim=true 表示会话可继续（下一轮同一 session 直接续上）。

        安全修复 P1：运营面端点，要求 ops scope。
        """
        authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        ticket = components.handoff_board.get(ticket_id)
        authenticate_user(request, str(body.get("user_id", "")))
        if ticket is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        resolved = components.handoff_board.resolve(ticket_id, body.get("resolution", {}))
        return {
            "ticket_id": ticket_id,
            "status": resolved.status,
            "reclaim_session": bool(body.get("reclaim", True)),
        }

    @app.get("/v1/messages/search", tags=["handoff"])
    async def message_search(request: Request, user_id: str = "", q: str = "",
                             session_id: str = "", limit: int = 50):
        """阶段八：对话全文检索（坐席接手/质检）。

        安全修复 P1：运营面端点，要求 ops scope。
        归属围栏：user_id 一律取自认证（auth 关闭时回退参数），只搜本用户；
        ES 不可达 → 200 + degraded=true（检索侧降级，不影响 Agent 主流程）。
        """
        authorize_scopes(request, SCOPE_OPS)
        components: PodComponents = request.app.state.components
        user_id = authenticate_user(request, user_id)
        _validate_request_ids(user_id, session_id)
        if components.es_client is None or not components.message_index:
            return {"hits": [], "degraded": True, "reason": "ES 未配置/不可达"}
        if not q.strip():
            return {"hits": [], "degraded": False}

        must = [{"match": {"content": q}}]
        filters = [{"term": {"user_id": user_id}}]
        if session_id:
            filters.append({"term": {"session_key": f"{user_id}/{session_id}"}})
        try:
            resp = components.es_client.search(
                index=components.message_index,
                query={"bool": {"must": must, "filter": filters}},
                sort=[{"ts": "desc"}],
                size=min(max(limit, 1), 200),
                source=["session_key", "seq", "role", "content", "ts"],
            )
        except Exception:  # noqa: BLE001 —— 检索失败降级为空结果
            return {"hits": [], "degraded": True, "reason": "ES 查询失败"}
        return {
            "hits": [
                {
                    "session_id": h["_source"].get("session_key", "").split("/", 1)[-1],
                    "seq": h["_source"].get("seq"),
                    "role": h["_source"].get("role", ""),
                    "content": h["_source"].get("content", ""),
                    "ts": h["_source"].get("ts"),
                }
                for h in resp.get("hits", {}).get("hits", [])
            ],
            "degraded": False,
        }

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
        with SessionLease(
            components.locks, user_id, body.session_id or "session",
        ) as lease:
            if lease.token is None:
                raise HTTPException(
                    status_code=409,
                    detail="该会话正在处理中，请稍候再试",
                )
            try:
                await runtime.run_agent_reset(agent)
            finally:
                await runtime.run_agent_close(agent)
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

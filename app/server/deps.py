"""Agent 工厂与 pod 级组件（阶段一 1.2）。

- build_pod_components：启动时建一次，挂到 app.state（OpenAI client /
  SkillManager / MCP client 共享连接）；
- build_agent：每请求构造 Agent，pod 级组件注入而非全局单例。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import HTTPException, Request
from openai import OpenAI

from app.config.settings import settings


def authenticate_user(request: Request, body_user_id: str) -> str:
    """阶段三 3.1：auth_enabled 时 user_id 一律从 Bearer JWT 解出。

    auth_enabled=False（开发/内部试用）：回退请求体 user_id。
    """
    if not settings.auth_enabled:
        return body_user_id
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Bearer token")
    from app.security.jwt import TokenValidationError, decode_token

    try:
        return decode_token(auth[len("Bearer "):])
    except TokenValidationError as e:
        raise HTTPException(status_code=401, detail=f"认证失败: {e}") from e


@dataclass
class PodComponents:
    """pod 级共享资源：启动时构建一次，进程生命周期内复用。"""

    client: OpenAI
    skill_manager: object  # SkillManager
    mcp_client: object | None = None  # MCPClient（未启用时为 None）
    redis: object | None = None  # Redis 客户端；不可用时 None（降级文件实现）
    session_store: object | None = None  # SessionStore
    ltm_store: object | None = None  # LTMStore
    locks: object | None = None  # SessionLockManager
    turns_archive: object | None = None  # ObjectStore（turns 归档）
    limiter: object | None = None  # 阶段三 3.7 UserLimiter
    usage_tracker: object | None = None  # 阶段三 3.7 UsageTracker
    handoff_board: object | None = None  # 阶段六 6.1 转人工板
    db_engine: object | None = None  # 阶段八：SQL 正本引擎（db_url 配置时启用）
    es_provider: object | None = None  # 修复计划·二轮 5：可恢复 ES provider（Callable[[], ES|None]）
    message_index: str = ""  # 阶段八：对话全文检索索引名（{prefix}-messages）
    tool_executor: object | None = None  # Agent能力强化计划：pod 级工具批次执行器（无状态单例）
    upload_service: object | None = None  # KB 文档上传编排（kb_upload_enabled 且 DB 可用时）
    object_store: object | None = None  # 4.1：S3/OSS 共享客户端（turns/分片/原件；未配置 None）
    memory_job_worker: object | None = None  # 阶段F：异步记忆 worker（首次用时惰性装配）
    kb_job_store: object | None = None  # KB 异步建库任务队列（多实例异步改造；DB 可用时）
    human_knowledge_store: object | None = None  # 人工会话知识链路正本（迁移008；DB 可用时）

    def es(self):
        """当前 ES 客户端（修复计划·二轮 5：单一可恢复 provider）。

        provider 每次调用都反映当前可用性（冷却后自动重建），不缓存启动时结果。
        """
        if self.es_provider is not None:
            return self.es_provider()
        from app.agent.rag.es_util import get_es_client

        return get_es_client()


def build_openai_client() -> OpenAI:
    """pod 级共享 LLM 客户端（安全修复 P2：显式 timeout 接线）。

    历史缺陷：不传 timeout，SDK 默认 600s 生效，llm_timeout_seconds=60
    是死配置。max_retries=0：重试由韧性包装（ResilientLLM）按幂等策略
    负责，SDK 内建重试会与其叠加放大墙钟时间。
    """
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def build_pod_components() -> PodComponents:
    """构建 pod 级组件；MCP 启用时建立共享连接（失败即降级 None，走本地工具）。"""
    from app.stores.locks import SessionLockManager
    from app.stores.memory_store import LocalFileLTMStore
    from app.stores.redis_client import get_redis
    from app.stores.session_store import LocalFileSessionStore, RedisSessionStore

    client = build_openai_client()

    from app.agent.skills import SkillManager
    skill_manager = SkillManager(
        skills_dir=settings.skills_dir,
        enabled=settings.skills_enabled,
    )

    mcp_client = None
    if settings.mcp_enabled and settings.mcp_server_url:
        from app.mcp_client import MCPClient
        from app.mcp_client.actor import validate_mcp_security

        # 修复计划：生产启动校验（缺服务 token / actor 密钥 → 启动失败）
        validate_mcp_security()
        try:
            # Pod 共享 MCPClient：必须携带服务级 Bearer（修复计划：先前遗漏）
            mcp_client = MCPClient(
                settings.mcp_server_url, auth_token=settings.mcp_auth_token,
            )
            mcp_client.connect()
        except Exception:
            mcp_client = None

    # 阶段二 2.1/2.3：Redis 可用 → 外置；否则本地文件（处处降级）
    redis = get_redis()

    # 阶段二 2.1/2.3 + 阶段八 + 安全修复 P2（session_store_backend 生效）：
    # db_url 配置 → SQL 正本（消息行式追加 + CAS；Redis 保留并发角色：
    # 锁/限流照旧，另做会话热缓存 write-through）；
    # 否则按 session_store_backend 选择：auto=历史行为（Redis 可用→Redis，
    # 否则文件）；file=显式文件（Redis 仅锁/限流）；redis=强制 Redis
    #（不可用且未 redis_required → 降级文件并告警）。
    #
    # 记忆系统重构：LTM 不再以 Redis 为唯一主存储——SQL 正本 + Redis
    # cache-aside 读缓存（与会话热缓存同模式）；无 SQL 时 LTM 落文件，
    # Redis 只做会话/锁/限流（eviction 静默丢记忆 + 跨存储无事务两个坑）。
    from app.stores.sql.engine import get_engine

    engine = get_engine()
    backend = settings.session_store_backend
    if engine is not None:
        from app.stores.memory_store import CachedLTMStore
        from app.stores.sql.memory_store import SqlLTMStore
        from app.stores.sql.session_store import SqlSessionStore

        session_store = SqlSessionStore(engine, redis=redis)
        # redis=None → 裸 SqlLTMStore（无缓存层，语义与 CachedLTMStore 无缓存等价）
        ltm_store = (
            CachedLTMStore(SqlLTMStore(engine), redis)
            if redis is not None else SqlLTMStore(engine)
        )
    elif backend == "file":
        if redis is not None:
            logging.getLogger("app.server.deps").info(
                "SESSION_STORE_BACKEND=file：会话/记忆落本地文件，"
                "Redis 仅保留锁/限流角色"
            )
        session_store = LocalFileSessionStore(settings.session_dir)
        ltm_store = LocalFileLTMStore(settings.memory_dir)
    elif backend == "redis" and redis is None:
        logging.getLogger("app.server.deps").warning(
            "SESSION_STORE_BACKEND=redis 但 Redis 不可用：降级本地文件"
        )
        session_store = LocalFileSessionStore(settings.session_dir)
        ltm_store = LocalFileLTMStore(settings.memory_dir)
    elif redis is not None:
        logging.getLogger("app.server.deps").warning(
            "SESSION_STORE_BACKEND=redis：会话走 Redis，但记忆不再以 Redis "
            "为唯一主存储（eviction 静默丢记忆），LTM 落本地文件——"
            "生产请配置 DB_URL 走 SQL 正本"
        )
        session_store = RedisSessionStore(redis)
        ltm_store = LocalFileLTMStore(settings.memory_dir)
    else:
        session_store = LocalFileSessionStore(settings.session_dir)
        ltm_store = LocalFileLTMStore(settings.memory_dir)
    # 修复计划·一：生产 redis_required=true 时锁后端故障 fail-closed（503），
    # 不再降级进程内锁（多 Pod 各自本地锁 = 同一会话可被并发写）
    locks = SessionLockManager(
        redis,
        ttl_seconds=settings.session_lock_ttl_seconds,
        redis_required=settings.redis_required,
    )

    # 2.4/2.8：S3/OSS 兼容对象存储共享客户端——turns 归档、上传分片、上传原件
    # 复用同一 S3 client，通过不同 prefix 隔离（turns/、uploads/、originals/）。
    # s3 未配置/不可用 → None，各组件降级本地。
    from app.stores.object_store import ObjectStoreUnavailable

    object_store = None
    if settings.s3_bucket and settings.s3_endpoint_url:
        from app.stores.object_store import S3ObjectStore

        try:
            object_store = S3ObjectStore(
                settings.s3_bucket,
                endpoint_url=settings.s3_endpoint_url,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
            )
        except ObjectStoreUnavailable as e:
            logging.getLogger("app.server.deps").warning(
                "S3 对象存储初始化失败（%s），turns/分片/原件降级本地", e,
            )

    # 2.4：turns 归档对象存储（复用共享 client；未配置/不可用时 → None）
    from app.evolution.turn_sync import build_turns_archive

    try:
        turns_archive = build_turns_archive()
    except ObjectStoreUnavailable as e:
        turns_archive = None
        logging.getLogger("app.server.deps").warning(
            "turns 归档 S3 不可用（%s），recorder 降级本地", e,
        )

    # 阶段三 3.7：用户级限流/配额 + LLM 用量归集
    from app.security.ratelimit import UsageTracker, UserLimiter, install_usage_tracking

    limiter = UserLimiter(redis, redis_required=settings.redis_required)
    usage_tracker = UsageTracker()
    install_usage_tracking(client, usage_tracker)
    # 阶段四 4.4：韧性包装（重试/超时/pod 级信号量/降级链/廉价任务路由），
    # 装在 usage 外层 → 重试前的用量已归集，指标同时反映真实记账。
    # 修复计划·四：注入 limiter → 每次真实尝试原子预留/按 usage 结算。
    from app.llm.client import install_resilience

    install_resilience(client, settings.model_name, limiter=limiter)
    from app.handoff.board import get_board

    handoff_board = get_board(redis)

    # 阶段八：ES（对话全文检索 / KB 检索后端）——修复计划·二轮 5：
    # 不保存启动时固定客户端，注入可恢复 provider（readiness/搜索/Outbox/RAG 共用）
    from app.agent.rag.es_util import get_es_client

    es_provider = get_es_client

    # Agent能力强化计划·改造二：pod 级工具批次执行器（无状态，仅持全局
    # 并发信号量）；CLI/评估自建并负责 shutdown
    from app.agent.tools.batch_executor import ToolBatchExecutor

    tool_executor = ToolBatchExecutor()

    # KB 文档上传编排（v7：依赖 SQL 正本；Redis 用于断点状态与 generation 指针；
    # 2.8：共享 S3 client 注入分片/原件存储）
    kb_job_store = None
    human_knowledge_store = None
    if engine is not None:
        from app.agent.rag.job_store import KbIndexJobStore

        kb_job_store = KbIndexJobStore(engine)
        from app.evolution.human_store import HumanKnowledgeStore

        human_knowledge_store = HumanKnowledgeStore(engine)
    upload_service = _build_upload_service(engine, redis, object_store,
                                           job_store=kb_job_store)

    return PodComponents(
        client=client,
        skill_manager=skill_manager,
        mcp_client=mcp_client,
        redis=redis,
        session_store=session_store,
        ltm_store=ltm_store,
        locks=locks,
        turns_archive=turns_archive,
        limiter=limiter,
        usage_tracker=usage_tracker,
        handoff_board=handoff_board,
        db_engine=engine,
        es_provider=es_provider,
        message_index=f"{settings.es_index_prefix}-messages",
        tool_executor=tool_executor,
        upload_service=upload_service,
        object_store=object_store,
        kb_job_store=kb_job_store,
        human_knowledge_store=human_knowledge_store,
    )


def _build_upload_service(engine, redis, object_store=None, job_store=None):
    """构造 KB 上传编排；无 DB/未启用/初始化失败 → None（端点 503 fail-closed）。

    注意（冻结约束）：生产路径（引擎/Redis 任一可用时）GenerationStore 走
    strict_shared（指针永不降级文件）；上传状态在 Redis 不可用时退化为进程内
    ——仅开发单机，多 Pod 必须 Redis（部署要求见 docs）。

    2.8：object_store 注入时（KB_UPLOAD_STORAGE=s3），分片与原件都走对象
    存储（uploads/ 与 originals/ 前缀隔离）；None 时维持本地目录。
    job_store：KbIndexJobStore（多实例异步改造；None = 仅同步语义）。
    """
    import logging

    log = logging.getLogger("app.server.deps")

    if not settings.kb_upload_enabled:
        return None
    if engine is None:
        log.info("KB 上传未启用：DB 未配置（kb_documents 在 SQL 正本上）")
        return None
    try:
        from pathlib import Path

        from app.agent.rag.embedder import create_embedder
        from app.agent.rag.parsers import chunk_kb_dir
        from app.agent.rag.upload_service import DocumentUploadService
        from app.evolution.generation import GenerationStore
        from app.evolution.index_service import IndexBuildService
        from app.stores.sql.document_store import KbControlStore, SqlDocumentStore
        from app.stores.upload_storage import (
            OriginalStore,
            S3ChunkStorage,
            build_chunk_storage,
        )

        # deps.py 位于 app/server/ → 项目根 = 上溯 3 级（此前 2 级导致
        # kb_root 解析成 app/app/agent/rag/knowledge，容器内只读且路径错误）
        root = Path(__file__).resolve().parent.parent.parent
        gen_store = GenerationStore(
            root / settings.kb_generation_path, redis_client=redis,
            strict_shared=redis is not None,
        )
        index_service = IndexBuildService(
            embedder=create_embedder(),
            kb_dir=root / settings.kb_dir,
            generation_store=gen_store,
            chunker=chunk_kb_dir,
            strict_build=True,
        )
        # 修复计划·三：KB_UPLOAD_STORAGE=s3 时对象存储不可用 → 上传端点 503，
        # 绝不退回 Pod 本地文件（多 Pod 数据分裂）。未配置 s3 才用本地开发存储。
        if settings.kb_upload_storage == "s3":
            if object_store is None:
                log.warning(
                    "KB_UPLOAD_STORAGE=s3 但对象存储不可用：上传服务不装配"
                    "（端点返回 503，不退回本地文件）"
                )
                return None
            chunk_storage = S3ChunkStorage(object_store, prefix="uploads")
            originals = OriginalStore(object_store=object_store)
        else:
            chunk_storage = build_chunk_storage()
            originals = OriginalStore()
        return DocumentUploadService(
            doc_store=SqlDocumentStore(engine),
            control_store=KbControlStore(engine),
            generation_store=gen_store,
            index_service=index_service,
            engine=engine,
            redis=redis,
            kb_root=root / settings.kb_dir,
            chunk_storage=chunk_storage,
            originals=originals,
            job_store=job_store,
        )
    except Exception as e:
        log.warning("KB 上传服务初始化失败（端点将返回 503）: %s", e)
        return None


def build_agent(
    user_id: str,
    session_id: str = "",
    components: PodComponents | None = None,
    *,
    memory_enabled: bool | None = None,
    use_mcp: bool | None = None,
    temperature: float | None = None,
    credentials: dict | None = None,
    enforce_order_ownership: bool | None = None,
):
    """按 user_id/session_id 构造 EcomAgent（单 Agent）。

    credentials 为请求级外部凭证（3.2），随 ToolContext 注入，永不进 prompt/日志；
    enforce_order_ownership 由配置注入（不传则 Agent 内跟随全局 settings）。
    """
    components = components or build_pod_components()
    from app.agent.chat import EcomAgent

    return EcomAgent(
        user_id=user_id,
        session_id=session_id or None,
        client=components.client,
        skill_manager=components.skill_manager,
        mcp_client=components.mcp_client,
        memory_enabled=memory_enabled,
        use_mcp=use_mcp,
        temperature=temperature,
        session_store=components.session_store,
        ltm_store=components.ltm_store,
        turns_archive=components.turns_archive,
        credentials=credentials,
        tool_executor=components.tool_executor,
        enforce_order_ownership=(
            enforce_order_ownership if enforce_order_ownership is not None
            else settings.enforce_order_ownership
        ),
    )

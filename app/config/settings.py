import os


def _resolve_env_file() -> str | None:
    """工作目录 .env 的显式开关：ECOM_ENV_FILE=none/空 → 不读任何 .env。

    测试基线（单 Agent 全量优化计划）：测试配置不得读取工作目录 .env，
    tests/conftest.py 预设 ECOM_ENV_FILE=none；生产/CLI 不设置该变量，
    保持读 .env 的历史行为。
    """
    value = os.environ.get("ECOM_ENV_FILE", ".env")
    if value.strip().lower() in ("", "0", "false", "none"):
        return None
    return value


from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置，从 .env 文件读取"""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7

    # ReAct 循环（Agent能力强化计划·改造一）
    max_react_steps: int = 8  # 5→8：配合 turn_budget_seconds 熔断，步数不再是墙钟上界
    turn_budget_seconds: float = 120  # 单轮墙钟预算（安全熔断值；与 15s P95 SLO 无关）

    # 上下文 token 水位（单 Agent 全量优化计划·阶段B）
    # 份额：system+skill 20% / memory 10% / dialog+tool 50% / 输出预留 20%
    context_window_tokens: int = 32768
    fact_guard_enabled: bool = True  # 声明级事实接地（阶段D）；无证据数字删除并转核实
    memory_job_worker_enabled: bool = True  # 服务端异步记忆 worker（SQL 模式）
    memory_job_max_attempts: int = 5  # memory job 重试上限（超过置 failed 死信）
    memory_job_lease_seconds: int = 300  # memory job 租约时长（接管崩溃任务）
    memory_fact_ttl_days: int = 365  # 记忆注入有效期（阶段F；超期事实不注入）
    memory_relevance_threshold: float = 0.02  # 记忆注入相关性下限（dice；阶段F）

    # MCP 配置
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9123/mcp"
    mcp_auth_token: str = ""  # 3.4 mTLS/OAuth 之前的过渡：HTTP 认证头（客户端与服务端两侧同一 token；配置后 MCP server 拒绝匿名请求）
    mcp_actor_user_id: str = "mcp-actor"  # 兼容旧字段：无 ToolContext 时兜底的操作者身份（新路径一律走 actor token）
    mcp_actor_secret: str = (
        ""  # agent 侧签发/服务端校验的短期 actor token 密钥（HS256）
    )
    mcp_actor_ttl_seconds: int = 60  # actor token 有效期（短 TTL：单轮调用面）

    # RAG 配置（第5期）
    embedding_model: str = "text-embedding-3-small"
    # Embedding 提供方：openai（默认，走 OPENAI_BASE_URL 的标准 /embeddings）；
    # sophnet = SophNet EasyLLM（如 bge-m3，请求体 input_texts/easyllm_id，非 OpenAI 兼容）
    embedding_provider: str = "openai"
    sophnet_embedding_url: str = (
        ""  # 完整 URL（以 /embeddings 结尾，{projectId} 替换为实际值）
    )
    sophnet_api_key: str = ""  # Bearer token；留空回退 openai_api_key
    sophnet_easyllm_id: str = ""  # EasyLLM 实例 ID
    sophnet_embedding_dimensions: int = 1024  # bge-m3 输出维度
    kb_dir: str = "app/agent/rag/knowledge"
    # 向量后端：numpy（手写余弦，教学透明，零依赖，默认）/ chroma（向量数据库，生产代表，需 pip install chromadb）
    rag_backend: str = "numpy"
    # NumpyBackend 的 JSON 索引路径
    kb_index_path: str = "app/sessions/kb_index.json"
    # ChromaBackend 的持久化目录与 collection 名
    chroma_persist_dir: str = "app/sessions/chroma"
    chroma_collection: str = "ecom_kb"

    # RAG 生产化（阶段七）
    rag_hybrid: bool = False  # 混合检索：向量 + BM25 + RRF（7.2）
    rag_hybrid_recall_k: int = 30  # 混合检索的每路召回量（rerank 前候选数）
    rag_rerank: str = (
        "none"  # 精排器：none 不启用；cohere/jina/bge-reranker-v2-m3（7.3）
    )
    rerank_endpoint_url: str = ""  # 7.3 自部署端点（bge-reranker-v2-m3，TEI 协议）
    rerank_api_key: str = ""  # 7.3 托管 API 密钥（cohere/jina）
    rerank_model: str = ""  # 7.3 模型名（默认按 provider 取官方默认）
    # 最终检索结果的相关度下限；None 表示尚未校准、保持兼容行为。
    # 分数尺度依赖 embedding/backend/hybrid/reranker 组合，切换配置后必须重校准。
    rag_min_relevance_score: float | None = None
    retrieval_eval_dataset_path: str = (
        "app/evaluation/retrieval_cases.json"  # 检索质量评估集（7.6）
    )

    # Multi-Agent 配置（第6期）
    multi_agent_enabled: bool = False

    # Memory 配置（第7期）
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50

    # Skill 配置（第8期）
    skills_enabled: bool = True
    skills_dir: str = "app/agent/skills/definitions"

    # Evaluation 配置（第9期，离线评估工具，无聊天开关）
    eval_dataset_path: str = "app/evaluation/cases.json"
    eval_use_judge: bool = True  # 是否启用 LLM-as-judge（质量/幻觉/过程合理性）
    eval_pass_threshold: float = 0.6  # 单维度通过阈值（judge 归一化到 0-1 后比较）
    # 3.2：不同于被测模型的 Judge 模型（正式发布评测强制；空=拒绝运行）
    eval_judge_model: str = ""
    # 3.1：v2 评测报告目录（artifacts/eval/v2/<run-id>/）
    eval_output_dir: str = "artifacts/eval/v2"

    # 多轮对话管理
    session_path: str = "app/sessions/session.json"  # 保留字段：显式指定单会话文件（阶段一后由 session_dir+user_id 派生为主）
    session_dir: str = (
        "app/sessions"  # 阶段一：按 {user_id}/{session_id}.json 派生会话文件
    )
    history_threshold: int = 10  # 消息压缩策略通常为上下文达到一定的token数，例如claude code通常为达到最大上下文窗口的70%左右，此处简略为原始消息条数超过10轮
    history_keep_recent: int = 3  # 压缩时保留最近 3 条原始消息

    # 服务端（阶段一：1.7 同步/异步执行模型——Agent 全链路是同步代码，
    # FastAPI 事件循环内一律经线程池执行，禁止直接同步调用 LLM/工具）
    server_agent_threads: int = (
        32  # pod 级 Agent 执行线程上限（与阶段四 4.4 信号量衔接）
    )

    # 工具并发（Agent能力强化计划·改造二，双层限流）
    tool_parallelism: int = 2  # 单批次并行度（原序只读段内）
    tool_max_concurrent: int = (
        16  # pod 全局工具并发上限（固定池 max_workers + 提交许可）
    )
    tool_write_timeout_seconds: float = (
        60  # 远端写操作超时：超时即「结果未知」（indeterminate），禁止自动重试
    )
    # 工具调用守卫（2.5，可配置可量化）：签名去重 + 单工具轮内次数上限
    tool_call_guard_enabled: bool = (
        True  # 总开关（消融实验 baseline 关 / candidate 开）
    )
    tool_max_calls_per_name: int = 2  # 普通工具每轮最多调用次数
    tool_search_max_calls: int = 3  # search_knowledge 上限（允许一次关键词改写重检索）

    # 工具结果结构化精简（记忆/摘要转录层；digest.py）
    tool_digest_enabled: bool = True  # 总开关：False 整体回退旧版前缀截断
    tool_digest_budget_chars: int = 200  # 每条 tool 消息的转录预算（字符）

    # 状态外置（阶段二）
    session_store_backend: str = "auto"  # auto（db_url>Redis>文件，历史行为）| file（显式文件，Redis 仅锁/限流）| redis（强制 Redis）
    redis_url: str = "redis://localhost:6379/0"
    redis_required: bool = False  # 生产 true：Redis 连不上启动 fast-fail（2.7）
    session_lock_ttl_seconds: int = 300  # 会话锁 TTL（≥ 单轮最长耗时，lease 自动续期）
    memory_consolidate_every: int = (
        10  # 2.3：每 N 轮增量巩固长期记忆（close() 不可靠的 K8s 兜底）
    )
    memory_consolidate_idle_minutes: int = (
        30  # 2.3：会话静默超过该时长由后台 job 兜底巩固
    )
    memory_consolidate_scan_minutes: int = 5  # 静默兜底扫描间隔
    turns_archive_backend: str = (
        "local"  # local | s3（evolution turns 落盘对象存储，2.4）
    )
    s3_bucket: str = ""
    s3_endpoint_url: str = ""
    s3_access_key: str = ""  # 4.3：MinIO/OSS 凭证（留空则匿名/环境变量链）
    s3_secret_key: str = ""

    # 鉴权与安全（阶段三）
    auth_enabled: bool = (
        False  # true：Bearer JWT 校验，user_id 从 token 解出，请求体不再接受
    )
    jwt_secret: str = ""  # K8s Secret 注入；留空仅开发（使用固定 dev 密钥并给出警告）
    jwt_issuer: str = "ecom-agent"
    jwt_audience: str = "ecom-agent-api"
    jwt_ttl_minutes: int = 60 * 24
    guardrails_enabled: bool = True  # 3.5 运行时 guardrails（输入注入/PII、输出敏感词）
    guardrail_block_terms: str = ""  # 输出侧额外敏感词（逗号分隔）
    business_only_scope: bool = (
        False  # 业务范围闸门：非业务/闲聊 → 固定引导话术（默认关）
    )
    retrieval_fence_enabled: bool = True  # 检索内容来源围栏（KB 块视为不可信数据）
    citation_check_enabled: bool = True  # 引用来源真实性校验（降置信/转人工，不硬拦）
    refund_confirmation_required: bool = (
        False  # 3.3 退款两段式（生产 true；开发/CLI 保持直退）
    )
    refund_confirm_ttl_seconds: int = 300  # 一次性确认 token 有效期
    refund_idempotency_ttl_seconds: int = (
        7 * 24 * 3600
    )  # 幂等结果账本 TTL（refund_id → 首次结果）
    enforce_order_ownership: bool = False  # 3.1/3.2 工具级授权：查询/退款校验订单归属
    rate_limit_rps: int = 5  # 3.7 per-user RPS（0=不限）

    # 商家业务网关（2.3）：mock（开发/评测）| http（生产，fail-fast）
    commerce_backend: str = "mock"
    commerce_base_url: str = ""  # 如 http://commerce:8080（kind fake commerce 服务）
    commerce_timeout_seconds: float = 5.0  # 读取超时（退款超时 → indeterminate）
    commerce_connect_timeout_seconds: float = 2.0  # 连接超时（内网服务）
    commerce_api_key: str = (
        ""  # 服务级 Bearer（仅 HTTP 后端；经 ToolContext.credentials 注入）
    )
    rate_limit_window_seconds: int = 60
    rate_limit_daily_tokens: int = 0  # 每用户日 token/费用预算（0=不限）
    ops_rbac_required: bool = (
        False  # 安全修复 P1：true 时运营端点即使 auth 关闭也强制 JWT+ops scope
    )

    # 知识审核后台（阶段六 6.3；安全修复 P1：认证 + CSRF）
    review_admin_token: str = ""  # 审核后台管理令牌；留空 → 整站 503 fail-closed
    review_admin_cookie_secure: bool = (
        False  # TLS 部署置 true（会话 cookie 加 Secure 位）
    )

    # 运营闭环（阶段六）
    knowledge_aging_days: int = (
        365  # 6.4 知识时效阈值：effective_date 超期 → aging 标注
    )

    # 阶段八：MySQL 正本 + ES 检索（Redis 保留并发角色：锁/限流/会话热缓存）
    db_url: str = ""  # 如 mysql+pymysql://user:pwd@localhost:3306/ecom 或 sqlite:///app/sessions/ecom.sqlite；空=沿用文件/Redis 存储（开发默认）
    session_hot_cache_ttl_seconds: int = (
        1800  # 会话热缓存 TTL（write-through，Redis 可用时生效）
    )
    message_index_outbox_enabled: bool = True  # 消息入库出站箱→ES message_search 同步
    es_url: str = ""  # 如 http://localhost:9200；空=不启用 ES 后端
    es_user: str = ""
    es_password: str = ""
    es_index_prefix: str = (
        "ecom"  # KB 索引前缀（alias: {prefix}-kb-active；消息索引: {prefix}-messages）
    )

    # 可观测性（阶段四）
    log_format: str = "json"  # json（生产，stdout 结构化）| console（CLI 可读）
    otel_exporter_endpoint: str = ""  # OTLP HTTP 端点（空=不导出，埋点空跑）
    otel_service_name: str = "ecom-agent"

    # LLM 韧性客户端（阶段四 4.4）
    llm_timeout_seconds: float = 60  # 显式 timeout（取代 SDK 默认 600s）
    llm_max_retries: int = 2  # 指数退避+抖动重试（仅幂等读）
    llm_max_concurrent: int = 16  # pod 级并发信号量（护系统；用户级配额见 3.7）
    llm_fallback_model: str = ""  # 降级链备用模型（空=不降级）
    llm_max_tokens: int = 2048  # 单次回复 token 上限（单轮预算的粒度）
    extraction_model: str = (
        ""  # 便宜模型：结构化提取/STM/摘要/路由（解决每轮双调用成本）
    )

    # Evolution 配置（第10期，QA 自动沉淀）
    evolve_capture_enabled: bool = True  # 每轮对话是否落盘 turn 原始记录
    self_evolve_enabled: bool = (
        False  # 是否允许自动沉淀写入知识库（安全开关，默认关闭）
    )
    evolve_turns_dir: str = "app/sessions/evolution/turns"
    evolve_state_dir: str = "app/sessions/evolution/state"
    evolve_output_dir: str = "app/sessions/evolution/output"
    evolve_min_confidence: float = 0.8  # 只有置信度不低于该值的 turn 才进入挖掘
    evolve_min_quality: float = 0.8  # 价值 Judge 质量分低于该值不自动发布（4/5 分闸门）
    evolve_pre_dedup_threshold: float = 0.95  # 预去重相似度阈值（原问题层面）
    evolve_dedup_threshold: float = 0.9  # 最终去重相似度阈值（规范问题+答案）
    evolve_max_judge_per_run: int = 50  # 单次运行最多送审的候选数（Judge API 成本上限）
    evolve_max_per_run: int = 20  # 单次运行最多发布的文档数
    evolve_pending_aging_days: int = 30  # pending 目录中超过该天数未审核视为 aging
    evolve_turn_retention_days: int = 90  # turn 原始记录保留天数，超期可 prune
    evolve_lock_stale_seconds: int = 7200  # 运行锁超过该秒数且进程已死视为 stale
    evolve_effective_days: int = 180  # 自进化文档 effective_date = 发布日 + 该天数
    evolve_max_reground_per_run: int = (
        20  # 单次重接地最多核对的自进化文档数（成本上限）
    )
    evolve_revalidate_enabled: bool = True  # 人工知识变更后是否自动触发存量沉淀重接地
    # 人工客服问答沉淀（独立人工知识支路）：cronjob 自动收集 resolved 工单
    # （resolve 勾选 knowledge_candidate 为加急通道）→ Evolution 导入 pending
    # → 夜间 LLM 评审 → 人工审核 → 现有发布事务。
    # 首次部署保持 False；开启后自动扫描已解决工单（ledger 终态幂等去重）。
    human_qa_evolution_enabled: bool = False
    human_conversation_retention_days: int = 180
    # 人工链路语义去重阈值（与 evolve_dedup_threshold 彻底分离；初始值待校准）。
    # 相似度归一化：norm = min(max(cos, 0), 1)（截断而非仿射）——0.90 语义即
    # 「余弦 0.90」，正交 → 0 而非 0.5，既有阈值可直接沿用为初始值。
    human_dedup_question_threshold: float = 0.90
    human_dedup_answer_threshold: float = 0.90
    # 新知识复合分门槛（初始冻结值 0.70；2026-09 真实环境校准：bge-m3 下域内
    # 答案侧 top-1 ∈ [0.65, 0.75] ⇒ 新颖度 ≤ 0.35 ⇒ composite 上限 ≈ 0.67，
    # 0.70 在真实空间不可达——验收/生产经该校准值下发）。
    human_composite_threshold: float = 0.70
    # 人工会话栅栏（MySQL GET_LOCK）：接入侧/发布侧等待超时秒数。
    human_fence_ingest_timeout_seconds: float = 3.0
    human_fence_publish_timeout_seconds: float = 10.0
    # Generation 指针文件：记录每个后端当前激活的索引版本
    kb_generation_path: str = "app/sessions/kb_generations.json"

    # KB 统一写锁（v7 冻结：部署期选择后端，运行时不降级）
    app_env: str = "dev"  # 仅锁名/前缀用途（如 ecom-agent:dev:kb_write）
    kb_write_lock_backend: str = (
        "auto"  # auto | mysql | redis | file（file 仅显式配置，单机开发）
    )
    kb_write_lock_timeout: int = 30  # MySQL GET_LOCK 等待秒数

    # KB 文档上传（分片断点续传 + MySQL 元数据 + 版本化重建入库）
    kb_upload_enabled: bool = True  # 总开关
    kb_upload_storage: str = (
        "local"  # 分片存储：local（单 Pod 或共享卷）| s3（对象存储）
    )
    kb_upload_tmp_dir: str = (
        "app/sessions/uploads"  # 分片落盘目录（运行时数据，gitignore）
    )
    # 以下三个目录必须与知识库同挂载点（原子 rename 的前提）：
    kb_upload_staging_dir: str = (
        "app/agent/rag/knowledge/.staging"  # complete 暂存/failed 保留区
    )
    kb_uploads_dir: str = (
        "app/agent/rag/knowledge/uploads"  # 入库收纳目录（多 Pod 需共享卷）
    )
    kb_upload_trash_dir: str = "app/agent/rag/knowledge/.trash"  # 下架隔离目录
    kb_upload_original_dir: str = "app/sessions/upload_originals"  # 原件（原格式）保存
    kb_upload_max_bytes: int = 20 * 1024 * 1024  # 单文件上限
    kb_upload_chunk_size: int = 1024 * 1024  # 客户端缺省分片大小
    kb_upload_min_chunk_size: int = 64 * 1024  # 分片钳制下限（防海量 Redis 字段）
    kb_upload_max_chunk_size: int = 4 * 1024 * 1024  # 分片钳制上限
    kb_upload_max_chunks: int = 1000  # 单会话分片数上限
    kb_upload_max_pages: int = 200  # PDF 页数上限（解析炸弹防护）
    kb_upload_max_zip_entries: int = 2000  # docx zip 条目上限
    kb_upload_max_zip_bytes: int = 50 * 1024 * 1024  # docx zip 累计解压字节上限
    kb_upload_max_zip_ratio: int = 200  # docx zip 压缩比上限
    kb_upload_max_text_chars: int = 200_000  # 解析后文本长度上限
    kb_upload_parse_timeout: int = 30  # 子进程解析超时（秒，可终止）
    kb_upload_publish_timeout: int = 60  # publishing 分片超时接管阈值（秒）
    kb_upload_session_ttl: int = 7 * 24 * 3600  # 断点会话 Redis TTL
    kb_upload_retention_original_days: int = 90  # deleted 原件保留（indexed 永留）
    kb_upload_retention_trash_days: int = 30  # 下架 trash 文件保留
    kb_upload_retention_failed_days: int = 30  # failed staging 现场保留
    kb_gc_interval_seconds: int = 6 * 3600  # 定时 GC 间隔
    kb_gc_max_items: int = 100  # 定时 GC 处理上限
    kb_gc_opportunistic_max: int = 10  # 请求内机会式 GC 上限

    # KB 异步建库任务（多实例异步改造）：MySQL 任务队列正本 + 独立 Worker
    # 两阶段上线：默认 False（同步语义），由共享 kb_control 开关
    # kb_async_enabled=1 一次性启用异步 API 与 Worker（避免滚动期新旧语义混用）
    kb_async_api: bool = False  # 无共享开关时的默认值（开发/测试直开）
    kb_async_control_key: str = "kb_async_enabled"  # 共享开关（kb_control 表）
    kb_worker_enabled: bool = (
        False  # KB Worker 进程总开关（仅 Worker Deployment 置 true）
    )
    kb_worker_concurrency: int = 1  # 单实例并发（全局单写者由 kb_write 锁保证）
    kb_job_lease_seconds: int = 120  # 任务租约（MySQL 服务端时间计算）
    kb_job_heartbeat_seconds: int = 30  # heartbeat 续租间隔
    kb_job_poll_seconds: int = 2  # Worker 轮询间隔
    kb_job_max_attempts: int = 5  # 自动尝试上限（锁等待不计失败次数）
    kb_job_backoff_base_seconds: int = 5  # 指数退避基数（+随机抖动）
    kb_job_backoff_max_seconds: int = 300  # 退避上限
    kb_job_lock_wait_seconds: int = 2  # 锁等待重回队列的再执行延迟
    kb_worker_grace_seconds: int = 120  # SIGTERM 终止宽限期（超时由租约接管）
    kb_job_backlog_alert_seconds: int = 600  # 积压（最老任务年龄）告警阈值
    kb_job_retention_done_days: int = 90  # succeeded/cancelled 记录保留
    kb_job_retention_failed_days: int = 180  # failed/blocked 记录保留

    model_config = {"env_file": _resolve_env_file()}


settings = Settings()

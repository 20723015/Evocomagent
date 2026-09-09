"""MySQL 正本 schema（阶段八）：Sessions/消息/LTM/Outbox。

与 deploy/sql/001_init.sql 保持同构（那里是给 DBA/迁移用的原生 MySQL DDL，
此处 SQLAlchemy metadata 供 create_all 与 sqlite 测试方言复用）。生产 MySQL、
本地/测试 SQLite：主键自增、唯一约束、索引语义一致。
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.sqlite import INTEGER as SQLITE_INTEGER

metadata = MetaData()

# 会话元数据：一行一会话；version 是 CAS 乐观锁依据
sessions = Table(
    "sessions",
    metadata,
    Column(
        "session_key", String(200), primary_key=True
    ),  # user_id/session_id（缺省 session）
    Column("user_id", String(64), nullable=False, index=True),
    Column("session_uuid", String(32), default=""),
    Column("version", Integer, nullable=False, default=0),
    Column(
        "summary", Text, nullable=True
    ),  # 历史压缩摘要（正本永续，摘要只是 LLM 窗口）
    Column("stm_json", Text, nullable=True),  # 短期记忆序列化
    Column(
        "consolidated_len", Integer, nullable=False, default=0
    ),  # 安全修复 P2：增量巩固水位（防重启/兜底重复巩固）
    Column(
        "status", String(12), default="active"
    ),  # active | idle | closed（idle 巩固扫描依据）
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("expires_at", DateTime, nullable=True),
)

# 消息：append-only（只 INSERT；压缩只作用于发给 LLM 的窗口，正本不删）
chat_messages = Table(
    "chat_messages",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("session_key", String(200), nullable=False, index=True),
    Column("user_id", String(64), nullable=False, index=True),
    Column("turn_id", String(36), default="", index=True),
    Column("seq", Integer, nullable=False),
    Column("role", String(16), nullable=False),  # user/assistant/tool/system（查询用）
    Column("content", Text, nullable=False),  # 完整消息 dict 的 JSON（含 tool_calls）
    Column("tool_name", String(64), nullable=True),  # tool 消息的工具名（审计/质检用）
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    UniqueConstraint("session_key", "seq", name="uq_session_seq"),
)

# 长期记忆行式化
memory_facts = Table(
    "memory_facts",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("user_id", String(64), nullable=False, index=True),
    Column("category", String(32), default="other"),
    Column("content", Text, nullable=False),
    Column("source_session", String(64), default=""),
    Column("fact_id", String(64), nullable=False, default=""),
    Column("fact_key", String(128), nullable=False, default=""),
    Column("status", String(16), nullable=False, default="active"),
    Column("confidence", Float, nullable=False, default=1.0),
    Column("supersedes_id", String(64), nullable=False, default=""),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("evidence", Text, nullable=False, default=""),  # 2.4：用户原话依据（可审计）
)
Index("idx_facts_user_status", memory_facts.c.user_id, memory_facts.c.status)
Index("idx_facts_user_key", memory_facts.c.user_id, memory_facts.c.fact_key)

interaction_summaries = Table(
    "interaction_summaries",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("user_id", String(64), nullable=False, index=True),
    Column("summary", Text, nullable=False),
    Column("source_session", String(64), nullable=False, default=""),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)

# KB 控制表：全局阻塞标记等（值走 SQL，不依赖 Redis/文件，避免多 Pod 分裂）
kb_control = Table(
    "kb_control",
    metadata,
    Column("key", String(128), primary_key=True),
    Column("value", Text, nullable=False, default=""),
    Column("version", Integer, nullable=False, default=0),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
)

# KB 文档上传元数据（正本）：状态机
# uploading→validating→indexing→indexed；输入错误→failed(终态)；系统错误→回退 uploading
# indexed→deleting→deleted（deleting 仅提交点前可回滚 indexed）
kb_documents = Table(
    "kb_documents",
    metadata,
    Column("doc_id", String(64), primary_key=True),  # uuid4.hex
    Column(
        "upload_id", String(64), nullable=False, unique=True
    ),  # 断点会话 id（幂等锚点）
    Column(
        "uploader", String(64), nullable=False, default=""
    ),  # 上传主体（认证身份；幂等比对）
    Column(
        "storage_key", String(255), nullable=False, unique=True
    ),  # 规范化 .md 入库文件名（不含 /）
    Column(
        "original_key", String(255), nullable=False, default=""
    ),  # 原件存放 key（原格式）
    Column("filename", String(255), nullable=False),  # 展示用原始文件名（绝不拼路径）
    Column("content_type", String(100), nullable=False, default=""),
    Column("format", String(16), nullable=False),  # 原始格式（白名单后缀）
    Column(
        "chunk_size", Integer, nullable=False, default=0
    ),  # 会话分片大小（幂等比对）
    Column("size_bytes", BigInteger, nullable=False),
    Column("sha256", String(64), nullable=False),  # 合并后全文件摘要
    Column("status", String(16), nullable=False, default="uploading"),
    Column(
        "operation_id", String(64), nullable=False, default=""
    ),  # 服务端内部处理令牌（CAS 三条件）
    Column(
        "pending_generation_id", String(64), nullable=False, default=""
    ),  # 提交点前的目标代
    Column("upload_chunk_count", Integer, nullable=False, default=0),
    Column("indexed_chunk_count", Integer, nullable=False, default=0),
    Column("generation_id", String(64), nullable=False, default=""),
    Column("owner", String(32), nullable=False, default="ops"),
    Column("provenance", String(128), nullable=False, default=""),  # upload:{upload_id}
    Column("error", String(500), nullable=False, default=""),
    Column("version", Integer, nullable=False, default=0),  # CAS 乐观锁
    Column("status_changed_at", DateTime, nullable=True),  # 保留期计算（GC 依据）
    Column("expires_at", DateTime, nullable=True),  # 上传会话过期哨兵
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
)

# KB 建库异步任务（多实例异步改造·迁移007）：MySQL 是任务队列正本。
# 状态机：queued → running → succeeded | failed | blocked；
# retry_wait（退避中，到点可再领取）；cancelled（仅 queued/retry_wait 可取消）。
# 唯一键 (operation, doc_id)：重复 complete/delete 返回同一任务。
# 领取 = SKIP LOCKED + lease（owner/token/until）；MySQL 服务端时间计算租约。
kb_index_jobs = Table(
    "kb_index_jobs",
    metadata,
    Column("job_id", String(64), primary_key=True),  # uuid4.hex
    Column("operation", String(16), nullable=False),  # upload | delete
    Column("doc_id", String(64), nullable=False, index=True),
    Column("upload_id", String(64), nullable=False, default=""),
    Column("requested_by", String(64), nullable=False, default=""),  # 审计：请求主体
    Column("status", String(16), nullable=False, default="queued"),
    Column(
        "stage", String(32), nullable=False, default=""
    ),  # validating|parsing|waiting_for_lock|chunking|embedding|writing_index|activating|finalizing
    Column(
        "progress", Integer, nullable=False, default=0
    ),  # 0-100（embedding 按批推进）
    Column(
        "attempts", Integer, nullable=False, default=0
    ),  # 自动尝试次数（锁等待不计）
    Column(
        "manual_retry_count", Integer, nullable=False, default=0
    ),  # 人工重试审计数（只增）
    Column(
        "retryable", Integer, nullable=False, default=1
    ),  # 0=永久输入错误（人工重试亦无意义）
    Column("error", Text, nullable=False, default=""),  # 脱敏错误摘要
    Column("next_run_at", DateTime, nullable=True),  # 退避后可执行时间
    Column("lease_owner", String(64), nullable=False, default=""),
    Column("lease_token", String(64), nullable=False, default=""),  # 提交前所有权校验
    Column("lease_until", DateTime, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("started_at", DateTime, nullable=True),  # 首次开始执行
    Column("finished_at", DateTime, nullable=True),  # 进入终态时间
    UniqueConstraint("operation", "doc_id", name="uq_kb_job_op_doc"),
)
Index("idx_kb_jobs_status_next", kb_index_jobs.c.status, kb_index_jobs.c.next_run_at)

# 人工客服知识自进化（迁移008）：外部会话批量接入 → 每日评审 → 人工批量发布。
# 正本在 MySQL；LLM/embedding/RAG 评审不占 KB 写锁，短事务提交结果。
human_conversations = Table(
    "human_conversations",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("source", String(64), nullable=False),
    Column("external_conversation_id", String(128), nullable=False),
    Column("source_version", Integer, nullable=False, default=1),
    Column("agent_id", String(64), nullable=False, default=""),
    Column("started_at", DateTime, nullable=True),
    Column("ended_at", DateTime, nullable=True),
    Column("message_count", Integer, nullable=False, default=0),
    Column("conversation_digest", String(64), nullable=False),
    Column("transcript_json", Text, nullable=False),  # 脱敏后消息正本
    Column("transcript_purged_at", DateTime, nullable=True),
    Column("eval_status", String(16), nullable=False, default="pending"),
    Column("evaluation_finished_at", DateTime, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    UniqueConstraint(
        "source", "external_conversation_id", "source_version", name="uq_human_conv"
    ),
)

human_evaluation_jobs = Table(
    "human_evaluation_jobs",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("job_type", String(16), nullable=False, default="conversation"),
    Column("conversation_id", BigInteger, nullable=True),
    Column("candidate_id", BigInteger, nullable=True),
    Column("status", String(16), nullable=False, default="queued"),
    Column("attempts", Integer, nullable=False, default=0),
    Column("next_run_at", DateTime, nullable=True),
    Column("lease_owner", String(64), nullable=False, default=""),
    Column("lease_token", String(64), nullable=False, default=""),
    Column("lease_until", DateTime, nullable=True),
    Column("error", Text, nullable=False, default=""),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("finished_at", DateTime, nullable=True),
)

human_knowledge_candidates = Table(
    "human_knowledge_candidates",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("conversation_id", BigInteger, nullable=False),
    Column("status", String(24), nullable=False, default="pending_review"),
    Column("reject_reason", String(64), nullable=False, default=""),
    Column("question", String(200), nullable=False),
    Column("answer", Text, nullable=False),
    Column("evidence_message_ids", Text, nullable=False, default="[]"),
    Column("value_score", Float, nullable=True),
    Column("worth_saving", Integer, nullable=True),
    Column("value_reason", String(500), nullable=False, default=""),
    Column("max_similarity", Float, nullable=True),
    Column("novelty_score", Float, nullable=True),
    Column("composite_score", Float, nullable=True),
    Column("rag_hit_path", String(255), nullable=False, default=""),
    Column("rag_hit_kind", String(32), nullable=False, default=""),
    Column("classification", String(24), nullable=False, default=""),
    Column("score_stale", Integer, nullable=False, default=0),
    Column("revision", Integer, nullable=False, default=0),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("eval_model", String(64), nullable=False, default=""),
    Column("eval_prompt_version", String(32), nullable=False, default=""),
    Column("eval_embedding_version", String(64), nullable=False, default=""),
    Column("eval_kb_generation", String(64), nullable=False, default=""),
    Column("published_filename", String(255), nullable=False, default=""),
    Column("publish_batch_id", BigInteger, nullable=True),
    # 迁移 010：证据链/去重快照与生命周期（见 010_human_knowledge_lifecycle.sql）
    Column("source_snapshot_json", Text, nullable=True),
    Column("evidence_snapshot_json", Text, nullable=True),
    Column("evidence_state", String(24), nullable=False, default="legacy_evidence_missing"),
    Column("dedup_snapshot_json", Text, nullable=True),
    Column("lifecycle_revision", Integer, nullable=False, default=0),
    Column("published_at", DateTime, nullable=True),
    Column("published_generation", String(64), nullable=False, default=""),
    Column("retired_at", DateTime, nullable=True),
    Column("retire_reason", String(255), nullable=False, default=""),
    Column("replaced_by_candidate_id", BigInteger, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)

human_publish_batches = Table(
    "human_publish_batches",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("status", String(16), nullable=False, default="queued"),
    Column("requested_by", String(64), nullable=False, default=""),
    Column("item_count", Integer, nullable=False, default=0),
    Column("operation", String(8), nullable=False, default="publish"),
    Column("reason", String(255), nullable=False, default=""),
    Column("generation_id", String(64), nullable=False, default=""),
    Column("error", Text, nullable=False, default=""),
    Column("attempts", Integer, nullable=False, default=0),
    Column("next_run_at", DateTime, nullable=True),
    Column("lease_owner", String(64), nullable=False, default=""),
    Column("lease_token", String(64), nullable=False, default=""),
    Column("lease_until", DateTime, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("finished_at", DateTime, nullable=True),
)

human_publish_items = Table(
    "human_publish_items",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("batch_id", BigInteger, nullable=False),
    Column("candidate_id", BigInteger, nullable=False),
    # 迁移 010：不可变审批快照（发布内容取自快照而非候选现行值）
    Column("candidate_revision", Integer, nullable=True),
    Column("source_version", Integer, nullable=True),
    Column("question", String(200), nullable=True),
    Column("answer", Text, nullable=True),
    Column("value_score", Float, nullable=True),
    Column("classification", String(24), nullable=True),
    Column("dedup_target_path", String(255), nullable=False, default=""),
    Column("approved_by", String(64), nullable=False, default=""),
    Column("approved_at", DateTime, nullable=True),
    Column("approval_digest", String(64), nullable=False, default=""),
    Column("status", String(16), nullable=False, default="queued"),
    Column("filename", String(255), nullable=False, default=""),
    Column("detail", String(500), nullable=False, default=""),
    Index("idx_human_pub_item", "batch_id"),
)

# Outbox：消息 INSERT 同事务写入；后台任务搬去 ES 后标记 synced_at
outbox_rows = Table(
    "outbox_rows",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("session_key", String(200), nullable=False),
    Column("seq", Integer, nullable=False),
    Column("payload", Text, nullable=False),  # 完整消息 JSON
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("synced_at", DateTime, nullable=True),
    Column("sync_error", Text, nullable=True),
    UniqueConstraint("session_key", "seq", name="uq_outbox_session_seq"),
)

# 记忆巩固任务（单 Agent 全量优化计划·阶段F）：消息保存同事务入队；
# 唯一键 session_key + through_seq（至少一次投递下按水位幂等）。
# 状态机：pending → processing → done | failed；租约接管崩溃任务。
memory_jobs = Table(
    "memory_jobs",
    metadata,
    Column(
        "id",
        BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    Column("session_key", String(200), nullable=False, index=True),
    Column("user_id", String(64), nullable=False, index=True),
    Column(
        "session_uuid", String(32), nullable=False, default=""
    ),  # 006：创建任务时的会话实例（reset 后旧任务 obsolete）
    Column("through_seq", Integer, nullable=False),  # 已入库消息水位（含本条）
    Column("status", String(16), nullable=False, default="pending"),
    Column("attempts", Integer, nullable=False, default=0),
    Column("next_run_at", DateTime, nullable=True),
    Column("lease_until", DateTime, nullable=True),
    Column("leased_by", String(64), nullable=False, default=""),
    Column("error", Text, nullable=False, default=""),  # 脱敏错误摘要
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    UniqueConstraint("session_key", "through_seq", name="uq_memory_job_session_seq"),
)

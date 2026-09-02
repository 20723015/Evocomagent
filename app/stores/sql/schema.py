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
    "sessions", metadata,
    Column("session_key", String(200), primary_key=True),  # user_id/session_id（缺省 session）
    Column("user_id", String(64), nullable=False, index=True),
    Column("session_uuid", String(32), default=""),
    Column("version", Integer, nullable=False, default=0),
    Column("summary", Text, nullable=True),       # 历史压缩摘要（正本永续，摘要只是 LLM 窗口）
    Column("stm_json", Text, nullable=True),      # 短期记忆序列化
    Column("consolidated_len", Integer, nullable=False, default=0),  # 安全修复 P2：增量巩固水位（防重启/兜底重复巩固）
    Column("status", String(12), default="active"),  # active | idle | closed（idle 巩固扫描依据）
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("expires_at", DateTime, nullable=True),
)

# 消息：append-only（只 INSERT；压缩只作用于发给 LLM 的窗口，正本不删）
chat_messages = Table(
    "chat_messages", metadata,
    Column("id", BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
           primary_key=True, autoincrement=True),
    Column("session_key", String(200), nullable=False, index=True),
    Column("user_id", String(64), nullable=False, index=True),
    Column("turn_id", String(36), default="", index=True),
    Column("seq", Integer, nullable=False),
    Column("role", String(16), nullable=False),       # user/assistant/tool/system（查询用）
    Column("content", Text, nullable=False),          # 完整消息 dict 的 JSON（含 tool_calls）
    Column("tool_name", String(64), nullable=True),   # tool 消息的工具名（审计/质检用）
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    UniqueConstraint("session_key", "seq", name="uq_session_seq"),
)

# 长期记忆行式化
memory_facts = Table(
    "memory_facts", metadata,
    Column("id", BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
           primary_key=True, autoincrement=True),
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
    "interaction_summaries", metadata,
    Column("id", BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
           primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False, index=True),
    Column("summary", Text, nullable=False),
    Column("source_session", String(64), nullable=False, default=""),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)

# KB 控制表：全局阻塞标记等（值走 SQL，不依赖 Redis/文件，避免多 Pod 分裂）
kb_control = Table(
    "kb_control", metadata,
    Column("key", String(128), primary_key=True),
    Column("value", Text, nullable=False, default=""),
    Column("version", Integer, nullable=False, default=0),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
)

# KB 文档上传元数据（正本）：状态机
# uploading→validating→indexing→indexed；输入错误→failed(终态)；系统错误→回退 uploading
# indexed→deleting→deleted（deleting 仅提交点前可回滚 indexed）
kb_documents = Table(
    "kb_documents", metadata,
    Column("doc_id", String(64), primary_key=True),            # uuid4.hex
    Column("upload_id", String(64), nullable=False, unique=True),  # 断点会话 id（幂等锚点）
    Column("uploader", String(64), nullable=False, default=""),     # 上传主体（认证身份；幂等比对）
    Column("storage_key", String(255), nullable=False, unique=True),  # 规范化 .md 入库文件名（不含 /）
    Column("original_key", String(255), nullable=False, default=""),  # 原件存放 key（原格式）
    Column("filename", String(255), nullable=False),           # 展示用原始文件名（绝不拼路径）
    Column("content_type", String(100), nullable=False, default=""),
    Column("format", String(16), nullable=False),              # 原始格式（白名单后缀）
    Column("chunk_size", Integer, nullable=False, default=0),  # 会话分片大小（幂等比对）
    Column("size_bytes", BigInteger, nullable=False),
    Column("sha256", String(64), nullable=False),              # 合并后全文件摘要
    Column("status", String(16), nullable=False, default="uploading"),
    Column("operation_id", String(64), nullable=False, default=""),  # 服务端内部处理令牌（CAS 三条件）
    Column("pending_generation_id", String(64), nullable=False, default=""),  # 提交点前的目标代
    Column("upload_chunk_count", Integer, nullable=False, default=0),
    Column("indexed_chunk_count", Integer, nullable=False, default=0),
    Column("generation_id", String(64), nullable=False, default=""),
    Column("owner", String(32), nullable=False, default="ops"),
    Column("provenance", String(128), nullable=False, default=""),      # upload:{upload_id}
    Column("error", String(500), nullable=False, default=""),
    Column("version", Integer, nullable=False, default=0),     # CAS 乐观锁
    Column("status_changed_at", DateTime, nullable=True),      # 保留期计算（GC 依据）
    Column("expires_at", DateTime, nullable=True),             # 上传会话过期哨兵
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
)

# Outbox：消息 INSERT 同事务写入；后台任务搬去 ES 后标记 synced_at
outbox_rows = Table(
    "outbox_rows", metadata,
    Column("id", BigInteger().with_variant(SQLITE_INTEGER, "sqlite"),
           primary_key=True, autoincrement=True),
    Column("session_key", String(200), nullable=False),
    Column("seq", Integer, nullable=False),
    Column("payload", Text, nullable=False),   # 完整消息 JSON
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("synced_at", DateTime, nullable=True),
    Column("sync_error", Text, nullable=True),
    UniqueConstraint("session_key", "seq", name="uq_outbox_session_seq"),
)

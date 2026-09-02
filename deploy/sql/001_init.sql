-- 阶段八：MySQL 正本迁移（与 app/stores/sql/schema.py 同构；本文件供 DBA/迁移用）
-- 生产连接串示例：mysql+pymysql://user:pwd@localhost:3306/ecom?charset=utf8mb4

CREATE DATABASE IF NOT EXISTS ecom DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ecom;

-- 会话元数据：一行一会话；version 是 CAS 乐观锁依据
CREATE TABLE IF NOT EXISTS sessions (
    session_key  VARCHAR(200) PRIMARY KEY COMMENT 'user_id/session_id（缺省 session）',
    user_id      VARCHAR(64)  NOT NULL COMMENT '归属校验(403)依据，建索引',
    session_uuid CHAR(32)     NOT NULL DEFAULT '',
    version      INT          NOT NULL DEFAULT 0 COMMENT 'CAS 乐观锁',
    summary      MEDIUMTEXT   NULL COMMENT '历史压缩摘要（正本永续，摘要只给 LLM 窗口）',
    stm_json     MEDIUMTEXT   NULL COMMENT '短期记忆序列化',
    consolidated_len INT      NOT NULL DEFAULT 0 COMMENT '增量巩固水位（安全修复 P2：防重启/兜底重复巩固）',
    status       VARCHAR(12)  NOT NULL DEFAULT 'active' COMMENT 'active|idle|closed',
    updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at   DATETIME     NULL,
    KEY idx_sessions_user (user_id),
    KEY idx_sessions_status (status)
) ENGINE=InnoDB;

-- 消息：append-only（只 INSERT，不 UPDATE；压缩只作用于 LLM 窗口，正本不删）
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGINT       NOT NULL AUTO_INCREMENT,
    session_key VARCHAR(200) NOT NULL,
    user_id     VARCHAR(64)  NOT NULL,
    turn_id     VARCHAR(36)  NOT NULL DEFAULT '' COMMENT '一轮一个 turn，关联进化/审计',
    seq         INT          NOT NULL COMMENT '会话内序号：ORDER BY seq 重建会话',
    role        VARCHAR(16)  NOT NULL COMMENT 'user/assistant/tool/system',
    content     MEDIUMTEXT   NOT NULL COMMENT '完整消息 dict 的 JSON（含 tool_calls）',
    tool_name   VARCHAR(64)  NULL COMMENT '工具消息的工具名（审计/质检用）',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_session_seq (session_key, seq),
    KEY idx_messages_user_time (user_id, created_at),
    KEY idx_messages_turn (turn_id)
) ENGINE=InnoDB;

-- 长期记忆行式化
CREATE TABLE IF NOT EXISTS memory_facts (
      id              BIGINT      NOT NULL AUTO_INCREMENT,
      user_id         VARCHAR(64) NOT NULL,
      category        VARCHAR(32) NOT NULL DEFAULT 'other' COMMENT 'identity/preference/behavior/issue/other',
      content         TEXT        NOT NULL,
      source_session  VARCHAR(64) NOT NULL DEFAULT '',
      fact_id         VARCHAR(64) NOT NULL DEFAULT '',
      fact_key        VARCHAR(128) NOT NULL DEFAULT '',
      status          VARCHAR(16) NOT NULL DEFAULT 'active',
      confidence      DOUBLE      NOT NULL DEFAULT 1.0,
      supersedes_id   VARCHAR(64) NOT NULL DEFAULT '',
      created_at      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      KEY idx_facts_user (user_id),
      KEY idx_facts_user_status (user_id, status),
      KEY idx_facts_user_key (user_id, fact_key)
  ) ENGINE=InnoDB;

  CREATE TABLE IF NOT EXISTS interaction_summaries (
      id         BIGINT      NOT NULL AUTO_INCREMENT,
      user_id    VARCHAR(64) NOT NULL,
      summary    TEXT        NOT NULL,
      source_session VARCHAR(64) NOT NULL DEFAULT '',
      created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_summaries_user (user_id)
) ENGINE=InnoDB;

-- Outbox：消息 INSERT 同事务写入；后台任务搬去 ES 后标记 synced_at（可重建/最终一致）
CREATE TABLE IF NOT EXISTS outbox_rows (
    id         BIGINT      NOT NULL AUTO_INCREMENT,
    session_key VARCHAR(200) NOT NULL,
    seq        INT         NOT NULL,
    payload    MEDIUMTEXT  NOT NULL COMMENT '完整消息 JSON',
    created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    synced_at  DATETIME    NULL,
    sync_error TEXT        NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_outbox_session_seq (session_key, seq),
    KEY idx_outbox_unsynced (synced_at)
) ENGINE=InnoDB;

-- KB 控制表：全局阻塞标记等（值走 SQL，不依赖 Redis/文件，避免多 Pod 分裂）
CREATE TABLE IF NOT EXISTS kb_control (
    `key`       VARCHAR(128) NOT NULL,
    value       TEXT         NOT NULL,
    version     INT          NOT NULL DEFAULT 0,
    updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`key`)
) ENGINE=InnoDB;

-- KB 文档上传元数据（正本）：状态机
-- uploading→validating→indexing→indexed；输入错误→failed(终态)；系统错误→回退 uploading
-- indexed→deleting→deleted（deleting 仅提交点前可回滚 indexed）
CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id                VARCHAR(64)  NOT NULL,
    upload_id             VARCHAR(64)  NOT NULL COMMENT '断点会话 id（幂等锚点）',
    uploader              VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '上传主体（认证身份；幂等比对）',
    storage_key           VARCHAR(255) NOT NULL COMMENT '规范化 .md 入库文件名（不含 /）',
    original_key          VARCHAR(255) NOT NULL DEFAULT '' COMMENT '原件存放 key（原格式）',
    filename              VARCHAR(255) NOT NULL COMMENT '展示用原始文件名（绝不拼路径）',
    content_type          VARCHAR(100) NOT NULL DEFAULT '',
    format                VARCHAR(16)  NOT NULL COMMENT '原始格式（白名单后缀）',
    chunk_size            INT          NOT NULL DEFAULT 0 COMMENT '会话分片大小（幂等比对）',
    size_bytes            BIGINT       NOT NULL,
    sha256                VARCHAR(64)  NOT NULL COMMENT '合并后全文件摘要',
    status                VARCHAR(16)  NOT NULL DEFAULT 'uploading',
    operation_id          VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '服务端内部处理令牌（CAS 三条件）',
    pending_generation_id VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '提交点前的目标代',
    upload_chunk_count    INT          NOT NULL DEFAULT 0,
    indexed_chunk_count   INT          NOT NULL DEFAULT 0 COMMENT '入库时该文档切出的 chunk 数',
    generation_id         VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '入库代际（可溯源）',
    owner                 VARCHAR(32)  NOT NULL DEFAULT 'ops',
    provenance            VARCHAR(128) NOT NULL DEFAULT '' COMMENT 'upload:{upload_id}',
    error                 VARCHAR(500) NOT NULL DEFAULT '',
    version               INT          NOT NULL DEFAULT 0 COMMENT 'CAS 乐观锁',
    status_changed_at     DATETIME     NULL COMMENT '保留期计算（GC 依据）',
    expires_at            DATETIME     NULL COMMENT '上传会话过期哨兵',
    created_at            DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id),
    UNIQUE KEY uq_docs_upload (upload_id),
    UNIQUE KEY uq_docs_storage (storage_key),
    KEY idx_docs_status (status),
    KEY idx_docs_updated (updated_at),
    KEY idx_docs_status_changed (status_changed_at)
) ENGINE=InnoDB;

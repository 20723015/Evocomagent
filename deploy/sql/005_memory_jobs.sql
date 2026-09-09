-- 005：memory_jobs（单 Agent 全量优化计划·阶段F）
-- 持久化记忆巩固任务：消息保存同事务入队；唯一键 session_key+through_seq
-- 保证至少一次投递下的幂等；租约接管崩溃任务；重试上限后进入 failed 死信。
-- 应用侧水位：sessions.consolidated_len 即 memory_consolidated_seq
-- （worker 成功后单调推进；重复任务按水位直接确认完成）。
-- schema_migrations 记录由 migrate_db.py 写入（本文件只含 DDL）。

CREATE TABLE IF NOT EXISTS memory_jobs (
    id          BIGINT       NOT NULL AUTO_INCREMENT,
    session_key VARCHAR(200) NOT NULL COMMENT 'user_id/session_id',
    user_id     VARCHAR(64)  NOT NULL,
    through_seq INT          NOT NULL COMMENT '已入库消息水位（含本条）',
    status      VARCHAR(16)  NOT NULL DEFAULT 'pending' COMMENT 'pending|processing|done|failed',
    attempts    INT          NOT NULL DEFAULT 0,
    next_run_at DATETIME     NULL COMMENT '退避后可执行时间',
    lease_until DATETIME     NULL COMMENT 'processing 租约到期（接管崩溃任务）',
    leased_by   VARCHAR(64)  NOT NULL DEFAULT '',
    error       TEXT         NULL COMMENT '脱敏错误摘要',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_memory_job_session_seq (session_key, through_seq),
    KEY idx_memory_jobs_user (user_id),
    KEY idx_memory_jobs_status (status)
) ENGINE=InnoDB;

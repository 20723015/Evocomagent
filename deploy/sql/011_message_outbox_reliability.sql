-- 011：消息 Outbox 可靠性 + Reset 后 ES 删除事件（单 Agent 链路全量修复·二）
--
-- 背景：原 outbox 无重试状态/租约，单个 Pod 崩溃或坏行会阻塞后续；
-- reset 只删 MySQL 正本，ES 旧消息仍可被运营搜索命中。
--
-- 本迁移（修复计划·二轮 8：本轮内直接修正，尚未发布）：
-- 1) outbox_rows 增加 session_uuid 与重试/租约/dead-letter 字段（含 obsolete 终态）；
-- 2) 新建 message_delete_outbox：Reset 时同事务写入的 ES 删除事件（相同重试语义）；
-- 3) 存量行回填：session_uuid 从 sessions 取；按 synced_at 回填 status；清空租约字段；
-- 4) 唯一键调整为 (session_key, session_uuid, seq)，允许 Reset 后新会话重新使用 seq。
-- schema_migrations 记录由 migrate_db.py 写入（本文件只含 DDL/回填）。

ALTER TABLE outbox_rows
    ADD COLUMN session_uuid CHAR(32) NOT NULL DEFAULT '' COMMENT '写入时的会话实例（删除事件按 UUID 精确匹配）',
    ADD COLUMN status VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT 'pending|processing|done|dead_letter|obsolete',
    ADD COLUMN attempts INT NOT NULL DEFAULT 0 COMMENT '已尝试次数（含 crash 接管）',
    ADD COLUMN next_run_at DATETIME NULL COMMENT '退避后可执行时间（NULL=立即）',
    ADD COLUMN lease_owner VARCHAR(64) NOT NULL DEFAULT '',
    ADD COLUMN lease_token VARCHAR(64) NOT NULL DEFAULT '' COMMENT '结算前所有权校验',
    ADD COLUMN lease_until DATETIME NULL,
    ADD COLUMN dead_lettered_at DATETIME NULL COMMENT '进入死信时间';

UPDATE outbox_rows o
JOIN sessions s ON s.session_key = o.session_key
SET o.session_uuid = s.session_uuid
WHERE o.session_uuid = '';

UPDATE outbox_rows
SET status = CASE WHEN synced_at IS NOT NULL THEN 'done' ELSE 'pending' END,
    lease_owner = '',
    lease_token = '',
    lease_until = NULL;

ALTER TABLE outbox_rows
    DROP INDEX uq_outbox_session_seq,
    ADD UNIQUE KEY uq_outbox_session_seq (session_key, session_uuid, seq);

CREATE INDEX idx_outbox_status_next ON outbox_rows (status, next_run_at);

CREATE TABLE IF NOT EXISTS message_delete_outbox (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_key VARCHAR(200) NOT NULL,
    session_uuid CHAR(32) NOT NULL COMMENT '被重置会话的实例（新会话用新 UUID，不会被误删）',
    status VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT 'pending|processing|done|dead_letter|obsolete',
    attempts INT NOT NULL DEFAULT 0,
    next_run_at DATETIME NULL,
    lease_owner VARCHAR(64) NOT NULL DEFAULT '',
    lease_token VARCHAR(64) NOT NULL DEFAULT '',
    lease_until DATETIME NULL,
    error TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at DATETIME NULL,
    UNIQUE KEY uq_delete_outbox_session_uuid (session_key, session_uuid),
    KEY idx_delete_outbox_status_next (status, next_run_at)
) ENGINE=InnoDB;

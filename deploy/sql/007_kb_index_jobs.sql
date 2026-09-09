-- 007：kb_index_jobs（多实例异步知识库建库改造）
-- MySQL 是任务队列正本：上传 complete / 文档下架改为持久化异步任务，
-- API 立即返回 202；独立 KB Worker Deployment 领取执行。
-- - 唯一键 (operation, doc_id)：重复请求返回同一任务（幂等锚点）；
-- - 领取：SKIP LOCKED + lease（owner/token/until，MySQL 服务端时间计算租约）；
--   lease_token 随领取重生成——提交前所有权检查，失去所有权的旧 Worker 不得提交；
-- - 锁等待不计失败次数（attempts 回退）；自动尝试上限 5 次，指数退避上限 300s；
-- - 已越过提交点的可确定故障持续重试（不进入普通死信）。
-- schema_migrations 记录由 migrate_db.py 写入（本文件只含 DDL）。

CREATE TABLE IF NOT EXISTS kb_index_jobs (
    job_id             VARCHAR(64)  NOT NULL COMMENT 'uuid4.hex',
    operation          VARCHAR(16)  NOT NULL COMMENT 'upload|delete',
    doc_id             VARCHAR(64)  NOT NULL,
    upload_id          VARCHAR(64)  NOT NULL DEFAULT '',
    requested_by       VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '请求主体（审计）',
    status             VARCHAR(16)  NOT NULL DEFAULT 'queued'
        COMMENT 'queued|running|retry_wait|succeeded|failed|cancelled|blocked',
    stage              VARCHAR(32)  NOT NULL DEFAULT ''
        COMMENT 'validating|parsing|waiting_for_lock|chunking|embedding|writing_index|activating|finalizing',
    progress           INT          NOT NULL DEFAULT 0 COMMENT '0-100，embedding 每批推进',
    attempts           INT          NOT NULL DEFAULT 0 COMMENT '自动尝试次数（锁等待不计）',
    manual_retry_count INT          NOT NULL DEFAULT 0 COMMENT '人工重试审计数（只增）',
    retryable          TINYINT      NOT NULL DEFAULT 1 COMMENT '0=永久输入错误',
    error              TEXT         NOT NULL COMMENT '脱敏错误摘要',
    next_run_at        DATETIME     NULL COMMENT '退避后可执行时间',
    lease_owner        VARCHAR(64)  NOT NULL DEFAULT '',
    lease_token        VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '提交前所有权校验',
    lease_until        DATETIME     NULL COMMENT '租约到期（MySQL 服务端时间）',
    created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at         DATETIME     NULL COMMENT '首次开始执行',
    finished_at        DATETIME     NULL COMMENT '进入终态时间',
    PRIMARY KEY (job_id),
    UNIQUE KEY uq_kb_job_op_doc (operation, doc_id),
    KEY idx_kb_jobs_doc (doc_id),
    KEY idx_kb_jobs_status_next (status, next_run_at)
) ENGINE=InnoDB;

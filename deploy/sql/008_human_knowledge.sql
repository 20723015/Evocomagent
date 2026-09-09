-- 008：人工客服知识自进化链路（外部会话批量接入 → 评审 → 人工批量发布）
-- 正本全部在 MySQL；LLM/embedding/RAG 评审不占 KB 写锁，只在短事务提交结果。
-- schema_migrations 记录由 migrate_db.py 写入（本文件只含 DDL）。

-- 1) 脱敏会话正本：(source, external_conversation_id, source_version) 唯一；
--    conversation_digest 幂等（同摘要重复返回原记录，同版本不同内容 409）；
--    脱敏原文保留 180 天（evaluation_finished_at 起算），候选与审计长期保留。
CREATE TABLE IF NOT EXISTS human_conversations (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    source          VARCHAR(64)  NOT NULL COMMENT '外部系统标识',
    external_conversation_id VARCHAR(128) NOT NULL,
    source_version  INT          NOT NULL DEFAULT 1 COMMENT '外部修订版本（更高版本=重新评审）',
    agent_id        VARCHAR(64)  NOT NULL DEFAULT '',
    started_at      DATETIME     NULL,
    ended_at        DATETIME     NULL COMMENT '每日 Cron 处理 ended_at < 当日 00:00 的会话',
    message_count   INT          NOT NULL DEFAULT 0,
    conversation_digest CHAR(64) NOT NULL COMMENT '脱敏后消息正本 SHA-256（幂等/冲突判定）',
    transcript_json MEDIUMTEXT   NOT NULL COMMENT '脱敏后消息正本（PII 已替换占位符）',
    eval_status     VARCHAR(16)  NOT NULL DEFAULT 'pending'
        COMMENT 'pending|evaluated|failed（迟到数据自动补收）',
    evaluation_finished_at DATETIME NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_human_conv (source, external_conversation_id, source_version),
    KEY idx_human_conv_cron (eval_status, ended_at)
) ENGINE=InnoDB;

-- 2) 评审任务：SKIP LOCKED + lease token；退避 5m/30m/2h/6h/12h/24h，
--    6 次后 blocked（人工可重试）；LLM/RAG 失败不写 completed。
CREATE TABLE IF NOT EXISTS human_evaluation_jobs (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    job_type        VARCHAR(16)  NOT NULL DEFAULT 'conversation'
        COMMENT 'conversation=整段抽取 | candidate=编辑后重评',
    conversation_id BIGINT       NULL,
    candidate_id    BIGINT       NULL,
    status          VARCHAR(16)  NOT NULL DEFAULT 'queued'
        COMMENT 'queued|running|retry_wait|completed|blocked',
    attempts        INT          NOT NULL DEFAULT 0,
    next_run_at     DATETIME     NULL,
    lease_owner     VARCHAR(64)  NOT NULL DEFAULT '',
    lease_token     VARCHAR(64)  NOT NULL DEFAULT '',
    lease_until     DATETIME     NULL,
    error           TEXT         NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     DATETIME     NULL,
    PRIMARY KEY (id),
    KEY idx_human_eval_claim (status, next_run_at),
    KEY idx_human_eval_conv (conversation_id),
    KEY idx_human_eval_cand (candidate_id)
) ENGINE=InnoDB;

-- 3) 人工知识候选：问答/证据消息/评分/RAG 命中/revision/审计。
--    status: pending_review|rejected|superseded|publish_queued|published
--    评审元数据（模型/prompt 版本/embedding 版本/KB generation）随行审计。
CREATE TABLE IF NOT EXISTS human_knowledge_candidates (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    conversation_id BIGINT       NOT NULL,
    status          VARCHAR(24)  NOT NULL DEFAULT 'pending_review',
    reject_reason   VARCHAR(64)  NOT NULL DEFAULT '',
    question        VARCHAR(200) NOT NULL,
    answer          TEXT         NOT NULL,
    evidence_message_ids JSON   NOT NULL,
    value_score     FLOAT        NULL,
    worth_saving    TINYINT      NULL,
    value_reason    VARCHAR(500) NOT NULL DEFAULT '',
    max_similarity  FLOAT        NULL,
    novelty_score   FLOAT        NULL,
    composite_score FLOAT        NULL,
    rag_hit_path    VARCHAR(255) NOT NULL DEFAULT '',
    rag_hit_kind    VARCHAR(32)  NOT NULL DEFAULT '' COMMENT 'new|duplicate|update|authoritative_conflict',
    classification  VARCHAR(24)  NOT NULL DEFAULT '' COMMENT 'new|duplicate|update',
    score_stale     TINYINT      NOT NULL DEFAULT 0 COMMENT '编辑后旧评分过期（重评前禁止批准）',
    revision        INT          NOT NULL DEFAULT 0,
    updated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    eval_model      VARCHAR(64)  NOT NULL DEFAULT '',
    eval_prompt_version VARCHAR(32) NOT NULL DEFAULT '',
    eval_embedding_version VARCHAR(64) NOT NULL DEFAULT '',
    eval_kb_generation  VARCHAR(64) NOT NULL DEFAULT '',
    published_filename VARCHAR(255) NOT NULL DEFAULT '',
    publish_batch_id BIGINT      NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_human_cand_conv (conversation_id),
    KEY idx_human_cand_status (status)
) ENGINE=InnoDB;

-- 4) 人工批量发布：先整体校验 revision，同事务建批次并置 publish_queued；
--    多实例 Worker 租约领取批次，一个批次只构建一个 staging generation。
CREATE TABLE IF NOT EXISTS human_publish_batches (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    status          VARCHAR(16)  NOT NULL DEFAULT 'queued'
        COMMENT 'queued|running|published|failed',
    requested_by    VARCHAR(64)  NOT NULL DEFAULT '',
    item_count      INT          NOT NULL DEFAULT 0,
    generation_id   VARCHAR(64)  NOT NULL DEFAULT '',
    error           TEXT         NULL,
    attempts        INT          NOT NULL DEFAULT 0,
    lease_owner     VARCHAR(64)  NOT NULL DEFAULT '',
    lease_token     VARCHAR(64)  NOT NULL DEFAULT '',
    lease_until     DATETIME     NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     DATETIME     NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS human_publish_items (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    batch_id        BIGINT       NOT NULL,
    candidate_id    BIGINT       NOT NULL,
    status          VARCHAR(16)  NOT NULL DEFAULT 'queued'
        COMMENT 'queued|published|rejected|failed',
    filename        VARCHAR(255) NOT NULL DEFAULT '',
    detail          VARCHAR(500) NOT NULL DEFAULT '',
    PRIMARY KEY (id),
    KEY idx_human_pub_item (batch_id)
) ENGINE=InnoDB;

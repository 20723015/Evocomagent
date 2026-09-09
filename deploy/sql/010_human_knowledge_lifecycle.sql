-- 010：人工知识生命周期（证据链/审批快照/CAS 结算/下架/替换结算）。
-- 全部新列一次建齐：PR-1 启用证据与去重快照列，PR-2 启用审批快照与
-- lifecycle_revision，PR-3 启用 operation（下架批次）与 retired 列。
-- 历史行：evidence_state 默认 legacy_evidence_missing（禁止批准，回填脚本置 ok）。

-- 会话保留期只清正文、不删版本头；版本头永久参与 latest-version 校验。
ALTER TABLE human_conversations
    ADD COLUMN transcript_purged_at DATETIME NULL AFTER transcript_json;

ALTER TABLE human_knowledge_candidates
    ADD COLUMN source_snapshot_json TEXT NULL AFTER eval_kb_generation,
    ADD COLUMN evidence_snapshot_json TEXT NULL AFTER source_snapshot_json,
    ADD COLUMN evidence_state VARCHAR(24) NOT NULL DEFAULT 'legacy_evidence_missing' AFTER evidence_snapshot_json,
    ADD COLUMN dedup_snapshot_json TEXT NULL AFTER evidence_state,
    ADD COLUMN lifecycle_revision INT NOT NULL DEFAULT 0 AFTER dedup_snapshot_json,
    ADD COLUMN published_at DATETIME NULL AFTER lifecycle_revision,
    ADD COLUMN published_generation VARCHAR(64) NOT NULL DEFAULT '' AFTER published_at,
    ADD COLUMN retired_at DATETIME NULL AFTER published_generation,
    ADD COLUMN retire_reason VARCHAR(255) NOT NULL DEFAULT '' AFTER retired_at,
    ADD COLUMN replaced_by_candidate_id BIGINT NULL AFTER retire_reason;

ALTER TABLE human_publish_batches
    ADD COLUMN operation VARCHAR(8) NOT NULL DEFAULT 'publish' AFTER item_count,
    ADD COLUMN reason VARCHAR(255) NOT NULL DEFAULT '' AFTER operation;

-- 审批快照（不可变）：发布内容取自快照而非候选现行值——批准什么就发什么。
ALTER TABLE human_publish_items
    ADD COLUMN candidate_revision INT NULL AFTER candidate_id,
    ADD COLUMN source_version INT NULL AFTER candidate_revision,
    ADD COLUMN question VARCHAR(200) NULL AFTER source_version,
    ADD COLUMN answer TEXT NULL AFTER question,
    ADD COLUMN value_score FLOAT NULL AFTER answer,
    ADD COLUMN classification VARCHAR(24) NULL AFTER value_score,
    ADD COLUMN dedup_target_path VARCHAR(255) NOT NULL DEFAULT '' AFTER classification,
    ADD COLUMN approved_by VARCHAR(64) NOT NULL DEFAULT '' AFTER dedup_target_path,
    ADD COLUMN approved_at DATETIME NULL AFTER approved_by,
    ADD COLUMN approval_digest VARCHAR(64) NOT NULL DEFAULT '' AFTER approved_at;

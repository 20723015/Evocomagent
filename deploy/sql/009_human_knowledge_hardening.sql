-- 009：人工知识多实例安全加固（发布退避 + error 约束统一）。
UPDATE human_evaluation_jobs SET error = '' WHERE error IS NULL;
ALTER TABLE human_evaluation_jobs
    MODIFY COLUMN error TEXT NOT NULL;

UPDATE human_publish_batches SET error = '' WHERE error IS NULL;
ALTER TABLE human_publish_batches
    MODIFY COLUMN error TEXT NOT NULL,
    ADD COLUMN next_run_at DATETIME NULL AFTER attempts,
    ADD KEY idx_human_publish_claim (status, next_run_at);

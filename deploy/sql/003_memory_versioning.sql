-- 结构化记忆 v2：为已有 memory_facts 增加版本与状态字段（2.9 起幂等条件执行）。
-- 旧库缺列时补列并刷 legacy 标识；全新库（当前 001 已含全部列/索引）自动跳过。

-- ---------- memory_facts：逐列条件补列 ----------
-- 对缺失列逐条 ALTER（information_schema 预检，列已存在则跳过）
SET @c = 'fact_id';
SET @has = (SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND COLUMN_NAME = 'fact_id');
SET @ddl = IF(@has = 0, 'ALTER TABLE memory_facts ADD COLUMN fact_id VARCHAR(64) NOT NULL DEFAULT ''''', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @c = 'fact_key';
SET @has = (SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND COLUMN_NAME = 'fact_key');
SET @ddl = IF(@has = 0, 'ALTER TABLE memory_facts ADD COLUMN fact_key VARCHAR(128) NOT NULL DEFAULT ''''', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @c = 'status';
SET @has = (SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND COLUMN_NAME = 'status');
SET @ddl = IF(@has = 0, 'ALTER TABLE memory_facts ADD COLUMN status VARCHAR(16) NOT NULL DEFAULT ''active''', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @c = 'confidence';
SET @has = (SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND COLUMN_NAME = 'confidence');
SET @ddl = IF(@has = 0, 'ALTER TABLE memory_facts ADD COLUMN confidence DOUBLE NOT NULL DEFAULT 1.0', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @c = 'supersedes_id';
SET @has = (SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND COLUMN_NAME = 'supersedes_id');
SET @ddl = IF(@has = 0, 'ALTER TABLE memory_facts ADD COLUMN supersedes_id VARCHAR(64) NOT NULL DEFAULT ''''', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @c = 'updated_at';
SET @has = (SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND COLUMN_NAME = 'updated_at');
SET @ddl = IF(@has = 0, 'ALTER TABLE memory_facts ADD COLUMN updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 旧数据刷 legacy 标识（幂等：仅对空值行生效）
UPDATE memory_facts
SET fact_id = CONCAT('legacy-', id),
    fact_key = CONCAT('legacy.', LPAD(LOWER(HEX(id)), 16, '0')),
    updated_at = created_at
WHERE fact_id = '' OR fact_key = '';

-- ---------- 索引：存在则跳过 ----------
SET @has = (SELECT COUNT(*) FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND INDEX_NAME = 'idx_facts_user_status');
SET @ddl = IF(@has = 0, 'CREATE INDEX idx_facts_user_status ON memory_facts (user_id, status)', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has = (SELECT COUNT(*) FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'memory_facts'
              AND INDEX_NAME = 'idx_facts_user_key');
SET @ddl = IF(@has = 0, 'CREATE INDEX idx_facts_user_key ON memory_facts (user_id, fact_key)', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ---------- interaction_summaries：条件补列 ----------
SET @has = (SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'interaction_summaries'
              AND COLUMN_NAME = 'source_session');
SET @ddl = IF(@has = 0, 'ALTER TABLE interaction_summaries ADD COLUMN source_session VARCHAR(64) NOT NULL DEFAULT ''''', 'SELECT 1');
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
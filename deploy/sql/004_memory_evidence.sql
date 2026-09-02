-- 2.4：memory_facts 增加 evidence（用户原话依据，可审计）
-- 幂等：仅在列不存在时 ADD（MySQL 8 不支持 ADD COLUMN IF NOT EXISTS，
-- 用 information_schema 预检 + 动态 DDL 实现）。旧数据自动补空串，无损升级。

SET @has_evidence = (
    SELECT COUNT(*)
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'memory_facts'
      AND COLUMN_NAME = 'evidence'
);

SET @ddl = IF(
    @has_evidence = 0,
    'ALTER TABLE memory_facts ADD COLUMN evidence TEXT NOT NULL COMMENT ''用户原话依据（可审计）''',
    'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
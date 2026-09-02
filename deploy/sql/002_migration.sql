-- 0.3.0 升级迁移（评审二轮 A：升级路径破坏修复；2.9 起幂等条件执行）
-- 适用：旧库（阶段八按初版 001 建库）缺 sessions.consolidated_len。
-- 全新库（当前 001 已含该列）自动跳过；幂等由 information_schema 预检保证。
-- 服务端 dev/test 另有轻量自动补列（engine.py），本文件供迁移执行器正式留档。

SET @has_col = (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'sessions'
      AND COLUMN_NAME = 'consolidated_len'
);

SET @ddl = IF(
    @has_col = 0,
    'ALTER TABLE sessions ADD COLUMN consolidated_len INT NOT NULL DEFAULT 0 COMMENT ''增量巩固水位（安全修复 P2：防重启/兜底重复巩固）''',
    'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
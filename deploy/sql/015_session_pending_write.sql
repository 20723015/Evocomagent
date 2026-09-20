-- 015：写操作两阶段协议草稿（能力补全计划·P1-2）
--
-- 背景：退款提交的「用户确认」原先只写在 skills/process-return/SKILL.md 的
-- 提示词里，属模型自律——LLM 总可能无视提示词直接提交写操作。政策执行必须
-- 发生在工具内部（LangGraph 官方客服教程的硬原则）。
--
-- 本迁移：sessions 增加 pending_write_json（可空草稿）
--   {tool, client_request_id, arguments, display, created_at, confirmed_turn}
-- 语义：首次写调用不落库、只登记草稿并请求用户确认；下一轮判定为确认后，
--       复用草稿的 client_request_id（申请级幂等键）真正执行写；取消或
--       reset 时清空。遗留非空 = 有一个待用户确认的写操作。
--       纯可空列，旧行自动为 NULL（等价「无待确认写」），无需回填。
-- schema_migrations 记录由 migrate_db.py 写入（本文件只含 DDL）。

ALTER TABLE sessions
    ADD COLUMN pending_write_json TEXT NULL COMMENT '待确认写草稿 JSON（非空=有待用户确认的写操作）';

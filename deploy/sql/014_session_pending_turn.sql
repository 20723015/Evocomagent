-- 014：会话掉线恢复草稿（记忆系统重构·Step6）
--
-- 背景：一轮对话的 user 消息原先只在轮末收尾时随整轮一起落库；若 Pod 在
-- ReAct 中途被 kill（发布/OOM/超时），用户消息与已生成内容全部丢失，客户端
-- 重连后无从判断「上一条是否已被处理」。
--
-- 本迁移：sessions 增加 pending_turn_json（可空草稿标记）
--   {turn_id, user_message, started_at}
-- 语义：轮次开始随 user 消息同事务写入；成功收尾时清空。
--       遗留非空 = 上次回复未完成（API 带回该字段，客户端可提示重发）。
--       纯可空列，旧行自动为 NULL（等价「无草稿」），无需回填。
-- schema_migrations 记录由 migrate_db.py 写入（本文件只含 DDL）。

ALTER TABLE sessions
    ADD COLUMN pending_turn_json TEXT NULL COMMENT '进行中轮次草稿 JSON（非空=上次回复未完成）';

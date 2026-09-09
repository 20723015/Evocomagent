-- 006：memory_jobs.session_uuid（Review 修复：异步记忆可靠性）
-- 任务与创建时的会话实例绑定：reset（同 session_id 重建）后，旧任务按
-- obsolete 处理，绝不读取新会话消息。唯一键保持 session_key+through_seq
-- 不变；存量行从 sessions 回填。
-- schema_migrations 记录由 migrate_db.py 写入（本文件只含 DDL/回填）。

ALTER TABLE memory_jobs
    ADD COLUMN session_uuid CHAR(32) NOT NULL DEFAULT '' COMMENT '创建任务时的会话实例（reset 后旧任务判 obsolete）';

UPDATE memory_jobs mj
JOIN sessions s ON s.session_key = mj.session_key
SET mj.session_uuid = s.session_uuid
WHERE mj.session_uuid = '';

-- 013：人工知识候选「旧版本已发布」stale 标记（P2-3，推荐口径：标记可见、不自动下架）。
-- 同一 (source, external_conversation_id) 接入更高 source_version 时，该会话
-- 旧版本产出的、仍处于 published 状态的候选置 version_stale=1：
--   - 审核台列表加「新版本已到达」徽标、详情页提示；
--   - 下架动作复用既有 retire 入口，由人工决策（无知识空窗）。
-- version_stale 在替换结算 / retire 结算 / 重审失败重置时清零；不参与
-- 发布校验（stale 不是错误状态，仅提示）。

ALTER TABLE human_knowledge_candidates
    ADD COLUMN version_stale TINYINT(1) NOT NULL DEFAULT 0 AFTER retired_at;

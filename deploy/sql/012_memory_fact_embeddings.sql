-- 记忆系统重构·阶段2.2：记忆嵌入派生存储（与 memory_facts 正本分离）。
-- 向量是 content 的函数（派生数据）：丢失/损坏可按 model + content_hash
-- 经 backfill_embeddings 重建；不引入向量扩展，应用侧暴力余弦
-- （单用户 active 事实 ≤80 量级足够）。

CREATE TABLE IF NOT EXISTS memory_fact_embeddings (
    id           BIGINT       NOT NULL AUTO_INCREMENT,
    user_id      VARCHAR(64)  NOT NULL,
    fact_id      VARCHAR(64)  NOT NULL DEFAULT '',
    model        VARCHAR(128) NOT NULL DEFAULT '' COMMENT '嵌入模型（换版自动失效重算）',
    content_hash VARCHAR(64)  NOT NULL DEFAULT '' COMMENT 'sha256(model+content)，内容寻址',
    vector       MEDIUMTEXT   NOT NULL COMMENT 'JSON 数组（派生数据，可重建）',
    updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_memory_fact_embedding (user_id, fact_id, model),
    KEY idx_memory_embedding_user (user_id)
) ENGINE=InnoDB;

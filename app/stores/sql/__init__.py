"""app.stores.sql：MySQL 正本存储（阶段八）。

- SqlSessionStore：会话元数据 + 消息行式追加（append-only），
  CAS 版本冲突同事务生效；可选 Redis 会话热缓存（write-through）；
- SqlLTMStore：长期记忆行式化（memory_facts / interaction_summaries）；
- outbox：消息入库同事务写出箱，后台同步 ES message_search（可重建）。

Dialect 策略：生产 mysql+pymysql；本地/测试 sqlite（DDL 由 SQLAlchemy
按方言生成，Schema 语义一致——主键自增、唯一约束、索引同构）。
"""

from app.stores.sql.schema import metadata
from app.stores.sql.session_store import SqlSessionStore
from app.stores.sql.memory_store import SqlLTMStore
from app.stores.sql.outbox import sync_outbox_to_es

__all__ = ["metadata", "SqlSessionStore", "SqlLTMStore", "sync_outbox_to_es"]

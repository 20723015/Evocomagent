"""Outbox → ES 同步（阶段八）：MySQL 为正本，ES message_search 是可重建衍生品。

- 消息 INSERT 与 outbox 同事务（见 SqlSessionStore.save）；
- 本模块按序搬运：未同步行 → ES bulk（文档 id = {session_key}:{seq}，幂等）
  → 成功后标记 synced_at；失败标记 sync_error 下轮重试；
- ES 挂掉时本模块只记录错误退出，**不影响 Agent 主流程**（检索侧另有降级）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from app.observability.logging import get_logger
from app.stores.sql.schema import outbox_rows

log = get_logger("app.stores.sql.outbox")


def count_pending_outbox(engine) -> int:
    """5.x：待同步 outbox 条数（积压指标；统计失败返回 0 不阻断）。"""
    from sqlalchemy import func, select

    try:
        with engine.connect() as conn:
            return int(conn.execute(
                select(func.count())
                .select_from(outbox_rows)
                .where(outbox_rows.c.synced_at.is_(None))
            ).scalar() or 0)
    except Exception:  # noqa: BLE001 —— 统计失败按 0 处理
        return 0


def sync_outbox_to_es(engine, es_client, index: str, batch_size: int = 200) -> int:
    """搬运一批未同步 outbox 到 ES，返回处理条数；无待处理返回 0。"""
    from sqlalchemy import select

    try:
        with engine.begin() as conn:
            rows = conn.execute(
                select(outbox_rows).where(outbox_rows.c.synced_at.is_(None))
                .order_by(outbox_rows.c.id).limit(batch_size)
            ).mappings().all()
            if not rows:
                return 0
            actions = []
            for row in rows:
                msg = json.loads(row["payload"])
                user_id = row["session_key"].split("/", 1)[0]
                actions.append({"index": {
                    "_index": index,
                    "_id": f"{row['session_key']}:{row['seq']}",
                }})
                actions.append({
                    "session_key": row["session_key"],
                    "user_id": user_id,  # 归属过滤：坐席/质检只能搜到权限内用户
                    "seq": row["seq"],
                    "role": msg.get("role", ""),
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls"),
                    "ts": row["created_at"].isoformat() if row["created_at"] else None,
                })
            resp = es_client.bulk(operations=actions, index=index,
                                  refresh=False)
            failed = resp.get("errors", False)
            now = datetime.now()
            if failed:
                # 失败原因落 sync_error 列（审计/排查可见），synced_at 留空下轮重试
                n_failed = sum(
                    1 for item in resp.get("items", [])
                    if any(v.get("status", 200) >= 300 for v in item.values())
                )
                err_detail = f"es bulk errors=true（{n_failed} 项失败）"
                for row in rows:
                    conn.execute(
                        outbox_rows.update().where(outbox_rows.c.id == row["id"]).values(
                            synced_at=None, sync_error=err_detail,
                        )
                    )
                log.warning("outbox→ES 批量同步存在失败项: %s，下轮重试", err_detail)
                return 0
            for row in rows:
                conn.execute(
                    outbox_rows.update().where(outbox_rows.c.id == row["id"]).values(
                        synced_at=now, sync_error=None,
                    )
                )
            return len(rows)
    except Exception as e:  # noqa: BLE001 —— ES 故障：保留 outbox，下轮重试
        log.warning("outbox→ES 同步失败: %s（正本不受影响，可重建）", e)
        return 0


def ensure_message_index(es_client, index: str) -> None:
    """消息检索索引（若不存在）：keyword 字段聚合/过滤 + standard analyzer 全文。"""
    exists = es_client.indices.exists(index=index)
    if exists:
        return
    es_client.indices.create(
        index=index,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={
            "properties": {
                "session_key": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "seq": {"type": "integer"},
                "role": {"type": "keyword"},
                "content": {"type": "text"},
                "tool_calls": {"type": "object", "enabled": False},
                "ts": {"type": "date"},
            }
        },
    )

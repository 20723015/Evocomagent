"""outbox_admin.py：Outbox dead-letter 查询与重放 CLI（修复计划·二）。

用法：
    python -m app.scripts.outbox_admin list [--kind message|delete|all] [--limit 50]
    python -m app.scripts.outbox_admin replay --kind message --id 12
    python -m app.scripts.outbox_admin replay --kind delete --all

重放语义：把 dead_letter 行复位为 pending（attempts=0、error 清空、租约清空），
交由 outbox worker 正常重试。只允许重放 dead_letter 状态，避免误动进行中的行。
"""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import select, update

from app.stores.sql.outbox import _error_column
from app.stores.sql.schema import message_delete_outbox, outbox_rows


def _tables(kind: str):
    if kind == "message":
        return [("message", outbox_rows)]
    if kind == "delete":
        return [("delete", message_delete_outbox)]
    return [("message", outbox_rows), ("delete", message_delete_outbox)]


def _cmd_list(engine, kind: str, limit: int) -> int:
    out = []
    for label, table in _tables(kind):
        with engine.connect() as conn:
            rows = conn.execute(
                select(table).where(table.c.status == "dead_letter")
                .order_by(table.c.id).limit(limit)
            ).mappings().all()
        for row in rows:
            item = {"kind": label, "id": row["id"]}
            for key in ("session_key", "session_uuid", "seq", "attempts",
                        "dead_lettered_at", "finished_at", "created_at"):
                if key in row:
                    value = row[key]
                    item[key] = value.isoformat() if hasattr(value, "isoformat") else value
            err = row.get(_error_column(table))
            item["error"] = err
            out.append(item)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_replay(engine, kind: str, row_id: int | None, replay_all: bool) -> int:
    total = 0
    for label, table in _tables(kind):
        where = [table.c.status == "dead_letter"]
        if not replay_all:
            if row_id is None:
                print(f"[{label}] 需要 --id 或 --all", file=sys.stderr)
                return 2
            where.append(table.c.id == row_id)
        with engine.begin() as conn:
            result = conn.execute(
                update(table).where(*where).values(
                    status="pending", attempts=0, next_run_at=None,
                    lease_owner="", lease_token="", lease_until=None,
                    **{_error_column(table): None},
                )
            )
        count = int(result.rowcount or 0)
        total += count
        print(f"[{label}] 已重放 {count} 条 dead-letter → pending")
    return 0 if total or replay_all else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Outbox dead-letter 管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出 dead-letter")
    p_list.add_argument("--kind", choices=["message", "delete", "all"], default="all")
    p_list.add_argument("--limit", type=int, default=50)

    p_replay = sub.add_parser("replay", help="重放 dead-letter → pending")
    p_replay.add_argument("--kind", choices=["message", "delete", "all"], default="message")
    p_replay.add_argument("--id", type=int, default=None)
    p_replay.add_argument("--all", action="store_true")

    args = parser.parse_args(argv)

    from app.stores.sql.engine import get_engine

    engine = get_engine()
    if engine is None:
        print("未配置 DB（settings.db_url 为空）", file=sys.stderr)
        return 2

    if args.cmd == "list":
        return _cmd_list(engine, args.kind, max(int(args.limit), 1))
    if args.cmd == "replay":
        return _cmd_replay(engine, args.kind, args.id, args.all)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

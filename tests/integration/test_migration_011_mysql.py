"""真实 MySQL 迁移 011 升级测试（修复计划·三轮 P2-2）。

构造 010 版旧表与存量数据，执行 011，验证：
- session_uuid 回填、status 按 synced_at 回填、历史租约字段清空；
- 旧唯一键 (session_key, seq) 替换为 (session_key, session_uuid, seq)，
  跨 UUID 可复用同一 seq；
- message_delete_outbox 新建（含空 UUID legacy 事件）；
- 重复执行 011 幂等（已应用则跳过）。

无 MySQL 时本地跳过；CI 提供 TEST_MYSQL_URL（compose mysql）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app.scripts.migrate_db import (
    checksum_of,
    ensure_schema_migrations,
    migration_files,
    record_applied,
    run_migrations,
)
from app.stores.sql.schema import sessions

_LEGACY_OUTBOX_DDL = """
CREATE TABLE outbox_rows (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_key VARCHAR(200) NOT NULL,
    seq INT NOT NULL,
    payload TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    synced_at DATETIME NULL,
    sync_error TEXT NULL,
    UNIQUE KEY uq_outbox_session_seq (session_key, seq)
) ENGINE=InnoDB
"""


def _mysql_url() -> str:
    return os.environ.get(
        "TEST_MYSQL_URL",
        "mysql+pymysql://ecom:ecom@localhost:13306/ecom?charset=utf8mb4",
    )


def _engine_or_skip():
    try:
        engine = create_engine(_mysql_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # noqa: BLE001
        msg = f"真实 MySQL 不可用: {exc}"
        if os.environ.get("TEST_MYSQL_URL") or os.environ.get("CI", "").lower() == "true":
            pytest.fail(msg)
        pytest.skip(msg)


def test_migration_011_upgrades_legacy_outbox_mysql(tmp_path):
    engine = _engine_or_skip()
    root = Path(__file__).resolve().parents[2]
    sql_dir = root / "deploy" / "sql"

    # ---- 构造 010 版旧状态 ----
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS message_delete_outbox"))
        conn.execute(text("DROP TABLE IF EXISTS outbox_rows"))
        conn.execute(text("DROP TABLE IF EXISTS schema_migrations"))
    sessions.drop(engine, checkfirst=True)
    sessions.create(engine)
    with engine.begin() as conn:
        conn.execute(text(_LEGACY_OUTBOX_DDL))
        conn.execute(text(
            "INSERT INTO sessions (session_key, user_id, session_uuid, version, "
            "consolidated_len, status) VALUES ('u1/s1', 'u1', 'uuid-1', 0, 0, 'active')"
        ))
        # 一条已同步（synced_at 非空）、一条未同步
        conn.execute(text(
            "INSERT INTO outbox_rows (session_key, seq, payload, synced_at) "
            "VALUES ('u1/s1', 1, '{}', NOW())"
        ))
        conn.execute(text(
            "INSERT INTO outbox_rows (session_key, seq, payload, synced_at) "
            "VALUES ('u1/s1', 2, '{}', NULL)"
        ))
    # 标记 001..010 已应用（真实 checksum），使 run_migrations 只执行 011
    with engine.connect() as conn:
        ensure_schema_migrations(conn)
        conn.commit()
        for version, path in migration_files(sql_dir):
            if version <= 10:
                record_applied(conn, version, checksum_of(path))
        conn.commit()

    # ---- 执行迁移（含 011） ----
    with engine.connect() as conn:
        result = run_migrations(
            lambda sql: conn.execute(text(sql)), sql_dir, conn=conn,
        )
        conn.commit()
    assert result["applied"] == [11]

    # ---- 回填与租约清空 ----
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT seq, session_uuid, status, lease_owner, lease_token, lease_until "
            "FROM outbox_rows ORDER BY seq"
        )).mappings().all()
    assert rows[0]["status"] == "done"      # synced_at 非空 → done
    assert rows[1]["status"] == "pending"   # 未同步 → pending
    assert all(r["session_uuid"] == "uuid-1" for r in rows)
    assert all(r["lease_owner"] == "" and r["lease_token"] == "" for r in rows)
    assert all(r["lease_until"] is None for r in rows)

    # ---- 唯一键替换（列顺序 = session_key, session_uuid, seq） ----
    with engine.connect() as conn:
        idx = conn.execute(text(
            "SELECT column_name FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = 'outbox_rows' "
            "AND index_name = 'uq_outbox_session_seq' ORDER BY seq_in_index"
        )).scalars().all()
    assert list(idx) == ["session_key", "session_uuid", "seq"]

    # ---- 跨 UUID 复用同一 seq 可写 ----
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO outbox_rows (session_key, seq, payload, session_uuid, status) "
            "VALUES ('u1/s1', 1, '{}', 'uuid-2', 'pending')"
        ))
        # 新删除表存在，且允许空 UUID（legacy）事件
        conn.execute(text(
            "INSERT INTO message_delete_outbox (session_key, session_uuid, status) "
            "VALUES ('u1/s9', '', 'pending')"
        ))

    # ---- 重复执行幂等 ----
    with engine.connect() as conn:
        again = run_migrations(
            lambda sql: conn.execute(text(sql)), sql_dir, conn=conn,
        )
        conn.commit()
    assert again["applied"] == []

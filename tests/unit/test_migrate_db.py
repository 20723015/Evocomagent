"""2.9 数据库迁移执行器 + 生产 schema 校验测试。

迁移 SQL 本体是 MySQL 语法（无法在 sqlite 执行），框架逻辑
（顺序/幂等/checksum 校验/版本记录）用 sqlite 验证；
execute_sql 在测试里用记录型替身，不真正执行 DDL。
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from app.scripts.migrate_db import (
    applied_versions,
    checksum_of,
    ensure_schema_migrations,
    migration_files,
    record_applied,
    run_migrations,
)
from app.stores.sql.engine import EXPECTED_SCHEMA_VERSION, _verify_schema_version


def _fake_sql_dir(tmp_path, versions=(1, 2, 3, 4)):
    for v in versions:
        (tmp_path / f"{v:03d}_migration.sql").write_text(
            f"-- migration {v}\nCREATE TABLE t{v} (id INT);\n",
            encoding="utf-8",
        )
    return tmp_path


def test_migration_files_sorted_and_filtered(tmp_path):
    (tmp_path / "001_a.sql").write_text("x", encoding="utf-8")
    (tmp_path / "004_evidence.sql").write_text("x", encoding="utf-8")
    (tmp_path / "notes.sql").write_text("x", encoding="utf-8")  # 非编号忽略
    files = migration_files(tmp_path)
    assert [v for v, _ in files] == [1, 4]


def test_run_migrations_applies_in_order_then_idempotent(tmp_path):
    sql_dir = _fake_sql_dir(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    executed = []

    with engine.connect() as conn:
        result = run_migrations(
            lambda sql: executed.append(sql[:30]),
            sql_dir, conn=conn,
        )
        assert result["applied"] == [1, 2, 3, 4]
        assert result["skipped"] == []
        assert len(executed) == 4

        # 重复运行：全部跳过（幂等）
        result2 = run_migrations(
            lambda sql: executed.append(sql[:30]),
            sql_dir, conn=conn,
        )
        assert result2["applied"] == []
        assert result2["skipped"] == [1, 2, 3, 4]
        assert len(executed) == 4  # 未再执行任何 SQL


def test_run_migrations_partial_apply(tmp_path):
    """旧库已有 1-3，只补 4。"""
    sql_dir = _fake_sql_dir(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        ensure_schema_migrations(conn)
        conn.commit()
        for v in (1, 2, 3):
            record_applied(conn, v, checksum_of(sql_dir / f"{v:03d}_migration.sql"))
        conn.commit()
        executed = []
        result = run_migrations(
            lambda sql: executed.append(sql),
            sql_dir, conn=conn,
        )
        assert result["applied"] == [4]
        assert result["skipped"] == [1, 2, 3]
        assert len(executed) == 1  # 只执行了 004 的语句


def test_checksum_change_fails(tmp_path):
    sql_dir = _fake_sql_dir(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        ensure_schema_migrations(conn)
        conn.commit()
        record_applied(conn, 2, "old-checksum")
        conn.commit()
        with pytest.raises(RuntimeError, match="checksum 变化"):
            run_migrations(lambda sql: None, sql_dir, conn=conn)


def test_applied_versions_roundtrip(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        ensure_schema_migrations(conn)
        conn.commit()
        record_applied(conn, 1, "abc")
        record_applied(conn, 4, "def")
        conn.commit()
        assert applied_versions(conn) == {1: "abc", 4: "def"}


# ============================================================
# 生产引擎：只校验版本、不自动 DDL
# ============================================================
def test_prod_verify_fails_without_schema_migrations(tmp_path):
    """生产缺少 schema_migrations → 拒绝启动。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.sqlite'}")
    with pytest.raises(RuntimeError, match="schema_migrations"):
        _verify_schema_version(engine)


def test_prod_verify_fails_when_schema_behind(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE TABLE schema_migrations (version INT PRIMARY KEY, "
            f"checksum VARCHAR(64), applied_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        ))
        conn.execute(text(
            "INSERT INTO schema_migrations (version, checksum) VALUES (2, 'x')"
        ))
    with pytest.raises(RuntimeError, match=f"{EXPECTED_SCHEMA_VERSION}"):
        _verify_schema_version(engine)


def test_prod_verify_passes_when_schema_current(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'current.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE schema_migrations (version INT PRIMARY KEY, "
            "checksum VARCHAR(64), applied_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        ))
        conn.execute(text(
            "INSERT INTO schema_migrations (version, checksum) "
            f"VALUES ({EXPECTED_SCHEMA_VERSION}, 'x')"
        ))
    # 版本达标 → 不抛错
    _verify_schema_version(engine)


def test_prod_engine_skips_auto_ddl(tmp_path, monkeypatch, reset_settings):
    """APP_ENV=prod：create_all/_ensure_upgrades 均不执行（只校验版本）。"""
    from app.config.settings import settings

    monkeypatch.setattr(settings, "app_env", "prod")
    engine = create_engine(f"sqlite:///{tmp_path / 'prod.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE schema_migrations (version INT PRIMARY KEY, "
            "checksum VARCHAR(64), applied_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        ))
        conn.execute(text(
            "INSERT INTO schema_migrations (version, checksum) "
            f"VALUES ({EXPECTED_SCHEMA_VERSION}, 'x')"
        ))
    # 直接调 _build：应只校验、不建业务表
    from app.stores.sql import engine as engine_mod

    built = engine_mod._build(f"sqlite:///{tmp_path / 'prod.sqlite'}")
    from sqlalchemy import inspect

    inspector = inspect(built)
    assert not inspector.has_table("sessions")  # 未执行 create_all
    assert not inspector.has_table("memory_facts")
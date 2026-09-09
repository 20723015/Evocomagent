"""数据库引擎（阶段八）：按 settings.db_url 创建，无 db_url 时返回 None（沿用文件/Redis）。"""

from __future__ import annotations

import threading

from app.config.settings import settings
from app.stores.sql.schema import metadata

_engine = _UNSET = object()
_engine_lock = threading.Lock()


def get_engine():
    """pod 级共享 SQLAlchemy Engine；未配置 db_url 返回 None（降级现有存储）。

    注意：初始值必须与 _UNSET 同对象（None 会撞「不可用缓存」分支，
    导致 db_url 配置后永远不构建引擎——阶段八曾因此静默失效）。
    """
    global _engine
    if _engine is _UNSET:
        with _engine_lock:
            if _engine is _UNSET:
                _engine = _build(settings.db_url)
    return None if _engine is _UNSET or _engine is None else _engine


def _build(url: str):
    if not url:
        return None
    from sqlalchemy import create_engine

    engine = create_engine(
        url,
        pool_pre_ping=True,
        # sqlite 文件库需要检查线程复用（服务在多个线程里存取）
        connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
    )
    if settings.app_env.lower() == "prod":
        # 2.9 生产：只校验 schema 版本，绝不自动 DDL（多 Pod 并发 DDL 会冲突）；
        # schema 落后 → 启动失败，要求先跑 migrate_db（Helm migration Job）
        _verify_schema_version(engine)
        return engine
    metadata.create_all(engine)
    _ensure_upgrades(engine)
    return engine


# 期望的 schema 版本（= deploy/sql 最高迁移编号；009=human_knowledge_hardening）
EXPECTED_SCHEMA_VERSION = 10


def _verify_schema_version(engine) -> None:
    """生产启动校验：schema_migrations 存在且版本达标；否则拒绝启动。

    只读校验：不建表、不 DDL——由 migrate_db.py（Helm migration Job）负责。
    MySQL 方言才校验；sqlite 开发库不阻塞（prod 不会配 sqlite）。
    """
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        if not inspector.has_table("schema_migrations"):
            raise RuntimeError(
                "生产数据库缺少 schema_migrations 表：请先执行 "
                "`python -m app.scripts.migrate_db`（或 Helm migration Job）"
            )
        rows = (
            engine.connect()
            .execute(text("SELECT MAX(version) FROM schema_migrations"))
            .scalar()
        )
        latest = int(rows or 0)
        if latest < EXPECTED_SCHEMA_VERSION:
            raise RuntimeError(
                f"数据库 schema 版本落后（{latest} < {EXPECTED_SCHEMA_VERSION}）："
                "请先执行 `python -m app.scripts.migrate_db`，禁止带旧 schema 启动"
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"数据库 schema 版本校验失败（拒绝启动）: {e}") from e


def _ensure_upgrades(engine) -> None:
    """轻量兼容迁移；正式环境仍应预执行 deploy/sql 迁移脚本。"""
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        changed = False
        with engine.begin() as conn:
            if inspector.has_table("sessions"):
                session_columns = {c["name"] for c in inspector.get_columns("sessions")}
                if "consolidated_len" not in session_columns:
                    conn.execute(
                        text(
                            "ALTER TABLE sessions ADD COLUMN consolidated_len "
                            "INT NOT NULL DEFAULT 0"
                        )
                    )
                    changed = True

            if inspector.has_table("memory_facts"):
                fact_columns = {
                    c["name"] for c in inspector.get_columns("memory_facts")
                }
                additions = {
                    "fact_id": "VARCHAR(64) NOT NULL DEFAULT ''",
                    "fact_key": "VARCHAR(128) NOT NULL DEFAULT ''",
                    "status": "VARCHAR(16) NOT NULL DEFAULT 'active'",
                    "confidence": "DOUBLE NOT NULL DEFAULT 1.0",
                    "supersedes_id": "VARCHAR(64) NOT NULL DEFAULT ''",
                    # SQLite cannot add a column with CURRENT_TIMESTAMP default.
                    "updated_at": "DATETIME NULL",
                    # 2.4：用户原话依据（可审计），v2 旧库轻量补列
                    "evidence": "TEXT NOT NULL DEFAULT ''",
                }
                for name, ddl in additions.items():
                    if name not in fact_columns:
                        conn.execute(
                            text(f"ALTER TABLE memory_facts ADD COLUMN {name} {ddl}")
                        )
                        changed = True

            if inspector.has_table("interaction_summaries"):
                summary_columns = {
                    c["name"] for c in inspector.get_columns("interaction_summaries")
                }
                if "source_session" not in summary_columns:
                    conn.execute(
                        text(
                            "ALTER TABLE interaction_summaries ADD COLUMN "
                            "source_session VARCHAR(64) NOT NULL DEFAULT ''"
                        )
                    )
                    changed = True
        if not changed:
            return
        import logging

        logging.getLogger("app.stores.sql").warning(
            "已完成数据库兼容性字段检查；生产请以 deploy/sql 迁移脚本留档",
        )
    except Exception as e:
        raise RuntimeError(
            "数据库兼容迁移失败；请执行 deploy/sql/002_migration.sql 与 "
            "deploy/sql/003_memory_versioning.sql 后重启"
        ) from e


def set_engine_for_test(engine) -> None:
    """测试注入（sqlite 内存/文件 engine）；传 None 恢复探测。"""
    global _engine
    _engine = engine


def reset_engine() -> None:
    global _engine
    _engine = _UNSET

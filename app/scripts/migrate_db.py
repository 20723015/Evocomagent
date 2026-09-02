"""migrate_db.py：数据库迁移执行器（2.9）。

- MySQL advisory lock（GET_LOCK）串行迁移（多 Pod/多实例不并发 DDL）；
- schema_migrations(version, checksum, applied_at) 记录已执行迁移；
- 按编号执行 deploy/sql/{NNN}_*.sql（001 → 004），重复运行幂等（跳过已执行）；
- 已执行迁移 checksum 变化 → 直接失败（禁止篡改历史迁移）；
- 新库由最新 schema 初始化（001→…→004 依次执行）；
  旧库跳过已执行部分，只补缺失迁移。

用法：
    python -m app.scripts.migrate_db                 # 使用 settings.db_url
    python -m app.scripts.migrate_db --db-url "mysql+pymysql://user:pwd@host:3306/ecom"
    python -m app.scripts.migrate_db --sql-dir deploy/sql
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import create_engine, text

log = logging.getLogger("app.scripts.migrate_db")

MIGRATION_RE = re.compile(r"^(\d{3})_.+\.sql$")
LOCK_NAME = "ecom_schema_migrations"
LOCK_TIMEOUT = 30  # 等待迁移锁的秒数（>= 单次迁移最长时间）


def split_sql_statements(sql_text: str) -> list[str]:
    """按分号拆分迁移 SQL（行尾分号边界；忽略空语句与纯注释块）。

    约束：迁移文件里的字符串常量不含分号（004 的 'SELECT 1' 无分号），
    语句均以分号结束——满足本仓库迁移脚本即可。
    """
    statements: list[str] = []
    buf: list[str] = []
    for line in sql_text.splitlines():
        buf.append(line)
        if line.rstrip().endswith(";"):
            statements.append("\n".join(buf))
            buf = []
    if buf and "".join(buf).strip():
        statements.append("\n".join(buf))
    out = []
    for stmt in statements:
        stripped = stmt.strip()
        if not stripped:
            continue
        # 纯注释块（-- 开头）跳过
        if all(ln.strip().startswith("--") or not ln.strip()
               for ln in stripped.splitlines()):
            continue
        out.append(stripped)
    return out


def migration_files(sql_dir: str | Path) -> list[tuple[int, Path]]:
    """扫描编号迁移文件，按编号升序；非 3 位编号文件忽略。"""
    out: list[tuple[int, Path]] = []
    for p in sorted(Path(sql_dir).glob("*.sql")):
        m = MIGRATION_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def checksum_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_schema_migrations(conn) -> None:
    """幂等建表（MySQL 8：IF NOT EXISTS；sqlite 测试方言去 ENGINE 子句）。"""
    dialect = conn.dialect.name if hasattr(conn, "dialect") else "mysql"
    engine_clause = "" if dialect == "sqlite" else " ENGINE=InnoDB"
    conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INT          NOT NULL PRIMARY KEY,
            checksum   VARCHAR(64)  NOT NULL,
            applied_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
        ){engine_clause}
    """))


def applied_versions(conn) -> dict[int, str]:
    """已执行迁移：{version: checksum}。"""
    rows = conn.execute(
        text("SELECT version, checksum FROM schema_migrations ORDER BY version")
    ).all()
    return {int(r[0]): str(r[1]) for r in rows}


def record_applied(conn, version: int, checksum: str) -> None:
    """写入已执行记录（按方言：mysql upsert / sqlite replace，幂等）。"""
    dialect = conn.dialect.name if hasattr(conn, "dialect") else "mysql"
    if dialect == "sqlite":
        conn.execute(text(
            "INSERT OR REPLACE INTO schema_migrations (version, checksum) "
            "VALUES (:v, :c)"
        ), {"v": version, "c": checksum})
    else:
        conn.execute(text(
            "INSERT INTO schema_migrations (version, checksum) VALUES (:v, :c) "
            "ON DUPLICATE KEY UPDATE checksum = VALUES(checksum)"
        ), {"v": version, "c": checksum})


def run_migrations(
    execute_sql: Callable[[str], None],
    sql_dir: str | Path,
    *,
    conn=None,
    verbose: bool = False,
) -> dict:
    """顺序执行未迁移文件；返回 {"applied": [..], "skipped": [..]}。

    execute_sql(text)：在**同一连接**上执行整段 SQL（多语句）。
    conn：提供 schema_migrations 读写；None 时 execute_sql 需自行处理
    （测试注入）。
    """
    files = migration_files(sql_dir)
    applied: dict[int, str] = {}
    if conn is not None:
        ensure_schema_migrations(conn)
        conn.commit()
        applied = applied_versions(conn)

    result = {"applied": [], "skipped": []}
    for version, path in files:
        checksum = checksum_of(path)
        if version in applied:
            if applied[version] != checksum:
                raise RuntimeError(
                    f"迁移 {version:03d}（{path.name}）已执行但 checksum 变化："
                    f"旧 {applied[version][:12]}… ≠ 新 {checksum[:12]}…；"
                    "禁止修改已执行的历史迁移，请新建后续迁移"
                )
            result["skipped"].append(version)
            if verbose:
                log.info("跳过已执行迁移 %03d（%s）", version, path.name)
            continue
        if verbose:
            log.info("执行迁移 %03d（%s）", version, path.name)
        for stmt in split_sql_statements(path.read_text(encoding="utf-8")):
            execute_sql(stmt)
        if conn is not None:
            record_applied(conn, version, checksum)
            conn.commit()
        result["applied"].append(version)
    return result


def acquire_lock(conn, timeout: int = LOCK_TIMEOUT) -> None:
    """GET_LOCK 串行（同连接内释放）；拿不到直接失败（fail-closed）。"""
    got = conn.execute(text("SELECT GET_LOCK(:name, :timeout)"),
                       {"name": LOCK_NAME, "timeout": timeout}).scalar()
    if got != 1:
        raise RuntimeError(
            f"无法取得迁移锁 {LOCK_NAME}（另一实例正在迁移？）等待 {timeout}s 超时"
        )


def release_lock(conn) -> None:
    try:
        conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": LOCK_NAME})
    except Exception:  # noqa: BLE001 —— 释放失败由连接关闭兜底
        pass


def migrate(db_url: str, sql_dir: str | Path, verbose: bool = False) -> dict:
    """生产入口：MySQL advisory lock + 事务内迁移。"""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        acquire_lock(conn)
        try:
            result = run_migrations(
                lambda sql: conn.execute(text(sql)),
                sql_dir, conn=conn, verbose=verbose,
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            release_lock(conn)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="数据库迁移执行器（2.9）")
    parser.add_argument("--db-url", default="",
                        help="SQLAlchemy URL；缺省用 settings.db_url")
    parser.add_argument("--sql-dir", default="deploy/sql",
                        help="迁移 SQL 目录（默认 deploy/sql）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db_url = args.db_url
    if not db_url:
        from app.config.settings import settings

        db_url = settings.db_url
    if not db_url:
        log.error("未配置 db_url（--db-url 或 settings.db_url）")
        return 2
    sql_dir = Path(args.sql_dir)
    if not sql_dir.is_dir():
        log.error("迁移目录不存在: %s", sql_dir)
        return 2
    try:
        result = migrate(db_url, sql_dir, verbose=args.verbose)
    except Exception as e:  # noqa: BLE001 —— CLI 顶层：打印并失败退出
        log.error("迁移失败: %s", e)
        return 1
    log.info("迁移完成：应用 %s 个，跳过 %s 个",
             len(result["applied"]), len(result["skipped"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

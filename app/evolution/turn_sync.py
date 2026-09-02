"""turn_sync.py：S3 turns 归档 → 本地矿工目录的幂等同步（多 Pod 静默挖空修复）。

背景：server 侧在 turns_archive_backend=s3 时只写对象存储（recorder 无本地副本），
CLI/miner 只读本地 turns 目录 —— 多 Pod 部署下 run_evolution 会静默挖空。

- build_turns_archive：共享工厂，复用 turns_archive_backend/s3_bucket/
  s3_endpoint_url 三元组构造 S3ObjectStore（server deps 与 CLI 同一构造路径）。
- sync_turns_from_archive：store.list("turns/")（字典序=时间序），游标
  state/sync_cursor.json 记上次最大 key，只处理新增；本地已存在跳过（幂等）；
  下载走 tmp + os.replace 原子写（与 recorder 同模式）。最近 SYNC_WINDOW_DAYS
  天的 key 不受游标限制（时钟回拨/迟到写入保护）。
- ObjectStoreUnavailable → RuntimeError fail-closed：宁可报错也绝不静默空跑。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.config.settings import settings

SYNC_WINDOW_DAYS = 3  # 游标之外仍回查的最近天数（时钟回拨/迟到写入保护）


def build_turns_archive() -> Optional[object]:
    """按配置构造 turns 归档对象存储；非 s3 后端（或未配 bucket）返回 None（本地落盘）。

    S3 初始化失败（boto3 缺失/连接失败）抛 ObjectStoreUnavailable：
    server deps 捕获后降级本地，CLI 的调用方应 fail-closed 拒绝空跑。
    """
    if settings.turns_archive_backend != "s3" or not settings.s3_bucket:
        return None
    from app.stores.object_store import S3ObjectStore

    return S3ObjectStore(
        settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
    )


def _read_cursor(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("last_key", ""))
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        return ""


def _write_cursor(path: Path, last_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"last_key": last_key}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sync_turns_from_archive(store, turns_dir, state_dir) -> int:
    """把对象存储 turns/ 下的新增记录同步到本地 turns_dir；返回本次同步条数。

    - key ``turns/YYYYMMDD/<turn_id>.json`` → 本地 ``<turns_dir>/YYYYMMDD/<turn_id>.json``。
    - 幂等：本地已存在直接跳过（连同推进游标）；只下载游标之后的新增 key。
    - 任何 ObjectStoreUnavailable → RuntimeError（fail-closed，杜绝静默挖空）。
    """
    from app.stores.object_store import ObjectStoreUnavailable

    turns_dir = Path(turns_dir)
    cursor_path = Path(state_dir) / "sync_cursor.json"
    cursor = _read_cursor(cursor_path)

    try:
        keys = sorted(store.list("turns/"))
    except ObjectStoreUnavailable as e:
        raise RuntimeError(
            f"S3 turns 归档不可用（{e}），拒绝静默空跑："
            "请检查 TURNS_ARCHIVE_BACKEND/S3_BUCKET/S3_ENDPOINT_URL 配置，"
            "或改用 TURNS_ARCHIVE_BACKEND=local"
        ) from e

    cutoff = (datetime.now() - timedelta(days=SYNC_WINDOW_DAYS)).strftime("%Y%m%d")
    synced = 0
    last = cursor
    for key in keys:
        parts = key.split("/")
        date_part = parts[1] if len(parts) >= 3 else ""
        # 游标跳过 + 窗口例外：最近 N 天的 key 即便早于游标也回查
        # （多 Pod 时钟回拨/迟到写入会把 key 落在旧日期目录，纯游标会永久漏掉）
        if cursor and key <= cursor and not (date_part and date_part >= cutoff):
            continue
        rel = key[len("turns/"):] if key.startswith("turns/") else key
        target = turns_dir / rel
        if target.exists():
            if key > last:  # 幂等跳过；游标只前进（窗口回查可能遇到更小的 key）
                last = key
            continue
        try:
            data = store.get(key)
        except ObjectStoreUnavailable as e:
            raise RuntimeError(
                f"S3 turns 下载失败（{key}: {e}），拒绝静默空跑，请检查 S3 配置"
            ) from e
        if data is None:
            continue  # 远端已删：不推进游标也无妨（下次仍会检查）
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        if key > last:
            last = key
        synced += 1

    if last != cursor:
        _write_cursor(cursor_path, last)
    return synced

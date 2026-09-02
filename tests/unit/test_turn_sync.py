"""turn_sync：S3 turns 归档 → 本地矿工目录的幂等同步（多 Pod 静默挖空修复）。"""

from __future__ import annotations

import json

import pytest

from app.evolution.turn_sync import build_turns_archive, sync_turns_from_archive
from app.stores.object_store import (
    LocalDirObjectStore,
    ObjectStoreUnavailable,
    S3ObjectStore,
)


def _put(store: LocalDirObjectStore, day: str, turn_id: str, question: str = "能退吗") -> str:
    key = f"turns/{day}/{turn_id}.json"
    store.put(key, json.dumps(
        {"turn_id": turn_id, "session_id": "s", "question": question},
        ensure_ascii=False,
    ).encode("utf-8"))
    return key


def test_sync_downloads_new_turns(tmp_path):
    """新增 key 全量下载到本地（tmp + os.replace 原子写），返回同步条数。"""
    remote = LocalDirObjectStore(tmp_path / "archive")
    _put(remote, "20260828", "t1")
    _put(remote, "20260828", "t2")

    turns = tmp_path / "turns"
    state = tmp_path / "state"
    synced = sync_turns_from_archive(remote, turns, state)
    assert synced == 2
    assert (turns / "20260828" / "t1.json").exists()
    assert (turns / "20260828" / "t2.json").exists()
    assert (turns / "20260828" / "t2.json").read_text(encoding="utf-8").startswith("{")
    # 游标推进到最大 key
    cursor = json.loads((state / "sync_cursor.json").read_text(encoding="utf-8"))
    assert cursor["last_key"] == "turns/20260828/t2.json"
    # 无残留 tmp 文件
    assert not list((turns / "20260828").glob("*.tmp"))


def test_sync_idempotent_skip_existing(tmp_path):
    """本地已存在 → 跳过但推进游标；重复同步返回 0 条。"""
    remote = LocalDirObjectStore(tmp_path / "archive")
    _put(remote, "20260828", "t1")
    turns = tmp_path / "turns"
    state = tmp_path / "state"
    assert sync_turns_from_archive(remote, turns, state) == 1
    # 远程追加一条 → 只同步新增
    _put(remote, "20260828", "t2")
    assert sync_turns_from_archive(remote, turns, state) == 1
    assert not (turns / "20260828" / "t1.json").stat().st_size == 0  # t1 未被覆写
    assert sync_turns_from_archive(remote, turns, state) == 0  # 幂等：第三次 0 条


def test_sync_cursor_resume(tmp_path):
    """游标只处理新增；清空本地后游标仍生效（已存在的 key 不重复下载）。

    日期用窗口外的旧日期（SYNC_WINDOW_DAYS 之外），隔离「游标跳过」语义。
    """
    remote = LocalDirObjectStore(tmp_path / "archive")
    _put(remote, "20200101", "t1")
    _put(remote, "20200101", "t2")
    turns = tmp_path / "turns"
    state = tmp_path / "state"
    assert sync_turns_from_archive(remote, turns, state) == 2

    # 手工删除本地 t1 → 游标之后的不再下载（t1 <= cursor 且在窗口外，被跳过）
    (turns / "20200101" / "t1.json").unlink()
    assert sync_turns_from_archive(remote, turns, state) == 0
    assert not (turns / "20200101" / "t1.json").exists()


def test_sync_recent_backdated_key_still_synced(tmp_path):
    """时钟回拨/迟到写入：落在游标之前但日期在近 N 天内的 key 仍被同步；游标不回退。"""
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y%m%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    remote = LocalDirObjectStore(tmp_path / "archive")
    _put(remote, today, "t1")
    turns = tmp_path / "turns"
    state = tmp_path / "state"
    assert sync_turns_from_archive(remote, turns, state) == 1
    cursor_before = json.loads((state / "sync_cursor.json").read_text(encoding="utf-8"))

    # 迟到写入：key 排序在游标之前，但日期在窗口内 → 仍要同步
    _put(remote, yesterday, "late1")
    assert sync_turns_from_archive(remote, turns, state) == 1
    assert (turns / yesterday / "late1.json").exists()

    # 窗口之外的旧 key（远早于 SYNC_WINDOW_DAYS）→ 依旧跳过
    _put(remote, "20200101", "ancient")
    assert sync_turns_from_archive(remote, turns, state) == 0
    assert not (turns / "20200101" / "ancient.json").exists()

    # 游标只前进：窗口回查不把 last_key 拉回昨天
    cursor_after = json.loads((state / "sync_cursor.json").read_text(encoding="utf-8"))
    assert cursor_after["last_key"] == cursor_before["last_key"]


def test_sync_unavailable_raises(tmp_path):
    """ObjectStoreUnavailable → RuntimeError fail-closed（杜绝静默挖空）。"""

    class BrokenStore(LocalDirObjectStore):
        def list(self, prefix: str = "") -> list[str]:
            raise ObjectStoreUnavailable("bucket 不可达")

    with pytest.raises(RuntimeError, match="S3 turns 归档不可用"):
        sync_turns_from_archive(BrokenStore(tmp_path), tmp_path / "turns",
                                tmp_path / "state")


def test_sync_get_unavailable_raises(tmp_path):
    """list 成功但 get 失败同样 fail-closed。"""

    class BrokenGetStore(LocalDirObjectStore):
        def get(self, key: str):
            raise ObjectStoreUnavailable("read timeout")

    store = BrokenGetStore(tmp_path / "archive")
    _put(store, "20260828", "t1")
    with pytest.raises(RuntimeError, match="下载失败"):
        sync_turns_from_archive(store, tmp_path / "turns", tmp_path / "state")


def test_s3_get_network_error_is_not_reported_as_missing():
    """真实 S3 适配器必须上抛网络错误，供同步层 fail-closed。"""

    class BrokenClient:
        def get_object(self, **kwargs):
            raise TimeoutError("read timeout")

    store = S3ObjectStore.__new__(S3ObjectStore)
    store._client = BrokenClient()
    store._bucket = "bucket"
    with pytest.raises(ObjectStoreUnavailable, match="S3 get 失败"):
        store.get("turns/20260828/t1.json")


def test_s3_get_explicit_404_returns_none():
    """只有明确的对象不存在响应可以转换为 None。"""

    class MissingError(Exception):
        response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }

    class MissingClient:
        def get_object(self, **kwargs):
            raise MissingError("not found")

    store = S3ObjectStore.__new__(S3ObjectStore)
    store._client = MissingClient()
    store._bucket = "bucket"
    assert store.get("turns/20260828/missing.json") is None


def test_build_turns_archive_local_backend_returns_none(monkeypatch):
    from app.config.settings import settings

    monkeypatch.setattr(settings, "turns_archive_backend", "local", raising=False)
    monkeypatch.setattr(settings, "s3_bucket", "", raising=False)
    monkeypatch.setattr(settings, "s3_endpoint_url", "", raising=False)
    assert build_turns_archive() is None


def test_build_turns_archive_s3_backend(tmp_path, monkeypatch):
    from app.config.settings import settings

    try:
        import boto3  # noqa: F401
    except ImportError:
        pytest.skip("本机未装 boto3，跳过 s3 构造路径")

    monkeypatch.setattr(settings, "turns_archive_backend", "s3", raising=False)
    monkeypatch.setattr(settings, "s3_bucket", "my-bucket", raising=False)
    monkeypatch.setattr(settings, "s3_endpoint_url", "http://localhost:9000", raising=False)
    store = build_turns_archive()
    assert store is not None  # 与 deps 构造走同一路径（S3ObjectStore 实例）

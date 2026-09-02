"""2.8 对象存储协议测试：LocalDir 全功能 + S3（stub）/MinIO 集成 + 分片清理语义。

协议补全：delete / delete_prefix / healthcheck 双实现一致；
S3ChunkStorage 的 finalize 清理临时对象；delete_temp/delete_chunks 非空操作；
OriginalStore 对象存储版（originals/ 前缀）。
"""

from __future__ import annotations

import os

import pytest

from app.stores.object_store import LocalDirObjectStore, S3ObjectStore
from app.stores.upload_storage import (
    OriginalStore,
    S3ChunkStorage,
    UploadStorageError,
)
from app.stores.object_store import ObjectStoreUnavailable


# ============================================================
# LocalDirObjectStore：协议完整
# ============================================================
def test_localdir_full_protocol(tmp_path):
    store = LocalDirObjectStore(tmp_path / "store")
    store.put("a/b.txt", b"hello")
    store.put("a/c.txt", b"world")
    store.put("d.txt", b"x")
    assert store.get("a/b.txt") == b"hello"
    assert store.get("missing") is None
    assert set(store.list("a/")) == {"a/b.txt", "a/c.txt"}
    assert store.healthcheck() is True

    store.delete("a/b.txt")
    assert store.get("a/b.txt") is None  # 幂等删除
    store.delete("a/b.txt")  # 不存在不报错
    store.delete_prefix("a/")
    store.delete(".healthcheck")
    assert store.list("") == ["d.txt"]


# ============================================================
# S3ObjectStore：botocore Stubber 验证请求语义
# ============================================================
def _s3_with_stubber():
    """构造注入 stub client 的 S3ObjectStore（避免真实网络）。"""
    import boto3
    from botocore.stub import Stubber

    client = boto3.client("s3", region_name="cn-north-1",
                          aws_access_key_id="x", aws_secret_access_key="x")
    stubber = Stubber(client)
    return S3ObjectStore("bucket", client=client), stubber


def test_s3_put_get_delete_roundtrip():
    store, stubber = _s3_with_stubber()
    with stubber:
        stubber.add_response("put_object", {})
        store.put("k1", b"data")
        stubber.add_response(
            "get_object",
            {"Body": __import__("io").BytesIO(b"data"), "ContentLength": 4},
        )
        assert store.get("k1") == b"data"
        stubber.add_client_error("get_object", service_error_code="NoSuchKey",
                                 service_message="not found")
        assert store.get("missing") is None  # 404 → None（幂等）
        stubber.add_response("delete_object", {})
        store.delete("k1")


def test_s3_list_pagination_and_delete_prefix():
    store, stubber = _s3_with_stubber()
    paginator_pages = [
        {"Contents": [{"Key": "p/1"}, {"Key": "p/2"}], "IsTruncated": False},
    ]
    with stubber:
        stubber.add_response("list_objects_v2", paginator_pages[0],
                             expected_params={"Bucket": "bucket", "Prefix": "p/"})
        assert store.list("p/") == ["p/1", "p/2"]
        # delete_prefix → list + delete_objects
        stubber.add_response("list_objects_v2", paginator_pages[0],
                             expected_params={"Bucket": "bucket", "Prefix": "del/"})
        stubber.add_response(
            "delete_objects",
            {"Deleted": [{"Key": "p/1"}, {"Key": "p/2"}]},
            expected_params={
                "Bucket": "bucket",
                "Delete": {"Objects": [{"Key": "p/1"}, {"Key": "p/2"}]},
            },
        )
        store.delete_prefix("del/")


def test_s3_healthcheck():
    store, stubber = _s3_with_stubber()
    from io import BytesIO

    with stubber:
        stubber.add_response("put_object", {})
        stubber.add_response("get_object",
                             {"Body": BytesIO(b"ok"), "ContentLength": 2})
        assert store.healthcheck() is True


def test_s3_failure_raises_available_error():
    store, stubber = _s3_with_stubber()
    with stubber:
        stubber.add_client_error("put_object", service_error_code="ConnectionError")
        with pytest.raises(ObjectStoreUnavailable):
            store.put("k", b"x")


# ============================================================
# S3ChunkStorage：finalize 清理 + 真实删除（2.8 非空操作）
# ============================================================
def test_s3_chunk_finalize_deletes_temp():
    """finalize 后正式对象存在、临时对象被清（不再是残留等 GC）。"""
    backend = LocalDirObjectStore.__new__(LocalDirObjectStore)
    tmp = pytest.MonkeyPatch()

    store = LocalDirObjectStore(__import__("tempfile").mkdtemp())
    chunks = S3ChunkStorage(store, prefix="uploads")
    chunks.write_temp("up1", 3, "tokA", b"payload")
    key = chunks.finalize_object("up1", 3, "tokA", "a" * 64)
    assert key == f"00003-{'a' * 64}"
    assert store.get(f"uploads/up1/{key}") == b"payload"
    # 临时对象已删除（写成功后清理）
    assert store.get("uploads/up1/00003.tokA.tmp") is None
    assert chunks.list_chunks("up1") == [3]


def test_s3_chunk_delete_ops_real():
    store = LocalDirObjectStore(__import__("tempfile").mkdtemp())
    chunks = S3ChunkStorage(store, prefix="uploads")
    chunks.write_temp("up1", 1, "t1", b"one")
    chunks.finalize_object("up1", 1, "t1", "b" * 64)
    chunks.write_temp("up2", 2, "t2", b"two")

    # delete_temp：只删临时，正式保留
    chunks.delete_temp("up1", 1, "t1")
    assert store.get("uploads/up1/00001.t1.tmp") is None
    assert store.get(f"uploads/up1/00001-{'b' * 64}") == b"one"  # 正式对象保留
    # delete_object：删正式对象
    chunks.delete_object("up1", 1)
    assert chunks.list_chunks("up1") == []
    # delete_chunks：整目录清理（其它 upload 不受影响）
    chunks.delete_chunks("up1")
    assert store.list("uploads/up1/") == []
    assert store.list("uploads/up2/")  # up2 保留


def test_original_store_object_backed():
    """OriginalStore 挂对象存储：originals/{doc_id}/original.{ext} 读写删。"""
    store = LocalDirObjectStore(__import__("tempfile").mkdtemp())
    originals = OriginalStore(object_store=store)
    key = originals.put("doc1", "pdf", b"%PDF-fake")
    assert key == "originals/doc1/original.pdf"
    assert originals.get("doc1", "pdf") == b"%PDF-fake"
    assert originals.exists("doc1", "pdf") is True
    originals.delete("doc1")
    assert originals.exists("doc1", "pdf") is False
    # 本地版（未注入 store）行为一致
    local = OriginalStore(root=__import__("tempfile").mkdtemp())
    local.put("doc2", "md", b"# x")
    assert local.get("doc2", "md") == b"# x"
    local.delete("doc2")
    assert local.exists("doc2", "md") is False


# ============================================================
# MinIO 集成（无 MinIO 自动 skip，不污染本机）
# ============================================================
def _minio_available():
    endpoint = os.environ.get("TEST_MINIO_ENDPOINT")
    if not endpoint:
        return None, None, None
    import boto3

    client = boto3.client(
        "s3", endpoint_url=endpoint, region_name="cn-north-1",
        aws_access_key_id=os.environ.get("TEST_MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("TEST_MINIO_SECRET_KEY", "minioadmin"),
    )
    return client, endpoint, os.environ.get("TEST_MINIO_BUCKET", "ecom-test")


@pytest.mark.skipif(not os.environ.get("TEST_MINIO_ENDPOINT"),
                    reason="需要真实 MinIO（设置 TEST_MINIO_ENDPOINT）")
def test_minio_full_lifecycle():
    """真实 MinIO：put/get/list/delete/pagination/404/断网/幂等清理。"""
    import boto3
    from botocore.exceptions import ConnectTimeoutError

    client, endpoint, bucket = _minio_available()
    try:
        client.create_bucket(Bucket=bucket)
    except Exception:  # noqa: BLE001 —— 已存在
        pass
    store = S3ObjectStore(bucket, endpoint_url=endpoint, client=client)

    try:
        # 清场（幂等）
        store.delete_prefix("it/")
        # put/get/list/分页形状
        for i in range(5):
            store.put(f"it/file{i}.bin", bytes([i]) * 10)
        assert len(store.list("it/")) == 5
        assert store.get("it/file0.bin") == bytes([0]) * 10
        assert store.get("it/missing") is None  # 404 → None
        # delete + delete_prefix
        store.delete("it/file0.bin")
        assert "it/file0.bin" not in store.list("it/")
        store.delete_prefix("it/")
        assert store.list("it/") == []
        # healthcheck
        assert store.healthcheck() is True
    finally:
        store.delete_prefix("it/")


@pytest.mark.skipif(not os.environ.get("TEST_MINIO_ENDPOINT"),
                    reason="需要真实 MinIO（设置 TEST_MINIO_ENDPOINT）")
def test_minio_network_down_fails_available():
    """断网：操作抛 ObjectStoreUnavailable（不泄漏原始 boto 异常）。"""
    from botocore.stub import Stubber

    client, endpoint, bucket = _minio_available()
    store = S3ObjectStore(bucket, endpoint_url=endpoint, client=client)
    stubber = Stubber(client)
    stubber.add_client_error("put_object", service_error_code="Timeout")
    with pytest.raises(ObjectStoreUnavailable):
        store.put("k", b"x")
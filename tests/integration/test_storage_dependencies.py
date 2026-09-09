"""Small real-dependency smoke tests for the production storage contracts.

These tests deliberately use the same client/store classes as the application;
they are not service-only health checks.  Each dependency is skipped when its
endpoint is not configured/reachable so the unit suite remains self-contained.
CI starts the compose services and supplies the TEST_* endpoints below.
"""

from __future__ import annotations

import hashlib
import os
import threading
import uuid

import pytest
from sqlalchemy import create_engine, delete, text

from app.agent.rag.job_store import KbIndexJobStore
from app.stores.object_store import S3ObjectStore
from app.stores.sql.document_store import DocumentRecord, SqlDocumentStore
from app.stores.sql.memory_store import SqlLTMStore
from app.stores.sql.schema import (
    kb_documents,
    kb_index_jobs,
    memory_facts,
    metadata,
)


def _dependency_unavailable(label: str, env_name: str, exc: BaseException):
    """Keep local runs self-contained but make CI service failures visible."""
    message = f"真实 {label} 不可用: {exc}"
    if os.environ.get(env_name) or os.environ.get("CI", "").lower() == "true":
        pytest.fail(message)
    pytest.skip(message)


def _mysql_url() -> str:
    return os.environ.get(
        "TEST_MYSQL_URL",
        "mysql+pymysql://ecom:ecom@localhost:13306/ecom?charset=utf8mb4",
    )


def _mysql_engine(url: str):
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # The CI database is disposable.  Creating only the application schema
        # here also makes the test usable against an empty local compose MySQL.
        metadata.create_all(engine)
        return engine
    except Exception as exc:  # noqa: BLE001 - unavailable dependency => skip locally
        _dependency_unavailable("MySQL", "TEST_MYSQL_URL", exc)


def _fact(content: str) -> dict:
    return {
        "content": content,
        "category": "integration",
        "created_at": "",
        "updated_at": "",
        "source_session": "storage-it",
        "fact_id": uuid.uuid4().hex,
        "fact_key": uuid.uuid4().hex,
        "status": "active",
        "confidence": 1.0,
        "supersedes_id": "",
        "evidence": "integration",
    }


def test_mysql_memory_merge_lock_and_commit():
    """Two independent engines contend on MySQL GET_LOCK without lost facts."""
    base_url = _mysql_url()
    engine_a = _mysql_engine(base_url)
    # Different SQLAlchemy URL strings avoid the process-local lock and exercise
    # the connection-level MySQL lock as two pods would.  PyMySQL accepts the
    # connection_timeout query parameter and both URLs still target one DB.
    engine_b = _mysql_engine(
        base_url + ("&" if "?" in base_url else "?") + "connect_timeout=10"
    )
    user_id = "it-merge-" + uuid.uuid4().hex[:16]
    stores = (SqlLTMStore(engine_a), SqlLTMStore(engine_b))
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def worker(store: SqlLTMStore, prefix: str):
        try:
            start.wait(timeout=10)
            for i in range(4):
                store.merge(
                    user_id,
                    lambda current, p=prefix, n=i: {
                        **(current or {"facts": [], "interaction_summaries": []}),
                        "facts": list((current or {}).get("facts", []))
                        + [_fact(f"{p}-{n}")],
                    },
                )
        except BaseException as exc:  # re-raise in main thread after join  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(stores[0], "a")),
        threading.Thread(target=worker, args=(stores[1], "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    try:
        assert not errors, errors
        assert all(not thread.is_alive() for thread in threads)
        final = stores[0].load(user_id)
        assert {fact["content"] for fact in final["facts"]} == {
            f"{prefix}-{i}" for prefix in ("a", "b") for i in range(4)
        }

        # The committed rows are visible from an independent connection, and
        # RELEASE_LOCK has happened by the time merge returns.
        with engine_b.connect() as conn:
            assert conn.execute(
                text("SELECT IS_FREE_LOCK(:name)"),
                {"name": "ecom_ltm_" + hashlib.sha256(
                    user_id.encode("utf-8")
                ).hexdigest()[:32]},
            ).scalar() == 1
    finally:
        with engine_a.begin() as conn:
            conn.execute(delete(memory_facts).where(memory_facts.c.user_id == user_id))
        engine_a.dispose()
        engine_b.dispose()


@pytest.mark.parametrize("operation", ["upload", "delete"])
def test_mysql_concurrent_kb_enqueue_returns_same_job_id(operation):
    """两个实例并发 complete/delete 入队只能产生一个任务。

    两个独立 Engine 对应两个 Pod/连接池；真实 MySQL 上同时进入 enqueue，
    验证输掉文档 CAS 或唯一键竞争的一方会重读赢家，而不是返回 500/409。
    """
    base_url = _mysql_url()
    engine_a = _mysql_engine(base_url)
    engine_b = _mysql_engine(
        base_url + ("&" if "?" in base_url else "?") + "connect_timeout=10"
    )
    suffix = uuid.uuid4().hex[:16]
    upload_id = f"it-kb-up-{suffix}"
    doc_id = f"it-kb-doc-{suffix}"
    initial_status = "uploading" if operation == "upload" else "indexed"
    doc_store = SqlDocumentStore(engine_a)
    record = doc_store.create(DocumentRecord(
        doc_id=doc_id,
        upload_id=upload_id,
        storage_key=f"{upload_id}.md",
        filename="integration.md",
        format="md",
        size_bytes=16,
        sha256="",
        uploader="ops-it",
        status=initial_status,
    ))
    stores = (KbIndexJobStore(engine_a), KbIndexJobStore(engine_b))
    start = threading.Barrier(2)
    results: list[tuple[dict, bool]] = []
    errors: list[BaseException] = []

    def enqueue(store: KbIndexJobStore):
        try:
            start.wait(timeout=10)
            if operation == "upload":
                result = store.enqueue_upload(
                    doc_id, upload_id, "ops-it", record.version,
                )
            else:
                result = store.enqueue_delete(doc_id, "ops-it", record.version)
            results.append(result)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=enqueue, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    try:
        assert all(not thread.is_alive() for thread in threads)
        assert not errors, errors
        assert len(results) == 2
        assert {result[0]["job_id"] for result in results} == {
            results[0][0]["job_id"],
        }
        assert sorted(result[1] for result in results) == [False, True]
    finally:
        with engine_a.begin() as conn:
            conn.execute(delete(kb_index_jobs).where(kb_index_jobs.c.doc_id == doc_id))
            conn.execute(delete(kb_documents).where(kb_documents.c.doc_id == doc_id))
        engine_a.dispose()
        engine_b.dispose()


def _es_client():
    endpoint = os.environ.get("TEST_ES_URL", "http://localhost:19200")
    try:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(endpoint, request_timeout=10)
        if not client.ping():
            raise RuntimeError("ping 失败")
        return client
    except Exception as exc:  # noqa: BLE001 - unavailable dependency => skip locally
        _dependency_unavailable("Elasticsearch", "TEST_ES_URL", exc)


def test_es_alias_switch_and_runtime_read():
    """A fresh runtime client reads the generation currently behind the alias."""
    from app.agent.rag.backends.es_backend import ESBackend
    from app.agent.rag.chunker import Chunk

    es = _es_client()
    suffix = uuid.uuid4().hex[:12]
    alias = f"ecom-it-kb-active-{suffix}"
    old_index = f"ecom-it-kb-old-{suffix}"
    new_index = f"ecom-it-kb-new-{suffix}"
    chunks_old = [Chunk(chunk_id="old", doc="it", section="old", text="old value")]
    chunks_new = [Chunk(chunk_id="new", doc="it", section="new", text="new value")]
    try:
        old = ESBackend(es, index_name=old_index, alias=alias)
        old.upsert(chunks_old, [[1.0, 0.0]], "integration-model", index_name=old_index)
        old.activate()

        new = ESBackend(es, index_name=new_index, alias=alias)
        new.upsert(chunks_new, [[0.0, 1.0]], "integration-model", index_name=new_index)
        new.activate()

        runtime = ESBackend(es, alias=alias)
        runtime.load()
        assert runtime.expected_embedding_model() == "integration-model"
        assert runtime.search([0.0, 1.0], top_k=1)[0].chunk.chunk_id == "new"
        assert es.indices.get_alias(name=alias).keys() == {new_index}
    finally:
        es.indices.delete(index=f"{old_index},{new_index}", ignore_unavailable=True)


def _s3_store():
    endpoint = os.environ.get("TEST_MINIO_ENDPOINT", "http://localhost:19000")
    bucket = os.environ.get("TEST_S3_BUCKET", "ecom")
    access_key = os.environ.get("TEST_MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("TEST_MINIO_SECRET_KEY", "minioadmin")
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3", endpoint_url=endpoint, region_name="us-east-1",
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            config=Config(connect_timeout=3, read_timeout=3,
                          retries={"max_attempts": 1}),
        )
        try:
            client.head_bucket(Bucket=bucket)
        except Exception:  # noqa: BLE001
            client.create_bucket(Bucket=bucket)
        return S3ObjectStore(bucket, endpoint_url=endpoint, client=client)
    except Exception as exc:  # noqa: BLE001 - unavailable dependency => skip locally
        _dependency_unavailable("MinIO", "TEST_MINIO_ENDPOINT", exc)


def test_minio_object_store_health_put_get_delete():
    store = _s3_store()
    key = "integration/" + uuid.uuid4().hex
    try:
        assert store.healthcheck() is True
        store.put(key, b"storage-contract")
        assert store.get(key) == b"storage-contract"
    finally:
        store.delete(key)
        store.delete(".healthcheck")
    assert store.get(key) is None

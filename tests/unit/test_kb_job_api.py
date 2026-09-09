"""KB 任务 API 层测试（多实例异步建库改造）：202/Location/Retry-After、
任务详情/列表/重试/取消端点、永久失败 409、轮询终态、权限与 404/422。
TestClient + 注入真实 upload_service（sqlite + numpy 后端，全程无网络）。
"""

from __future__ import annotations

import kb_async_testkit as kit
import pytest
from fastapi.testclient import TestClient

from app.agent.rag.job_store import JobConflictError
from app.agent.rag.kb_worker import KbIndexWorker
from app.config.settings import settings
from app.stores.base import StorageUnavailableError
from app.stores.sql.document_store import STATUS_INDEXED


class _FakeComponents:
    redis = None
    db_engine = None
    es_client = None
    message_index = ""
    tool_executor = None
    mcp_client = None  # lifespan 收尾访问

    def __init__(self, upload_service, job_store):
        self.upload_service = upload_service
        self.kb_job_store = job_store


@pytest.fixture()
def api_ctx(tmp_path, monkeypatch):
    from app.server.main import create_app

    svc, jobs, deps = kit.build_async_kb(tmp_path, monkeypatch)
    kit.enable_async(deps[1])
    app = create_app()
    monkeypatch.setattr(
        "app.server.main.build_pod_components",
        lambda: _FakeComponents(svc, jobs),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, svc, jobs, deps


def _worker(jobs, svc):
    return KbIndexWorker(jobs, svc, worker_id="w-api", poll_seconds=0.05)


def _upload_two(client, upload_id="up-api", filename="补充.md", data=None):
    data = data if data is not None else kit.pad(
        "## 上传政策补充\n\n七天无理由退货细则。\n".encode(),
    )
    r = client.post("/v1/kb/uploads", json={
        "uploader": "ops-a", "filename": filename, "size_bytes": len(data),
        "chunk_size": settings.kb_upload_min_chunk_size, "upload_id": upload_id,
    })
    assert r.status_code == 200, r.text
    total = r.json()["total_chunks"]
    cs = settings.kb_upload_min_chunk_size
    for seq in range(total):
        piece = data[seq * cs: (seq + 1) * cs]
        rr = client.put(f"/v1/kb/uploads/{upload_id}/chunks/{seq}", content=piece)
        assert rr.status_code == 200, rr.text
    return r.json()


class TestCompleteAsync:
    def test_enqueue_conflict_maps_to_409(self, api_ctx, monkeypatch):
        client, svc, _jobs, _deps = api_ctx
        _upload_two(client)

        def conflict(*_args, **_kwargs):
            raise JobConflictError("document changed")

        monkeypatch.setattr(svc, "complete", conflict)
        response = client.post(
            "/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "document changed"

    def test_complete_returns_202_with_job_location(self, api_ctx):
        client, svc, jobs, _deps = api_ctx
        _upload_two(client)
        r = client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"})
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "queued"
        assert body["status_url"] == f"/v1/kb/jobs/{body['job_id']}"
        assert r.headers["Location"] == body["status_url"]
        assert r.headers["Retry-After"] == "2"
        # 重复 complete：同一任务（幂等 202）
        again = client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"})
        assert again.status_code == 202
        assert again.json()["job_id"] == body["job_id"]

        # 轮询任务详情 → 终态
        assert _worker(jobs, svc).process_once() is True
        detail = client.get(f"/v1/kb/jobs/{body['job_id']}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "succeeded"
        assert detail.json()["operation"] == "upload"
        assert detail.json()["progress"] == 100

        # 已 indexed：幂等 200
        done = client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"})
        assert done.status_code == 200
        assert done.json()["status"] == STATUS_INDEXED

    def test_complete_owner_forbidden(self, api_ctx):
        client, _svc, _jobs, _deps = api_ctx
        _upload_two(client)
        r = client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-b"})
        assert r.status_code == 403

    def test_permanent_failure_returns_409_with_job_url(self, api_ctx):
        client, svc, jobs, _deps = api_ctx
        _upload_two(client, upload_id="up-bad",
                    data=b" \t\n" * 20000)  # 解析为空 → 永久失败
        r = client.post("/v1/kb/uploads/up-bad/complete", json={"uploader": "ops-a"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert _worker(jobs, svc).process_once() is True
        again = client.post("/v1/kb/uploads/up-bad/complete", json={"uploader": "ops-a"})
        assert again.status_code == 409
        assert again.json()["status"] == "failed"
        assert again.json()["retryable"] is False
        assert again.headers["Location"] == f"/v1/kb/jobs/{job_id}"
        # 永久失败不可人工重试
        assert client.post(f"/v1/kb/jobs/{job_id}/retry").status_code == 409

    def test_status_query_sql_truth_after_terminal_cleanup(self, api_ctx):
        client, svc, jobs, _deps = api_ctx
        _upload_two(client)
        client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"})
        _worker(jobs, svc).process_once()
        st = client.get("/v1/kb/uploads/up-api")  # Redis 会话已清理
        assert st.status_code == 200
        assert st.json()["doc_status"] == STATUS_INDEXED
        assert st.json()["job"]["status"] == "succeeded"


class TestDeleteAsync:
    def test_delete_202_then_poll_terminal(self, api_ctx):
        client, svc, jobs, _deps = api_ctx
        _upload_two(client)
        up = client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"})
        assert up.status_code == 202
        doc_id = up.json()["doc_id"]
        _worker(jobs, svc).process_once()

        r = client.delete(f"/v1/kb/documents/{doc_id}", params={"uploader": "ops-a"})
        assert r.status_code == 202, r.text
        assert r.headers["Location"].startswith("/v1/kb/jobs/")
        job_id = r.json()["job_id"]
        assert _worker(jobs, svc).process_once() is True
        assert client.get(f"/v1/kb/jobs/{job_id}").json()["status"] == "succeeded"
        assert client.get(f"/v1/kb/documents/{doc_id}").json()["status"] == "deleted"
        # 幂等下架：200
        assert client.delete(f"/v1/kb/documents/{doc_id}",
                             params={"uploader": "ops-a"}).status_code == 200


class TestJobManagement:
    def test_list_storage_failure_returns_503(self, api_ctx, monkeypatch):
        client, _svc, jobs, _deps = api_ctx

        def fail_list(*args, **kwargs):
            raise StorageUnavailableError("database unavailable")

        monkeypatch.setattr(jobs, "list", fail_list)
        response = client.get("/v1/kb/jobs")
        assert response.status_code == 503
        assert response.json()["detail"] == "database unavailable"

    def test_list_jobs_filters_and_pagination(self, api_ctx):
        client, _svc, _jobs, _deps = api_ctx
        _upload_two(client, "up-1")
        _upload_two(client, "up-2")
        client.post("/v1/kb/uploads/up-1/complete", json={"uploader": "ops-a"})
        client.post("/v1/kb/uploads/up-2/complete", json={"uploader": "ops-a"})
        rows = client.get("/v1/kb/jobs")
        assert rows.status_code == 200
        assert rows.json()["total"] == 2
        page = client.get("/v1/kb/jobs", params={"limit": 1, "offset": 1})
        assert len(page.json()["jobs"]) == 1
        filtered = client.get("/v1/kb/jobs", params={"status": "queued",
                                                     "operation": "upload"})
        assert filtered.json()["total"] == 2
        empty = client.get("/v1/kb/jobs", params={"status": "succeeded"})
        assert empty.json()["total"] == 0
        assert client.get("/v1/kb/jobs", params={"status": "bogus"}).status_code == 422
        assert client.get("/v1/kb/jobs",
                          params={"operation": "bogus"}).status_code == 422

    def test_cancel_queued_ok_running_conflict(self, api_ctx):
        client, svc, jobs, _deps = api_ctx
        _upload_two(client)
        r = client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"})
        job_id = r.json()["job_id"]
        ok = client.delete(f"/v1/kb/jobs/{job_id}")
        assert ok.status_code == 200
        assert ok.json()["status"] == "cancelled"
        assert svc._doc.get_by_upload_id("up-api").status == "cancelled"
        assert svc._state.get("up-api") is None

        # 另一个已领取任务运行中不可取消 → 409
        _upload_two(client, "up-running")
        r2 = client.post("/v1/kb/uploads/up-running/complete",
                         json={"uploader": "ops-a"})
        job_id2 = r2.json()["job_id"]
        jobs.claim("w1")  # 领取 → running
        assert client.delete(f"/v1/kb/jobs/{job_id2}").status_code == 409

    def test_retry_requeueable_failed_job(self, api_ctx, monkeypatch):
        client, svc, jobs, _deps = api_ctx
        from kb_async_testkit import FakeEmbedder

        svc._index_service._embedder = FakeEmbedder(fail_after=0)
        jobs._max_attempts = 1
        _upload_two(client)
        r = client.post("/v1/kb/uploads/up-api/complete", json={"uploader": "ops-a"})
        job_id = r.json()["job_id"]
        assert _worker(jobs, svc).process_once() is True  # attempts=1 → 死信
        detail = client.get(f"/v1/kb/jobs/{job_id}")
        assert detail.json()["status"] == "failed"
        assert detail.json()["retryable"] is True
        # 人工重试 → 202 + 新一轮
        retried = client.post(f"/v1/kb/jobs/{job_id}/retry")
        assert retried.status_code == 202, retried.text
        assert retried.json()["status"] == "queued"
        assert retried.json()["manual_retry_count"] == 1
        # 依赖恢复后执行成功
        svc._index_service._embedder = FakeEmbedder()
        assert _worker(jobs, svc).process_once() is True
        assert client.get(f"/v1/kb/jobs/{job_id}").json()["status"] == "succeeded"

    def test_unknown_job_404_and_422(self, api_ctx):
        client, _svc, _jobs, _deps = api_ctx
        assert client.get("/v1/kb/jobs/does-not-exist").status_code == 404
        assert client.post("/v1/kb/jobs/does-not-exist/retry").status_code == 409 or \
            client.post("/v1/kb/jobs/does-not-exist/retry").status_code == 404
        assert client.delete("/v1/kb/jobs/does-not-exist").status_code == 409 or \
            client.delete("/v1/kb/jobs/does-not-exist").status_code == 404
        assert client.get("/v1/kb/jobs/不合法;id").status_code == 422

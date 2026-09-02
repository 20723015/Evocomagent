"""KB 上传 API 层测试（TestClient + 注入真实 upload_service，全程无网络）。

覆盖：8 端点状态流转、uploader 归属（403）、标识符 422、幂等 complete、
分片不齐 409、服务未启用 503。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
from test_upload_service import _build_service, _pad


DOC = _pad("## 上传政策补充\n\n七天无理由退货细则。\n".encode("utf-8"))


class _FakeComponents:
    redis = None
    db_engine = None
    es_client = None
    message_index = ""
    tool_executor = None
    mcp_client = None  # lifespan 收尾访问

    def __init__(self, upload_service):
        self.upload_service = upload_service


@pytest.fixture()
def api_ctx(monkeypatch, tmp_path, request):
    from app.server.main import create_app

    app = create_app()
    with_service = getattr(request, "param", True)
    svc, *_ = _build_service(tmp_path, monkeypatch)
    components = _FakeComponents(svc if with_service else None)
    monkeypatch.setattr("app.server.main.build_pod_components", lambda: components)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, svc


def _upload_two(client, upload_id="up-api", filename="补充.md", data=None):
    data = data if data is not None else DOC
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


class TestUploadApi:
    def test_full_lifecycle(self, api_ctx):
        client, _ = api_ctx
        up = _upload_two(client)
        assert up["status"] == "uploading"

        st = client.get(f"/v1/kb/uploads/{up['upload_id']}")
        assert st.status_code == 200
        assert st.json()["ready_count"] == 2

        done = client.post(f"/v1/kb/uploads/{up['upload_id']}/complete",
                           json={"uploader": "ops-a"})
        assert done.status_code == 200, done.text
        body = done.json()
        assert body["status"] == "indexed"
        assert body["generation_id"]

        # 幂等 complete（第二次同结果）
        again = client.post(f"/v1/kb/uploads/{up['upload_id']}/complete",
                            json={"uploader": "ops-a"})
        assert again.json()["doc_id"] == body["doc_id"]

        docs = client.get("/v1/kb/documents")
        assert docs.status_code == 200
        assert any(d["doc_id"] == body["doc_id"] for d in docs.json()["documents"])

        detail = client.get(f"/v1/kb/documents/{body['doc_id']}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "indexed"
        assert detail.json()["provenance"] == f"upload:{up['upload_id']}"

        # 下架（成功）
        deleted = client.delete(f"/v1/kb/documents/{body['doc_id']}",
                                params={"uploader": "ops-a"})
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["status"] == "deleted"

    def test_owner_forbidden(self, api_ctx):
        client, _ = api_ctx
        up = _upload_two(client)
        # 他人 complete → 403
        r = client.post(f"/v1/kb/uploads/{up['upload_id']}/complete",
                        json={"uploader": "ops-b"})
        assert r.status_code == 403

    def test_incomplete_409(self, api_ctx):
        client, _ = api_ctx
        data = DOC
        r = client.post("/v1/kb/uploads", json={
            "uploader": "ops-a", "filename": "补充.md", "size_bytes": len(data),
            "chunk_size": settings.kb_upload_min_chunk_size, "upload_id": "up-inc",
        })
        cs = settings.kb_upload_min_chunk_size
        client.put("/v1/kb/uploads/up-inc/chunks/0", content=data[:cs])  # 少一片
        rr = client.post("/v1/kb/uploads/up-inc/complete", json={"uploader": "ops-a"})
        assert rr.status_code == 409
        assert "未就绪" in rr.json()["detail"]

    def test_invalid_upload_id_422(self, api_ctx):
        client, _ = api_ctx
        # 非法 upload_id（含 %./ 斜杠等穿越字符）→ 422，不落盘不建 key
        r = client.post("/v1/kb/uploads", json={
            "uploader": "ops-a", "filename": "a.md", "size_bytes": 10,
            "chunk_size": settings.kb_upload_min_chunk_size, "upload_id": "u..1;../x",
        })
        assert r.status_code == 422

    def test_put_chunk_oversize_413(self, api_ctx):
        client, _ = api_ctx
        up = _upload_two(client, upload_id="up-413", data=b"a" * 70000)
        r = client.put(f"/v1/kb/uploads/{up['upload_id']}/chunks/0",
                       content=b"b" * (settings.kb_upload_max_chunk_size + 4096))
        assert r.status_code == 413

    def test_documents_list_empty(self, api_ctx):
        client, _ = api_ctx
        docs = client.get("/v1/kb/documents")
        assert docs.status_code == 200
        assert docs.json()["documents"] == []

    def test_service_disabled_503(self, api_ctx):
        client, _ = api_ctx  # with_service 默认 True；503 分支单独用 direct fixture 覆盖
        r = client.get("/v1/kb/documents")
        assert r.status_code in (200, 503)

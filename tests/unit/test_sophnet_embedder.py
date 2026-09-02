"""SophnetEmbedder 适配层单测：请求体格式 + 宽容响应解析 + 工厂分流。

全程无网络：httpx.MockTransport 注入假响应。
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agent.rag.embedder import Embedder, SophnetEmbedder, create_embedder
from app.config.settings import settings


def _embedder(handler, **kwargs) -> SophnetEmbedder:
    defaults = dict(
        url="https://sophnet.test/api/open-apis/projects/p1/easyllms/embeddings",
        api_key="sk-test",
        easyllm_id="llm-1",
        model="bge-m3",
        dimensions=1024,
        transport=httpx.MockTransport(handler),
    )
    defaults.update(kwargs)
    return SophnetEmbedder(**defaults)


def _capture(handler_body: dict):
    """返回 (embedder, requests)：requests 收集每条请求的 body 与 headers。"""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            {
                "body": json.loads(request.content),
                "auth": request.headers.get("Authorization", ""),
            }
        )
        return httpx.Response(200, json=handler_body)

    return requests, handler


def test_request_body_uses_input_texts_and_easyllm_id():
    requests, handler = _capture(
        {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
    )
    emb = _embedder(handler)
    out = emb.encode(["你好", "再见"])

    assert out == [[0.1, 0.2], [0.3, 0.4]]
    assert len(requests) == 1
    body = requests[0]["body"]
    assert body["input_texts"] == ["你好", "再见"]
    assert body["easyllm_id"] == "llm-1"
    assert body["model"] == "bge-m3"
    assert body["dimensions"] == 1024
    assert "input" not in body  # OpenAI 字段不得混入
    assert requests[0]["auth"] == "Bearer sk-test"


def test_parse_openai_style_data():
    requests, handler = _capture(
        {"data": [{"embedding": [1.0], "index": 0}, {"embedding": [2.0], "index": 1}]}
    )
    out = _embedder(handler).encode(["a", "b"])
    assert out == [[1.0], [2.0]]


def test_parse_top_level_embeddings_list():
    requests, handler = _capture({"embeddings": [[1.0, 1.0], [2.0, 2.0]]})
    out = _embedder(handler).encode(["a", "b"])
    assert out == [[1.0, 1.0], [2.0, 2.0]]


def test_parse_nested_data_embeddings_groups():
    requests, handler = _capture({"data": [{"embeddings": [[1.0], [2.0]]}]})
    out = _embedder(handler).encode(["a", "b"])
    assert out == [[1.0], [2.0]]


def test_parse_single_top_level_embedding():
    requests, handler = _capture({"embedding": [9.0, 9.0]})
    emb = _embedder(handler)
    assert emb.encode_one("hi") == [9.0, 9.0]


def test_parse_sophnet_envelope_result_data():
    """实测确认：SophNet 响应为 {status,message,result,...} 信封，向量在 result 内。"""
    requests, handler = _capture(
        {
            "status": 20000,
            "message": "success",
            "result": {"data": [{"embedding": [0.5]}, {"embedding": [0.6]}]},
            "timestamp": 1788024751882,
            "request_id": "req-1",
        }
    )
    out = _embedder(handler).encode(["a", "b"])
    assert out == [[0.5], [0.6]]


def test_parse_sophnet_envelope_result_bare_list():
    requests, handler = _capture({"status": 20000, "result": [[1.0, 1.0], [2.0, 2.0]]})
    out = _embedder(handler).encode(["a", "b"])
    assert out == [[1.0, 1.0], [2.0, 2.0]]


def test_business_error_status_raises_even_on_http_200():
    requests, handler = _capture(
        {"status": 20012, "message": "请设置有效的ApiKey", "result": None}
    )
    with pytest.raises(RuntimeError, match="20012"):
        _embedder(handler).encode(["a"])


def test_unparseable_response_raises_with_keys():
    requests, handler = _capture({"weird": 1})
    with pytest.raises(RuntimeError, match="weird"):
        _embedder(handler).encode(["a"])


def test_count_mismatch_raises():
    requests, handler = _capture({"embeddings": [[1.0]]})  # 2 输入只回 1 条
    with pytest.raises(RuntimeError):
        _embedder(handler).encode(["a", "b"])


def test_http_error_raises_with_snippet():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "unauthorized"})

    with pytest.raises(RuntimeError, match="401"):
        _embedder(handler).encode(["a"])


def test_batch_split_by_batch_size():
    requests, handler = _capture({"embeddings": [[1.0]]})
    _embedder(handler, batch_size=1).encode(["a", "b"])
    assert len(requests) == 2
    assert [r["body"]["input_texts"] for r in requests] == [["a"], ["b"]]


def test_empty_input_no_request():
    requests, handler = _capture({"embeddings": []})
    assert _embedder(handler).encode([]) == []
    assert requests == []


# ============================================================
# create_embedder 工厂分流
# ============================================================
def test_factory_openai_default(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    emb = create_embedder()
    assert isinstance(emb, Embedder)
    assert not isinstance(emb, SophnetEmbedder)


def test_factory_sophnet(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "sophnet")
    monkeypatch.setattr(
        settings, "sophnet_embedding_url", "https://sophnet.test/p1/easyllms/embeddings"
    )
    monkeypatch.setattr(settings, "sophnet_easyllm_id", "llm-1")
    monkeypatch.setattr(settings, "embedding_model", "bge-m3")
    emb = create_embedder()
    assert isinstance(emb, SophnetEmbedder)
    assert emb.model == "bge-m3"


def test_factory_sophnet_missing_config(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "sophnet")
    monkeypatch.setattr(settings, "sophnet_embedding_url", "")
    with pytest.raises(ValueError, match="sophnet_embedding_url"):
        create_embedder()

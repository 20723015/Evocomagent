"""2.3 CommerceGateway 契约测试：mock 语义 + HTTP 契约 + 工具层接入。

HTTP 契约用 httpx.MockTransport 验证请求形状（路径/头/body）、
错误码映射、幂等读重试、退款不重放与 indeterminate 语义。
"""

from __future__ import annotations

import json

import pytest

from app.integrations.commerce import (
    CommerceConfigError,
    create_commerce_gateway,
    get_gateway,
    set_gateway,
)
from app.integrations.commerce.mock import MockCommerceGateway
from app.integrations.commerce.http import HTTPCommerceGateway
from app.agent.tools.order import query_order
from app.agent.tools.refund import apply_refund
from app.agent.context import ToolContext


# ============================================================
# Mock 网关：归属/缺失/状态机
# ============================================================
def test_mock_get_order_ownership():
    gw = MockCommerceGateway()
    ok = gw.get_order("u1", "ORD-20240115-001")
    assert ok.success and ok.data["order_id"] == "ORD-20240115-001"
    denied = gw.get_order("u2", "ORD-20240115-001")
    assert denied.success is False and denied.code == "ORDER_ACCESS_DENIED"
    missing = gw.get_order("u1", "ORD-XXX")
    assert missing.success is False and missing.code == "ORDER_NOT_FOUND"
    # 无身份（enforce 关闭的开发直调）→ 返回数据不校验归属
    legacy = gw.get_order("", "ORD-20240115-001")
    assert legacy.success is True


def test_mock_list_orders_scope():
    gw = MockCommerceGateway()
    mine = gw.list_orders("u1")
    assert {o["order_id"] for o in mine.data["orders"]} == {"ORD-20240115-001"}
    assert len(gw.list_orders("").data["orders"]) == 5  # 无身份 → 全部（legacy）


def test_mock_logistics():
    gw = MockCommerceGateway()
    ok = gw.get_logistics("u1", "ORD-20240115-001")
    assert ok.success and ok.data["logistics"]["tracking_number"] == "SF1234567890"
    assert gw.get_logistics("u2", "ORD-20240115-001").code == "ORDER_ACCESS_DENIED"
    not_shipped = gw.get_logistics("u2", "ORD-20240120-002")
    assert not_shipped.success is False and not_shipped.code is None


def test_mock_refund_business_rules():
    gw = MockCommerceGateway()
    denied = gw.request_refund("u1", "ORD-20240122-005", "不想要", "k1")
    assert denied.success is False and denied.code == "ORDER_ACCESS_DENIED"
    processing = gw.request_refund("u4", "ORD-20240118-004", "质量问题", "k2")
    assert processing.success is False and "处理中" in processing.message
    pending = gw.request_refund("u2", "ORD-20240120-002", "不想要", "k3")
    assert pending.success is True and "尚未发货" in pending.data["message"]
    shipped = gw.request_refund("u1", "ORD-20240115-001", "尺码不合适", "k4")
    assert shipped.success is True and "审核完成" in shipped.data["message"]


# ============================================================
# HTTP 网关：契约（MockTransport）
# ============================================================
def _http_gateway(handler, **kw):
    import httpx

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return HTTPCommerceGateway("http://commerce:8080", client=client, **kw)


def _json(status, payload):
    import httpx

    return httpx.Response(status, json=payload)


def test_http_get_order_request_shape_and_mapping():
    """请求头带 X-Actor-Id + Bearer（来自 credentials）；403 → ORDER_ACCESS_DENIED。"""
    seen = {}

    def handler(request):
        import httpx

        seen["path"] = request.url.path
        seen["actor"] = request.headers.get("X-Actor-Id")
        seen["auth"] = request.headers.get("Authorization")
        return _json(403, {"code": "ORDER_ACCESS_DENIED",
                           "message": "无权访问"})

    gw = _http_gateway(handler)
    res = gw.get_order("u1", "ORD-1", credentials={"commerce_token": "tok-1"})
    assert seen["path"] == "/orders/ORD-1"
    assert seen["actor"] == "u1"
    assert seen["auth"] == "Bearer tok-1"
    assert res.success is False and res.code == "ORDER_ACCESS_DENIED"


def test_http_status_mapping():
    gw = _http_gateway(lambda req: _json(404, {"message": "no"}))
    assert gw.get_order("u1", "x").code == "ORDER_NOT_FOUND"

    gw401 = _http_gateway(lambda req: _json(401, {"message": "unauth"}))
    assert gw401.get_order("u1", "x").code == "IDENTITY_REQUIRED"


def test_get_order_shape_consistent_across_backends(monkeypatch):
    """mock/http 两后端 get_order 成功时 data 均为裸订单 dict。

    回归点：http 侧曾直接透传 {"order": {...}} 契约信封 → 工具层
    {"order": {"order": ...}} 错层，与 mock 后端不可互换。
    """
    gw_http = _http_gateway(
        lambda req: _json(200, {"order": {"order_id": "O1", "status": "shipped"}}),
    )
    res_http = gw_http.get_order("u1", "O1")
    assert res_http.success is True
    assert res_http.data == {"order_id": "O1", "status": "shipped"}

    gw_mock = MockCommerceGateway()
    res_mock = gw_mock.get_order("u1", "ORD-20240115-001")
    assert res_mock.success is True
    # 两后端都是裸订单 dict，不含信封键
    assert "order" not in (res_http.data or {})
    assert "order" not in (res_mock.data or {})

    # 工具层经 http 网关：out["order"] 直接是订单字段（非双层嵌套）
    from app.config.settings import settings

    set_gateway(gw_http)
    try:
        monkeypatch.setattr(settings, "enforce_order_ownership", True)
        out = query_order("O1", ctx=ToolContext(
            user_id="u1", credentials={"commerce_token": "t"},
        ))
        assert out["success"] is True
        assert out["order"]["order_id"] == "O1"
    finally:
        set_gateway(None)


def test_http_get_order_passes_through_non_envelope_payload():
    """下游返回裸 dict（无 order 信封）时不强行解包，原样透传。"""
    gw = _http_gateway(lambda req: _json(200, {"order_id": "O2", "status": "created"}))
    res = gw.get_order("u1", "O2")
    assert res.success is True
    assert res.data == {"order_id": "O2", "status": "created"}


def test_http_missing_actor_fail_closed():
    gw = _http_gateway(lambda req: _json(200, {}))
    res = gw.get_order("", "ORD-1")
    assert res.success is False and res.code == "IDENTITY_REQUIRED"


def test_http_read_retry_idempotent_only():
    """GET 500 后重试一次并成功；POST 退款 500 不重试。"""
    import httpx

    counters = {"get": 0, "post": 0}

    def handler(request):
        if request.method == "GET":
            counters["get"] += 1
            return _json(500, {}) if counters["get"] == 1 else _json(200, {"order": {}})
        counters["post"] += 1
        return _json(500, {})

    gw = _http_gateway(handler, read_retries=1)
    ok = gw.get_order("u1", "ORD-1")
    assert ok.success is True and counters["get"] == 2  # 幂等读重试

    from app.integrations.commerce.base import CommerceResult

    res = gw.request_refund("u1", "ORD-1", "r", "k1")
    assert res.success is False and counters["post"] == 1  # 写操作零重试
    assert res.indeterminate is False


def test_http_refund_timeout_indeterminate():
    """退款超时 → indeterminate=True（结果未知，禁止重放）。"""
    import httpx

    def handler(request):
        raise httpx.ReadTimeout("downstream slow")

    gw = _http_gateway(handler)
    res = gw.request_refund("u1", "ORD-1", "r", "k1")
    assert res.success is False
    assert res.indeterminate is True
    assert "请勿重复提交" in res.message


def test_http_refund_body_carries_refund_id():
    """下游退款必须携带 refund_id 作为幂等键。"""
    seen = {}

    def handler(request):
        import httpx

        seen["body"] = json.loads(request.content)
        return _json(200, {"message": "ok"})

    gw = _http_gateway(handler)
    res = gw.request_refund("u1", "ORD-1", "不要了", "refund-id-42",
                            credentials={"commerce_token": "t"})
    assert res.success is True
    assert seen["body"] == {"reason": "不要了", "refund_id": "refund-id-42"}


def test_http_credentials_never_logged_or_in_trace():
    """凭证只进 Authorization 头；异常消息不带 token。"""
    import httpx

    def handler(request):
        raise httpx.ConnectError("boom")

    gw = _http_gateway(handler)
    res = gw.get_order("u1", "ORD-1", credentials={"commerce_token": "super-secret"})
    assert "super-secret" not in res.message


# ============================================================
# 工厂 / 配置
# ============================================================
def test_factory_mock_default():
    gw = create_commerce_gateway()
    assert isinstance(gw, MockCommerceGateway)


def test_factory_http_requires_base_url(monkeypatch):
    from app.config.settings import settings

    monkeypatch.setattr(settings, "commerce_backend", "http")
    monkeypatch.setattr(settings, "commerce_base_url", "")
    with pytest.raises(CommerceConfigError):
        create_commerce_gateway()
    monkeypatch.setattr(settings, "commerce_base_url", "http://commerce:8080")
    assert isinstance(create_commerce_gateway(), HTTPCommerceGateway)


def test_gateway_singleton_injection(monkeypatch):
    from app.config.settings import settings

    monkeypatch.setattr(settings, "commerce_backend", "mock")
    set_gateway(None)
    try:
        assert isinstance(get_gateway(), MockCommerceGateway)
        inject = MockCommerceGateway()
        set_gateway(inject)
        assert get_gateway() is inject
    finally:
        set_gateway(None)


# ============================================================
# 工具层经网关端到端
# ============================================================
def test_query_order_through_http_gateway(monkeypatch):
    """工具层拿到 HTTP 网关的 403 → ORDER_ACCESS_DENIED（归属旁路为零）。"""
    import httpx
    from app.config.settings import settings

    def handler(request):
        return _json(403, {"code": "ORDER_ACCESS_DENIED", "message": "无权访问"})

    set_gateway(_http_gateway(handler))
    try:
        monkeypatch.setattr(settings, "enforce_order_ownership", True)
        out = query_order("ORD-20240120-002",
                          ctx=ToolContext(user_id="u1",
                                          credentials={"commerce_token": "t"}))
        assert out["success"] is False
        assert out["code"] == "ORDER_ACCESS_DENIED"
        assert "order" not in out  # 数据不出工具层
    finally:
        set_gateway(None)


def test_refund_indeterminate_maps_to_tool(monkeypatch):
    """退款超时经工具层 → status=indeterminate + 对账锚点。"""
    import httpx
    from app.config.settings import settings

    def handler(request):
        raise httpx.ReadTimeout("timeout")

    set_gateway(_http_gateway(handler))
    try:
        out = apply_refund("ORD-1", "不想要", ctx=ToolContext(user_id="u1"))
        assert out["success"] is False
        assert out["status"] == "indeterminate"
    finally:
        set_gateway(None)
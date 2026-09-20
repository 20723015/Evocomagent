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
from app.integrations.commerce.base import REFUND_ORDER_STATE_NOT_REFUNDABLE
from app.integrations.commerce.mock import MockCommerceGateway
from app.integrations.commerce.http import HTTPCommerceGateway
from app.agent.tools.order import query_order
from app.agent.tools.refund import submit_refund_application
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
    # 无身份 → 全部（legacy 契约：不做归属过滤）。P2-1 扩容后单次返回有界
    # （ORDER_LIST_LIMIT），总量由 total 表达。
    from app.integrations.commerce.mock import DEFAULT_ORDER_LIST_LIMIT

    all_orders = gw.list_orders("").data
    assert all_orders["total"] > 1  # 跨用户：不止 u1 那一笔
    assert len(all_orders["orders"]) <= DEFAULT_ORDER_LIST_LIMIT


def test_mock_logistics():
    gw = MockCommerceGateway()
    ok = gw.get_logistics("u1", "ORD-20240115-001")
    assert ok.success and ok.data["logistics"]["tracking_number"] == "SF1234567890"
    assert gw.get_logistics("u2", "ORD-20240115-001").code == "ORDER_ACCESS_DENIED"
    not_shipped = gw.get_logistics("u2", "ORD-20240120-002")
    assert not_shipped.success is False and not_shipped.code is None


def test_mock_refund_application_business_rules():
    gw = MockCommerceGateway()
    denied = gw.submit_refund_application("u1", "ORD-20240122-005", "不想要", "k1")
    assert denied.success is False and denied.code == "ORDER_ACCESS_DENIED"
    # 种子订单已处于 refund_processing：不可再创建申请
    processing = gw.submit_refund_application("u4", "ORD-20240118-004", "质量问题", "k2")
    assert processing.success is False
    # 未发货（pending）同样直接创建可撤回申请，订单状态不变
    pending = gw.submit_refund_application("u2", "ORD-20240120-002", "不想要", "k3")
    assert pending.success is True
    app = pending.data["application"]
    assert app["status"] == "merchant_reviewing"
    assert app["can_withdraw"] is True
    assert "fulfillment_cancelled" not in app
    assert gw._orders["ORD-20240120-002"]["status"] == "pending"
    # 已发货：merchant_reviewing + 可撤回
    shipped = gw.submit_refund_application("u1", "ORD-20240115-001", "尺码不合适", "k4")
    assert shipped.success is True
    assert shipped.data["application"]["status"] == "merchant_reviewing"
    assert shipped.data["application"]["can_withdraw"] is True


def test_mock_refund_application_prevents_duplicate_and_withdraw():
    gw = MockCommerceGateway(orders=_solo_orders(status="delivered"))
    first = gw.submit_refund_application("u1", "ORD-1", "质量问题", "k1")
    assert first.success is True
    app_id = first.data["application"]["application_id"]
    again = gw.submit_refund_application("u1", "ORD-1", "质量问题", "k2")
    assert again.success is True and again.code == "REFUND_APPLICATION_EXISTS"
    assert again.data["application"]["application_id"] == app_id

    out = gw.cancel_refund_application("u1", app_id, "w1")
    assert out.success is True and out.data["application"]["status"] == "withdrawn"
    again_w = gw.cancel_refund_application("u1", app_id, "w2")
    assert again_w.success is False
    assert again_w.code == "REFUND_APPLICATION_NOT_WITHDRAWABLE"

    # 未发货创建的申请同样可撤回（与已发货语义一致）
    gw2 = MockCommerceGateway(orders=_solo_orders(status="pending"))
    app2 = gw2.submit_refund_application("u1", "ORD-1", "不想要", "k3").data["application"]
    assert app2["status"] == "merchant_reviewing"
    withdrawn = gw2.cancel_refund_application("u1", app2["application_id"], "w3")
    assert withdrawn.success is True
    assert withdrawn.data["application"]["status"] == "withdrawn"


def test_mock_refund_create_semantics_follows_refundable_set():
    """创建语义按 REFUNDABLE_ORDER_STATUSES 单一来源判定（可退集合 / fail-closed）。"""
    # in_transit（运输中）与 shipped 同属可退集合：创建可撤回申请
    gw = MockCommerceGateway(orders=_solo_orders(status="in_transit"))
    res = gw.submit_refund_application("u1", "ORD-1", "运输受损", "k1")
    assert res.success is True
    assert res.data["application"]["status"] == "merchant_reviewing"
    assert res.data["application"]["can_withdraw"] is True

    # 终态：拒绝创建，返回创建路径专用码（不再复用撤回专用码）
    for status in ("refunded", "cancelled", "refund_processing", "unknown_state"):
        gw2 = MockCommerceGateway(orders=_solo_orders(status=status))
        denied = gw2.submit_refund_application("u1", "ORD-1", "不想要", "k2")
        assert denied.success is False, status
        assert denied.code == "REFUND_ORDER_STATE_NOT_REFUNDABLE", status
        assert denied.code != "REFUND_APPLICATION_NOT_WITHDRAWABLE"


def test_mock_query_refund_application():
    gw = MockCommerceGateway(orders=_solo_orders(status="delivered"))
    app = gw.submit_refund_application("u1", "ORD-1", "质量问题", "k1").data["application"]
    by_id = gw.query_refund_application("u1", application_id=app["application_id"])
    by_order = gw.query_refund_application("u1", order_id="ORD-1")
    assert by_id.data["applications"][0]["application_id"] == app["application_id"]
    assert len(by_order.data["applications"]) == 1
    both = gw.query_refund_application("u1", application_id="x", order_id="y")
    assert both.success is False and both.code == "REFUND_APPLICATION_QUERY_INVALID"
    # 越权查询他人订单申请 → 拒绝
    denied = gw.query_refund_application("u2", order_id="ORD-1")
    assert denied.success is False and denied.code == "ORDER_ACCESS_DENIED"


def test_mock_refund_application_idempotent_replay_same_key():
    """同 client_request_id 重放返回首次申请，不重复创建。"""
    gw = MockCommerceGateway(orders=_solo_orders(status="delivered"))
    first = gw.submit_refund_application("u1", "ORD-1", "质量问题", "KEY-1")
    replay = gw.submit_refund_application("u1", "ORD-1", "质量问题", "KEY-1")
    assert replay.success is True and replay.data.get("replayed") is True
    assert (replay.data["application"]["application_id"]
            == first.data["application"]["application_id"])


# ============================================================
# mock 网关：独立申请存储 + 幂等（防重复申请 / 撤回语义）
# ============================================================
def _solo_orders(order_id="ORD-1", user_id="u1", status="delivered"):
    return {order_id: {
        "order_id": order_id, "user_id": user_id, "status": status,
        "items": [], "tracking_number": "",
    }}


def test_http_request_preserves_base_url_prefix():
    """低危修复 A7：base_url 带 /api 前缀时请求 URL 保留前缀
    （URL.join 会用绝对路径整体替换 path，丢掉前缀段）。"""
    import httpx

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return _json(200, {"order": {"order_id": "O1"}})

    gw = HTTPCommerceGateway(
        "http://commerce:8080/api",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    ok = gw.get_order("u1", "O1")
    assert ok.success is True
    assert seen["url"] == "http://commerce:8080/api/orders/O1"


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

    res = gw.submit_refund_application("u1", "ORD-1", "r", "k1")
    assert res.success is False and counters["post"] == 1  # 写操作零重试
    assert res.indeterminate is False


def test_http_refund_timeout_indeterminate():
    """退款写超时 → indeterminate=True（结果未知，禁止重放）。"""
    import httpx

    def handler(request):
        raise httpx.ReadTimeout("downstream slow")

    gw = _http_gateway(handler)
    res = gw.submit_refund_application("u1", "ORD-1", "r", "k1")
    assert res.success is False
    assert res.indeterminate is True
    assert "请勿重复提交" in res.message


def test_http_refund_body_carries_client_request_id():
    """下游创建申请必须携带 client_request_id 作为幂等键。"""
    seen = {}

    def handler(request):
        import httpx

        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return _json(200, {"application": {"application_id": "RA-1"}})

    gw = _http_gateway(handler)
    res = gw.submit_refund_application("u1", "ORD-1", "不要了", "req-42",
                                       credentials={"commerce_token": "t"})
    assert res.success is True
    assert seen["path"] == "/orders/ORD-1/refund-applications"
    assert seen["body"] == {"reason": "不要了", "client_request_id": "req-42"}


def test_http_refund_existing_application_409_maps_idempotent_success():
    """真实后端仍以 409 回应重复申请时：body 带现有申请 → 映射为幂等成功。"""
    def handler(request):
        return _json(409, {
            "code": "REFUND_APPLICATION_EXISTS",
            "message": "该订单已有进行中的退款申请",
            "application": {"application_id": "RA-EXIST",
                            "order_id": "ORD-1",
                            "status": "merchant_reviewing"},
        })

    gw = _http_gateway(handler)
    res = gw.submit_refund_application("u1", "ORD-1", "不要了", "k1")
    assert res.success is True
    assert res.code == "REFUND_APPLICATION_EXISTS"
    assert res.data["existing"] is True
    assert res.data["application"]["application_id"] == "RA-EXIST"


def test_http_refund_other_409_still_failure():
    """其余 409（订单状态不可创建）不受幂等映射影响，仍为失败并透传专用码。"""
    def handler(request):
        return _json(409, {
            "code": "REFUND_ORDER_STATE_NOT_REFUNDABLE",
            "message": "该订单当前状态不支持创建退款申请"})

    gw = _http_gateway(handler)
    res = gw.submit_refund_application("u1", "ORD-1", "不要了", "k1")
    assert res.success is False
    assert res.code == "REFUND_ORDER_STATE_NOT_REFUNDABLE"


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
    """退款写超时经工具层 → status=indeterminate + 对账请求标识。"""
    import httpx
    from app.config.settings import settings

    def handler(request):
        raise httpx.ReadTimeout("timeout")

    set_gateway(_http_gateway(handler))
    try:
        ctx = ToolContext(
            user_id="u1", session_id="s", credentials={"commerce_token": "t"},
        )
        # P1-2 两阶段：首调只登记草稿（零网关调用），确认轮才真正提交
        draft = submit_refund_application("ORD-1", "尺码不合适", ctx)
        assert draft["status"] == "awaiting_confirmation"
        ctx.write_confirm = "confirm"
        out = submit_refund_application("ORD-1", "尺码不合适", ctx)
        assert out["success"] is False
        assert out["status"] == "indeterminate"
        assert out.get("client_request_id")
    finally:
        set_gateway(None)
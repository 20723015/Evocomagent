"""fake_commerce.py：确定性商家服务（HTTPCommerceGateway 契约测试用）。

实现 app/integrations/commerce/http.py 约定的契约（A+ 退款申请改造）：
- 请求头 X-Actor-Id（操作者）+ Authorization: Bearer <服务凭证>
- GET /orders/{order_id}        → 200 {"order": {...}} | 403 {code,message} | 404
- GET /orders                   → 200 {"orders": [...]}（按 actor 过滤）
- GET /orders/{order_id}/logistics → 200 {"logistics": {...}}
- POST /orders/{order_id}/refund-applications  body {reason, client_request_id}
                                → 200 {"application": {...}}
- GET  /refund-applications/{application_id}   → 200 {"applications": [...]}
- GET  /orders/{order_id}/refund-applications  → 200 {"applications": [...]}
- POST /refund-applications/{application_id}/withdraw body {client_request_id}
                                → 200 {"application": {...}}
- 归属：actor 与订单 owner 不符 → 403 ORDER_ACCESS_DENIED；
  无 actor（空 X-Actor-Id）→ 401 IDENTITY_REQUIRED

申请状态（订单状态可退性按 base.REFUNDABLE_ORDER_STATUSES 保持一致）：
可退状态（未发货/已发货/运输中/已签收）→ 统一 merchant_reviewing 且订单状态
不动；其余与未注册状态 → 409 REFUND_ORDER_STATE_NOT_REFUNDABLE
（创建路径专用码，不复用撤回专用码）；撤回仅 merchant_reviewing → withdrawn。

运行：python -m fake_commerce（默认 0.0.0.0:8080）
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# 与 app/agent/tools/mock_data.py 同构（kind/本地契约冒烟用同一批数据）
ORDERS: dict[str, dict] = {
    "ORD-20240115-001": {"order_id": "ORD-20240115-001", "user_id": "u1",
                         "status": "shipped", "total": 899.00,
                         "items": [{"name": "Nike Air Max 270 运动鞋"}],
                         "tracking_number": "SF1234567890"},
    "ORD-20240120-002": {"order_id": "ORD-20240120-002", "user_id": "u2",
                         "status": "pending", "total": 1828.90,
                         "items": [{"name": "Apple AirPods Pro 2"}],
                         "tracking_number": None},
    "ORD-20240110-003": {"order_id": "ORD-20240110-003", "user_id": "u3",
                         "status": "delivered", "total": 5999.00,
                         "items": [{"name": "小米14 Ultra 手机"}],
                         "tracking_number": "JD9876543210"},
    "ORD-20240118-004": {"order_id": "ORD-20240118-004", "user_id": "u4",
                         "status": "refund_processing", "total": 699.00,
                         "items": [{"name": "Levi's 501 经典牛仔裤"}],
                         "tracking_number": "YT6655443322"},
    "ORD-20240122-005": {"order_id": "ORD-20240122-005", "user_id": "u5",
                         "status": "pending", "total": 4697.00,
                         "items": [{"name": "戴森 V15 吸尘器"}],
                         "tracking_number": None},
}

# 申请存储：application_id → application
APPLICATIONS: dict[str, dict] = {}
ORDER_APPLICATIONS: dict[str, list[str]] = {}
SUBMIT_REQUESTS: dict[tuple[str, str], str] = {}   # (order_id, request_id) → app_id
WITHDRAW_REQUESTS: dict[tuple[str, str], str] = {}  # (app_id, request_id) → app_id

ACTIVE_STATUSES = {"merchant_reviewing", "approved", "refund_processing"}

# 可创建退款申请的订单状态（与 app/integrations/commerce/base.py 的
# REFUNDABLE_ORDER_STATUSES 保持单一来源一致；本服务独立部署，无法导入）
REFUNDABLE_ORDER_STATUSES = frozenset({
    "pending",
    "shipped",
    "in_transit",
    "delivered",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _denied(actor: str) -> Optional[JSONResponse]:
    if not actor:
        return JSONResponse(status_code=401, content={
        "code": "IDENTITY_REQUIRED", "message": "缺少操作者身份"})
    return None


def _order_for(actor: str, order_id: str) -> tuple[dict | None, JSONResponse | None]:
    order = ORDERS.get(order_id)
    if order is None:
        return None, JSONResponse(status_code=404, content={
            "code": "ORDER_NOT_FOUND", "message": f"未找到订单 {order_id}"})
    if order["user_id"] != actor:
        return None, JSONResponse(status_code=403, content={
            "code": "ORDER_ACCESS_DENIED", "message": "无权访问该订单"})
    return order, None


def _application_for(actor: str, application_id: str):
    app_obj = APPLICATIONS.get(application_id)
    if app_obj is None:
        return None, JSONResponse(status_code=404, content={
            "code": "REFUND_APPLICATION_NOT_FOUND",
            "message": f"未找到退款申请 {application_id}"})
    order, denied = _order_for(actor, app_obj["order_id"])
    if denied is not None:
        if denied.status_code == 403:
            return None, denied
        return None, JSONResponse(status_code=404, content={
            "code": "REFUND_APPLICATION_NOT_FOUND",
            "message": f"未找到退款申请 {application_id}"})
    return app_obj, None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/orders")
def list_orders(x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    return {"orders": [o for o in ORDERS.values()
                       if o["user_id"] == x_actor_id]}


@app.get("/orders/{order_id}")
def get_order(order_id: str, x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    _order, err = _order_for(x_actor_id, order_id)
    if err is not None:
        return err
    return {"order": _order}


@app.get("/orders/{order_id}/logistics")
def get_logistics(order_id: str, x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    order, err = _order_for(x_actor_id, order_id)
    if err is not None:
        return err
    tracking = order.get("tracking_number")
    if not tracking:
        return JSONResponse(200, {"logistics": {"status": "not_shipped"}})
    return {"logistics": {"tracking_number": tracking,
                          "status": "in_transit",
                          "events": [{"time": "2024-01-01 10:00",
                                      "location": "深圳", "description": "已揽收"}]}}


@app.post("/orders/{order_id}/refund-applications")
async def submit_refund_application(order_id: str, request: Request,
                                    x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    order, err = _order_for(x_actor_id, order_id)
    if err is not None:
        return err
    body = await request.json()
    client_request_id = body.get("client_request_id", "")
    if not client_request_id:
        return JSONResponse(status_code=422, content={
            "code": "CLIENT_REQUEST_ID_REQUIRED",
            "message": "缺少请求标识 client_request_id"})
    prior_id = SUBMIT_REQUESTS.get((order_id, client_request_id))
    if prior_id and prior_id in APPLICATIONS:
        return {"application": APPLICATIONS[prior_id], "replayed": True}
    for app_id in ORDER_APPLICATIONS.get(order_id, []):
        existing = APPLICATIONS.get(app_id)
        if existing and existing["status"] in ACTIVE_STATUSES:
            # 已有有效申请：与 mock 网关对齐，200 返回现有申请（幂等），
            # 不再用 409（避免真实后端把 body 里的 application 丢弃）。
            return {"application": existing, "existing": True}
    status = order["status"]
    if status not in REFUNDABLE_ORDER_STATUSES:
        # 不可退与未注册状态：创建路径专用码（不复用撤回专用码）
        return JSONResponse(status_code=409, content={
            "code": "REFUND_ORDER_STATE_NOT_REFUNDABLE",
            "message": "该订单当前状态不支持创建退款申请"})
    app_id = f"RA-{uuid.uuid4().hex[:12]}"
    now = _now()
    application = {
        "application_id": app_id, "order_id": order_id,
        "status": "merchant_reviewing",
        "reason": body.get("reason", ""), "created_at": now, "updated_at": now,
        "can_withdraw": True,
    }
    APPLICATIONS[app_id] = application
    ORDER_APPLICATIONS.setdefault(order_id, []).append(app_id)
    SUBMIT_REQUESTS[(order_id, client_request_id)] = app_id
    return {"application": application}


@app.get("/refund-applications/{application_id}")
def query_application(application_id: str, x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    application, err = _application_for(x_actor_id, application_id)
    if err is not None:
        return err
    return {"applications": [application]}


@app.get("/orders/{order_id}/refund-applications")
def query_order_applications(order_id: str, x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    _order, err = _order_for(x_actor_id, order_id)
    if err is not None:
        return err
    apps = [APPLICATIONS[a] for a in ORDER_APPLICATIONS.get(order_id, [])
            if a in APPLICATIONS]
    return {"applications": apps}


@app.post("/refund-applications/{application_id}/withdraw")
async def cancel_refund_application(application_id: str, request: Request,
                                    x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    application, err = _application_for(x_actor_id, application_id)
    if err is not None:
        return err
    body = await request.json()
    client_request_id = body.get("client_request_id", "")
    if client_request_id and WITHDRAW_REQUESTS.get(
        (application_id, client_request_id)
    ) == application_id:
        return {"application": application, "replayed": True}
    if application["status"] != "merchant_reviewing":
        return JSONResponse(status_code=409, content={
            "code": "REFUND_APPLICATION_NOT_WITHDRAWABLE",
            "message": "只有审核中的退款申请可以撤回"})
    application["status"] = "withdrawn"
    application["can_withdraw"] = False
    application["updated_at"] = _now()
    if client_request_id:
        WITHDRAW_REQUESTS[(application_id, client_request_id)] = application_id
    return {"application": application}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

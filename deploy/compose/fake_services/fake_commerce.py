"""fake_commerce.py：确定性商家服务（HTTPCommerceGateway 契约测试用）。

实现 app/integrations/commerce/http.py 约定的契约：
- 请求头 X-Actor-Id（操作者）+ Authorization: Bearer <服务凭证>
- GET /orders/{order_id}        → 200 {"order": {...}} | 403 {code,message} | 404
- GET /orders                   → 200 {"orders": [...]}（按 actor 过滤）
- GET /orders/{order_id}/logistics → 200 {"logistics": {...}}
- POST /orders/{order_id}/refunds body {reason, refund_id} → 200 {"message": ...}
- 归属：actor 与订单 owner 不符 → 403 ORDER_ACCESS_DENIED；
  无 actor（空 X-Actor-Id）→ 401 IDENTITY_REQUIRED

运行：python -m fake_commerce（默认 0.0.0.0:8080）
"""

from __future__ import annotations

import json
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


def _denied(actor: str) -> Optional[JSONResponse]:
    if not actor:
        return JSONResponse(status_code=401, content={
        "code": "IDENTITY_REQUIRED", "message": "缺少操作者身份"})
    return None


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
    order = ORDERS.get(order_id)
    if order is None:
        return JSONResponse(status_code=404, content={
            "code": "ORDER_NOT_FOUND", "message": f"未找到订单 {order_id}"})
    if order["user_id"] != x_actor_id:
        return JSONResponse(status_code=403, content={
            "code": "ORDER_ACCESS_DENIED", "message": "无权访问该订单"})
    return {"order": order}


@app.get("/orders/{order_id}/logistics")
def get_logistics(order_id: str, x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    order = ORDERS.get(order_id)
    if order is None:
        return JSONResponse(status_code=404, content={
            "code": "ORDER_NOT_FOUND", "message": f"未找到订单 {order_id}"})
    if order["user_id"] != x_actor_id:
        return JSONResponse(status_code=403, content={
            "code": "ORDER_ACCESS_DENIED", "message": "无权访问该订单"})
    tracking = order.get("tracking_number")
    if not tracking:
        return JSONResponse(200, {"logistics": {"status": "not_shipped"}})
    return {"logistics": {"tracking_number": tracking,
                          "status": "in_transit",
                          "events": [{"time": "2024-01-01 10:00",
                                      "location": "深圳", "description": "已揽收"}]}}


@app.post("/orders/{order_id}/refunds")
async def request_refund(order_id: str, request: Request,
                         x_actor_id: str = Header("")):
    denied = _denied(x_actor_id)
    if denied:
        return denied
    order = ORDERS.get(order_id)
    if order is None:
        return JSONResponse(status_code=404, content={
            "code": "ORDER_NOT_FOUND", "message": f"未找到订单 {order_id}"})
    if order["user_id"] != x_actor_id:
        return JSONResponse(status_code=403, content={
            "code": "ORDER_ACCESS_DENIED", "message": "无权操作该订单"})
    body = await request.json()
    refund_id = body.get("refund_id", "")
    if not refund_id:
        return JSONResponse(status_code=422, content={
            "code": "REFUND_ID_REQUIRED", "message": "缺少幂等键 refund_id"})
    return {"message": f"退款申请已提交（refund_id={refund_id}）",
            "refund_id": refund_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
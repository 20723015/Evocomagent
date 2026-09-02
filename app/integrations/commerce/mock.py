"""MockCommerceGateway：开发与离线评测用（2.3）。

数据来自 app/agent/tools/mock_data（ORDERS/LOGISTICS），语义与历史工具
直读完全一致——仅把「归属校验 + 状态机文案」收进网关，工具层不再碰数据。

归属语义（与 2.2 对齐）：
- actor_id 非空 → 校验订单归属（ORDER_ACCESS_DENIED / ORDER_NOT_FOUND）；
- actor_id 为空 → 无身份直调（enforce 关闭时的 CLI/旧脚本兼容），返回数据。
"""

from __future__ import annotations

from typing import Optional

from app.agent.tools.ownership import ORDER_ACCESS_DENIED, ORDER_NOT_FOUND
from app.integrations.commerce.base import CommerceGateway, CommerceResult

from app.agent.tools.mock_data import LOGISTICS, ORDERS


class MockCommerceGateway(CommerceGateway):
    """内存实现：唯一数据源为 mock_data.ORDERS / LOGISTICS。"""

    def __init__(self, orders=None, logistics=None):
        self._orders = orders if orders is not None else ORDERS
        self._logistics = logistics if logistics is not None else LOGISTICS

    # ---------------- 内部辅助 ----------------
    def _order(self, actor_id: str, order_id: str) -> CommerceResult | None:
        """归属校验后的订单。失败返回结果；越权/缺失时返回相应结果。"""
        order = self._orders.get(order_id)
        if order is None:
            return CommerceResult(
                False, ORDER_NOT_FOUND,
                message=f"未找到订单 {order_id}，请核实订单号",
            )
        if actor_id and order.get("user_id") != actor_id:
            return CommerceResult(
                False, ORDER_ACCESS_DENIED,
                message="无权访问该订单（不属于当前用户）",
            )
        return None  # 通过

    # ---------------- 契约实现 ----------------
    def get_order(self, actor_id: str, order_id: str,
                  credentials: Optional[dict] = None) -> CommerceResult:
        denied = self._order(actor_id, order_id)
        if denied is not None:
            return denied
        return CommerceResult(True, data=self._orders[order_id])

    def list_orders(self, actor_id: str,
                    credentials: Optional[dict] = None) -> CommerceResult:
        orders = [
            o for o in self._orders.values()
            if not actor_id or o.get("user_id") == actor_id
        ]
        return CommerceResult(True, data={"orders": orders})

    def get_logistics(self, actor_id: str, order_id: str,
                      credentials: Optional[dict] = None) -> CommerceResult:
        denied = self._order(actor_id, order_id)
        if denied is not None:
            return denied
        tracking_number = self._orders[order_id].get("tracking_number")
        if not tracking_number:
            return CommerceResult(False, message="该订单尚未发货，暂无物流信息")
        logistics = self._logistics.get(tracking_number)
        if not logistics:
            return CommerceResult(
                False, message=f"物流单号 {tracking_number} 暂无轨迹信息",
            )
        return CommerceResult(True, data={"logistics": logistics})

    def request_refund(
        self,
        actor_id: str,
        order_id: str,
        reason: str,
        idempotency_key: str,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        denied = self._order(actor_id, order_id)
        if denied is not None:
            return denied
        order = self._orders[order_id]
        if order["status"] == "refund_processing":
            return CommerceResult(
                False, message="该订单已有退款申请正在处理中，请耐心等待",
            )
        if order["status"] == "pending":
            return CommerceResult(True, data={
                "message": (
                    f"订单 {order_id} 尚未发货，已直接取消并发起退款。"
                    f"退款原因：{reason}。退款将在 1-3 个工作日内原路退回。"
                ),
            })
        return CommerceResult(True, data={
            "message": (
                f"退款申请已提交。订单 {order_id}，退款原因：{reason}。"
                f"预计 1-3 个工作日内审核完成，届时会通知您退货地址。"
            ),
        })
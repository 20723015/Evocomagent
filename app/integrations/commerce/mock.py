"""MockCommerceGateway：开发与离线评测用（2.3；A+ 退款申请改造）。

数据来自 app/agent/tools/mock_data（ORDERS/LOGISTICS）。退款改为独立的
**申请存储**：创建 / 查询 / 撤回均按申请对象操作，并用申请级幂等
（order_id + client_request_id）去重，不再用订单状态代替完整申请状态。

申请语义（订单状态可退性按 base.REFUNDABLE_ORDER_STATUSES 单一来源判定）：
- 可退状态（未发货/已发货/运输中/已签收）：统一创建可撤回的
  ``merchant_reviewing`` 申请（``can_withdraw=True``），订单状态不做任何
  前置流转；只有权威到账回执才能进入 ``refunded``；
- 其余状态（refund_processing/refunded/cancelled 等）与未注册状态：拒绝创建
  （``REFUND_ORDER_STATE_NOT_REFUNDABLE``，不复用撤回专用码）；
- 撤回：仅 ``merchant_reviewing`` 可撤回 → ``withdrawn``；
- 已有进行中申请：返回现有申请（REFUND_APPLICATION_EXISTS），不重复创建。

归属语义（与 2.2 对齐）：
- actor_id 非空 → 校验订单归属（ORDER_ACCESS_DENIED / ORDER_NOT_FOUND）；
- actor_id 为空 → 无身份直调（enforce 关闭时的 CLI/旧脚本兼容），返回数据。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from app.agent.tools.ownership import ORDER_ACCESS_DENIED, ORDER_NOT_FOUND
from app.integrations.commerce.base import (
    ACTIVE_APPLICATION_STATUSES,
    APPLICATION_MERCHANT_REVIEWING,
    APPLICATION_WITHDRAWN,
    REFUNDABLE_ORDER_STATUSES,
    REFUND_APPLICATION_EXISTS,
    REFUND_APPLICATION_NOT_FOUND,
    REFUND_APPLICATION_NOT_WITHDRAWABLE,
    REFUND_ORDER_STATE_NOT_REFUNDABLE,
    CommerceGateway,
    CommerceResult,
)

from app.agent.tools.mock_data import LOGISTICS, ORDERS, get_dataset


# 订单列表默认返回上限（P2-1 扩容后全量可达千级）：网关层也设界，避免
# 任何直接调用方（脚本/测试/新工具）意外拿到千级列表。None = 不截断。
DEFAULT_ORDER_LIST_LIMIT = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MockCommerceGateway(CommerceGateway):
    """内存实现：订单来自 mock_data；退款申请存独立字典。"""

    def __init__(self, orders=None, logistics=None):
        # 深拷贝种子数据：退款状态更新只影响本实例，不污染模块级种子
        import copy

        if orders is None or logistics is None:
            # 扩容数据集（P2-1）：种子 + 生成条目（缺生成文件时即内置种子，
            # 行为与扩容前一致）。订单/物流同源取用，避免「商品已扩容、
            # 订单仍只有 5 条」的空转接线。
            dataset = get_dataset()
            if orders is None:
                orders = dataset["orders"]
            if logistics is None:
                logistics = dataset["logistics"]
        self._orders = copy.deepcopy(orders)
        self._logistics = logistics
        # 申请存储：application_id → application
        self._applications: dict[str, dict] = {}
        # order_id → [application_id]（按创建时间倒序追加）
        self._order_applications: dict[str, list[str]] = {}
        # (order_id, client_request_id) → application_id（创建幂等）
        self._submit_requests: dict[tuple[str, str], str] = {}
        # (application_id, client_request_id) → application_id（撤回幂等）
        self._withdraw_requests: dict[tuple[str, str], str] = {}

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

    def _application_of(self, actor_id: str, application_id: str) -> CommerceResult:
        app = self._applications.get(application_id)
        if app is None:
            return CommerceResult(
                False, REFUND_APPLICATION_NOT_FOUND,
                message=f"未找到退款申请 {application_id}",
            )
        denied = self._order(actor_id, str(app.get("order_id", "")))
        if denied is not None:
            # 订单缺失同样按申请不存在处理，避免泄露他人申请存在性
            if denied.code == ORDER_ACCESS_DENIED:
                return denied
            return CommerceResult(
                False, REFUND_APPLICATION_NOT_FOUND,
                message=f"未找到退款申请 {application_id}",
            )
        return CommerceResult(True, data=app)

    def _active_application(self, order_id: str) -> dict | None:
        for app_id in self._order_applications.get(order_id, []):
            app = self._applications.get(app_id)
            if app and app.get("status") in ACTIVE_APPLICATION_STATUSES:
                return app
        return None

    # ---------------- 契约实现 ----------------
    def get_order(self, actor_id: str, order_id: str,
                  credentials: Optional[dict] = None) -> CommerceResult:
        denied = self._order(actor_id, order_id)
        if denied is not None:
            return denied
        return CommerceResult(True, data=self._orders[order_id])

    def list_orders(self, actor_id: str,
                    credentials: Optional[dict] = None,
                    limit: Optional[int] = DEFAULT_ORDER_LIST_LIMIT,
                    ) -> CommerceResult:
        """订单概要列表：按创建时间倒序（最近优先），受 limit 约束。

        P2-1 扩容后单用户订单可达数十条、全量可达千级——不设界会让工具结果
        撑爆 Agent 上下文。limit=None 表示不截断（仅内部/测试用）。
        """
        orders = [
            o for o in self._orders.values()
            if not actor_id or o.get("user_id") == actor_id
        ]
        orders.sort(key=lambda o: str(o.get("created_at", "")), reverse=True)
        total = len(orders)
        if limit is not None and limit > 0:
            orders = orders[:limit]
        return CommerceResult(
            True, data={"orders": orders, "total": total},
        )

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

    def submit_refund_application(
        self,
        actor_id: str,
        order_id: str,
        reason: str,
        client_request_id: str,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        denied = self._order(actor_id, order_id)
        if denied is not None:
            return denied

        # 申请级幂等：同 (order, request) 重放返回首次创建结果
        if client_request_id:
            prior_id = self._submit_requests.get((order_id, client_request_id))
            if prior_id and prior_id in self._applications:
                app = self._applications[prior_id]
                return CommerceResult(
                    True, data={"application": app, "replayed": True},
                    message="该退款申请已创建（幂等重放）",
                )

        existing = self._active_application(order_id)
        if existing is not None:
            return CommerceResult(
                True, REFUND_APPLICATION_EXISTS,
                data={"application": existing, "existing": True},
                message="该订单已有进行中的退款申请，返回现有申请",
            )

        order = self._orders[order_id]
        status = str(order.get("status", ""))
        now = _now()
        # 创建语义：可退状态统一进入商家审核且订单状态不动；其余 fail-closed
        if status not in REFUNDABLE_ORDER_STATUSES:
            return CommerceResult(
                False, REFUND_ORDER_STATE_NOT_REFUNDABLE,
                message="该订单当前状态不支持创建退款申请",
            )

        application_id = f"RA-{uuid.uuid4().hex[:12]}"
        application = {
            "application_id": application_id,
            "order_id": order_id,
            "status": APPLICATION_MERCHANT_REVIEWING,
            "reason": reason,
            "created_at": now,
            "updated_at": now,
            "can_withdraw": True,
        }
        self._applications[application_id] = application
        self._order_applications.setdefault(order_id, []).append(application_id)
        if client_request_id:
            self._submit_requests[(order_id, client_request_id)] = application_id
        return CommerceResult(
            True, data={"application": application},
            message="退款申请已创建",
        )

    def query_refund_application(
        self,
        actor_id: str,
        application_id: Optional[str] = None,
        order_id: Optional[str] = None,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        if bool(application_id) == bool(order_id):
            return CommerceResult(
                False, "REFUND_APPLICATION_QUERY_INVALID",
                message="必须且只能提供一个查询条件（application_id 或 order_id）",
            )
        if application_id:
            found = self._application_of(actor_id, application_id)
            if not found.success:
                return found
            return CommerceResult(
                True, data={"applications": [found.data]},
            )
        # 按订单号：返回该订单下全部申请（可能为空数组）
        denied = self._order(actor_id, order_id or "")
        if denied is not None:
            return denied
        apps = [
            self._applications[app_id]
            for app_id in self._order_applications.get(order_id or "", [])
            if app_id in self._applications
        ]
        return CommerceResult(True, data={"applications": apps})

    def cancel_refund_application(
        self,
        actor_id: str,
        application_id: str,
        client_request_id: str,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        found = self._application_of(actor_id, application_id)
        if not found.success:
            return found
        # 撤回幂等：同 (application, request) 重放
        if client_request_id:
            prior = self._withdraw_requests.get((application_id, client_request_id))
            if prior == application_id:
                return CommerceResult(
                    True, data={"application": self._applications[application_id],
                                "replayed": True},
                    message="撤回已提交（幂等重放）",
                )
        app = self._applications[application_id]
        if app.get("status") != APPLICATION_MERCHANT_REVIEWING:
            return CommerceResult(
                False, REFUND_APPLICATION_NOT_WITHDRAWABLE,
                message="只有审核中的退款申请可以撤回",
            )
        app["status"] = APPLICATION_WITHDRAWN
        app["can_withdraw"] = False
        app["updated_at"] = _now()
        if client_request_id:
            self._withdraw_requests[(application_id, client_request_id)] = application_id
        return CommerceResult(
            True, data={"application": app},
            message="退款申请已撤回",
        )

"""CommerceGateway 协议与结果类型（2.3；A+ 退款申请改造）。

错误码复用工具层的机器可判定枚举（IDENTITY_REQUIRED /
ORDER_ACCESS_DENIED / ORDER_NOT_FOUND / REFUND_APPLICATION_NOT_FOUND /
REFUND_APPLICATION_NOT_WITHDRAWABLE），保证端到端判定一致。

退款业务改为「申请制」：Agent 只创建、查询、撤回申请，商家审核与资金到账
由外部业务系统推进。申请状态固定为：
merchant_reviewing / approved / rejected / refund_processing / refunded / withdrawn。
只有权威到账回执才能进入 refunded。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# 退款申请状态（冻结枚举）
APPLICATION_MERCHANT_REVIEWING = "merchant_reviewing"
APPLICATION_APPROVED = "approved"
APPLICATION_REJECTED = "rejected"
APPLICATION_REFUND_PROCESSING = "refund_processing"
APPLICATION_REFUNDED = "refunded"
APPLICATION_WITHDRAWN = "withdrawn"

APPLICATION_STATUSES = frozenset({
    APPLICATION_MERCHANT_REVIEWING,
    APPLICATION_APPROVED,
    APPLICATION_REJECTED,
    APPLICATION_REFUND_PROCESSING,
    APPLICATION_REFUNDED,
    APPLICATION_WITHDRAWN,
})

# 申请错误码
REFUND_APPLICATION_NOT_FOUND = "REFUND_APPLICATION_NOT_FOUND"
REFUND_APPLICATION_NOT_WITHDRAWABLE = "REFUND_APPLICATION_NOT_WITHDRAWABLE"
REFUND_APPLICATION_EXISTS = "REFUND_APPLICATION_EXISTS"
REFUND_APPLICATION_QUERY_INVALID = "REFUND_APPLICATION_QUERY_INVALID"
# 创建路径专用：订单状态不支持创建申请（不复用撤回专用码 NOT_WITHDRAWABLE）
REFUND_ORDER_STATE_NOT_REFUNDABLE = "REFUND_ORDER_STATE_NOT_REFUNDABLE"

# 可创建退款申请的订单状态（单一来源；各网关均从此派生）：
# 无论是否发货，创建路径一致——统一生成可撤回的 merchant_reviewing 申请，
# 订单状态不做任何前置流转。未列出的状态一律 fail-closed，拒绝创建
# （REFUND_ORDER_STATE_NOT_REFUNDABLE，不复用撤回专用码）。
REFUNDABLE_ORDER_STATUSES = frozenset({
    "pending",
    "shipped",
    "in_transit",
    "delivered",
})

# 仍有进行中申请的状态（创建幂等：返回现有申请）
ACTIVE_APPLICATION_STATUSES = frozenset({
    APPLICATION_MERCHANT_REVIEWING,
    APPLICATION_APPROVED,
    APPLICATION_REFUND_PROCESSING,
})


@dataclass
class CommerceResult:
    """网关调用结果。

    - success：业务是否成功；
    - code：机器可判定错误码（None=无特定码，如后端 500）；
    - data：成功时的业务数据；
    - message：人类可读结果/错误文案；
    - indeterminate：写入类操作结果未知——调用方必须停止自动重试，
      转人工对账（2.3 契约）。
    """

    success: bool
    code: Optional[str] = None
    data: Optional[dict] = None
    message: str = ""
    indeterminate: bool = False


class CommerceGateway(ABC):
    """商家业务网关契约。

    所有方法第一参数为 actor_id（操作者身份，由工具层从
    ToolContext.user_id 传入；空串=无身份直调，仅 mock 后端兼容开发）。
    credentials 为请求级凭证（Bearer），禁止进入 prompt/轨迹/日志。

    写请求（submit/cancel）由调用方显式传入系统生成的 client_request_id，
    网关不得内置回放语义之外的自动重试。
    """

    @abstractmethod
    def get_order(self, actor_id: str, order_id: str,
                  credentials: Optional[dict] = None) -> CommerceResult:
        """查询单个订单详情（含归属校验）。"""

    @abstractmethod
    def list_orders(self, actor_id: str,
                    credentials: Optional[dict] = None,
                    limit: Optional[int] = None) -> CommerceResult:
        """查询操作者的订单列表（最近优先、有界；limit=None 表示不截断）。

        实现须给默认上限：P2-1 扩容后全量订单可达千级，无界返回会撑爆
        Agent 上下文。
        """

    @abstractmethod
    def get_logistics(self, actor_id: str, order_id: str,
                      credentials: Optional[dict] = None) -> CommerceResult:
        """查询订单物流轨迹。"""

    @abstractmethod
    def submit_refund_application(
        self, actor_id: str, order_id: str, reason: str,
        client_request_id: str,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        """创建退款申请（幂等键 client_request_id 由调用方显式传入）。"""

    @abstractmethod
    def query_refund_application(
        self, actor_id: str,
        application_id: Optional[str] = None,
        order_id: Optional[str] = None,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        """查询退款申请（application_id 与 order_id 必须且只能提供一个）。

        统一返回 ``{"applications": [...]}``。
        """

    @abstractmethod
    def cancel_refund_application(
        self, actor_id: str, application_id: str,
        client_request_id: str,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        """撤回退款申请（仅 merchant_reviewing 可撤回）。"""

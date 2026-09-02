"""CommerceGateway 协议与结果类型（2.3）。

错误码复用工具层 2.2 的机器可判定枚举（IDENTITY_REQUIRED /
ORDER_ACCESS_DENIED / ORDER_NOT_FOUND / CONFIRMATION_REQUIRED /
CONFIRMATION_INVALID），保证端到端判定一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class CommerceResult:
    """网关调用结果。

    - success：业务是否成功；
    - code：机器可判定错误码（None=无特定码，如后端 500）；
    - data：成功时的业务数据；
    - message：人类可读结果/错误文案；
    - indeterminate：写入类操作（退款）结果未知——调用方必须停止自动重试，
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
    """

    @abstractmethod
    def get_order(self, actor_id: str, order_id: str,
                  credentials: Optional[dict] = None) -> CommerceResult:
        """查询单个订单详情（含归属校验）。"""

    @abstractmethod
    def list_orders(self, actor_id: str,
                    credentials: Optional[dict] = None) -> CommerceResult:
        """查询操作者的订单列表。"""

    @abstractmethod
    def get_logistics(self, actor_id: str, order_id: str,
                      credentials: Optional[dict] = None) -> CommerceResult:
        """查询订单物流轨迹。"""

    @abstractmethod
    def request_refund(
        self,
        actor_id: str,
        order_id: str,
        reason: str,
        idempotency_key: str,
        credentials: Optional[dict] = None,
    ) -> CommerceResult:
        """发起退款（幂等键由调用方显式传入，禁止网关内置回放语义）。"""
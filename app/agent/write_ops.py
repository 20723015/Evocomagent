"""写操作状态机（单 Agent 全量优化计划·阶段C）：高风险写操作程序化约束。

退款流程状态机：
    collecting → ownership_verified → awaiting_confirmation
              → executing → committed | rejected | indeterminate

规则（程序强制，不依赖模型自觉遵守流程）：
- 缺订单号或原因：只追问（状态不推进），工具不执行；
- 未验证订单归属：不得生成确认令牌（归属校验失败单加入拒绝名单）；
- 未取得有效确认令牌：不得提交写操作（apply_refund 两段式天然强制，
  状态机额外拦截「无 token 却要求执行」的幻觉参数）；
- 无本轮 committed 工具证据：最终回复禁止声称退款成功（确定性改写）；
- indeterminate：强制 requires_human，记录订单号与幂等键供人工对账；
- 越权/拒绝结果：本轮内禁止对该订单继续重试；
- 后续新增写工具必须注册状态机，否则默认禁止执行。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum


class WritePhase(str, Enum):
    COLLECTING = "collecting"
    OWNERSHIP_VERIFIED = "ownership_verified"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    COMMITTED = "committed"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


# 写工具 → 状态机注册表（新写工具未注册 → 默认禁止执行）
WRITE_TOOL_STATE_MACHINES: dict[str, str] = {
    "apply_refund": "refund",
}

_MISSING_PARAMS_ERROR = json.dumps({
    "error": "REFUND_PARAMS_REQUIRED",
    "message": "缺少退款必需参数（订单号或原因）：不要执行退款，先向用户确认订单号与退款原因",
}, ensure_ascii=False)

_DENIED_RETRY_ERROR = json.dumps({
    "error": "WRITE_RETRY_FORBIDDEN",
    "message": "该订单已被拒绝操作，禁止重试；也不要向用户披露该订单内容，建议转人工",
}, ensure_ascii=False)

_UNREGISTERED_ERROR = json.dumps({
    "error": "WRITE_TOOL_UNREGISTERED",
    "message": "该写工具未注册状态机，禁止执行",
}, ensure_ascii=False)

_COMMIT_REQUIRED_ERROR = json.dumps({
    "error": "CONFIRMATION_REQUIRED",
    "message": "提交写操作必须携带有效 confirmation_token；请先完成用户确认",
}, ensure_ascii=False)

# 最终回复中的「退款成功」宣称（无 committed 证据时禁止出现）
_SUCCESS_CLAIM_RE = re.compile(
    r"(退款成功|已退款|退款已(到账|处理|完成|退回|发起)|已为您(办理|完成|处理)退款"
    r"|已成功(退|办理)|退款申请已(通过|完成)|钱已退)"
)
_SAFE_REWRITE_REPLY = (
    "您的退款请求我已记录，但目前系统尚未确认退款完成。"
    "为避免误导，请以退款到账通知为准；如需立即确认，我将为您转接人工客服核实处理进度。"
)


@dataclass
class WriteOpRecord:
    """一次写操作的状态轨迹（审计/对账用，脱敏后进日志）。"""

    tool: str
    order_id: str = ""
    idempotency_key: str = ""
    phase: WritePhase = WritePhase.COLLECTING
    confirmation_issued: bool = False
    detail: dict = field(default_factory=dict)


class RefundWriteTracker:
    """单轮退款状态机（每轮新建；跨轮状态由确认令牌与幂等账本承担）。"""

    tool_name = "apply_refund"

    def __init__(self) -> None:
        self.phase: WritePhase = WritePhase.COLLECTING
        self.records: list[WriteOpRecord] = []
        self.denied_orders: set[str] = set()
        self.indeterminate: list[dict] = []
        self.committed: list[dict] = []
        self.tokens_issued: set[str] = set()
        self.ownership_verified_orders: set[str] = set()

    # ---------- 执行前拦截 ----------
    def check(self, name: str, arguments: dict) -> str | None:
        """写工具执行前校验；返回 None=放行，返回 JSON 字符串=拦截（工具不执行）。"""
        if WRITE_TOOL_STATE_MACHINES.get(name) != "refund":
            return _UNREGISTERED_ERROR
        order_id = str(arguments.get("order_id", "") or "").strip()
        reason = str(arguments.get("reason", "") or "").strip()
        token = str(arguments.get("confirmation_token", "") or "").strip()

        if not order_id or not reason:
            # 缺订单号或原因：只追问，不执行退款
            return _MISSING_PARAMS_ERROR
        if order_id in self.denied_orders:
            # 越权/拒绝后禁止模型继续重试
            return _DENIED_RETRY_ERROR
        if token:
            # 携带确认令牌 = 提交段；令牌有效性由两段式工具校验
            record = self._current(order_id)
            if record is not None:
                record.phase = WritePhase.EXECUTING
            return None
        # 无令牌 = 签发段（第一段）：归属校验由工具侧两段式强制（签发前先过
        # 网关 get_order 归属检查，fail-closed）——状态机不重复设门，只记录
        # 阶段推进，保证跨轮确认（令牌在上一轮签发）不被误拦。
        return None

    # ---------- 结果观察 ----------
    def observe_query_order(self, order_id: str, ok: bool, access_denied: bool) -> None:
        """query_order 结果回填：归属验证 / 拒绝名单。"""
        if not order_id:
            return
        if access_denied:
            self.denied_orders.add(order_id)
            return
        if ok:
            self.ownership_verified_orders.add(order_id)
            if self.phase == WritePhase.COLLECTING:
                self.phase = WritePhase.OWNERSHIP_VERIFIED

    def observe_refund(self, arguments: dict, result: dict) -> None:
        """apply_refund 结果回填；推进状态机。"""
        order_id = str(arguments.get("order_id", "") or "").strip()
        token = str(arguments.get("confirmation_token", "") or "").strip()
        record = self._current(order_id) or self._new_record(order_id)

        status = str(result.get("status", "") or "")
        code = str(result.get("code", "") or "")
        success = result.get("success") is True

        if token:
            self.tokens_issued.discard(token)

        if status == "pending_confirmation" or code == "CONFIRMATION_REQUIRED":
            record.phase = WritePhase.AWAITING_CONFIRMATION
            record.confirmation_issued = True
            record.idempotency_key = str(
                result.get("idempotency_key") or result.get("refund_id") or ""
            )
            self.tokens_issued.add(str(result.get("confirmation_token") or ""))
            self.phase = WritePhase.AWAITING_CONFIRMATION
            return
        if status == "indeterminate":
            record.phase = WritePhase.INDETERMINATE
            self.phase = WritePhase.INDETERMINATE
            entry = {
                "tool": self.tool_name,
                "order_id": order_id,
                "idempotency_key": str(
                    arguments.get("idempotency_key")
                    or arguments.get("refund_id")
                    or record.idempotency_key or ""
                ),
            }
            self.indeterminate.append(entry)
            return
        if success and result.get("confirmed"):
            record.phase = WritePhase.COMMITTED
            self.phase = WritePhase.COMMITTED
            self.committed.append({
                "order_id": order_id,
                "idempotency_key": str(result.get("idempotency_key") or ""),
            })
            return
        if success:
            # 未走确认段的直接成功（refund_confirmation_required=False 开发路径）
            record.phase = WritePhase.COMMITTED
            self.phase = WritePhase.COMMITTED
            self.committed.append({"order_id": order_id, "idempotency_key": ""})
            return
        # 失败：越权/身份 → 拒绝名单；其余 rejected
        if code in ("ORDER_ACCESS_DENIED", "IDENTITY_REQUIRED"):
            self.denied_orders.add(order_id)
        record.phase = WritePhase.REJECTED
        if self.phase != WritePhase.COMMITTED:
            self.phase = WritePhase.REJECTED

    # ---------- 终答校验 ----------
    def final_reply_guard(self, reply: str) -> tuple[str, bool]:
        """无 committed 证据时禁止声称退款成功。

        返回 (safe_reply, rewritten)；未改写时原样返回。
        """
        if self.committed or not _SUCCESS_CLAIM_RE.search(reply or ""):
            return reply, False
        return _SAFE_REWRITE_REPLY, True

    @property
    def has_indeterminate(self) -> bool:
        return bool(self.indeterminate)

    def _current(self, order_id: str) -> WriteOpRecord | None:
        for record in reversed(self.records):
            if record.order_id == order_id:
                return record
        return None

    def _new_record(self, order_id: str) -> WriteOpRecord:
        record = WriteOpRecord(tool=self.tool_name, order_id=order_id)
        self.records.append(record)
        return record


class WriteOpTracker:
    """单轮写操作总闸：按注册表分发到具体状态机；未注册写工具一律拦截。"""

    def __init__(self) -> None:
        self.refund = RefundWriteTracker()
        self.blocked_calls: list[dict] = []  # 被拦截的写调用（审计）

    def check(self, name: str, arguments: dict) -> str | None:
        if name not in WRITE_TOOL_STATE_MACHINES:
            return _UNREGISTERED_ERROR
        return self.refund.check(name, arguments)

    def observe(self, name: str, arguments: dict, result_payload: dict) -> None:
        """执行结果回填（读工具负责转发：query_order 归属、apply_refund 主流程）。"""
        if name == "query_order":
            order_id = str(arguments.get("order_id", "") or "").strip()
            access_denied = str(result_payload.get("code", "")) in (
                "ORDER_ACCESS_DENIED", "IDENTITY_REQUIRED",
            ) or result_payload.get("error") == "ORDER_ACCESS_DENIED"
            ok = result_payload.get("success") is True or (
                isinstance(result_payload.get("order"), dict)
            )
            self.refund.observe_query_order(order_id, bool(ok), bool(access_denied))
        elif name == "apply_refund":
            self.refund.observe_refund(arguments, result_payload)

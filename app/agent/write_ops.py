"""写操作状态机（申请制）：提交 / 业务结果两层。

退款链路由程序强制、不依赖模型自觉：

1. **提交层**（工具执行）：身份校验通过后直接调用网关创建/撤回申请；
   请求标识（client_request_id）由工具层为每次调用自生成，用作申请级
   幂等键与超时对账锚点。写超时保持 ``indeterminate``（结果未知），禁止
   自动换键重试，改用查询工具按订单号/申请编号对账。
2. **业务结果层**（本模块观察）：网关回执里的申请状态
   （merchant_reviewing / approved / rejected / refund_processing /
   refunded / withdrawn）是终答唯一可用的业务事实。

终答状态白名单（无对应证据禁止越级宣称）：

- ``merchant_reviewing``：只能说「申请已提交，等待审核」；
- ``refund_processing``：只能说「退款处理中」；
- ``refunded``：只有权威到账回执才能说「退款已完成」；
- 未知/indeterminate：不得宣称任何完成态，转人工对账。

新写工具必须注册 ``WRITE_TOOL_STATE_MACHINES``，否则默认禁止执行。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.agent.write_registry import state_machines, target_of

# 写工具 → 状态机注册表（单一来源在 write_registry；此处派生 + 兼容导出）
WRITE_TOOL_STATE_MACHINES: dict[str, str] = state_machines()

# 申请业务状态（网关冻结枚举）
APPLICATION_STATUSES = frozenset({
    "merchant_reviewing", "approved", "rejected",
    "refund_processing", "refunded", "withdrawn",
})
ACTIVE_STATUSES = frozenset({
    "merchant_reviewing", "approved", "refund_processing",
})
# 草稿态（P1-2 两阶段协议）：不是网关业务状态，只是「已登记、待用户确认」。
# 单列一个状态位是为了让终答守卫能区分「草稿已登记」（可如实说）与
# 「申请已提交」（必须有网关回执）——两阶段下后者是最常见的越级宣称。
DRAFT_STATUS = "draft"
# 草稿被用户取消（P1-2）：登记为独立状态位，供终答守卫放行「草稿已作废」类
# 如实话术。WriteOpTracker 是逐轮的，取消轮本身不会重新观察到草稿态——
# 缺这一位时，模型在取消轮说「已为您作废草稿」会被误判为越级宣称并改写升级。
DRAFT_CANCELLED_STATUS = "draft_cancelled"

_MISSING_PARAMS_ERROR = json.dumps({
    "error": "REFUND_PARAMS_REQUIRED",
    "message": "缺少退款写操作必需参数：不要执行，先向用户确认订单号/申请编号与原因",
}, ensure_ascii=False)

_DENIED_RETRY_ERROR = json.dumps({
    "error": "WRITE_RETRY_FORBIDDEN",
    "message": "该目标已被拒绝操作，禁止重试；也不要向用户披露该目标内容，建议转人工",
}, ensure_ascii=False)

_UNREGISTERED_ERROR = json.dumps({
    "error": "WRITE_TOOL_UNREGISTERED",
    "message": "该写工具未注册状态机，禁止执行",
}, ensure_ascii=False)

_SAFE_REWRITE_REPLY = (
    "您的退款申请我已记录，但目前系统尚未返回对应的业务回执。"
    "为避免误导，请以商家与支付系统的通知为准；"
    "如需立即确认，我可以为您查询申请状态或转接人工客服核实处理进度。"
)

# 终答宣称 → 需要的最小业务证据（观察到任一即可）
# 顺序敏感：先判「更强的宣称」，最后才是「草稿登记」这类弱措辞。
_CLAIM_RULES: tuple[tuple[re.Pattern, frozenset[str]], ...] = (
    (
        # 状态词后接「政策名词」时不算宣称：模型引用《退款到账时效分档表》这类
        # **文档标题**会被误判成「退款已到账」（2026-09-18 记忆消融实测：
        # memory_synonym_paraphrase 两臂同因改写失败）。负向前瞻排除
        # 时效/时间/周期/说明/规则/表/流程 等标题性后缀。
        re.compile(
            r"退款(?:已)?(?:完成|到账|退回)(?!(?:时效|时间|周期|说明|规则|表|流程|分档|政策|指南))|"
            r"已退款|钱已退|退款已成功|已成功退款|退款已经到账|款项已退回"
        ),
        frozenset({"refunded"}),
    ),
    (
        re.compile(
            r"(?:退款申请|退款)(?:已|已经)?(?:审核)?通过|退款已批准|"
            r"商家已(?:同意|批准)退款"
        ),
        frozenset({"approved", "refunded"}),
    ),
    (
        re.compile(
            r"退款(?:正在)?处理中|正在退款|正在处理退款|"
            r"履约已取消|订单已取消|已取消订单"
        ),
        frozenset({"refund_processing", "refunded"}),
    ),
    (
        # P1-2：两阶段协议下「已提交/已创建/已受理/已进入审核」必须有网关回执。
        # 草稿态（business_statuses={"draft"}）不满足本规则 → 确定性改写，
        # 用户不会在未确认时被告知「已提交」。
        re.compile(
            r"退款申请已(?:提交|创建|受理)|申请已提交|已提交退款申请|"
            r"等待(?:商家)?审核|商家审核中|已进入审核"
        ),
        frozenset({"merchant_reviewing", "approved", "refund_processing", "refunded"}),
    ),
    (
        # 取消类措辞（「草稿已作废/已为您取消该草稿/未提交」）：取消轮如实陈述。
        # 无任何状态时不放行（模型凭空说「已取消」仍要改写）。
        re.compile(
            r"草稿已(?:作废|取消|失效)|已(?:为您)?(?:作废|取消)(?:该|这笔)?(?:退款)?草稿|"
            r"退款申请草稿(?:已|将)?(?:作废|取消)|尚未提交|未提交任何申请"
        ),
        frozenset({
            DRAFT_CANCELLED_STATUS, DRAFT_STATUS, "merchant_reviewing",
            "approved", "refund_processing", "refunded",
        }),
    ),
    (
        # 草稿登记类措辞（「已为您登记/已生成草稿/待您确认」）：有草稿即可如实说，
        # 避免把合规的草稿话术误改写（P1-2 之前「登记」被归在强宣称里）。
        re.compile(
            r"退款申请已(?:登记|记录)|已(?:为您)?登记|已生成.{0,6}草稿|"
            r"草稿已生成|待您确认|请您确认|请确认后"
        ),
        frozenset({
            DRAFT_STATUS, DRAFT_CANCELLED_STATUS, "merchant_reviewing",
            "approved", "refund_processing", "refunded",
        }),
    ),
)


@dataclass
class WriteOpRecord:
    """一次写操作的三层轨迹（审计/对账用，脱敏后进日志）。"""

    tool: str
    target: str = ""
    client_request_id: str = ""
    phase: str = "collected"  # collected|submitted|rejected|indeterminate
    application_status: str = ""
    detail: dict = field(default_factory=dict)


class WriteOpTracker:
    """单轮写操作总闸：按注册表分发；未注册写工具一律拦截。

    只跟踪本轮（跨轮状态由会话草稿 ``SessionState.pending_write`` 承担）。
    """

    def __init__(self) -> None:
        self.records: list[WriteOpRecord] = []
        self.denied_targets: set[str] = set()
        self.indeterminate: list[dict] = []
        self.business_statuses: set[str] = set()
        self.blocked_calls: list[dict] = []  # 被拦截的写调用（审计）
        self.draft_targets: set[str] = set()  # 已登记待确认草稿的目标（P1-2）

    # ---------- 执行前拦截 ----------
    def check(self, name: str, arguments: dict) -> str | None:
        """写工具执行前校验；None=放行，JSON 字符串=拦截（工具不执行）。"""
        if name not in WRITE_TOOL_STATE_MACHINES:
            return _UNREGISTERED_ERROR
        if name == "submit_refund_application":
            if not str(arguments.get("order_id", "") or "").strip():
                return _MISSING_PARAMS_ERROR
            if not str(arguments.get("reason", "") or "").strip():
                return _MISSING_PARAMS_ERROR
        elif not str(arguments.get("application_id", "") or "").strip():
            return _MISSING_PARAMS_ERROR
        target = self._target_of(name, arguments)
        if target in self.denied_targets:
            return _DENIED_RETRY_ERROR
        return None

    # ---------- 结果观察 ----------
    def observe_query_order(self, order_id: str, ok: bool, access_denied: bool) -> None:
        """query_order 结果回填：归属验证 / 拒绝名单。"""
        if not order_id:
            return
        if access_denied:
            self.denied_targets.add(order_id)

    def observe(self, name: str, arguments: dict, result: dict) -> None:
        """写工具结果回填；推进三层状态。"""
        target = self._target_of(name, arguments)
        code = str(result.get("code", "") or "")
        status = str(result.get("status", "") or "")
        record = WriteOpRecord(
            tool=name, target=target,
            client_request_id=str(result.get("client_request_id", "") or ""),
        )

        # 草稿态（P1-2 两阶段协议）：首次调用只登记草稿、未落库。
        # 必须先于 success 分支判定——草稿结果 success=True 但没有业务回执，
        # 记成 submitted 会让终答守卫误放行「已提交」话术。
        if status == "awaiting_confirmation":
            record.phase = "draft"
            record.detail = {"code": code}
            self.draft_targets.add(target)
            self.business_statuses.add(DRAFT_STATUS)
            self.records.append(record)
            return

        # 草稿被取消（P1-2）：非业务失败，登记取消态供守卫放行如实话术
        if code == "REFUND_DRAFT_CANCELLED":
            record.phase = "cancelled"
            record.detail = {"code": code}
            self.business_statuses.add(DRAFT_CANCELLED_STATUS)
            self.records.append(record)
            return

        # 网关超时（gateway_failure）返回 success=False + status=indeterminate：
        # 结果未知而非被拒，必须先于失败分支判定，否则对账清单丢失。
        if status == "indeterminate":
            record.phase = "indeterminate"
            self.indeterminate.append({
                "tool": name,
                "order_id": str(arguments.get("order_id", "") or ""),
                "application_id": str(arguments.get("application_id", "") or ""),
                "client_request_id": str(
                    result.get("client_request_id")
                    or arguments.get("client_request_id") or ""
                ),
            })
            self.records.append(record)
            return

        if result.get("success") is False:
            if code in ("ORDER_ACCESS_DENIED", "IDENTITY_REQUIRED"):
                self.denied_targets.add(target)
            record.phase = "rejected"
            record.detail = {"code": code}
            self.records.append(record)
            return

        # 提交成功：记录业务结果层状态
        record.phase = "submitted"
        if status in APPLICATION_STATUSES:
            record.application_status = status
            self.business_statuses.add(status)
        self.records.append(record)

    # ---------- 终答校验 ----------
    def final_reply_guard(self, reply: str) -> tuple[str, bool]:
        """无对应业务证据时禁止越级宣称退款状态。

        返回 (safe_reply, rewritten)；未改写时原样返回。
        """
        text = reply or ""
        for pattern, required in _CLAIM_RULES:
            if not pattern.search(text):
                continue
            if self.business_statuses & required:
                continue
            return _SAFE_REWRITE_REPLY, True
        return reply, False

    @property
    def has_indeterminate(self) -> bool:
        return bool(self.indeterminate)

    @property
    def committed(self) -> list[WriteOpRecord]:
        """已提交成功的写记录（供持久化 pending/业务状态判别）。"""
        return [r for r in self.records if r.phase == "submitted"]

    @staticmethod
    def _target_of(name: str, arguments: dict) -> str:
        # 目标标识派生自 write_registry（target_arg 字段）
        return target_of(name, arguments)

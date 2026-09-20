"""写操作确认闸门（能力补全计划·P1-2）：服务端确定性判定用户对待确认写操作的表态。

**为什么在工具层**：LangGraph 官方客服教程的硬原则——「政策执行必须发生在
工具/API 内部，因为 LLM 总可能无视提示词」。此前「退款前必须与用户确认」只写在
``skills/process-return/SKILL.md`` 里，属模型自律；本模块把确认下沉为工具层的
强制位：**没有确认轮，写操作不可能落库**（不依赖模型是否记得带某个参数）。

两阶段协议：

1. **草稿轮**：首次调用 ``submit_refund_application`` 不调用网关，只登记草稿
   （``client_request_id`` 在此生成并复用为申请级幂等键），返回结构化草稿 +
   追问话术；草稿随 ``SessionState.pending_write`` 持久化（崩溃/重启后可恢复）。
2. **确认轮**：下一轮用户消息经 :func:`judge_write_confirmation` 三态判定
   （confirm / cancel / ambiguous，**确定性规则优先，否定优先于肯定**）；
   判定 confirm 时工具复用草稿的 ``client_request_id`` 真正执行写；
   cancel 立即作废草稿；ambiguous 一律禁止写。

判定纪律（沿用被删除的 write_gate 实现，2026-09-14 批次 4 收紧口径）：

- 明确确认词（"确认退款"/"同意这笔退款"）或紧接唯一待确认草稿的简短肯定；
- 弱确认（"好的"/"嗯"/"退"）仅对**创建类**生效——撤回类必须命中
  "确认撤回/撤销"或逐字包含 ``RA-`` 编号（语义相反，不得放行）；
- 否定优先："不要退款"/"还没确认" 不得被肯定词表吃掉；
- 疑问（"能退吗？"/"怎么确认"）不是授权；
- 用户点名了另一笔订单/申请 → 安全失败（ambiguous），不得解释成当前唯一草稿。

草稿时效：除会话 reset 外，草稿超过 :data:`DRAFT_TTL_SECONDS` 自动作废——
避免「很久以前登记的草稿」被后来一句无关的"确认"意外执行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

# 草稿有效期：超过即作废（防陈旧草稿被后来的确认词误触发）
DRAFT_TTL_SECONDS = 1800

# 否定/取消（优先判定）。这里仅把明确放弃写操作的意图视为 cancel。
_CANCEL_RE = re.compile(
    r"取消(?:退款|申请)?|不要了|不退了|先不退|暂不退|不用了|算了|别退|"
    r"撤回(?:退款|申请)?|撤销(?:退款|申请)?|放弃(?:退款|申请)?|不办了|"
    r"拒绝退款|"
    r"(?:不要|不用|不想|不再|无需|不需要)\s*"
    r"(?:这笔|该笔|本次|此次)?\s*(?:退款|退货|退钱|申请)|"
    r"(?:退款|退货|退钱|申请退款).{0,6}"
    r"(?:不要|不用|不想|不再|无需|不需要)"
)
# “不取消/别取消”是否定取消，不得因为包含“取消退款”而删除待确认项。
_NEGATED_CANCEL_RE = re.compile(
    r"(?:不|不要|别|无需|不用|不能|暂不)\s*(?:要\s*)?取消(?:退款|申请)?"
)
# 对确认动词的否定要在肯定判定前识别。
_NEGATED_CONFIRM_RE = re.compile(
    r"(?:不|没|未|不能|无法|暂时不能|暂不|还没|不想|不要|拒绝)\s*"
    r"(?:确认|同意|确定|批准|执行|提交|撤回|撤销|退(?:款|货|钱)?)"
)
# 带退款语境的明确确认：确认动作必须直接指向退款/退货/退款申请。
_CONFIRM_REFUND_RE = re.compile(
    r"(?:确认|同意|确定|批准|执行|提交)\s*"
    r"(?:一下|下|要\s*)?(?:这笔|该笔|本次|此次|这个|该)?\s*"
    r"(?:退款申请|申请退款|退款|退货|退钱)"
)
# 撤回类写操作的确认（不与 _CANCEL_RE 的“取消”语义冲突，故限定为撤回/撤销）。
_CONFIRM_WITHDRAW_RE = re.compile(
    r"(?:确认|同意|确定|批准|执行)\s*(?:一下|下)?\s*"
    r"(?:撤回|撤销)\s*(?:退款申请|申请|退款)?"
)
_CONFIRM_ORDER_RE = re.compile(
    r"(?:确认|同意|确定|批准|执行|提交)\s*"
    r"(?:一下|下|要\s*)?(?:订单|这笔订单|该订单)?\s*"
    r"ORD-[A-Za-z0-9\-]+"
)
# 肯定语（可单个，也可逗号连接两个："是的，退吧" / "对，退了吧"）
_AFFIRM_TOKEN = (
    r"(?:没错|就是这样|就这么办|退吧|退了吧|是的|对|可以退|给我退|好)"
)
_CONFIRM_AFFIRM_REFUND_RE = re.compile(
    r"^" + _AFFIRM_TOKEN + r"\s*[，,]?\s*" + _AFFIRM_TOKEN + r"?"
    r"\s*(?:这笔|该笔|本次|此次)?\s*(?:退款|退货|退钱)?"
    r"[。！~！.!\s]*$"
)
_CONFIRM_BARE_RE = re.compile(
    r"^(?:我\s*)?(?:确认|同意|确定|没错|就是这样|就这么办|退吧|执行|是的|"
    r"对[，,]?|可以退|给我退)(?:\s*(?:以上|这笔|该笔|本次|此次))?"
    r"[。！~！.!\s]*$",
    re.IGNORECASE,
)
_CONFIRM_WEAK_RE = re.compile(
    r"^(?:好的?|嗯+|可以|行|ok|yes|对|是|嗯嗯|要|退|好呀|好嘞|的确|当然)[。！~！.!\s]*$",
    re.IGNORECASE,
)
# 疑问/咨询确认流程不是授权。
_QUESTION_RE = re.compile(r"(?:吗|么|如何|怎么|怎样|是否|可否|能否|要不要|为什么|[？?])")
_ORDER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{5,}")
_CONTEXT_ORDER_ID_RE = re.compile(
    r"(?:订单|退款)\s*(?:号|ID|id)?\s*[:：#]?\s*"
    r"([A-Za-z][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)"
)
_ORD_PREFIX_RE = re.compile(r"ORD-[A-Za-z0-9\-]+")
# 申请编号（退款申请）：RA- 前缀
_APPLICATION_ID_RE = re.compile(r"RA-[A-Za-z0-9\-]+")


@dataclass
class PendingWrite:
    """会话级待确认写操作（来自 ``SessionState.pending_write`` 的脱敏视图）。"""

    tool: str
    client_request_id: str
    order_id: str = ""
    reason: str = ""
    application_id: str = ""
    summary: str = ""


@dataclass
class WriteConfirmationDecision:
    """本轮待确认写操作的表态判定。"""

    action: str  # confirm | cancel | ambiguous | none
    payload: PendingWrite | None = None
    clarification: str = ""  # ambiguous+多笔待确认时的确定性澄清问题
    matched_order_id: str = ""
    matched_application_id: str = ""


CLARIFICATION_MULTI = (
    "您当前有多笔待确认的退款操作。请告诉我要确认或取消哪一笔"
    "（请注明订单号或申请编号），我再为您继续处理。"
)


def mentioned_order_id(text: str) -> str:
    """用户消息中提到的订单号（ORD- 前缀优先；向后兼容宽松匹配）。"""
    text = text or ""
    match = _ORD_PREFIX_RE.search(text)
    if match:
        return match.group(0)
    match = _CONTEXT_ORDER_ID_RE.search(text)
    if match:
        return match.group(1)
    for match in _ORDER_ID_RE.finditer(text):
        candidate = match.group(0).strip().rstrip("。，,")
        # 普通英文单词不是订单号；宽松回退至少要求包含数字。
        if len(candidate) >= 6 and any(ch.isdigit() for ch in candidate):
            return candidate
    return ""


def mentioned_application_id(text: str) -> str:
    match = _APPLICATION_ID_RE.search(text or "")
    return match.group(0) if match else ""


def judge_write_confirmation(
    text: str, pending: list[PendingWrite],
) -> WriteConfirmationDecision:
    """三态判定（确定性；否定优先于肯定）。"""
    if not pending:
        return WriteConfirmationDecision(action="none")

    normalized = (text or "").strip()
    mentioned_order = mentioned_order_id(normalized)
    mentioned_app = mentioned_application_id(normalized)

    def _match() -> PendingWrite | None:
        if mentioned_app:
            for entry in pending:
                if entry.application_id and entry.application_id == mentioned_app:
                    return entry
            return None
        if mentioned_order:
            for entry in pending:
                if entry.order_id and entry.order_id == mentioned_order:
                    return entry
            return None
        return None

    matched = _match()
    if (mentioned_app or mentioned_order) and matched is None:
        # 用户明确点名了另一笔写操作：安全失败，不得解释成当前唯一待确认项。
        return WriteConfirmationDecision(action="ambiguous")
    target = matched or (pending[0] if len(pending) == 1 else None)
    if target is None:
        # 多笔待确认且无法定位 → 澄清问题（确定性）
        return WriteConfirmationDecision(
            action="ambiguous", clarification=CLARIFICATION_MULTI,
        )

    negated_cancel = _NEGATED_CANCEL_RE.search(normalized) is not None
    negated_confirm = _NEGATED_CONFIRM_RE.search(normalized) is not None
    question = _QUESTION_RE.search(normalized) is not None

    # 撤回类待确认：先识别「确认撤回/确认撤销」，避免被 _CANCEL_RE 的“撤回”误判。
    if (not negated_confirm and not question
            and _CONFIRM_WITHDRAW_RE.search(normalized)):
        return WriteConfirmationDecision(
            action="confirm", payload=target,
            matched_application_id=target.application_id,
            matched_order_id=target.order_id,
        )

    if _CANCEL_RE.search(normalized) and not negated_cancel:
        return WriteConfirmationDecision(
            action="cancel", payload=target, matched_order_id=target.order_id,
            matched_application_id=target.application_id,
        )

    # 否定确认（包括“不能确认”）保留待确认条目，但本轮禁止写工具。
    if negated_confirm:
        return WriteConfirmationDecision(action="ambiguous")
    if question:
        return WriteConfirmationDecision(action="ambiguous")

    # 确认闸门收紧（批次 4）：非创建类（撤回等）不得被泛化确认词放行——
    # 弱确认（"好的"/"退"/"要"）与裸"确认"对撤回类语义相反，用户回一个
    # "退"字不得确认"撤回退款申请"。撤回类必须命中"确认撤回/撤销"
    # （前置分支已覆盖）或消息逐字包含 RA- 编号。
    if target.tool != "submit_refund_application":
        if (target.application_id
                and target.application_id in normalized):
            return WriteConfirmationDecision(
                action="confirm", payload=target,
                matched_application_id=target.application_id,
                matched_order_id=target.order_id,
            )
        return WriteConfirmationDecision(action="ambiguous")

    if (_CONFIRM_REFUND_RE.search(normalized)
            or _CONFIRM_ORDER_RE.search(normalized)
            or (_CONFIRM_AFFIRM_REFUND_RE.match(normalized) is not None)
            or _CONFIRM_BARE_RE.fullmatch(normalized)
            or (_CONFIRM_WEAK_RE.fullmatch(normalized) and len(pending) == 1)):
        return WriteConfirmationDecision(
            action="confirm", payload=target, matched_order_id=target.order_id,
            matched_application_id=target.application_id,
        )

    # 答非所问/其他：本轮禁止写工具。
    return WriteConfirmationDecision(action="ambiguous")


# ============================================================
# 草稿记录（SessionState.pending_write 的构建/解析/时效）
# ============================================================
def build_draft(tool: str, client_request_id: str, arguments: dict,
                display: dict | None = None, *, now: datetime | None = None) -> dict:
    """构建可 JSON 序列化的草稿记录。

    只存执行写所必需的参数与展示用摘要——**不存任何授权凭证**：草稿本身即
    授权依据，落库在会话正本里，不进消息 metadata、日志或模型上下文。
    """
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    return {
        "tool": tool,
        "client_request_id": client_request_id,
        "arguments": dict(arguments or {}),
        "display": dict(display or {}),
        "created_at": stamp,
        "confirmed_turn": False,
    }


def pending_of(draft: dict | None) -> list[PendingWrite]:
    """草稿记录 → PendingWrite 视图（异形/损坏按「无草稿」处理）。"""
    if not isinstance(draft, dict) or not draft.get("tool"):
        return []
    arguments = draft.get("arguments") or {}
    display = draft.get("display") or {}
    return [PendingWrite(
        tool=str(draft.get("tool", "")),
        client_request_id=str(draft.get("client_request_id", "")),
        order_id=str(arguments.get("order_id") or display.get("order_id") or ""),
        reason=str(arguments.get("reason") or display.get("reason") or ""),
        application_id=str(
            arguments.get("application_id") or display.get("application_id") or ""
        ),
        summary=str(display.get("summary", "")),
    )]


def is_expired(draft: dict | None, *, now: datetime | None = None) -> bool:
    """草稿是否超期（解析失败视为超期——宁可让用户重新确认）。"""
    if not isinstance(draft, dict):
        return True
    raw = str(draft.get("created_at", "") or "")
    if not raw:
        return True
    try:
        created = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return True
    return (now or datetime.now()) - created > timedelta(seconds=DRAFT_TTL_SECONDS)


def mark_confirmed(draft: dict, *, now: datetime | None = None) -> dict:
    """标记草稿已出现确认轮（审计用；不改写参数）。"""
    updated = dict(draft)
    updated["confirmed_turn"] = True
    updated["confirmed_at"] = (now or datetime.now()).isoformat(timespec="seconds")
    return updated

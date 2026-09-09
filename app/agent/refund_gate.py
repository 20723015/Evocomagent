"""退款确认闸门（Review 问题全量修复计划·服务端退款确认）。

程序层三态判定用户对本轮待确认退款的表态：
- `confirm`：明确确认词，或紧接唯一待确认退款的简短肯定回复；
- `cancel`：否定/取消表达（优先于肯定判定）；
- `ambiguous`：答非所问、多笔待确认但未带订单号等——禁止写工具；
  多笔待确认且无订单号时返回确定性澄清问题（零 LLM 规则响应）。

token 管理约束：
- confirmation token 只存确认存储（user+session+refund_id 反向解析），
  不进消息 metadata、outbox、日志或模型上下文；
- 执行器仅在本轮判定为 confirm 时，经内部参数通道注入 token 与幂等键；
  模型提交 confirmation_token/refund_id 等保留字段一律拒绝。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 否定/取消（优先判定）。这里仅把明确撤回退款意图视为 cancel；
# “不能确认/暂时不能确认”不等于用户取消申请，单独落到 ambiguous，避免
# 误删待确认条目，同时也绝不触发写操作。
_CANCEL_RE = re.compile(
    r"取消(?:退款|申请)?|不要了|不退了|先不退|暂不退|不用了|算了|别退|"
    r"撤回(?:退款|申请)?|放弃(?:退款|申请)?|不办了|拒绝退款|"
    r"(?:不要|不用|不想|不再|无需|不需要)\s*"
    r"(?:这笔|该笔|本次|此次)?\s*(?:退款|退货|退钱|申请)|"
    r"(?:退款|退货|退钱|申请退款).{0,6}"
    r"(?:不要|不用|不想|不再|无需|不需要)"
)
# “不取消/别取消”是否定取消，不得因为包含“取消退款”而删除待确认项。
_NEGATED_CANCEL_RE = re.compile(
    r"(?:不|不要|别|无需|不用|不能|暂不)\s*(?:要\s*)?取消(?:退款|申请)?"
)
# 对确认动词的否定要在肯定判定前识别。原实现只看到了“确认”二字，
# “我不确认退款/暂时不能确认”因而被错误地判成 confirm。
_NEGATED_CONFIRM_RE = re.compile(
    r"(?:不|没|未|不能|无法|暂时不能|暂不|还没|不想|不要|拒绝)\s*"
    r"(?:确认|同意|确定|批准|执行|提交|退(?:款|货|钱)?)"
)
# 带退款语境的明确确认。必须是确认动作直接指向退款/退货/退款申请；
# “提交投诉材料”“我确定物流还没到”等一般性陈述不满足该条件。
_CONFIRM_REFUND_RE = re.compile(
    r"(?:确认|同意|确定|批准|执行|提交)\s*"
    r"(?:一下|下|要\s*)?(?:这笔|该笔|本次|此次|这个|该)?\s*"
    r"(?:退款申请|申请退款|退款|退货|退钱)"
)
_CONFIRM_ORDER_RE = re.compile(
    r"(?:确认|同意|确定|批准|执行|提交)\s*"
    r"(?:一下|下|要\s*)?(?:订单|这笔订单|该订单)?\s*"
    r"ORD-[A-Za-z0-9\-]+"
)
_CONFIRM_AFFIRM_REFUND_RE = re.compile(
    r"(?:没错|就是这样|就这么办|退吧|是的|对[，,]?|可以退|给我退)\s*"
    r"(?:这笔|该笔|本次|此次)?\s*(?:退款|退货|退钱)?"
)
# 明确肯定短语可以在唯一待确认退款时使用；限定为完整短语，不能被
# “我确定物流还没到”这类包含确认词的长句误触发。
_CONFIRM_BARE_RE = re.compile(
    r"^(?:我\s*)?(?:确认|同意|确定|没错|就是这样|就这么办|退吧|执行|是的|"
    r"对[，,]?|可以退|给我退)(?:\s*(?:以上|这笔|该笔|本次|此次))?"
    r"[。！~！.!\s]*$",
    re.IGNORECASE,
)
# 简短肯定（仅在唯一待确认退款时视为 confirm）
_CONFIRM_WEAK_RE = re.compile(
    r"^(?:好的?|嗯+|可以|行|ok|yes|对|是|嗯嗯|要|退|好呀|好嘞|的确|当然)[。！~！.!\s]*$",
    re.IGNORECASE,
)
# 疑问/咨询确认流程不是授权。即使句中出现“确认退款”，也必须保持待确认。
_QUESTION_RE = re.compile(r"(?:吗|么|如何|怎么|怎样|是否|可否|能否|要不要|为什么|[？?])")
# 订单号（工具示例格式 ORD-YYYYMMDD-NNN；宽松匹配字母数字连字符段）
_ORDER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{5,}")
_CONTEXT_ORDER_ID_RE = re.compile(
    r"(?:订单|退款)\s*(?:号|ID|id)?\s*[:：#]?\s*"
    r"([A-Za-z][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)"
)


@dataclass
class PendingRefund:
    """会话级待确认退款（来自确认存储注册表，已脱敏视图）。"""

    refund_id: str
    order_id: str
    reason: str = ""
    token: str = ""       # 仅执行器内部通道使用，绝不进模型上下文


@dataclass
class ConfirmationDecision:
    """本轮退款确认判定。"""

    action: str  # confirm | cancel | ambiguous | none
    payload: PendingRefund | None = None
    clarification: str = ""  # ambiguous+多笔待确认时的确定性澄清问题
    matched_order_id: str = ""


CLARIFICATION_MULTI = (
    "您当前有多笔待确认的退款申请。请告诉我要确认或取消哪一笔"
    "（请注明订单号），我再为您继续处理。"
)


_ORD_PREFIX_RE = re.compile(r"ORD-[A-Za-z0-9\-]+")


def _mentioned_order_id(text: str) -> str:
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


def judge_refund_confirmation(
    text: str, pending: list[PendingRefund],
) -> ConfirmationDecision:
    """三态判定（确定性；否定优先于肯定）。"""
    if not pending:
        return ConfirmationDecision(action="none")

    normalized = (text or "").strip()
    mentioned = _mentioned_order_id(normalized)

    def _match_by_order() -> PendingRefund | None:
        if not mentioned:
            return None
        for entry in pending:
            # 订单号必须精确匹配；不能因为用户提到另一笔订单而回退到
            # pending[0]，否则会把确认意图绑定到错误退款。
            if entry.order_id and entry.order_id == mentioned:
                return entry
        return None

    matched = _match_by_order()
    if mentioned and matched is None:
        # 用户明确点名了另一笔订单：安全失败，不得把它解释成当前唯一待确认
        # 退款的确认。
        return ConfirmationDecision(action="ambiguous")
    target = matched or (pending[0] if len(pending) == 1 else None)
    if target is None:
        # 多笔待确认且无法定位订单 → 澄清问题（确定性）
        return ConfirmationDecision(
            action="ambiguous", clarification=CLARIFICATION_MULTI,
        )

    negated_cancel = _NEGATED_CANCEL_RE.search(normalized) is not None
    if _CANCEL_RE.search(normalized) and not negated_cancel:
        return ConfirmationDecision(
            action="cancel", payload=target, matched_order_id=target.order_id,
        )

    # 否定确认（包括“不能确认”）保留待确认条目，但本轮禁止写工具。
    if _NEGATED_CONFIRM_RE.search(normalized):
        return ConfirmationDecision(action="ambiguous")

    if _QUESTION_RE.search(normalized):
        return ConfirmationDecision(action="ambiguous")

    if (_CONFIRM_REFUND_RE.search(normalized)
            or _CONFIRM_ORDER_RE.search(normalized)
            or (_CONFIRM_AFFIRM_REFUND_RE.fullmatch(normalized) is not None)
            or _CONFIRM_BARE_RE.fullmatch(normalized)
            or (_CONFIRM_WEAK_RE.fullmatch(normalized) and len(pending) == 1)):
        return ConfirmationDecision(
            action="confirm", payload=target, matched_order_id=target.order_id,
        )

    # 答非所问/其他：本轮禁止写工具（单笔待确认时 LLM 仍可正常回答问题）
    return ConfirmationDecision(action="ambiguous")


# ============================================================
# 确认存储访问（token 只在确认存储；user+session+refund_id 反向解析）
# ============================================================
def get_session_pending(store, user_id: str, session_id: str) -> list[PendingRefund]:
    getter = getattr(store, "get_session_refunds", None)
    if getter is None:
        return []
    out: list[PendingRefund] = []
    for entry in getter(user_id, session_id) or []:
        out.append(PendingRefund(
            refund_id=str(entry.get("refund_id", "")),
            order_id=str(entry.get("order_id", "")),
            reason=str(entry.get("reason", "")),
            token=str(entry.get("token", "")),
        ))
    return out


def cancel_pending(store, user_id: str, session_id: str,
                   entry: PendingRefund) -> None:
    """取消：删除待确认条目并让未消费 token 立即失效。"""
    take = getattr(store, "take_session_refund", None)
    if take is not None:
        take(user_id, session_id, entry.refund_id)
    drop = getattr(store, "drop_token", None)
    if drop is not None and entry.token:
        drop(entry.token)


def migrate_legacy_metadata_tokens(store, user_id: str, session_id: str,
                                   pending_writes: list[dict]) -> None:
    """旧会话 metadata 中尚未过期的 token：仅供服务端内部迁移进确认存储。

    迁移后模型上下文必须剔除（pending_write_note 只读 order/reason 字段，
    token 永不回注上下文）。
    """
    put = getattr(store, "put_session_refund", None)
    if put is None:
        return
    ttl = 60  # 迁移条目给一个保守短 TTL（原签发时的剩余时间无法还原）
    for entry in pending_writes or []:
        token = str(entry.get("confirmation_token", "") or "")
        refund_id = str(entry.get("idempotency_key", "") or "")
        if not token or not refund_id:
            continue
        put(user_id, session_id, {
            "refund_id": refund_id,
            "order_id": str(entry.get("order_id", "") or ""),
            "reason": str(entry.get("reason", "") or ""),
            "token": token,
            "user_id": user_id,
            "session_id": session_id,
        }, ttl)

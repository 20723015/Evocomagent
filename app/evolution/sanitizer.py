"""第10期 脱敏与规范化：PII 检测、注入检测、长度闸门、slug、candidate ID。

设计原则：
- 进知识库的文本必须过 sanitizer；命中 PII/注入即拒绝或替换。
- 政策数字（如 "7天"、"12元"）不当作 PII 误杀。
- normalize_* 兼做长度闸门：不合法返回 ""，由调用方丢弃。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# ============================================================
# PII 检测（含上下文关键词的防误杀设计）
# ============================================================
# 手机号：11 位，1[3-9] 开头，前后不跟数字
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 邮箱
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
# 身份证：18 位（末位可为 X/x）
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
# 银行卡：13-19 位连续数字，前面需出现"卡"类字段词（防误杀普通数字）。
# 注意：不能用 lookbehind（变宽），改为主模式组 1=字段词、组 2=数字。
_BANK_CARD = re.compile(r"(卡[号码：: ]{0,2})(\d{13,19})")
# 订单号/快递单号/运单号：长度 ≥ 8 的数字串，必须邻近字段词（组对：字段词|数字）
_ORDER_NO = re.compile(
    r"(订单[号码：: ]{0,2})(\d{8,})"
    r"|(单号[：: ])(\d{8,})"
    r"|(快递[单号：: ]{0,2})(\d{8,})"
    r"|(物流[单号：: ]{0,2})(\d{8,})"
    r"|(运单[单号：: ]{0,2})(\d{8,})"
)
# 联系方式：关键字邻近的账号样式（保守——账号段不含连字符且必须含数字，
# 规避 "wx widgets"、"vx-miniapp" 这类普通英文词/文件名的误杀）。
# 组 1 含关键字 + 分隔符（_mask 保留组 1 只替换账号段），组 2 = 账号。
_CONTACT = re.compile(
    r"(?<![a-zA-Z0-9_])((?:微信|vx|wx|qq)[号码:：\s]{0,3})"
    r"((?=[a-zA-Z0-9_]*\d)[a-zA-Z0-9_]{5,20})"
)
# 裸长数字（≥12 位）：覆盖无字段词的运单号/流水号；政策白名单优先
_RAW_LONG_NUMBER = re.compile(r"(?<!\d)\d{12,}(?!\d)")
# 政策数字白名单：带"天/日/元/个工作日/%"等单位数字是政策内容，不是 PII
POLICY_NUMBER_WHITELIST = re.compile(r"\d+\s*(天|日|元|个工作日|%)")

PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_PHONE, "【手机号】"),
    (_EMAIL, "【邮箱】"),
    (_ID_CARD, "【身份证号】"),
    (_BANK_CARD, "【银行卡号】"),
    (_ORDER_NO, "【单号】"),
    (_CONTACT, "【联系方式】"),
    (_RAW_LONG_NUMBER, "【单号】"),
]


def _is_policy_number(span: str) -> bool:
    """命中的数字片段若是政策数量（如 7天 / 12元）则不按 PII 处理。"""
    return bool(POLICY_NUMBER_WHITELIST.search(span))


def _mask(match: re.Match, placeholder: str) -> str:
    """替换命中；带上下文组的模式保留字段词只替换数字，政策数字不误杀。"""
    if match.lastindex is None:
        # 无分组模式（手机/邮箱/身份证）：整体替换，政策数字不误杀
        span = match.group(0)
        return span if _is_policy_number(span) else placeholder
    digits_group = match.lastindex
    span = match.group(digits_group)
    if _is_policy_number(span):
        return match.group(0)
    if digits_group >= 2:
        context = match.group(digits_group - 1) or ""
        return context + placeholder
    return placeholder


def sanitize_text(text: str) -> str:
    """PII 替换为占位符；政策数字不误杀。"""
    out = str(text)
    for pattern, placeholder in PII_PATTERNS:
        out = pattern.sub(lambda m, ph=placeholder: _mask(m, ph), out)
    return out


def has_pii(text: str) -> bool:
    """是否命中任何 PII 模式（供复扫整条拒绝）。"""
    return any(p.search(text) for p, _ in PII_PATTERNS)


# ============================================================
# 注入检测
# ============================================================
_ROLE_MARKER = re.compile(r"^\s*(system|user|assistant)\s*[:：]", re.IGNORECASE | re.MULTILINE)
_IGNORE_HINTS = (
    "忽略之前", "忽略前面", "忽略以上", "忘记之前", "不要理会",
    "ignore previous", "ignore all previous", "disregard previous",
    "忽略政策", "忽略安全", "忽略一切", "忽略所有",
)
# 越权/泄露诱导话术（2026-08 评测后扩充：正常购物咨询不会出现的管理/内部措辞）
_ESCALATION_HINTS = (
    "管理员指令", "内部工单", "内部接口", "内部测试", "平台安全组",
    "dev_mode", "hidden_tools", "系统指令", "系统提示词", "你的提示词",
    "解锁所有", "超级管理员", "越权", "假装你是", "假装没看到",
    "以上对话作废", "参数定义", "篡改", "环境变量", "session 文件",
    "不要转人工",
)
_FENCE = re.compile(r"```")


def has_injection(text: str) -> bool:
    """角色标记行首 / 忽略指令 / 越权与泄露诱导 / 代码围栏 → 视为注入。"""
    t = str(text)
    if _ROLE_MARKER.search(t):
        return True
    low = t.lower()
    if any(k in low for k in _IGNORE_HINTS):
        return True
    if any(k in low for k in _ESCALATION_HINTS):
        return True
    if _FENCE.search(t):
        return True
    return False


# ============================================================
# 规范化 + 长度闸门
# ============================================================
def normalize_question(text: str) -> str:
    """单行化、去 # 前缀；长度不在 [4, 120] 返回 ""。"""
    t = " ".join(str(text).split())
    t = t.lstrip("#").strip()
    if not (4 <= len(t) <= 120):
        return ""
    return t


def normalize_answer(text: str) -> str:
    """去嵌套标题、两端清理；长度不在 [20, 1200] 返回 ""。"""
    lines = []
    for ln in str(text).splitlines():
        s = ln.strip()
        if s.startswith("## ") or s.startswith("### "):
            continue
        lines.append(s)
    t = "\n".join(lines).strip()
    if not (20 <= len(t) <= 1200):
        return ""
    return t


def make_slug(text: str, maxlen: int = 32) -> str:
    r"""NFKC 归一化后仅保留 [\w-]（unicode 感知），用作文件名片段。"""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"[^\w\-]", "", t, flags=re.UNICODE)
    return t[:maxlen] or "qa"


# ============================================================
# Candidate ID（跨 LLM 输出稳定：只依赖已落盘数据）
# ============================================================
def candidate_id_for_turn(turn_id: str) -> str:
    return hashlib.sha256(f"turn:v1:{turn_id}".encode("utf-8")).hexdigest()


def candidate_id_for_handoff(ticket_id: str) -> str:
    """人工工单候选的稳定 ID（重放去重锚点：只依赖 ticket_id）。"""
    return hashlib.sha256(
        f"handoff:v1:{ticket_id}".encode("utf-8"),
    ).hexdigest()


def legacy_candidate_id(
    session_key: str,
    msg_index: int,
    raw_q: str,
    raw_a: str,
    source_ids: list[str],
) -> str:
    payload = "|".join([session_key, str(msg_index), raw_q, raw_a, ",".join(sorted(source_ids))])
    return hashlib.sha256(f"legacy:v1:{payload}".encode("utf-8")).hexdigest()
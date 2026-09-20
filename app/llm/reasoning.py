"""reasoning 捕获与双通道存储（推理模型全量适配计划·T3）。

双通道是唯一约定：

- **窗口通道**（`raw_messages`，回传模型）严格按画像
  `reasoning_return_policy`：`forbidden`（DeepSeek，明确禁止回传
  `reasoning_content`）与 `internal`（o 系列，厂商内部管理）一律剥离；
  `required_signed`（Claude thinking）原样回传 thinking block —— 丢 signature
  厂商会 400；
- **审计通道**（`full_turn_messages` → append_log / turns archive）**始终留存**，
  reasoning 作为消息上的附加字段（`REASONING_KEY`）随审计切片归档，additive、
  旧读取路径不受影响。

解耦原则：**捕获**不依赖画像（未登记的推理模型照样可审计）；**放行**依赖画像
（默认剥离 = 永不把推理塞回请求体）。两者分开，才不会因为画像漏配而把
reasoning 泄漏进模型窗口，也不会因为画像漏配而丢掉审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.llm.model_profile import ModelProfile

# 审计附加字段名（additive：旧读取路径忽略未知字段）
REASONING_KEY = "reasoning"

# SSE 透出的单条 reasoning 上限（T6）：推理原文可能极长，透出侧必须有界
SSE_MAX_CHARS = 4000

_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking", "reasoning")


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """dict / SDK 对象统一取值（测试里 message 常是 dict）。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_as_text(v) for v in value) if part)
    if isinstance(value, dict):
        for key in ("thinking", "text", "content", "reasoning_content", "summary"):
            if value.get(key):
                return _as_text(value[key])
        return ""
    return str(value)


def _thinking_blocks(content: Any) -> tuple[dict, ...]:
    """从 content 块列表里挑出 thinking / reasoning 块（保留 signature 原样）。"""
    if not isinstance(content, list):
        return ()
    blocks: list[dict] = []
    for item in content:
        if isinstance(item, dict) and str(item.get("type") or "") in _THINKING_BLOCK_TYPES:
            blocks.append(item)
    return tuple(blocks)


@dataclass(frozen=True)
class CapturedReasoning:
    """一次 assistant 响应里的推理内容（已归一化为文本 + 原样载荷）。"""

    field: str                          # reasoning_content | thinking_blocks
    text: str                           # 归一化文本（审计/SSE 用）
    raw: Any = None                     # 原始字段值（required_signed 回传用）
    raw_content: Any = None             # 原始 content（块列表形态时；回传用）
    blocks: tuple[dict, ...] = field(default_factory=tuple)

    def audit_payload(self) -> dict:
        """审计字段（additive）：文本 + 形状元信息，供对账与体量观测。"""
        payload: dict[str, Any] = {
            "field": self.field,
            "text": self.text,
            "chars": len(self.text),
        }
        if self.blocks:
            payload["blocks"] = [dict(b) for b in self.blocks]
        return payload

    def sse_text(self) -> str:
        """SSE 透出文本（有界；T6）。"""
        if len(self.text) <= SSE_MAX_CHARS:
            return self.text
        return self.text[:SSE_MAX_CHARS] + "…"


def capture_reasoning(message: Any) -> CapturedReasoning | None:
    """捕获 assistant 消息中的推理内容；没有则 None（不影响正常回复）。"""
    content = _field(message, "content")
    blocks = _thinking_blocks(content)
    if blocks:
        return CapturedReasoning(
            field="thinking_blocks",
            # text 只取推理块本身（不含正文块），审计/SSE 都不应把回复正文重复一遍
            text=_as_text([dict(b) for b in blocks]),
            raw=blocks,
            raw_content=content,
            blocks=blocks,
        )
    for key in ("reasoning_content", "reasoning"):
        value = _field(message, key)
        text = _as_text(value)
        if text:
            return CapturedReasoning(field="reasoning_content", text=text, raw=value)
    return None


def flatten_content(content: Any) -> str:
    """content 归一化为纯文本（块列表 → 文本拼接）。"""
    return _as_text(content)


def window_content(message: Any, profile: ModelProfile, captured: CapturedReasoning | None):
    """窗口通道的 assistant content。

    `required_signed` 且原响应为 thinking 块列表时**原样**返回块列表（含
    signature）；其余情况一律压平成文本 —— 绝不把 reasoning 文本混进 content。
    """
    content = _field(message, "content")
    if (
        captured is not None
        and profile.reasoning_return_policy == "required_signed"
        and captured.raw_content is not None
    ):
        return captured.raw_content
    return flatten_content(content)


def window_extra_fields(profile: ModelProfile,
                        captured: CapturedReasoning | None) -> dict:
    """窗口通道需要额外回传的 reasoning 字段（扁平形态的 required_signed）。"""
    if captured is None or profile.reasoning_return_policy != "required_signed":
        return {}
    if captured.field == "reasoning_content" and captured.raw is not None:
        return {"reasoning_content": captured.raw}
    return {}


def merge_audit(message: dict, captured: CapturedReasoning | None) -> dict:
    """审计副本：附加 reasoning 字段（无捕获时原样返回同一对象）。"""
    if captured is None:
        return message
    return {**message, REASONING_KEY: captured.audit_payload()}

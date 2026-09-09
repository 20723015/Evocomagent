"""final_response：仅供模型使用的虚拟终止工具（单 Agent 全量优化计划·阶段A2）。

设计：
- 不进 ToolManager、不进本地工具注册表；仅在发给模型的工具列表末尾追加，
  模型调用它即结束 ReAct 循环——结构化字段（intent/requires_human/
  follow_up_question）由模型在终答时一并给出，正常路径不再有第二次
  `_extract_structured_response` LLM 调用；
- 参数经 Pydantic 严格校验（禁止额外字段）；不合法时返回结构化纠错信息
  （tool 结果），模型可在剩余步数内修复一次或多次；
- 达到最大步数时：最后一轮只挂 final_response 工具强制收尾；预算耗尽仍走
  确定性 fallback（零 LLM）；
- 外部 confidence 不采纳模型自评，由程序可靠度计算给出（阶段E）。
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.response import IntentType

FINAL_RESPONSE_TOOL_NAME = "final_response"

FINAL_RESPONSE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": FINAL_RESPONSE_TOOL_NAME,
        "description": (
            "结束本轮处理并给出面向用户的最终答复。回复内容必须已经完整"
            "（包含基于工具结果/检索证据的答案），调用后本轮即结束，"
            "不要再调用其他工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [e.value for e in IntentType],
                    "description": "本轮用户请求的意图类型",
                },
                "reply": {
                    "type": "string",
                    "description": "给用户的最终回复全文（用户可见，简洁友好）",
                },
                "requires_human": {
                    "type": "boolean",
                    "description": "是否需要转接人工客服",
                },
                "follow_up_question": {
                    "type": ["string", "null"],
                    "description": "需要向用户进一步确认的问题，不需要则为 null",
                },
            },
            "required": ["intent", "reply", "requires_human"],
            "additionalProperties": False,
        },
    },
}

FINAL_RESPONSE_INSTRUCTIONS = """

## 最终答复协议（必须遵守）
处理完成后，你必须调用 `final_response` 工具给出最终答复并结束本轮：
- `reply`：面向用户的完整回复（用户可见的最终文本）；
- `intent`：本轮意图（order_query/return_request/product_consult/complaint/after_sale/promotion/account/greeting/other 之一）；
- `requires_human`：是否需要转人工；
- `follow_up_question`：需要用户确认的追问（无则 null）。
除 `final_response` 与业务工具外不要输出其他内容；信息足够时立即调用 `final_response`，不要继续调用工具。
"""


class FinalResponseArgs(BaseModel):
    """final_response 的严格参数模型（禁止额外字段）。"""

    model_config = ConfigDict(extra="forbid")

    intent: IntentType = Field(description="本轮用户请求的意图类型")
    reply: str = Field(min_length=1, description="给用户的最终回复全文")
    requires_human: bool = Field(default=False, description="是否需要转接人工")
    follow_up_question: str | None = Field(
        default=None, description="进一步确认的问题（无则 null）",
    )


def validate_final_response(
    payload: str | bytes | dict,
) -> tuple[FinalResponseArgs | None, str | None]:
    """校验 final_response 参数。

    返回 (args, None) 或 (None, 纠错 JSON 字符串)——纠错 JSON 作为 tool 结果
    回给模型（稳定错误码 + 字段说明，不暴露内部异常）。
    """
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, _correction("FINAL_RESPONSE_NOT_JSON", "参数必须是合法 JSON 对象")
    elif isinstance(payload, dict):
        data = payload
    else:
        return None, _correction("FINAL_RESPONSE_NOT_JSON", "参数必须是合法 JSON 对象")

    if not isinstance(data, dict):
        return None, _correction("FINAL_RESPONSE_NOT_JSON", "参数必须是 JSON 对象")
    try:
        return FinalResponseArgs.model_validate(data), None
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            errors.append(f"{loc}: {err.get('msg', 'invalid')}")
        return None, _correction(
            "FINAL_RESPONSE_INVALID",
            "字段不合法：" + "; ".join(errors[:5]),
        )


def _correction(code: str, message: str) -> str:
    return json.dumps(
        {
            "error": code,
            "message": (
                f"{message}。请重新调用 final_response："
                "intent 取 intent 枚举值之一，reply 为非空字符串，"
                "requires_human 为布尔值，不要携带其他字段。"
            ),
        },
        ensure_ascii=False,
    )

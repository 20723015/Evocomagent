"""工具注册表：OpenAI function calling schema + 分发执行"""

import json
from collections.abc import Callable

from app.agent.context import ToolContext
from app.agent.tools.knowledge import search_knowledge
from app.agent.tools.logistics import query_logistics
from app.agent.tools.memory_tool import recall_user_memory
from app.agent.tools.order import query_order
from app.agent.tools.product import query_product
from app.agent.tools.refund import apply_refund
from app.agent.tools.skill_tool import load_skill
from app.agent.tools.user_orders import list_user_orders

_TOOL_MAP: dict[str, Callable] = {
    "query_order": query_order,
    "query_product": query_product,
    "query_logistics": query_logistics,
    "apply_refund": apply_refund,
    "search_knowledge": search_knowledge,
    "list_user_orders": list_user_orders,
    "recall_user_memory": recall_user_memory,
    "load_skill": load_skill,
}

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 ORD-20240115-001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_product",
            "description": "根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。支持模糊搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "商品名称关键词或商品ID，例如「耳机」「运动鞋」「SHOE-270-BK-42」",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": "根据订单号查询物流轨迹信息，包括快递公司、运单号、运输状态和轨迹事件",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 ORD-20240115-001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索并夕夕的政策与帮助文档（退换货政策、配送说明、会员权益、常见问题 FAQ）。"
                "当顾客询问规则、流程、时效、是否支持等政策类问题时使用，"
                "比如「能退货吗」「多久到账」「钻石会员有什么权益」「偏远地区包邮吗」。"
                "返回 Top-K 命中片段及来源文档，请基于检索结果回答，不要编造政策。"
                "对比较、组合条件或跨文档问题，可用 queries 一次提交多个子查询"
                "（最多 3 个），结果会自动合并去重"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用顾客的原问题或一句简洁中文描述要查的政策点",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数，默认 3，最大 5",
                        "default": 3,
                    },
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "可选的子查询列表（最多 3 个）：比较/组合条件/跨文档问题时，"
                            "把问题拆成多个独立政策点分别检索"
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_orders",
            "description": (
                "查询当前用户的所有订单概要列表（订单号、状态、商品、金额、下单时间）。"
                "当用户想查订单但未提供订单号，或提供的订单号查不到时，"
                "调用此工具列出订单供用户确认"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_refund",
            "description": (
                "为指定订单申请退款（敏感操作，必须先与用户确认）。"
                "首次调用签发待确认请求；用户明确确认后再次调用（相同订单号与原因）"
                "即由系统自动完成提交——确认凭证与幂等键由系统内部保管与注入，"
                "你不需要也无法提交任何凭证字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "要退款的订单号",
                    },
                    "reason": {
                        "type": "string",
                        "description": "退款原因，例如「尺码不合适」「质量问题」「不想要了」",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_user_memory",
            "description": (
                "查询当前用户的记忆信息，包括本次对话提取的短期记忆和跨会话的长期记忆。"
                "当需要回顾用户的偏好、历史问题、会员信息等时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "可选的查询关键词，用于过滤记忆内容",
                        "default": "",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "加载指定技能的完整操作指令。"
                "当用户问题匹配某个可用技能时，调用此工具获取该技能的详细处理流程，"
                "然后按流程指引使用已有工具完成用户请求。"
                "可用技能会在系统提示中列出。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要加载的技能名称，如 process-return、track-order、product-recommend",
                    }
                },
                "required": ["skill_name"],
            },
        },
    },
]

# 阶段2.3：工具参数 schema 索引（严格校验用；与 TOOL_DEFINITIONS 同源）
_SCHEMA_BY_NAME: dict[str, dict] = {
    td["function"]["name"]: td["function"].get("parameters", {})
    for td in TOOL_DEFINITIONS
}


# Review 修复：模型不得提交的保留字段（确认凭证/幂等键由执行器内部注入）
_RESERVED_TOOL_ARGS: dict[str, frozenset[str]] = {
    "apply_refund": frozenset({"confirmation_token", "idempotency_key", "refund_id"}),
}


def _validate_arguments(name: str, arguments: dict) -> tuple[dict | None, str | None]:
    """阶段2.3：工具参数严格校验（Pydantic 语义的注册表驱动版）。

    - 参数必须是 JSON 对象（数组/标量拒绝）；
    - 未知参数不得进入实际工具函数（additionalProperties=False 语义）；
    - 保留字段（确认凭证等）模型提交一律拒绝（Review 修复）；
    - required 缺失 → 结构化错误（模型可自愈）；
    - 递归剔除 schema 未声明的嵌套额外字段（防注入膨胀）。
    """
    if not isinstance(arguments, dict):
        return None, json.dumps(
            {"error": "INVALID_TOOL_ARGUMENTS", "message": "工具参数必须是 JSON 对象"},
            ensure_ascii=False,
        )
    reserved = _RESERVED_TOOL_ARGS.get(name) or frozenset()
    for key in arguments:
        if key in reserved:
            from app.observability.metrics import record_refund_confirmation_blocked

            record_refund_confirmation_blocked("reserved_args")
            return None, json.dumps(
                {
                    "error": "RESERVED_TOOL_ARGUMENT",
                    "message": (
                        f"参数 {key!r} 由系统内部保管与注入，不接受模型提交；"
                        "请只传订单号与退款原因"
                    ),
                },
                ensure_ascii=False,
            )
    schema = _SCHEMA_BY_NAME.get(name)
    if schema is None:
        return dict(arguments), None
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    cleaned: dict = {}
    for key, value in arguments.items():
        if key not in properties:
            return None, json.dumps(
                {
                    "error": "UNKNOWN_TOOL_ARGUMENT",
                    "message": f"未知参数 {key!r}；只接受: {', '.join(sorted(properties)) or '无'}",
                },
                ensure_ascii=False,
            )
        cleaned[key] = _clean_value(properties[key], value)
    missing = required - set(cleaned)
    if missing:
        return None, json.dumps(
            {
                "error": "MISSING_TOOL_ARGUMENT",
                "message": f"缺少必需参数: {', '.join(sorted(missing))}",
            },
            ensure_ascii=False,
        )
    return cleaned, None


def _clean_value(prop_schema, value):
    """按属性 schema 递归清洗：数组/对象按 items/additionalProperties 裁剪。"""
    if not isinstance(prop_schema, dict):
        return value
    prop_type = prop_schema.get("type")
    if prop_type == "array" and isinstance(value, list):
        item_schema = prop_schema.get("items")
        if isinstance(item_schema, dict):
            return [_clean_value(item_schema, item) for item in value]
        return value
    if prop_type == "object" and isinstance(value, dict):
        properties = prop_schema.get("properties")
        if properties is None:
            return value
        return {
            k: _clean_value(properties.get(k, {}), v)
            for k, v in value.items()
            if k in properties
        }
    return value


def execute_tool(
    name: str, arguments: dict, ctx: ToolContext | None = None,
    timeout: float | None = None, internal_args: dict | None = None,
) -> str:
    """根据工具名称分发执行，返回 JSON 字符串结果。

    阶段一 1.3：ctx 全链路透传工具函数（user_id/session_id/memory 句柄）。
    timeout：轮次预算剩余（修复计划）——search_knowledge 继续下传到
    Embedder/远端 reranker/ES（Embeddings 不再用 SDK 600s 默认）；其余工具忽略。
    阶段2.3：参数严格校验——非法参数不进入实际工具函数（结构化错误回模型）。
    """
    func = _TOOL_MAP.get(name)
    if not func:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    cleaned, error = _validate_arguments(name, arguments)
    if error is not None:
        return error
    # Review 修复：内部参数通道在**校验之后**合并（仅执行器可写，
    # 不受保留字段校验约束，也绝不进入模型可见参数面）
    arguments = {**(cleaned or {}), **(internal_args or {})}
    try:
        if name == "search_knowledge":
            result = func(**arguments, ctx=ctx, timeout=timeout)
        else:
            result = func(**arguments, ctx=ctx)
    except TypeError as e:
        # 参数名/数量与函数签名不符（模型幻觉参数已在上游拦截，此处兜底）
        result = {"error": f"工具参数不匹配: {e}"}
    except Exception as e:
        result = {"error": f"工具执行出错: {e}"}
    return json.dumps(result, ensure_ascii=False)

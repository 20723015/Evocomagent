"""工具注册表：OpenAI function calling schema + 分发执行"""

import json
from collections.abc import Callable

from app.agent.context import ToolContext
from app.agent.tools.knowledge import search_knowledge
from app.agent.tools.logistics import query_logistics
from app.agent.tools.memory_tool import recall_user_memory
from app.agent.tools.order import query_order
from app.agent.tools.product import query_product
from app.agent.tools.refund import (
    cancel_refund_application,
    query_refund_application,
    submit_refund_application,
)
from app.agent.tools.skill_tool import load_skill
from app.agent.tools.user_orders import list_user_orders

_TOOL_MAP: dict[str, Callable] = {
    "query_order": query_order,
    "query_product": query_product,
    "query_logistics": query_logistics,
    "submit_refund_application": submit_refund_application,
    "query_refund_application": query_refund_application,
    "cancel_refund_application": cancel_refund_application,
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
            "name": "submit_refund_application",
            "description": (
                "为指定订单提交退款申请（敏感写操作，两阶段协议）。"
                "首次调用只生成草稿、不提交：请把订单号与退款原因复述给用户并请求确认；"
                "用户明确确认后，带 confirm=true 再次调用本工具才会真正提交。"
                "只有当用户当前消息明确要求退款或退货、且给出准确订单号时才调用；"
                "申请提交后进入商家审核，审核通过前可撤回。"
                "咨询、否定、条件句、订单指代（如「这单」）不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "要退款的订单号（必须是用户当前消息中逐字出现的订单号）",
                    },
                    "reason": {
                        "type": "string",
                        "description": "退款原因，例如「尺码不合适」「质量问题」「不想要了」",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": (
                            "仅当用户已在对话中明确确认该笔退款时才置 true；"
                            "首次调用（生成草稿）不要传或传 false。"
                        ),
                    },
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_refund_application",
            "description": (
                "查询退款申请的状态（订单号与申请编号必须且只能提供一个）。"
                "返回 applications 数组，每项含 application_id/order_id/status/"
                "reason/created_at/updated_at/can_withdraw。"
                "用户询问退款进度、或需要确认目标申请时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "application_id": {
                        "type": "string",
                        "description": "退款申请编号，例如 RA-1a2b3c4d5e6f",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 ORD-20240115-001",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_refund_application",
            "description": (
                "撤回退款申请（仅审核中的申请可撤回）。"
                "当用户当前消息明确表达撤回、并给出准确申请编号或订单号时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "application_id": {
                        "type": "string",
                        "description": "要撤回的退款申请编号，例如 RA-1a2b3c4d5e6f",
                    },
                },
                "required": ["application_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_user_memory",
            "description": (
                "查询当前用户的记忆信息，即跨会话的长期记忆。"
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


def _validate_arguments(name: str, arguments: dict) -> tuple[dict | None, str | None]:
    """阶段2.3：工具参数严格校验（Pydantic 语义的注册表驱动版）。

    - 参数必须是 JSON 对象（数组/标量拒绝）；
    - 未知参数不得进入实际工具函数（additionalProperties=False 语义）；
    - required 缺失 → 结构化错误（模型可自愈）；
    - 递归剔除 schema 未声明的嵌套额外字段（防注入膨胀）。
    """
    if not isinstance(arguments, dict):
        return None, json.dumps(
            {"error": "INVALID_TOOL_ARGUMENTS", "message": "工具参数必须是 JSON 对象"},
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
        cleaned_value = _clean_value(properties[key], value)
        if cleaned_value is _INVALID:
            # schema 声明 array 但值不是 list（如模型把 queries 传成字符串）：
            # 丢弃该键——可选参数回退默认路径，必需参数走下方 missing 报错，
            # 绝不把错误类型原样放行（历史缺陷：queries 字符串被逐字符当子查询）
            continue
        cleaned[key] = cleaned_value
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


# 类型不符的裁剪结果哨兵：该键不进入 cleaned（见 execute_tool 的清洗循环）
_INVALID = object()


def _clean_value(prop_schema, value):
    """按属性 schema 递归清洗：数组/对象按 items/additionalProperties 裁剪。

    schema 声明 array 但值不是 list → 返回哨兵 _INVALID（调用方丢键）。
    """
    if not isinstance(prop_schema, dict):
        return value
    prop_type = prop_schema.get("type")
    if prop_type == "array":
        if not isinstance(value, list):
            return _INVALID
        item_schema = prop_schema.get("items")
        if isinstance(item_schema, dict):
            cleaned_items = [_clean_value(item_schema, item) for item in value]
            if any(item is _INVALID for item in cleaned_items):
                return _INVALID
            return cleaned_items
        return value
    if prop_type == "object" and isinstance(value, dict):
        properties = prop_schema.get("properties")
        if properties is None:
            return value
        cleaned_obj = {}
        for k, v in value.items():
            if k not in properties:
                continue
            cleaned_child = _clean_value(properties[k], v)
            if cleaned_child is _INVALID:
                continue
            cleaned_obj[k] = cleaned_child
        return cleaned_obj
    return value


def execute_tool(
    name: str, arguments: dict, ctx: ToolContext | None = None,
    timeout: float | None = None,
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
    # 注：写工具租约门禁统一在 ToolManager.execute_tool（本地/MCP 分流前）执行，
    # 此处不再重复（修复计划·二轮 1：避免本地/MCP 行为不一致）。
    cleaned, error = _validate_arguments(name, arguments)
    if error is not None:
        return error
    arguments = cleaned or {}
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

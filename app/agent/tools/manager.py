"""ToolManager：统一管理本地工具和 MCP 工具。

当 MCP 启用时，通过 Streamable HTTP 连接 MCP Server 获取工具；
当 MCP 未启用或连接失败时，退回本地工具。
"""

import json

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.agent.tools.manager")


from app.agent.context import ToolContext
from app.agent.tools.registry import TOOL_DEFINITIONS as LOCAL_TOOL_DEFINITIONS
from app.agent.tools.registry import execute_tool as local_execute_tool

# 结果 JSON 中必须移除的临时秘密字段（值替换为占位说明）
_SECRET_RESULT_KEYS = ("confirmation_token", "access_token", "session_token")


def _scrub_secrets(result_str: str) -> str:
    """完整工具结果写入前移除 token/凭证等临时秘密（Review 修复）。

    审计价值保留：其余字段原样；秘密字段替换为「已由系统保管」占位。
    """
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return result_str
    if not isinstance(data, dict):
        return result_str
    hit = False
    for key in _SECRET_RESULT_KEYS:
        if key in data and data[key]:
            data[key] = "[已由系统保管]"
            hit = True
    if not hit:
        return result_str
    return json.dumps(data, ensure_ascii=False)


class ToolManager:
    """聚合本地工具和 MCP 工具，提供统一的工具定义和调度接口。

    阶段一 1.2/1.3：
    - `mcp_client` 可注入 pod 级共享 MCP 客户端（app.state 持有），
      注入时不负责 close（共享资源），自建时才 own；
    - execute_tool 透传 ToolContext，本地工具全链路带 ctx。
    """

    def __init__(
        self,
        use_mcp: bool = False,
        mcp_server_url: str = "",
        allowed_tools: set | None = None,
        mcp_client=None,
    ):
        self._mcp_client = mcp_client
        self._owns_mcp_client = mcp_client is None
        self._tool_source: dict[str, str] = {}
        self._tool_defs: list[dict] = []

        if use_mcp and mcp_server_url:
            self._init_mcp(mcp_server_url)
        else:
            self._init_local()

        if allowed_tools is not None:
            self._filter_tools(allowed_tools)

    def _init_local(self):
        """只加载本地工具。"""
        self._tool_defs = list(LOCAL_TOOL_DEFINITIONS)
        for td in self._tool_defs:
            self._tool_source[td["function"]["name"]] = "local"

    def _init_mcp(self, server_url: str):
        """连接 MCP Server 加载工具；失败时降级到本地工具。"""
        if self._mcp_client is None:
            from app.config.settings import settings
            from app.mcp_client import MCPClient

            self._mcp_client = MCPClient(server_url, auth_token=settings.mcp_auth_token)

        try:
            mcp_tools = self._mcp_client.connect()
            log.info("mcp.connected", server_url=server_url, tools=len(mcp_tools))

            mcp_names = set()
            for td in mcp_tools:
                name = td["function"]["name"]
                mcp_names.add(name)
                self._tool_source[name] = "mcp"
            self._tool_defs = list(mcp_tools)

            for td in LOCAL_TOOL_DEFINITIONS:
                name = td["function"]["name"]
                if name not in mcp_names:
                    self._tool_defs.append(td)
                    self._tool_source[name] = "local"

        except Exception as e:
            log.info(
                "mcp.connect_failed_fallback_local",
                error_type=type(e).__name__,
            )
            if self._mcp_client:
                self._mcp_client.close()
                self._mcp_client = None
            self._init_local()

    def _filter_tools(self, allowed: set):
        """只保留白名单中的工具，用于子 Agent 工具隔离。"""
        self._tool_defs = [
            d for d in self._tool_defs
            if d["function"]["name"] in allowed
        ]
        self._tool_source = {
            k: v for k, v in self._tool_source.items()
            if k in allowed
        }

    @property
    def tool_definitions(self) -> list[dict]:
        return self._tool_defs

    # 发送敏感 MCP 工具前签发短期 actor token（用户身份贯通；token 只在发送层注入）
    _SENSITIVE_MCP_TOOLS = frozenset({"query_order", "query_logistics", "apply_refund"})
    _WRITE_TOOLS = frozenset({"apply_refund"})

    def execute_tool(
        self, name: str, arguments: dict, ctx: ToolContext | None = None,
        timeout: float | None = None, internal_args: dict | None = None,
    ) -> str:
        """根据工具来源分发调用（ctx 透传本地工具）；4.3 工具成功率指标。

        timeout：轮次预算剩余（Agent能力强化计划·改造一 提交规则②），
        仅 MCP/HTTP 路径消费；本地工具即时返回，忽略。
        修复计划：敏感 MCP 工具（订单/物流/退款）发送前按 ToolContext 签发
        actor token（meta 通道）；商品/知识检索仍只依赖服务级 Bearer。
        """
        source = self._tool_source.get(name)

        if source == "mcp" and self._mcp_client:
            actor_token = None
            if name in self._SENSITIVE_MCP_TOOLS:
                from app.mcp_client.actor import (
                    SCOPE_ORDERS_READ,
                    SCOPE_REFUND_WRITE,
                    issue_actor_token,
                )

                user_id = ctx.user_id if ctx is not None else ""
                session_id = ctx.session_id if ctx is not None else ""
                if not user_id:
                    from app.config.settings import settings

                    user_id = settings.mcp_actor_user_id  # 兼容旧直调（无 ctx）
                scopes = (SCOPE_REFUND_WRITE,) if name == "apply_refund" else (SCOPE_ORDERS_READ,)
                actor_token = issue_actor_token(user_id, session_id, scopes)
            # 保留内部确认参数不进入 MCP 工具 schema。MCP 第二段退款的
            # confirmation_token/refund_id 通过独立 meta 通道发送，服务端再
            # 注入真实工具；把它们并入 mcp_args 会被 FastMCP schema 拒绝，
            # 或者意外暴露给模型可见参数面。
            mcp_kwargs = {
                "timeout": timeout,
                "actor_token": actor_token,
                "write": (name in self._WRITE_TOOLS),
            }
            if internal_args:
                mcp_kwargs["internal_args"] = dict(internal_args)
            result_str = self._mcp_client.call_tool(
                name, dict(arguments), **mcp_kwargs,
            )
            if name == "apply_refund":
                self._sync_mcp_refund_confirmation(result_str, arguments, ctx)
            # MCPToolResult 的 internal_meta 到此即完成消费；后续工具轨迹、
            # 审计与模型消息只能拿到普通 str，不能携带进程内隐藏属性。
            result_str = str(result_str)
        elif source == "local":
            result_str = local_execute_tool(
                name, arguments, ctx, timeout=timeout, internal_args=internal_args,
            )
        else:
            result_str = json.dumps(
                {"error": f"未知工具: {name}"}, ensure_ascii=False
            )
        # Review 修复：统一脱敏——确认凭证/临时秘密不进模型上下文、审计与 outbox
        result_str = _scrub_secrets(result_str)

        try:
            from app.observability.metrics import (
                record_tool_call,
                record_tool_result_code,
                record_tool_side_effect,
            )

            data = json.loads(result_str)
            # 阶段G修正：{"error": ...} 一律计为失败（历史 data.get("success",
            # True) 默认成功，把错误结果统计成了成功）
            failed = (
                data.get("success") is False
                or ("error" in data and data.get("success") is not True)
            )
            record_tool_call(name, "error" if failed else "ok")
            if not failed:
                record_tool_result_code(name, str(data.get("code", "") or "OK"))
            if isinstance(data, dict) and data.get("status") in (
                "pending_confirmation", "committed", "indeterminate",
            ):
                record_tool_side_effect(name, str(data["status"]))
        except Exception:  # noqa: BLE001 —— 指标失败不影响工具结果
            pass
        return result_str

    @staticmethod
    def _sync_mcp_refund_confirmation(result, arguments: dict,
                                      ctx: ToolContext | None) -> None:
        """消费 MCP 响应内部 meta，把待确认句柄同步到 Agent 侧存储。

        本地开发可能没有共享 Redis，Agent 与 MCP server 是两个进程；若不
        同步，首段 token 只存在服务端，下一轮闸门无法定位待确认退款。
        凭证仅存在 ``MCPToolResult.internal_meta``，不会进入返回正文。
        """
        meta = getattr(result, "internal_meta", None)
        item = meta.get("refund_confirmation") if isinstance(meta, dict) else None
        if not isinstance(item, dict) or ctx is None or not ctx.user_id:
            return
        raw_refund_id = item.get("refund_id")
        raw_token = item.get("confirmation_token")
        raw_order_id = item.get("order_id") or arguments.get("order_id")
        raw_reason = item.get("reason") or arguments.get("reason")
        if not all(isinstance(value, str) for value in (
            raw_refund_id, raw_token, raw_order_id, raw_reason,
        )):
            return
        refund_id, token = raw_refund_id, raw_token
        order_id, reason = raw_order_id, raw_reason
        if len(refund_id) > 512 or len(token) > 512:
            return
        if not refund_id or not token or not order_id:
            return
        # 元数据必须与本次公开调用绑定，防止异常/恶意 MCP 服务把别的退款
        # 句柄塞入本地确认注册表。
        if order_id != str(arguments.get("order_id") or ""):
            return
        if reason != str(arguments.get("reason") or ""):
            return
        from app.agent.tools.refund import _confirmation_store

        store = _confirmation_store()
        put = getattr(store, "put_session_refund", None)
        if put is None:
            return
        try:
            ttl = int(item.get("expires_in_seconds")
                      or settings.refund_confirm_ttl_seconds)
        except (TypeError, ValueError):
            return
        ttl = max(1, min(ttl, settings.refund_confirm_ttl_seconds))
        put(ctx.user_id, ctx.session_id, {
            "refund_id": refund_id,
            "order_id": order_id,
            "reason": reason,
            "token": token,
            "user_id": ctx.user_id,
            "session_id": ctx.session_id,
        }, max(ttl, 1))

    def close(self):
        """清理 MCP 连接（仅关闭自建连接；注入的共享连接交由 pod 生命周期）。"""
        if self._mcp_client and self._owns_mcp_client:
            self._mcp_client.close()
            self._mcp_client = None

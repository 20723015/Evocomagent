"""ToolManager：统一管理本地工具和 MCP 工具。

当 MCP 启用时，通过 Streamable HTTP 连接 MCP Server 获取工具；
当 MCP 未启用或连接失败时，退回本地工具。
"""

import json

from app.observability.logging import get_logger

log = get_logger("app.agent.tools.manager")


from app.agent.context import ToolContext
from app.agent.write_registry import confirmable_tools, write_tools
from app.agent.tools.registry import TOOL_DEFINITIONS as LOCAL_TOOL_DEFINITIONS
from app.agent.tools.registry import execute_tool as local_execute_tool

# 需用户确认的写工具（P1-2）：必须本地执行，不得路由到无会话状态的远端
_CONFIRMABLE_TOOLS = confirmable_tools()


def _lease_gate_error(name: str, ctx: ToolContext | None) -> str | None:
    """写工具统一租约门禁（本地/MCP 分流前）。

    修复计划·二轮 1：
    - 缺 ToolContext 或未绑定 lease_guard → SESSION_LOCK_REQUIRED；
    - SessionLockLost / SessionLockBackendUnavailable → 稳定结构化错误；
    - 其他异常（编程错误）继续抛出，不吞。
    """
    from app.agent.write_ops import WRITE_TOOL_STATE_MACHINES

    if name not in WRITE_TOOL_STATE_MACHINES:
        return None
    guard = getattr(ctx, "lease_guard", None) if ctx is not None else None
    if guard is None:
        return json.dumps({
            "error": "SESSION_LOCK_REQUIRED",
            "message": "写操作缺少会话租约上下文，禁止执行",
        }, ensure_ascii=False)
    from app.stores.locks import SessionLockBackendUnavailable, SessionLockLost

    try:
        guard()
    except SessionLockLost as e:
        return json.dumps({
            "error": "SESSION_LOCK_LOST",
            "message": f"会话租约已失效，禁止提交写操作：{e}",
        }, ensure_ascii=False)
    except SessionLockBackendUnavailable as e:
        return json.dumps({
            "error": "SESSION_LOCK_UNAVAILABLE",
            "message": f"会话锁后端不可用，禁止提交写操作：{e}",
        }, ensure_ascii=False)
    return None


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
                # 需用户确认的写工具（P1-2）一律走本地：确认闸门依赖会话状态
                # （SessionState.pending_write + 本轮表态判定），远端 MCP server
                # 是独立进程、看不到会话，路由过去等于绕过确认直接落库。
                if name in _CONFIRMABLE_TOOLS:
                    self._tool_source[name] = "local"
                else:
                    self._tool_source[name] = "mcp"
            self._tool_defs = list(mcp_tools)

            for td in LOCAL_TOOL_DEFINITIONS:
                name = td["function"]["name"]
                if name not in mcp_names:
                    self._tool_defs.append(td)
                    self._tool_source[name] = "local"

        except Exception as e:
            log.warning(
                "mcp.connect_failed_fallback_local",
                error_type=type(e).__name__,
                error=str(e),
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
    # 敏感 MCP 工具 = 敏感读（发 actor token）∪ 写工具（派生自 write_registry）
    _SENSITIVE_MCP_READ_TOOLS = frozenset({
        "query_order", "query_logistics", "query_refund_application",
    })
    _SENSITIVE_MCP_TOOLS = _SENSITIVE_MCP_READ_TOOLS | write_tools()
    _WRITE_TOOLS = write_tools()

    def execute_tool(
        self, name: str, arguments: dict, ctx: ToolContext | None = None,
        timeout: float | None = None,
    ) -> str:
        """根据工具来源分发调用（ctx 透传本地工具）；4.3 工具成功率指标。

        timeout：轮次预算剩余（Agent能力强化计划·改造一 提交规则②），
        仅 MCP/HTTP 路径消费；本地工具即时返回，忽略。
        修复计划：敏感 MCP 工具（订单/物流/退款）发送前按 ToolContext 签发
        actor token（meta 通道）；商品/知识检索仍只依赖服务级 Bearer。
        """
        source = self._tool_source.get(name)

        # 修复计划·二轮 1：写工具租约门禁统一在本地/MCP 分流前执行，
        # 两条路径行为一致；失败返回结构化错误（fail-closed）。
        gate_error = _lease_gate_error(name, ctx)
        if gate_error is not None:
            result_str = gate_error
        elif source == "mcp" and self._mcp_client:
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
                scopes = (
                    (SCOPE_REFUND_WRITE,) if name in self._WRITE_TOOLS
                    else (SCOPE_ORDERS_READ,)
                )
                actor_token = issue_actor_token(user_id, session_id, scopes)
            result_str = self._mcp_client.call_tool(
                name, dict(arguments),
                timeout=timeout,
                actor_token=actor_token,
                write=(name in self._WRITE_TOOLS),
            )
            result_str = str(result_str)
        elif source == "local":
            result_str = local_execute_tool(name, arguments, ctx, timeout=timeout)
        else:
            result_str = json.dumps(
                {"error": f"未知工具: {name}"}, ensure_ascii=False
            )

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
                "indeterminate", "merchant_reviewing", "approved", "rejected",
                "refund_processing", "refunded", "withdrawn",
            ):
                record_tool_side_effect(name, str(data["status"]))
        except Exception:  # noqa: BLE001 —— 指标失败不影响工具结果
            pass
        return result_str

    def close(self):
        """清理 MCP 连接（仅关闭自建连接；注入的共享连接交由 pod 生命周期）。"""
        if self._mcp_client and self._owns_mcp_client:
            self._mcp_client.close()
            self._mcp_client = None

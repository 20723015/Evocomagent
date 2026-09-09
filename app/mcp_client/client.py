"""MCP Client：同步封装，通过 Streamable HTTP 连接 MCP Server。

内部使用后台线程运行异步事件循环，对外暴露同步接口，
使得现有的同步 Agent 代码无需改动即可调用 MCP 工具。
"""

import asyncio
import json
import threading

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.mcp_client.converter import mcp_tools_to_openai


class MCPToolResult(str):
    """工具正文字符串 + 仅进程内可见的 MCP 响应元数据。

    保持 ``str`` 兼容现有调用方；ToolManager 会在返回模型前消费并剥离
    internal_meta。这样确认凭证不进入工具正文、审计消息或 prompt。
    """

    def __new__(cls, value: str, internal_meta: dict | None = None):
        obj = super().__new__(cls, value)
        obj.internal_meta = dict(internal_meta or {})
        return obj


class MCPClient:
    """同步 MCP 客户端，通过 Streamable HTTP 连接远程 MCP Server。"""

    def __init__(self, server_url: str, auth_token: str = ""):
        self._server_url = server_url
        self._auth_token = auth_token
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._connected = threading.Event()
        self._close_event: asyncio.Event | None = None
        self._tool_definitions: list[dict] = []

    def connect(self) -> list[dict]:
        """连接 MCP Server，发现工具，返回 OpenAI 格式的工具定义列表。

        阶段一 1.2：pod 级共享连接可被多个 ToolManager 复用——
        已连接时直接返回缓存定义，不再重建线程/连接。
        """
        if self._session is not None:
            return list(self._tool_definitions)

        self._loop = asyncio.new_event_loop()
        self._close_event = asyncio.Event()
        tool_definitions: list[dict] = []
        error_holder: list[Exception] = []

        def run_loop():
            self._loop.run_until_complete(self._run(tool_definitions, error_holder))

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        self._connected.wait(timeout=30)

        if error_holder:
            raise error_holder[0]

        return tool_definitions

    async def _run(self, tool_definitions: list[dict], error_holder: list[Exception]):
        """后台协程：建立连接 → 发现工具 → 保持存活等待调用。"""
        import httpx

        # 修复计划：mcp 1.29 transport 不接受 headers=，认证头经
        # http_client 的默认 headers 注入（每个请求都携带服务级 Bearer）
        http_client = None
        if self._auth_token:
            http_client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._auth_token}"},
            )
        try:
            async with streamable_http_client(
                self._server_url, http_client=http_client,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session

                    tools_result = await session.list_tools()
                    openai_tools = mcp_tools_to_openai(tools_result.tools)
                    tool_definitions.extend(openai_tools)
                    self._tool_definitions = list(openai_tools)

                    self._connected.set()
                    await self._close_event.wait()
        except Exception as e:
            error_holder.append(e)
            self._connected.set()
        finally:
            # 自建 client 由本协程负责生命周期（transport 只管 SDK 自建的）
            if http_client is not None:
                try:
                    await http_client.aclose()
                except Exception:  # noqa: BLE001 —— 清理失败不影响已捕获的错误
                    pass

    def call_tool(
        self, name: str, arguments: dict, timeout: float | None = None, *,
        actor_token: str | None = None, write: bool = False,
        internal_args: dict | None = None,
    ) -> str:
        """调用 MCP 工具，返回 JSON 字符串结果（修复计划：读/写策略分离）。

        - timeout：调用方传入的剩余预算（普通读调用；默认 30s）；
        - write=True：写调用——timeout 缺省取 settings.tool_write_timeout_seconds，
          超时取消 future 并返回 status=indeterminate（结果未知，禁止自动重试），
          不再把「无超时参数」隐式转成 30s 成功/失败语义；
        - actor_token：短期用户身份（meta 通道注入发送层，不进 schema/日志/结果）；
        - internal_args：执行器内部参数（仅退款确认段），同样走 meta 通道，
          不进入 MCP 工具 schema/模型消息。
        """
        if not self._session or not self._loop:
            return json.dumps({"error": "MCP 客户端未连接"}, ensure_ascii=False)

        kwargs: dict = {}
        if actor_token or internal_args:
            meta: dict = {}
            if actor_token:
                meta["actor"] = actor_token
            if internal_args:
                # 只允许确认执行器使用的三个内部字段，避免将未来新增
                # 参数任意透传到 MCP 服务。
                allowed = {"confirmation_token", "idempotency_key", "refund_id"}
                meta["internal_args"] = {
                    key: value for key, value in internal_args.items()
                    if key in allowed and isinstance(value, str)
                }
            kwargs["meta"] = meta
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments, **kwargs), self._loop
        )
        if write:
            from app.config.settings import settings

            wait = (
                timeout if timeout and timeout > 0
                else settings.tool_write_timeout_seconds
            )
        else:
            wait = timeout if timeout and timeout > 0 else 30
        try:
            result = future.result(timeout=wait)
        except (Exception, asyncio.CancelledError) as e:
            # CancelledError 是 BaseException（3.8+）：任务被取消（loop 关闭/
            # 竞争 close）同样归为轻量失败，不外溢冲击 Agent 主流程
            future.cancel()
            if write:
                return json.dumps({
                    "status": "indeterminate",
                    "tool": name,
                    "error": f"远端写操作执行超时（>{wait:.0f}s），结果未知，禁止自动重试",
                }, ensure_ascii=False)
            return json.dumps(
                {"error": f"MCP 工具调用出错: {e}"}, ensure_ascii=False
            )

        if result.isError:
            text = result.content[0].text if result.content else "未知错误"
            return json.dumps({"error": f"工具执行出错: {text}"}, ensure_ascii=False)

        text = result.content[0].text if result.content else "{}"
        return MCPToolResult(text, getattr(result, "meta", None))

    def close(self):
        """关闭 MCP 连接，清理后台线程。"""
        if self._close_event and self._loop:
            self._loop.call_soon_threadsafe(self._close_event.set)
        if self._thread:
            self._thread.join(timeout=5)
        self._session = None
        self._loop = None
        self._thread = None

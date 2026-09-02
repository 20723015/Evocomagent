"""MCP Server：通过 Streamable HTTP 暴露电商工具服务。

安全（修复计划·MCP 用户身份贯通）：
- 服务端 Bearer 校验：配置 MCP_AUTH_TOKEN 后，匿名请求一律 401；
- 订单/物流/退款为**敏感工具**：要求客户端附带短期 actor token（协议
  meta 通道，`call_tool(..., meta={"actor": jit})`），缺失/过期/签名不符/
  scope 不足一律拒绝，并以 token 里的 sub 重建 ToolContext（归属校验可
  正确 fail-closed）；商品与知识检索仍只要求服务级 Bearer；
- 生产（auth_enabled）缺 MCP_AUTH_TOKEN / MCP_ACTOR_SECRET → 启动失败。

启动方式：python mcp_server/server.py
默认监听：http://127.0.0.1:9123/mcp
依赖钉版：mcp>=1.8,<2（2.x 已改名 FastMCP 导入，迁移前不放开）
"""

import hmac
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import Context, FastMCP

from app.agent.context import ToolContext
from app.agent.tools.order import query_order as _query_order
from app.agent.tools.product import query_product as _query_product
from app.agent.tools.logistics import query_logistics as _query_logistics
from app.agent.tools.refund import apply_refund as _apply_refund
from app.agent.tools.knowledge import search_knowledge as _search_knowledge
from app.config.settings import settings

mcp = FastMCP("ecom-tools", host="127.0.0.1", port=9123)


def _actor_ctx(ctx: Context) -> ToolContext:
    """从请求 meta 读取并校验 actor token，重建 ToolContext（敏感工具专属）。

    失败抛 ValueError（FastMCP 转为工具错误返回，客户端收到结构化错误 JSON）。
    """
    from app.mcp_client.actor import (
        SCOPE_ORDERS_READ,
        ActorTokenError,
        require_scopes,
        validate_actor_token,
    )

    request_context = ctx.request_context if ctx is not None else None
    meta = request_context.meta if request_context is not None else None
    raw = getattr(meta, "actor", None) if meta is not None else None
    if not raw:
        raise ValueError("缺少 actor token：敏感工具要求短期用户身份（orders:read/refund:write）")
    try:
        claims = validate_actor_token(str(raw))
        require_scopes(claims, SCOPE_ORDERS_READ)
    except ActorTokenError as e:
        raise ValueError(f"actor token 校验失败: {e}") from e
    return ToolContext(
        user_id=str(claims.get("sub", "")),
        session_id=str(claims.get("session_id", "") or ""),
    )


def _actor_ctx_refund(ctx: Context) -> ToolContext:
    """退款专用：要求 refund:write scope（不含 orders:read 也可通过）。"""
    from app.mcp_client.actor import (
        SCOPE_REFUND_WRITE,
        ActorTokenError,
        require_scopes,
        validate_actor_token,
    )

    request_context = ctx.request_context if ctx is not None else None
    meta = request_context.meta if request_context is not None else None
    raw = getattr(meta, "actor", None) if meta is not None else None
    if not raw:
        raise ValueError("缺少 actor token：退款是敏感写操作（要求 refund:write）")
    try:
        claims = validate_actor_token(str(raw))
        require_scopes(claims, SCOPE_REFUND_WRITE)
    except ActorTokenError as e:
        raise ValueError(f"actor token 校验失败: {e}") from e
    return ToolContext(
        user_id=str(claims.get("sub", "")),
        session_id=str(claims.get("session_id", "") or ""),
    )


@mcp.tool()
def query_order(order_id: str, ctx: Context) -> str:
    """根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等"""
    user_ctx = _actor_ctx(ctx)
    result = _query_order(order_id, user_ctx)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_product(keyword: str) -> str:
    """根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。支持模糊搜索"""
    result = _query_product(keyword)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_logistics(order_id: str, ctx: Context) -> str:
    """根据订单号查询物流轨迹信息，包括快递公司、运单号、运输状态和轨迹事件"""
    user_ctx = _actor_ctx(ctx)
    result = _query_logistics(order_id, user_ctx)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def apply_refund(order_id: str, reason: str, ctx: Context) -> str:
    """为指定订单申请退款。注意：这是一个敏感操作，调用前应先与用户确认"""
    user_ctx = _actor_ctx_refund(ctx)
    result = _apply_refund(order_id, reason, user_ctx)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索并夕夕的政策与帮助文档（退换货政策、配送说明、会员权益、FAQ）。
    当顾客询问规则、流程、时效、是否支持等政策类问题时使用。
    返回 Top-K 命中片段及来源文档，请基于检索结果回答，不要编造政策。
    """
    result = _search_knowledge(query, top_k=top_k)
    return json.dumps(result, ensure_ascii=False)


class _BearerAuthMiddleware:
    """纯 ASGI 中间件：配置 token 后校验 /mcp 的 Authorization 头。

    健康路径（/healthz 若有）与 MCP 握手之外的一切请求都要求
    `Authorization: Bearer <MCP_AUTH_TOKEN>`，与客户端侧 app.mcp_client 的
    认证头同一 token。
    """

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/mcp":
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            expected = f"Bearer {self.token}"
            # 常量时间比较（评审二轮 D）：避免逐字节短路泄露 token 前缀
            if not hmac.compare_digest(headers.get("authorization", ""), expected):
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b'Bearer realm="mcp"'),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"error": "unauthorized"}',
                })
                return
        await self.app(scope, receive, send)


def create_server_app():
    """构建带认证中间件的 ASGI 应用（供 uvicorn / 测试复用）。

    生产（auth_enabled）且配置了 MCP 服务：缺服务 token / actor 密钥 → 启动失败。
    """
    from app.mcp_client.actor import validate_mcp_security

    validate_mcp_security()
    app = mcp.streamable_http_app()
    if settings.mcp_auth_token:
        return _BearerAuthMiddleware(app, settings.mcp_auth_token)
    print(
        "[WARN] MCP_AUTH_TOKEN 未配置：MCP server 允许匿名调用"
        "（仅限本机开发；生产必须配置）",
    )
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_server_app(), host="127.0.0.1", port=9123)

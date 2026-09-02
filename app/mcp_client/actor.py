"""MCP 短期 actor token（修复计划·MCP 用户身份贯通）。

为什么需要：MCP Client 的 transport 不支持 per-request 认证头
（mcp 1.29 streamable_http_client 仅支持一次性 headers），用户身份改为
**协议层 meta 通道**传递——client 侧 `session.call_tool(..., meta={"actor": jit})`，
服务端在 FastMCP Context 里经 `ctx.request_context.meta` 读取。

设计要点：
- 独立密钥（mcp_actor_secret）与定位（iss=ecom-agent / aud=ecom-mcp），
  与用户入口 JWT（jwt_secret）隔离，防止跨体系复用；
- claims：sub（user_id）/ session_id / scopes / iat / exp / jti；
- 短 TTL（mcp_actor_ttl_seconds=60）：只在单轮工具调用面有效；
- token 只在发送层注入（ToolManager → MCPClient.call_tool），不进模型
  schema、日志、ToolOutcome 或会话记录；
- 生产（auth_enabled=true）+ MCP_ENABLED 时缺服务 token 或 actor 密钥
  启动即失败（fail-fast）。

Scope 约定：订单/物流读取 requires `orders:read`；退款 requires `refund:write`；
商品/知识检索仍只要求服务级 Bearer（中间件）。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt as pyjwt

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.mcp_client.actor")

ALGORITHM = "HS256"
ACTOR_ISSUER = "ecom-agent"
ACTOR_AUDIENCE = "ecom-mcp"  # 固定 audience：与入口 JWT 的 ecom-agent-api 隔离
MIN_SECRET_BYTES = 32

SCOPE_ORDERS_READ = "orders:read"
SCOPE_REFUND_WRITE = "refund:write"
ALL_SCOPES = (SCOPE_ORDERS_READ, SCOPE_REFUND_WRITE)


class ActorTokenError(Exception):
    """actor token 缺失/过期/签名不符/受众不符/scope 不足。"""


class McpSecurityConfigError(Exception):
    """生产环境 MCP 安全配置不完整（缺服务 token / actor 密钥）。"""


def _sign_secret() -> str:
    if not settings.mcp_actor_secret:
        raise McpSecurityConfigError(
            "MCP_ACTOR_SECRET 未配置：无法签发/校验 actor token"
        )
    if len(settings.mcp_actor_secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise McpSecurityConfigError(
            f"MCP_ACTOR_SECRET 过短（需 ≥ {MIN_SECRET_BYTES} 字节，HS256）"
        )
    return settings.mcp_actor_secret


def issue_actor_token(
    user_id: str,
    session_id: str = "",
    scopes: tuple[str, ...] = (SCOPE_ORDERS_READ,),
    *,
    ttl_seconds: Optional[int] = None,
    now_utc: Optional[datetime] = None,
    jti: str = "",
) -> str:
    """签发短期 actor token（HS256；sub/session_id/scopes/iat/exp/jti）。

    仅由 ToolManager 在发送敏感 MCP 工具前调用；token 不落日志/存储。
    """
    secret = _sign_secret()
    now = now_utc or datetime.now(timezone.utc)
    ttl = ttl_seconds if ttl_seconds is not None else settings.mcp_actor_ttl_seconds
    claims: dict = {
        "sub": str(user_id),
        "session_id": str(session_id or ""),
        "scope": " ".join(scopes),
        "iss": ACTOR_ISSUER,
        "aud": ACTOR_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(seconds=max(ttl, 1)),
        "jti": jti or uuid.uuid4().hex,
    }
    return pyjwt.encode(claims, secret, algorithm=ALGORITHM)


def validate_actor_token(token: str) -> dict:
    """校验 actor token；失败抛 ActorTokenError（401 语义）。"""
    try:
        secret = _sign_secret()
    except McpSecurityConfigError as e:
        raise ActorTokenError(str(e)) from e
    try:
        payload = pyjwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            issuer=ACTOR_ISSUER,
            audience=ACTOR_AUDIENCE,
        )
    except pyjwt.ExpiredSignatureError as e:
        raise ActorTokenError("actor token 已过期") from e
    except pyjwt.PyJWTError as e:
        raise ActorTokenError(f"actor token 校验失败: {e}") from e
    sub = payload.get("sub")
    if not sub:
        raise ActorTokenError("actor token 缺少 sub")
    return payload


def require_scopes(claims: dict, *required: str) -> None:
    """scope 不足抛 ActorTokenError（403 语义）。"""
    granted = set(str(claims.get("scope", "") or "").split())
    missing = [s for s in required if s not in granted]
    if missing:
        raise ActorTokenError(
            f"actor token 权限不足（缺少 scope: {' '.join(missing)}）"
        )


def validate_mcp_security() -> None:
    """生产启动校验：MCP_ENABLED + auth_enabled 时缺服务 token/actor 密钥即失败。

    客户端（build_pod_components）与服务端（create_server_app）共用。
    """
    if not (settings.mcp_enabled and settings.mcp_server_url):
        return
    if not settings.auth_enabled:
        return  # 开发环境：允许退化（server 侧仍打警告）
    missing: list[str] = []
    if not settings.mcp_auth_token:
        missing.append("MCP_AUTH_TOKEN")
    if not settings.mcp_actor_secret:
        missing.append("MCP_ACTOR_SECRET")
    if missing:
        raise McpSecurityConfigError(
            "生产环境 MCP 安全配置不完整（拒绝启动）: "
            + "，".join(missing)
        )


def actor_token_subhash(token: str) -> str:
    """日志脱敏用：token 的短指纹（绝不落原始 token）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]

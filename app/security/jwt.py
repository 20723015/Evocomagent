"""入口 JWT（阶段三 3.1）：签发与校验，user_id 一律取自 token，请求体不再接受。

ISSUER/AUDIENCE 校验防跨应用复用；密钥从 settings.jwt_secret 读（K8s Secret 注入）。
auth_enabled=False 时（开发/内部试用）跳过校验，请求体 user_id 直通。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt as pyjwt

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.security.jwt")

ALGORITHM = "HS256"
MIN_SECRET_BYTES = 32  # HMAC-SHA256 最低推荐密钥长度（RFC 7518 §3.2）

# 仅开发兜底：auth_enabled=False 时允许空密钥，但必须显式警告（≥32 字节避免 pyjwt 警告）
DEV_FALLBACK_SECRET = "dev-secret-do-not-use-in-prod-0123456789"


class TokenValidationError(Exception):
    """token 缺失/过期/签名不符/受众不符。"""


class JwtConfigError(Exception):
    """jwt_secret 配置不合法（鉴权已启用时为空或过短）。"""


def validate_jwt_secret() -> str:
    """校验并返回签名密钥；鉴权启用时：空/过短直接失败（fail-fast）。

    auth_enabled=False（开发/内部试用）：回退开发密钥并打警告。
    """
    secret = settings.jwt_secret
    if settings.auth_enabled:
        if not secret:
            raise JwtConfigError(
                "AUTH_ENABLED=true 但 JWT_SECRET 未配置：拒绝启动/签发"
            )
        if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
            raise JwtConfigError(
                f"JWT_SECRET 过短（{len(secret.encode('utf-8'))} 字节，"
                f"需 ≥ {MIN_SECRET_BYTES}，RFC 7518 §3.2）"
            )
        return secret
    if not secret:
        log.warning(
            "JWT_SECRET 未配置且 AUTH_ENABLED=false：使用开发密钥，"
            "生产必须配置 32+ 字节随机密钥",
        )
        return DEV_FALLBACK_SECRET
    return secret


def create_token(
    user_id: str,
    ttl_minutes: Optional[int] = None,
    extra_claims: Optional[dict] = None,
    scopes: str = "",
) -> str:
    """签发用户 token（HS256）。仅测试/内部工具用；生产由网关或同路径签发。

    scopes：空格分隔的权限声明（P1 RBAC，如 "chat ops"）。
    """
    secret = validate_jwt_secret()
    now = datetime.now(timezone.utc)
    claims: dict = {
        "sub": user_id,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes or settings.jwt_ttl_minutes),
    }
    if scopes:
        claims["scope"] = scopes
    if extra_claims:
        claims.update(extra_claims)
    return pyjwt.encode(claims, secret, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """校验并解出 user_id；失败抛 TokenValidationError（→ 401）。"""
    return str(decode_claims(token).get("sub", ""))


def decode_claims(token: str) -> dict:
    """校验并解出全部 claims（P1 RBAC：运营端点需读 scope）；失败抛
    TokenValidationError（→ 401）。
    """
    try:
        secret = validate_jwt_secret()
    except JwtConfigError as e:
        raise TokenValidationError(str(e)) from e
    try:
        payload = pyjwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
    except pyjwt.ExpiredSignatureError as e:
        raise TokenValidationError("token 已过期") from e
    except pyjwt.PyJWTError as e:
        raise TokenValidationError(f"token 校验失败: {e}") from e
    sub = payload.get("sub")
    if not sub:
        raise TokenValidationError("token 缺少 sub")
    return payload

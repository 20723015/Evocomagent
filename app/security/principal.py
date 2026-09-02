"""请求主体与 RBAC（安全修复 P1：运营端点 scope 校验）。

- Principal：一次请求的认证身份（sub + scopes）。
- 用户面（/v1/chat 等）：沿用 authenticate_user——auth_enabled 时 sub 即 user_id；
- 运营面（/v1/handoffs*、/v1/messages/search）：要求 token 携带 `ops` scope。
  历史实现零 scope 校验，AUTH_ENABLED=false 时全网裸奔。

强制时机：
- auth_enabled=true：一律走 JWT（scope 缺失/不足 → 403）；
- auth_enabled=false：开发直通（测试/本地），除非 OPS_RBAC_REQUIRED=true
  单独强制运营面鉴权（不开启用户鉴权也能收紧运营面）。
配套工具：`python -m app.scripts.issue_token --user u1 --scopes chat,ops`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from fastapi import HTTPException, Request

from app.config.settings import settings

SCOPE_OPS = "ops"  # 运营面（handoffs / messages/search）


@dataclass(frozen=True)
class Principal:
    """认证身份：sub（主体）+ scopes（权限集）。"""

    sub: str
    scopes: frozenset = field(default_factory=frozenset)
    via: str = "jwt"  # jwt | anonymous（开发直通）

    def has(self, *scopes: str) -> bool:
        return all(s in self.scopes for s in scopes)


def _anonymous() -> Principal:
    return Principal(sub="anonymous", scopes=frozenset({SCOPE_OPS}), via="anonymous")


def _parse_scopes(raw: str) -> set[str]:
    # OAuth 约定空格分隔；同时容忍逗号（issue_token --scopes chat,ops）
    return {s for s in raw.replace(",", " ").split() if s}


def authorize_scopes(request: Request, *required: str) -> Principal:
    """校验请求身份与 scope；不足抛 401/403（HTTPException）。

    auth_enabled=false 且未强制运营面鉴权时返回匿名 Principal
    （scopes 含 ops，保持本地/测试直通）。
    """
    if not settings.auth_enabled and not settings.ops_rbac_required:
        return _anonymous()

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Bearer token")

    from app.security.jwt import TokenValidationError, decode_claims

    try:
        claims = decode_claims(auth[len("Bearer "):])
    except TokenValidationError as e:
        raise HTTPException(status_code=401, detail=f"认证失败: {e}") from e

    granted = _parse_scopes(str(claims.get("scope", "") or ""))
    missing = [s for s in required if s not in granted]
    if missing:
        raise HTTPException(
            status_code=403,
            detail=f"权限不足（缺少 scope: {' '.join(missing)}）",
        )
    return Principal(
        sub=str(claims.get("sub", "")),
        scopes=frozenset(granted),
    )


def issue_scopes_for_tests(scopes: Iterable[str]) -> str:
    """测试辅助：空格分隔的 scope 字符串（JWT scope claim 的格式）。"""
    return " ".join(scopes)

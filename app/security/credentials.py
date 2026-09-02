"""工具层凭证模式（阶段三 3.2）。

原则：
- 凭证只存在于 ToolContext.credentials，永远不注入 prompt/schema/日志
  （ToolContext 的 credentials 已设 repr=False）；
- HTTP 工具实现一律经 attach_credentials 附加请求头，代码里不出现明文；
- 读操作走服务账号 + 用户上下文（X-User-Id），写操作走 on-behalf-of
  （RFC 8693 token exchange，access_token 带 scope 明示权限）。

本项目订单/物流为 mock 实现，无真实 HTTP 后端；此模块为后端就绪后
「只换 baseURL」的对接点（见计划「真实后端不可用阻塞联调」风险应对）。
"""

from __future__ import annotations

from typing import Optional

from app.agent.context import ToolContext


def build_tool_credentials(user_id: str, scope: str = "read", claims: Optional[dict] = None) -> dict:
    """从认证上下文构造工具凭证（供 ToolContext.credentials）。

    scope: read / write（写操作才带 OBO token 交换语义）。
    """
    creds = {"sub": user_id, "scope": scope}
    if scope == "write":
        # RFC 8693 占位：真实后端的 token exchange 端点就绪后在此换取 access_token。
        # 明确约定：token 只进 header，不进 prompt/schema/日志。
        creds["obo"] = True
    if claims:
        creds["claims"] = dict(claims)
    return creds


def attach_credentials(headers: dict, credentials: Optional[dict]) -> dict:
    """给 HTTP 请求头附加凭证（唯一允许接触 credentials 的出口）。

    读：X-User-Id / Authorization: Bearer <服务账号>；
    写（OBO）：Authorization: Bearer <access_token> + X-OBO-Scope。
    """
    if not credentials:
        return headers
    headers["X-User-Id"] = credentials.get("sub", "")
    if credentials.get("obo"):
        headers["X-OBO-Scope"] = credentials.get("scope", "write")
    return headers


def credentials_from_ctx(ctx: ToolContext) -> dict:
    return ctx.credentials or {}

"""知识审核后台（阶段六 6.3）：pending 候选 → 编辑/批准/拒绝 → 索引重建。

独立可跑的轻量 Web（uvicorn app.review.webapp:create_review_app --factory）：
- GET  /login        登录页（REVIEW_ADMIN_TOKEN）
- POST /login        校验令牌 → 签发 HttpOnly 签名会话 cookie
- GET  /              pending 候选列表（ledger.list_pending + aging 标注）
- POST /approve/{cid} 批准并发布（走 run_evolution --approve 同路径：
                      发布 → build → verify → 切 generation）
- POST /reject/{cid}  拒绝（ledger.drop_pending，清理入 trash）
- GET  /aging         过期知识（frontmatter effective_date 超过阈值）

安全修复 P1（历史实现零认证 + XSS，任何人 POST /approve 即可写知识库，
再经检索进 prompt，等于公开注入入口）：
- REVIEW_ADMIN_TOKEN 未配置 → 整站 503 fail-closed；
- 登录签发 HMAC 签名会话 cookie（HttpOnly / SameSite=Lax）；
- 全部表单携带 CSRF token（HMAC 派生）并在 POST 校验；
- 全部 HTML 插值经 html.escape（候选问题/答案来自外部用户对话）。

批准路径自动触发 IndexBuildService「验证失败不切指针」：
验证不过 → generation 不切换，坏知识不会污染线上检索。
"""

from __future__ import annotations

import hashlib
import hmac
import html
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config.settings import settings

SESSION_COOKIE = "review_session"
SESSION_TTL_SECONDS = 8 * 3600


def _hmac(secret: str, message: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256,
    ).hexdigest()


def _make_services():
    from app.scripts.run_evolution import _build_services

    return _build_services()


def _pending_entries():
    svc = _make_services()
    ledger = svc["ledger"]
    aging_entries = []
    for cid, entry in ledger.list_pending(settings.evolve_pending_aging_days):
        aging_entries.append((cid, entry))
    return aging_entries


def _session_secret() -> str:
    # 会话/CSRF 签名密钥与管理令牌同源：管理令牌本身就是唯一的秘密
    return settings.review_admin_token


def _issue_session() -> tuple[str, str]:
    """返回 (cookie 值, csrf token)。cookie = <expiry>.<sig>。"""
    expiry = str(int(time.time()) + SESSION_TTL_SECONDS)
    cookie = f"{expiry}.{_hmac(_session_secret(), expiry)}"
    csrf = _hmac(_session_secret(), f"csrf:{expiry}")
    return cookie, csrf


def _validate_session(request: Request) -> bool:
    cookie = request.cookies.get(SESSION_COOKIE, "")
    if "." not in cookie:
        return False
    expiry, sig = cookie.split(".", 1)
    if not expiry.isdigit():
        return False
    if not hmac.compare_digest(sig, _hmac(_session_secret(), expiry)):
        return False
    return int(expiry) >= int(time.time())


async def _require_session(request: Request) -> tuple[str, str]:
    """所有路由入口：未配置令牌 → 503 fail-closed；未登录 → 401。

    返回 (csrf token, 登录态)；未登录时 csrf 为空串。
    """
    if not settings.review_admin_token:
        raise HTTPException(
            status_code=503,
            detail="审核后台未配置 REVIEW_ADMIN_TOKEN，拒绝服务（fail-closed）",
        )
    if not _validate_session(request):
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
    expiry = request.cookies.get(SESSION_COOKIE, "").split(".", 1)[0]
    return _hmac(_session_secret(), f"csrf:{expiry}")


def _check_csrf(request: Request, csrf_form: str) -> None:
    if not settings.review_admin_token:
        raise HTTPException(status_code=503, detail="审核后台未配置 REVIEW_ADMIN_TOKEN")
    if not _validate_session(request):
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
    expiry = request.cookies.get(SESSION_COOKIE, "").split(".", 1)[0]
    expected = _hmac(_session_secret(), f"csrf:{expiry}")
    if not csrf_form or not hmac.compare_digest(csrf_form, expected):
        raise HTTPException(status_code=403, detail="CSRF 校验失败")


LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>知识审核后台 · 登录</title></head>
<body><h2>知识审核后台</h2>
<form method="post" action="/login">
<input type="password" name="token" placeholder="管理令牌" style="width:320px">
<button type="submit">登录</button>
</form></body></html>"""

PENDING_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>知识审核后台</title></head>
<body><h2>待审核候选（{n}）</h2><table border="1" cellpadding="6">
<tr><th>候选</th><th>问题</th><th>答案</th><th>状态</th><th>操作</th></tr>
{rows}
</table><p><a href="/aging">查看过期知识</a> ·
<form method="post" action="/logout" style="display:inline">{csrf_field}<button>退出登录</button></form></p></body></html>"""

AGING_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>过期知识</title></head>
<body><h2>过期/临期知识文档（{n}）</h2><table border="1" cellpadding="6">
<tr><th>文档</th><th>effective_date</th><th>owner</th></tr>
{rows}</table><p><a href="/">返回审核列表</a></p></body></html>"""


def _csrf_field(csrf: str) -> str:
    return f"<input type='hidden' name='csrf' value='{html.escape(csrf, quote=True)}'>"


def create_review_app() -> FastAPI:
    app = FastAPI(title="知识审核后台", version="0.2.0")

    @app.get("/login", response_class=HTMLResponse)
    async def login_page():
        if not settings.review_admin_token:
            raise HTTPException(
                status_code=503,
                detail="审核后台未配置 REVIEW_ADMIN_TOKEN，拒绝服务（fail-closed）",
            )
        return LOGIN_HTML

    @app.post("/login")
    async def login(token: str = Form("")):
        if not settings.review_admin_token:
            raise HTTPException(status_code=503, detail="审核后台未配置 REVIEW_ADMIN_TOKEN")
        if not secrets.compare_digest(token, settings.review_admin_token):
            raise HTTPException(status_code=401, detail="管理令牌错误")
        cookie, _ = _issue_session()
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            SESSION_COOKIE, cookie,
            max_age=SESSION_TTL_SECONDS, httponly=True,
            # 评审二轮 D：SameSite 收紧为 Strict（CSRF token 仍保留为纵深防御）；
            # Secure 位仅在声明 TLS 部署时开启（明文 http 本地开发会收不到 cookie）
            samesite="strict", secure=settings.review_admin_cookie_secure,
        )
        return resp

    @app.post("/logout")
    async def logout(request: Request, csrf: str = Form("")):
        _check_csrf(request, csrf)
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        csrf = await _require_session(request)
        entries = _pending_entries()
        rows = "".join(
            f"<tr><td>{html.escape(str(cid))}</td>"
            f"<td>{html.escape(str(entry.get('question', ''))[:60])}</td>"
            f"<td>{html.escape(str(entry.get('answer', ''))[:120])}</td>"
            f"<td>{html.escape(str(entry.get('status', 'pending')))}</td>"
            f"<td>"
            f"<form method='post' action='/approve/{html.escape(str(cid), quote=True)}' style='display:inline'>"
            f"{_csrf_field(csrf)}<button>批准发布</button></form> "
            f"<form method='post' action='/reject/{html.escape(str(cid), quote=True)}' style='display:inline'>"
            f"{_csrf_field(csrf)}<button>拒绝</button></form></td></tr>"
            for cid, entry in entries
        )
        return PENDING_HTML.format(n=len(entries), rows=rows, csrf_field=_csrf_field(csrf))

    @app.post("/approve/{cid}")
    async def approve(cid: str, request: Request, csrf: str = Form("")):
        _check_csrf(request, csrf)
        from app.scripts.run_evolution import main as evo_main

        rc = evo_main(["--approve", cid])
        if rc != 0:
            raise HTTPException(status_code=400, detail=f"批准失败（exit {rc}）")
        return {"ok": True, "candidate_id": cid, "published": True}

    @app.post("/reject/{cid}")
    async def reject(cid: str, request: Request, csrf: str = Form("")):
        _check_csrf(request, csrf)
        from app.scripts.run_evolution import main as evo_main

        rc = evo_main(["--reject", cid])
        if rc != 0:
            raise HTTPException(status_code=400, detail=f"拒绝失败（exit {rc}）")
        return {"ok": True, "candidate_id": cid, "rejected": True}

    @app.get("/aging", response_class=HTMLResponse)
    async def aging(request: Request):
        await _require_session(request)
        from app.review.scan import scan_expired_knowledge

        expired = scan_expired_knowledge(
            Path(settings.kb_dir), settings.knowledge_aging_days,
        )
        rows = "".join(
            f"<tr><td>{html.escape(str(d['path']))}</td>"
            f"<td>{html.escape(str(d['effective_date']))}</td>"
            f"<td>{html.escape(str(d['owner'] or '-'))}</td></tr>"
            for d in expired
        )
        return AGING_HTML.format(n=len(expired), rows=rows)

    return app

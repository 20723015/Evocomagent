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
from app.review.service import ReviewLockHeld

SESSION_COOKIE = "review_session"
SESSION_TTL_SECONDS = 8 * 3600


def _hmac(secret: str, message: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256,
    ).hexdigest()


def _make_services():
    from app.scripts.run_evolution import _build_services

    return _build_services()


def _review_service():
    """每次请求独立装配（锁内 reload 保证读到磁盘正本，不做长期缓存）。"""
    from app.review.service import ReviewService

    svc = _make_services()
    return ReviewService(
        ledger=svc["ledger"], lock=svc["lock"], pipeline=svc["pipeline"],
    )


def _pending_entries():
    svc = _make_services()
    # GET 使用新鲜 ledger 快照，在返回副本上计算 aging；不要调用
    # Ledger.list_pending（它会把 aging 标记写回磁盘），否则无锁请求
    # 可能以陈旧内存态覆盖其他审核实例刚完成的编辑/批准。
    from app.scripts.run_evolution import _pending_entries_read_only

    return _pending_entries_read_only(
        svc["ledger"], settings.evolve_pending_aging_days,
    )


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
<body><h2>待审核候选（{n}）</h2>
{rows}
<p><a href="/aging">查看过期知识</a> ·
<form method="post" action="/logout" style="display:inline">{csrf_field}<button>退出登录</button></form></p></body></html>"""

# 每条候选一个卡片：来源标注（人工工单显示提交客服/知识依据/证据路径）、
# 可编辑的规范问题/标准答案（编辑走 /edit 复扫），批准/拒绝按钮。
CANDIDATE_CARD = """<fieldset style="margin:12px 0"><legend>{cid_short}</legend>
<p><b>来源</b>：{source_label}　<b>状态</b>：{status}　<b>创建</b>：{created_at}{human_meta}</p>
<form method="post" action="/edit/{cid_attr}" style="margin:6px 0">
{csrf_field}
<input type="hidden" name="expected_revision" value="{revision}">
<label>规范问题<br><textarea name="question" rows="2" cols="80">{question}</textarea></label><br>
<label>标准答案<br><textarea name="answer" rows="4" cols="80">{answer}</textarea></label><br>
<button type="submit">保存编辑（重新复扫）</button>
</form>
<form method="post" action="/approve/{cid_attr}" style="display:inline">
{csrf_field}<input type="hidden" name="expected_revision" value="{revision}"><button>批准发布</button></form>
<form method="post" action="/reject/{cid_attr}" style="display:inline">
{csrf_field}<input type="hidden" name="expected_revision" value="{revision}"><button>拒绝</button></form>
</fieldset>"""

AGING_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>过期知识</title></head>
<body><h2>过期/临期知识文档（{n}）</h2><table border="1" cellpadding="6">
<tr><th>文档</th><th>effective_date</th><th>owner</th></tr>
{rows}</table><p><a href="/">返回审核列表</a></p></body></html>"""


def _human_meta_html(entry: dict) -> str:
    """人工候选的审核元数据（提交客服/知识依据/证据路径；全部 HTML 转义）。"""
    if entry.get("source_kind") != "human_handoff":
        return ""
    basis = html.escape(str(entry.get("knowledge_basis", "")))
    submitted = html.escape(str(entry.get("submitted_by", "")))
    evidence = "、".join(
        html.escape(str(p)) for p in (entry.get("evidence_paths") or [])
    ) or "-"
    meta = (
        f"<br><b>提交客服</b>：{submitted}　"
        f"<b>知识依据</b>：{basis}　<b>证据路径</b>：{evidence}"
    )
    review = entry.get("llm_review") or {}
    if review:
        meta += _llm_review_html(review)
    return meta


def _llm_review_html(review: dict) -> str:
    """夜间 LLM 评审结论（建议性：新颖性/价值分；人工做最终接受/拒绝决定）。"""
    novel = review.get("novel")
    if novel is True:
        novel_text = "新知识（检索无命中）"
    elif novel is False:
        side = html.escape(str(review.get("duplicate_of") or "?"))
        novel_text = f"疑似重复（命中侧：{side}）"
    else:
        novel_text = "新颖性未知（检索不可用）"
    worth = review.get("worth_saving")
    score = review.get("quality_score")
    score_text = f"（{score}）" if isinstance(score, (int, float)) else ""
    if worth is True:
        worth_text = f"值得沉淀{score_text}"
    elif worth is False:
        worth_text = f"不建议沉淀{score_text}"
    else:
        worth_text = "未评分"
    reason = html.escape(str(review.get("reason", "")))
    reviewed_at = html.escape(str(review.get("reviewed_at", "")))
    return (
        f"<br><b>LLM 评审（建议）</b>：新颖性 {html.escape(novel_text)}　"
        f"价值 {html.escape(worth_text)}　{reason}　{reviewed_at}"
    )


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
        csrf_field = _csrf_field(csrf)
        rows = "".join(
            CANDIDATE_CARD.format(
                cid_short=html.escape(str(cid)[:16]),
                cid_attr=html.escape(str(cid), quote=True),
                source_label="人工工单" if entry.get("source_kind") == "human_handoff"
                else "机器人自进化",
                status=html.escape(str(entry.get("status", "pending"))),
                created_at=html.escape(str(entry.get("created_at", ""))),
                human_meta=_human_meta_html(entry),
                csrf_field=csrf_field,
                revision=html.escape(str(int(entry.get("revision", 0) or 0))),
                question=html.escape(str(entry.get("question", ""))),
                answer=html.escape(str(entry.get("answer", ""))),
            )
            for cid, entry in entries
        )
        return PENDING_HTML.format(n=len(entries), rows=rows, csrf_field=csrf_field)

    @app.post("/edit/{cid}")
    async def edit(cid: str, request: Request, csrf: str = Form(""),
                   question: str = Form(""), answer: str = Form(""),
                   expected_revision: str = Form("0")):
        """审核员编辑规范问题/标准答案：锁内 reload + 乐观锁 + 重新复扫。

        成功 → revision 递增并 303 返回列表页；revision 过期 → 409；
        内容不合法/命中 PII/注入 → 422（不落盘）。
        """
        _check_csrf(request, csrf)
        try:
            revision = int(expected_revision)
        except ValueError:
            raise HTTPException(status_code=422, detail="expected_revision 必须为整数")
        try:
            result = _review_service().edit(cid, question, answer, revision)
        except ReviewLockHeld as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        status = result["status"]
        if status == "edited":
            return RedirectResponse("/", status_code=303)
        if status == "not_found":
            raise HTTPException(status_code=404, detail="候选不存在或已出 pending")
        if status == "revision_conflict":
            raise HTTPException(
                status_code=409,
                detail=f"候选已被其他审核员修改（当前 revision "
                       f"{result.get('current_revision')}），请刷新后重试",
            )
        raise HTTPException(status_code=422, detail=result.get("detail", status))

    @app.post("/approve/{cid}")
    async def approve(cid: str, request: Request, csrf: str = Form(""),
                      expected_revision: str = Form("0")):
        """批准发布：结构化状态返回，绝不虚报发布成功。

        published → 200；duplicate_rejected → 200（published=false，候选已终结）；
        publish_forbidden（SELF_EVOLVE_ENABLED=false）→ 409；
        not_found → 404；revision_conflict → 409；publish_failed → 500。
        """
        _check_csrf(request, csrf)
        try:
            revision = int(expected_revision)
        except ValueError:
            raise HTTPException(status_code=422, detail="expected_revision 必须为整数")
        try:
            result = _review_service().approve(cid, revision)
        except ReviewLockHeld as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        status = result["status"]
        if status == "published":
            return {"status": status, "candidate_id": cid, "published": True,
                    "filename": result.get("filename", "")}
        if status == "duplicate_rejected":
            return {"status": status, "candidate_id": cid, "published": False,
                    "detail": result.get("detail", "")}
        if status == "publish_forbidden":
            raise HTTPException(status_code=409, detail=result.get("detail", status))
        if status == "not_found":
            raise HTTPException(status_code=404, detail="候选不存在或已出 pending")
        if status == "revision_conflict":
            raise HTTPException(
                status_code=409,
                detail=f"候选已被其他审核员修改（当前 revision "
                       f"{result.get('current_revision')}），请刷新后重试",
            )
        raise HTTPException(status_code=500,
                            detail=result.get("detail", "发布事务失败"))

    @app.post("/reject/{cid}")
    async def reject(cid: str, request: Request, csrf: str = Form(""),
                     expected_revision: str = Form("")):
        """手工拒绝：``manual_review_reject`` 原因持久化，候选出 pending。"""
        _check_csrf(request, csrf)
        revision = None
        if expected_revision.strip():
            try:
                revision = int(expected_revision)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail="expected_revision 必须为整数",
                )
        try:
            result = _review_service().reject(cid, revision)
        except ReviewLockHeld as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        if result["status"] == "rejected":
            return {"status": "rejected", "candidate_id": cid, "rejected": True}
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail="候选不存在或已出 pending")
        raise HTTPException(status_code=409, detail="候选已被其他审核员修改，请刷新后重试")

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

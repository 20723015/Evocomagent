"""Web 聊天界面挂载测试:GET / 返回页面,且 API 路由不被静态挂载遮蔽;
前端视觉/可访问性改造(2026-08)后,补充静态资源、关键 DOM 结构、
aria 属性与 CSS 变量一致性的结构断言(不引入构建链,直接检查源码)。"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from app.server.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _static() -> dict[str, str]:
    """静态资源源码映射(每用例内拉取,走 TestClient 覆盖真实挂载路径)。"""
    with _client() as client:
        return {
            "index": client.get("/").text,
            "ops": client.get("/ops.html").text,
            "css": client.get("/css/app.css").text,
            "common": client.get("/js/common.js").text,
            "chat": client.get("/js/chat.js").text,
            "ops_js": client.get("/js/ops.js").text,
        }


def test_root_serves_index_html():
    with _client() as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "小夕" in resp.text


def test_api_routes_not_shadowed_by_static_mount():
    """Mount("/") 在路由注册之后才不遮蔽 API——防止回归。"""
    with _client() as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        # 未匹配的 API 路径仍是 FastAPI 404（而非静态文件 404）
        resp = client.post("/v1/chat", json={})
        assert resp.status_code == 422


# ---------- 静态资源 ----------


def test_static_assets_served():
    with _client() as client:
        for path, mime in [
            ("/css/app.css", "text/css"),
            ("/js/common.js", "javascript"),
            ("/js/chat.js", "javascript"),
            ("/js/ops.js", "javascript"),
            ("/ops.html", "text/html"),
        ]:
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert mime in resp.headers["content-type"], path


# ---------- 对话页关键结构 ----------


def test_index_key_structure_and_aria():
    with _client() as client:
        html = client.get("/").text
    for dom_id in [
        "sidebar", "sidebarMask", "sessionList", "userIdInput",
        "healthDot", "healthText", "chatTitle", "btnReset",
        "chat", "inner", "input", "send", "btnNew", "srStatus",
    ]:
        assert f'id="{dom_id}"' in html, f"index 缺少 #{dom_id}"
    # 可访问性:输入框标签、消息区语义、状态播报、侧栏抽屉开关
    assert 'aria-label="输入问题"' in html
    assert 'role="log"' in html
    assert 'role="status"' in html
    assert 'aria-expanded="false"' in html  # 侧栏抽屉开关初始态


# ---------- 运营工作台关键结构 ----------


def test_ops_tabs_and_forms_structure():
    with _client() as client:
        html = client.get("/ops.html").text
    for dom_id in [
        "opsUser", "statusFilter", "refreshTickets", "ticketBox",
        "searchQ", "searchSession", "searchLimit", "doSearch",
        "searchBox", "srStatus",
        "tabBtnUpload", "kbFile", "kbUploadBtn", "kbProgress", "kbDocs",
    ]:
        assert f'id="{dom_id}"' in html, f"ops 缺少 #{dom_id}"
    # 标签语义:tablist / tab + aria-selected + tabpanel + roving tabindex
    assert 'role="tablist"' in html
    assert html.count('role="tab"') == 3
    assert 'aria-selected="true"' in html
    assert html.count('role="tabpanel"') == 3
    assert 'tabindex="0"' in html and 'tabindex="-1"' in html
    # 表单控件均有可访问名称(sr-only label 或 aria-label)
    assert 'for="searchQ"' in html
    assert 'aria-label="检索用户标识"' in html


# ---------- 界面脚本的状态/可访问性逻辑 ----------


def test_chat_js_ui_states():
    js = _static()["chat"]
    # 会话项 = 选择/删除并列按钮(无嵌套交互结构),当前会话有标记
    assert 'className = "s-main"' in js
    assert 'setAttribute("aria-label", "删除会话记录' in js
    assert 'setAttribute("aria-current", "true")' in js
    assert 'setAttribute("role", "button")' not in js
    # 过程卡展开状态与移动端抽屉(Escape/遮罩/滚动锁定/inert 焦点管理)
    assert "aria-expanded" in js
    assert '"Escape"' in js
    assert "modal-open" in js
    assert "aria-hidden" in js
    assert ".inert" in js
    assert "matchMedia" in js
    # 输入框:内容超过上限才开启内部滚动(移动端空态无滚动箭头)
    assert 'overflowY = inputEl.scrollHeight > 132 ? "auto" : "hidden"' in js
    # 状态播报(回复/失败/中断)
    assert "announce(" in js


def test_ops_js_ui_states():
    js = _static()["ops_js"]
    # 标签切换走 aria-selected + hidden;加载/空态复用公共组件
    assert 'setAttribute("aria-selected"' in js
    assert ".hidden = " in js
    assert "loadingHint(" in js
    assert "emptyState(" in js
    # roving tabindex + 方向键/Home/End 键盘导航
    assert "tabIndex = selected ? 0 : -1" in js
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert f'"{key}"' in js, key


# ---------- CSS 变量一致性与残留检查 ----------


def test_css_variables_all_defined():
    """app.css 里引用的 var(--x) 必须都在 :root 定义(防无效变量残留)。"""
    css = _static()["css"]
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
    used = set(re.findall(r"var\((--[a-z0-9-]+)[,)]", css))
    undefined = used - defined
    assert not undefined, f"未定义的 CSS 变量: {sorted(undefined)}"


def test_no_stale_legacy_tokens():
    """历史主题变量/类名不许回流(muted/primary 时代已废弃)。"""
    for name, src in _static().items():
        assert "var(--muted)" not in src, name
        assert "var(--primary" not in src, name
        assert "--primary:" not in src, name

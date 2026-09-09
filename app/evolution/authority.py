"""authority.py：文档权威性判定（人工知识链路修正 D 决策）。

只依据**位置 + frontmatter** 两个事实判定，owner 字段彻底不参与——
消除 owner=ops 误判（publisher 写 ops 与 uploads 冲突的根因）。

规则（fail-closed）：
- 非 ``evolved/`` 前缀（根目录 + uploads/）→ authoritative；
- ``evolved/`` 下读 frontmatter：source_kind ∈ {human_conversation, human_handoff}
  （human_handoff 仅作历史文档读取兼容）或自动沉淀 → managed；
- frontmatter 缺失（空 meta）/解析失败/文件不可读 → authoritative。
"""

from __future__ import annotations

from pathlib import Path

from app.config.settings import settings

AUTHORITATIVE = "authoritative"
MANAGED = "managed"

EVOLVED_PREFIX = "evolved/"
# human_handoff 仅作读取兼容（历史已发布文档）；新发布统一写 human_conversation
_MANAGED_SOURCE_KINDS = ("human_conversation", "human_handoff")


def authority_kind(source_path: str, *, kb_dir=None) -> str:
    """判定一个知识库文档的权威性：authoritative | managed。

    kb_dir 可注入（测试指向 tmp 知识库）；缺省用 settings.kb_dir。
    """
    path = str(source_path or "")
    if not path or not path.startswith(EVOLVED_PREFIX):
        return AUTHORITATIVE
    root = Path(kb_dir) if kb_dir is not None else Path(settings.kb_dir)
    doc = root / path
    try:
        from app.agent.rag.loader import parse_frontmatter

        meta, _body = parse_frontmatter(doc.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AUTHORITATIVE
    if not meta:
        # evolved/ 下无 frontmatter（正常发布必有）：fail-closed
        return AUTHORITATIVE
    if meta.get("source_kind") in _MANAGED_SOURCE_KINDS:
        return MANAGED
    # evolved/ 下其余合法 frontmatter = 自动沉淀文档 → managed
    return MANAGED

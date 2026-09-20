"""知识文档治理（RAG 修复计划·2）：索引范围与元数据校验。

规则：
- 只索引：知识库根目录的有效文档、`evolved/`、`uploads/`；
- 明确排除：`archive/`、`.trash/`、`.staging/`、隐藏目录/文件、编辑器临时文件；
- 元数据（md/txt frontmatter）：
    status: active | archived
    authority: platform | external_reference
    effective_date: ISO 日期（YYYY-MM-DD）
- `archived` 或 `external_reference` 一律**不进线上回答索引**；
- 校验开启（settings.rag_doc_metadata_required / strict 构建）时，缺失、非法或
  冲突的元数据直接失败，并输出具体文件与原因。
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

# 允许索引的顶层目录（"" = 根目录直属文件）
ALLOWED_ROOTS = ("", "evolved", "uploads")
# 明确排除的目录名（任意层级）
EXCLUDED_DIRS = frozenset({"archive", ".trash", ".staging", "__pycache__"})
_VALID_STATUS = ("active", "archived")
_VALID_AUTHORITY = ("platform", "external_reference")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_METADATA_SUFFIXES = (".md", ".txt")
SIDECAR_SUFFIX = ".meta.yaml"


class DocumentGovernanceError(ValueError):
    """文档治理校验失败：含具体文件与原因（strict 构建据此中止）。"""


def sidecar_path(doc_path: Path) -> Path:
    """同名 sidecar：`x.pdf` → `x.pdf.meta.yaml`（非 md/txt 的元数据载体）。"""
    return Path(str(doc_path) + SIDECAR_SUFFIX)


def parse_sidecar(text: str) -> dict:
    """解析 sidecar（简单 `key: value` 行；值去引号）。"""
    meta: dict = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key:
            meta[key] = value
    return meta


def load_sidecar(doc_path: Path) -> dict:
    path = sidecar_path(doc_path)
    try:
        return parse_sidecar(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def metadata_for(doc_path: Path, rel: str, text: str = "") -> dict:
    """取文档元数据：md/txt 用 frontmatter；其它格式用同名 sidecar。"""
    if Path(rel).suffix.lower() in _METADATA_SUFFIXES:
        from app.agent.rag.loader import parse_frontmatter

        meta, _ = parse_frontmatter(text or "")
        return meta
    return load_sidecar(doc_path)


def write_sidecar(doc_path: Path, *, status: str = "active",
                  authority: str = "platform", effective_date: str = "") -> Path:
    """为文档写 sidecar（上传发布链路用）。返回 sidecar 路径。"""
    path = sidecar_path(doc_path)
    path.write_text(
        f"status: {status}\nauthority: {authority}\n"
        f"effective_date: {effective_date}\n",
        encoding="utf-8",
    )
    return path


def is_excluded_path(rel: str) -> bool:
    """路径是否被排除（目录黑名单 / 隐藏段 / 临时文件）。"""
    parts = Path(rel).parts
    if not parts:
        return True
    for part in parts:
        if part in EXCLUDED_DIRS:
            return True
        if part.startswith("."):  # .trash/.staging/.git/隐藏文件
            return True
    name = parts[-1]
    if name.endswith(("~", ".tmp", ".bak", ".swp", ".orig")):
        return True
    return False


def is_allowed_scope(rel: str) -> bool:
    """是否在允许范围内（根目录文件 / evolved/ / uploads/ 子树）。"""
    parts = Path(rel).parts
    if len(parts) <= 1:
        return True  # 根目录直属文件
    return parts[0] in ("evolved", "uploads")


def is_indexable(rel: str) -> bool:
    """目录级可索引判定：在允许范围内且未被排除。"""
    return is_allowed_scope(rel) and not is_excluded_path(rel)


def _valid_date(value: str) -> bool:
    if not _DATE_RE.match(value or ""):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def validate_metadata(rel: str, meta: dict) -> None:
    """校验单文档元数据；不合法抛 DocumentGovernanceError（含文件与原因）。"""
    status = str(meta.get("status", "") or "").strip().lower()
    authority = str(meta.get("authority", "") or "").strip().lower()
    effective = str(meta.get("effective_date", "") or "").strip()

    problems: list[str] = []
    if not status:
        problems.append("缺少 status")
    elif status not in _VALID_STATUS:
        problems.append(f"status 非法({status})")
    if not authority:
        problems.append("缺少 authority")
    elif authority not in _VALID_AUTHORITY:
        problems.append(f"authority 非法({authority})")
    if not effective:
        problems.append("缺少 effective_date")
    elif not _valid_date(effective):
        problems.append(f"effective_date 非法({effective})")
    if status == "active" and authority == "external_reference":
        problems.append("冲突：active 文档不得为 external_reference")

    if problems:
        raise DocumentGovernanceError(f"{rel}: " + "；".join(problems))


def is_index_eligible(meta: dict) -> bool:
    """元数据决定是否进入线上回答索引（archived / external_reference 不进入）。"""
    status = str(meta.get("status", "") or "").strip().lower()
    authority = str(meta.get("authority", "") or "").strip().lower()
    if status == "archived":
        return False
    if authority == "external_reference":
        return False
    return True


def metadata_required_for(rel: str) -> bool:
    """当前仅对 md/txt 强校验元数据（其它格式无 frontmatter 载体）。"""
    return Path(rel).suffix.lower() in _METADATA_SUFFIXES

"""loader.py：文档加载与元数据解析（7.1 多格式接入 + 7.7 沉淀元数据补强）。

两个纯函数（无 IO，可直接单测）：
- normalize_document：统一文档原始文本（去 BOM、CRLF→LF、剔除行尾空白、
  去掉开头/结尾多余空行），多格式接入的公共前置步骤。
- parse_frontmatter：解析文件开头的 frontmatter 元数据块（如 provenance
  / owner），供 7.7 沉淀文档与 7.1 统一元数据规范识别使用。

frontmatter 格式示例：
    ---
    provenance: turn-abc123
    owner: system
    ---
    正文从这里开始……
"""

from __future__ import annotations

_BOM = "\ufeff"


def normalize_document(raw: str) -> str:
    """规范化文档原始文本：去 BOM、CRLF→LF、剔除行尾空白、去掉开头/结尾多余空行。

    示例：
        normalize_document("\\ufeff第1行  \\r\\n\\r\\n  第2行\\r\\n\\r\\n")
        # → "第1行\\n\\n  第2行"（BOM 去除、CRLF 归一、行尾空格剔除、两端空行清理）
    """
    text = str(raw)
    if text.startswith(_BOM):
        text = text[len(_BOM):]
    # CRLF / 单独 CR 一律归一为 LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 剔除每行行尾空白（保留行首缩进与内部空行结构）
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析文件开头的 frontmatter 块，返回 (meta, 正文)。

    仅处理**文件开头**的 ``---\\nkey: value\\n...\\n---\\n`` 块：
    - key 小写化、value 去首尾空白；只保留非空 key。
    - 无 frontmatter，或格式不完整（以 ``---`` 开头但找不到收尾 ``---``）
      → 返回 ({}, 原文本)。

    格式示例：
        ---
        provenance: turn-abc123
        owner: system
        ---
        正文从这里开始……

    返回：
        meta：key → value 的 dict（value 均为字符串，key 保持文件中的先后顺序）
        正文：frontmatter 块之后的文本（不含 frontmatter 行）
    """
    lines = text.splitlines()
    # 不以 --- 开头 → 不是 frontmatter 块
    if not lines or lines[0].strip() != "---":
        return {}, text

    meta: dict = {}
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            # 找到收尾 ---：其后即为正文
            body = "\n".join(lines[idx + 1:])
            return meta, body
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key:
                meta[key] = value
    # 以 --- 开头但未闭合 → 格式不完整，视为无 frontmatter
    return {}, text
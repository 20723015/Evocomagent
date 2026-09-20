"""loader.py：文档加载与元数据解析（7.1 多格式接入 + 7.7 沉淀元数据补强）。

纯函数（无 IO，可直接单测）：
- normalize_document：统一文档原始文本（去 BOM、CRLF→LF、剔除行尾空白、
  去掉开头/结尾多余空行），多格式接入的公共前置步骤。
- parse_frontmatter：解析文件开头的 frontmatter 元数据块（如 provenance
  / owner），供 7.7 沉淀文档与 7.1 统一元数据规范识别使用。
- serialize_frontmatter：meta → frontmatter 块（统一序列化正本——
  publisher / revalidate 刷新共用，杜绝三套实现漂移）。
- decode_text / read_text（批次4）：字节/文件 → 文本的编码回退链
  （utf-8 → gb18030），替代各处写死的 ``encoding="utf-8"``。

frontmatter 格式示例：
    ---
    provenance: turn-abc123
    owner: system
    ---
    正文从这里开始……

序列化安全约定：值含特殊字符（冒号/引号/井号/空白边界等）时用 JSON 字符串
字面量（合法 YAML 流标量，无 YAML 注入面）；解析端对称解码——
`submitted_by: "ops-a"` 读出 `ops-a`，绝不携带字面引号。值一律为字符串。
"""

from __future__ import annotations

import json
from pathlib import Path

_BOM = "\ufeff"

# 编码回退链（批次4）：utf-8 优先，其次 gb18030（GBK/GB2312 的超集）。
# 不含 utf-8-sig——BOM 本身是合法 utf-8 且 normalize_document 已剥，不可达。
_TEXT_ENCODINGS = ("utf-8", "gb18030")
# gb18030 能「成功」解码绝大多数随机字节，因此解码后还要按 NUL / 不可打印
# 控制字符占比再判一次，否则「二进制文件仍解析失败」的语义不成立。
_CONTROL_CHARS = frozenset(
    chr(c) for c in list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20))
)
_CONTROL_RATIO_LIMIT = 0.02


def _looks_binary(text: str) -> bool:
    """含 NUL 或不可打印控制字符占比超限 → 视为二进制（非文本）。"""
    if "\x00" in text:
        return True
    if not text:
        return False
    bad = sum(1 for ch in text if ch in _CONTROL_CHARS)
    return bad / len(text) > _CONTROL_RATIO_LIMIT


def decode_text(raw: bytes) -> str:
    """字节 → 文本：utf-8 → gb18030 依次尝试；全失败抛 ``ValueError``。

    回退顺序固定，不做编码探测（不引入 chardet）；GBK 文本因此不再抛
    ``UnicodeDecodeError`` 让 strict 构建中止。
    """
    for encoding in _TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if not _looks_binary(text):
            return text
    raise ValueError("文本解码失败（已尝试 utf-8 / gb18030，疑似二进制内容）")


def read_text(path) -> str:
    """读文件并按编码回退链解码（批次4；唯一文本读取入口）。"""
    return decode_text(Path(path).read_bytes())


# 值中出现这些字符（或首尾空白）时序列化加 JSON 引号。
# 注意逗号不在列：grounded_on 的单行逗号格式（a.md, b.md）保持既有字节形态，
# 行式解析（首个冒号切分）对逗号无歧义。
_NEEDS_QUOTE = (":", '"', "'", "#", "\n", "{", "}", "[", "]", "&", "*", "!",
                "|", ">", "%", "@", "`")


def _decode_scalar(raw: str) -> str:
    """解码标量：JSON 引号字面量 → 真实值；其余原样（去首尾空白）。"""
    value = raw.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return str(json.loads(value))
        except (json.JSONDecodeError, ValueError):
            return value[1:-1]
    return value


def _encode_scalar(value: str) -> str:
    """编码标量：含特殊字符 → JSON 字符串字面量；否则裸值（去首尾空白）。"""
    text = str(value).strip()
    if not text:
        return '""'
    if any(ch in text for ch in _NEEDS_QUOTE) or text != str(value):
        return json.dumps(text, ensure_ascii=False)
    return text


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
            if key:
                meta[key] = _decode_scalar(value)
    # 以 --- 开头但未闭合 → 格式不完整，视为无 frontmatter
    return {}, text


def serialize_frontmatter(meta: dict) -> str:
    """meta → frontmatter 块（统一序列化正本）。

    - 未知字段原样保留（refresh_last_validated 依赖此性质）；
    - 值经 _encode_scalar：特殊字符 JSON 引号化，其余裸值；
    - 空值跳过（与历史行为一致：frontmatter 不落空条目）。
    """
    lines = ["---"]
    for key, value in meta.items():
        if value is None or str(value).strip() == "":
            continue
        lines.append(f"{key}: {_encode_scalar(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n"
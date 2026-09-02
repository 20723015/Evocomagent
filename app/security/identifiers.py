"""标识符校验（安全修复 P2：user_id/session_id 路径穿越）。

user_id / session_id 会被拼进文件路径（{session_dir}/{user_id}/{session_id}.json、
{memory_dir}/{user_id}.json）与 Redis key。历史实现无字符集校验，
`../` 可逃逸目录。统一白名单：字母/数字开头，仅含 [A-Za-z0-9._-]，
长度 1..128；session_id 允许空串（缺省默认会话）。
"""

from __future__ import annotations

import re

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

MAX_IDENTIFIER_LENGTH = 128


class InvalidIdentifier(ValueError):
    """标识符含非法字符/超长（映射 4xx，不落盘、不建 key）。"""


def validate_identifier(value: str, field: str = "identifier",
                        *, allow_empty: bool = False) -> str:
    """校验并原样返回；非法抛 InvalidIdentifier。

    allow_empty=True 供 session_id（空 = 该用户默认会话）使用；
    user_id 一律非空。
    """
    if value is None:
        value = ""
    if not value:
        if allow_empty:
            return ""
        raise InvalidIdentifier(f"{field} 不能为空")
    if len(value) > MAX_IDENTIFIER_LENGTH or not IDENTIFIER_RE.match(value):
        raise InvalidIdentifier(
            f"{field} 含非法字符或超长（仅允许字母/数字开头，"
            f"[A-Za-z0-9._-]，≤{MAX_IDENTIFIER_LENGTH} 字符）"
        )
    return value

"""签发带 scope 的 JWT（安全修复 P1 配套开发工具）。

RBAC 落地后，运营端点（/v1/handoffs*、/v1/messages/search）要求 `ops` scope，
本地 curl 必须先签 token——没有这个脚本，第一个跑测试的人就会卡住（评审·坑5）。

用法：
    python -m app.scripts.issue_token --user u1 --scopes chat,ops --ttl 60
    curl -H "Authorization: Bearer <token>" http://localhost:8000/v1/handoffs
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="签发带 scope 的 JWT（开发/测试用）")
    parser.add_argument("--user", required=True, help="主体（sub，即 user_id）")
    parser.add_argument(
        "--scopes", default="chat",
        help="逗号或空格分隔的 scope（如 chat,ops；运营端点需要 ops）",
    )
    parser.add_argument("--ttl", type=int, default=60, help="有效期（分钟，默认 60）")
    args = parser.parse_args()

    from app.observability.logging import get_logger
    from app.security.jwt import create_token

    scopes = args.scopes.replace(",", " ").split()
    token = create_token(args.user, ttl_minutes=args.ttl, scopes=" ".join(scopes))
    # token 本身就是输出：写 stdout 供管道/变量捕获（CLI 工具，非日志流）
    import sys

    sys.stdout.write(token + "\n")


if __name__ == "__main__":
    main()

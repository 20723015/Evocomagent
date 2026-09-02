"""解析 worker（子进程入口）：python -m app.agent.rag.parse_worker <path>

输出协议：单行 JSON 到 stdout —— {"ok": "<解析文本>"} | {"err": "<message>"}；
非零退出码仅表示 worker 自身异常（结果仍以 JSON 为准）。
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        _emit({"err": "缺少文件路径参数"})
        return 2
    from pathlib import Path

    path = argv[1]
    try:
        from app.agent.rag.loader import normalize_document
        from app.agent.rag.parsers import parse_document
        from app.config.settings import settings

        text = normalize_document(parse_document(Path(path)))
        limit = settings.kb_upload_max_text_chars
        if len(text) > limit:
            text = text[:limit]
        _emit({"ok": text})
        return 0
    except BaseException as e:  # noqa: BLE001 —— 任何异常回传（含 KeyboardInterrupt）
        _emit({"err": f"{type(e).__name__}: {e}"})
        return 1


def _emit(payload: dict) -> None:
    """协议通道输出（单行 JSON 到 stdout——subprocess 捕获用，不是日志打印）。

    ensure_ascii=True：Windows 子进程 stdout 默认按本地代码页（如 GBK）编码，
    父进程固定按 UTF-8 解码，直接写中文会变成 mojibake；ASCII 转义后
    两条路径都能无损还原。
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""解析守护（v7 评审）：子进程执行 parse_document，超时可 terminate/kill。

背景：PDF/DOCX 解析是宿主进程内的 C 扩展调用（pypdf/python-docx），
线程超时只能「放弃等待」而不能终止恶意输入（解析炸弹继续耗尽 CPU/内存）。

实现：subprocess + `python -m app.agent.rag.parse_worker`（独立 worker 模块，
**不用 multiprocessing.spawn**——Windows 上 spawn 会对 sys.argv[0] 做
runpy.run_path，在 uvicorn/pytest 入口下会重跑主脚本，不可用）；
- 协议：worker 输出单行 JSON 到 stdout（{"ok": text} | {"err": message}）；
- 超时：subprocess.TimeoutExpired → 进程已被 run() 终止 → 抛 ParseTimeout；
- 子进程崩溃/非零退出 → 抛 ValueError（调用方按「解析失败」fail-fast）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading as _threading
from pathlib import Path
from typing import Optional

# parse_guard.py 位于 app/agent/rag/ → 项目根 = 上溯 4 层
ROOT = Path(__file__).resolve().parent.parent.parent.parent


class ParseTimeout(ValueError):
    """解析超过预算被终止（调用方按解析失败处理，fail-fast）。"""


def parse_with_timeout(path: Path, timeout: Optional[float] = None,
                       *, _impl=None) -> str:
    """在可终止子进程中解析文档；超时抛 ParseTimeout。

    _impl 仅供单测：显式传入时不走子进程（单进程直调 + 手动计时），用于
    验证超时异常路径；生产调用一律不传（保证真实子进程隔离）。
    """
    from app.config.settings import settings

    budget = float(timeout if timeout is not None else settings.kb_upload_parse_timeout)
    if _impl is not None:
        # 单测快捷路径：只验证超时语义，不验证进程隔离
        import time as _time

        from app.agent.rag.loader import normalize_document

        result: dict = {}
        exc: list[BaseException] = []

        def _run():
            try:
                result["text"] = normalize_document(_impl(path))
            except BaseException as e:  # noqa: BLE001
                exc.append(e)

        t = _threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(budget)
        if t.is_alive():
            raise ParseTimeout(f"解析超时（>{budget:.0f}s），已终止（{path.name}）")
        if exc:
            raise ValueError(f"解析失败: {exc[0]}")
        return result.get("text", "")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "app.agent.rag.parse_worker", str(path)],
            cwd=str(ROOT), capture_output=True, text=True,
            timeout=budget, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as e:
        raise ParseTimeout(f"解析超时（>{budget:.0f}s），已终止（{path.name}）") from e
    except OSError as e:
        raise ParseTimeout(f"解析子进程启动失败: {e}") from e

    stdout = (proc.stdout or "").strip()
    try:
        payload = json.loads(stdout.splitlines()[-1] if stdout else "{}")
    except json.JSONDecodeError:
        raise ValueError(f"解析子进程输出无效: {stdout[:200]}") from None
    if payload.get("ok") is not None:
        return str(payload["ok"])
    raise ValueError(f"解析失败: {payload.get('err') or '(无输出)'}")


def _is_win() -> bool:  # 保留给外部诊断
    import os

    return os.name == "nt"

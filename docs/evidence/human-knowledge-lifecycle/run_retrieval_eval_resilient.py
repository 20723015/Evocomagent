# ruff: noqa
"""检索评测弹性驱动：给 embedder 加指数退避重试后复用 run_retrieval_eval 全部逻辑。

背景：535 条用例串行调用远程 embedding（SophNet），任何一次瞬时 503/SSL EOF
都会让整跑崩溃。本驱动在 embedder 外层包 8 次指数退避重试（最长 45s 间隔），
评测逻辑、口径与门槛判断与原脚本完全一致。

用法：python tmp/run_retrieval_eval_resilient.py --json-out tmp/retrieval_baseline.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import app.agent.rag.embedder as embedder_mod

_real_create = embedder_mod.create_embedder


class _RetryingEmbedder:
    def __init__(self, inner):
        self._inner = inner
        self.model = inner.model

    def encode(self, texts, timeout=None):
        last = None
        for attempt in range(8):
            try:
                return self._inner.encode(texts, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - 瞬时网络故障统一退避重试
                last = exc
                wait = min(2**attempt, 45)
                print(
                    f"[embed-retry] {type(exc).__name__}: {str(exc)[:100]}"
                    f" — {wait}s 后重试 ({attempt + 1}/8)",
                    flush=True,
                )
                time.sleep(wait)
        raise last

    def encode_one(self, text, timeout=None):
        return self.encode([text], timeout=timeout)[0]


def _retrying_create(*args, **kwargs):
    return _RetryingEmbedder(_real_create(*args, **kwargs))


embedder_mod.create_embedder = _retrying_create

import app.scripts.run_retrieval_eval as rre  # noqa: E402

if __name__ == "__main__":
    rre.main()

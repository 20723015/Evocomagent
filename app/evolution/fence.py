"""fence.py：人工链路会话栅栏（MySQL GET_LOCK，连接级锁）。

分层语义（评审钉死）：fence 只**收敛竞态窗口**（避免「旧知识上线后再补偿
下架」），正确性硬保证来自发布结算的 CAS 条件更新——fence 失效不破坏正确性，
只让竞态窗口回到补偿路径。

- 连接级锁必须绑定同一专用连接（``engine.connect()`` 取出后全程持有，
  参照 migrate_db.py 的 GET_LOCK 先例）；连接断开时服务端自动释放（崩溃自愈）；
- 持有期间心跳用 ``IS_USED_LOCK(name) == CONNECTION_ID()`` 校验未被断连，
  丢失视为租约丢失（走 HumanLeaseLost 同路径中止）；
- 锁序防死锁：按 key 字典序获取；部分失败逆序全释放；
- RELEASE_LOCK 逆序释放；
- sqlite（测试）方言下 no-op。
"""

from __future__ import annotations

import hashlib
import json

from app.observability.logging import get_logger

log = get_logger("app.evolution.fence")


class FenceLost(RuntimeError):
    """栅栏丢失（连接断开/锁被服务端回收）；调用方按租约丢失同路径中止。"""


class FenceTimeout(RuntimeError):
    """栅栏获取超时（另一发布/接入正在持有同一会话）。"""


def conversation_fence_key(source: str, external_conversation_id: str) -> str:
    """返回固定长度、无拼接歧义的会话栅栏 key。

    MySQL ``GET_LOCK`` 的名称上限为 64 字节，而两个外部标识合计可达
    192 字符。对 canonical JSON 做 SHA-256 后截取 60 位，连同 ``hk:``
    前缀共 63 个 ASCII 字节；同一逻辑会话的所有版本仍共享同一把锁。
    """
    canonical = json.dumps(
        [str(source), str(external_conversation_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "hk:" + hashlib.sha256(canonical).hexdigest()[:60]


class ConversationFence:
    """MySQL 会话栅栏：一批会话 key 的全或无获取（字典序，防死锁）。"""

    def __init__(self, engine, timeout_seconds: float = 10.0):
        self._engine = engine
        self._timeout = float(timeout_seconds)
        self._conn = None
        self._held: list[str] = []
        self._noop = engine.dialect.name == "sqlite"

    # ---------- 获取/释放 ----------
    def acquire(self, keys: list[str], *, timeout: float | None = None) -> None:
        """获取全部会话锁；超时抛 FenceTimeout（调用方映射 503 / 批次 retry_wait）。

        幂等保护：重复 acquire 先释放既有持有（正常用法是每批一次）。
        """
        if self._held:
            self.release()
        if self._noop:
            self._held = list(dict.fromkeys(keys))
            return
        wanted = sorted(dict.fromkeys(str(k) for k in keys if str(k)))
        if not wanted:
            return
        timeout = self._timeout if timeout is None else float(timeout)
        self._conn = self._engine.connect()
        held: list[str] = []
        try:
            for key in wanted:
                got = self._conn.exec_driver_sql(
                    "SELECT GET_LOCK(%s, %s)", (key, timeout)
                ).scalar()
                if got != 1:
                    raise FenceTimeout(
                        f"会话栅栏获取超时（{len(held)}/{len(wanted)} 已持有）"
                    )
                held.append(key)
        except Exception:
            self._release_on(held)
            try:
                self._conn.close()
            except Exception as exc:  # noqa: BLE001 - close is best effort
                log.debug("fence.close_failed err=%s", type(exc).__name__)
            self._conn = None
            raise
        self._held = held

    def release(self) -> None:
        """逆序释放全部已持有锁并关闭专用连接（幂等）。"""
        if self._noop:
            self._held = []
            return
        self._release_on(self._held)
        self._held = []
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as exc:  # noqa: BLE001 - close is best effort
                log.debug("fence.close_failed err=%s", type(exc).__name__)
            self._conn = None

    def _release_on(self, held: list[str]) -> None:
        if self._conn is None:
            return
        for key in reversed(held):
            try:
                self._conn.exec_driver_sql("SELECT RELEASE_LOCK(%s)", (key,))
            except Exception as exc:  # noqa: BLE001 - 释放失败由连接关闭兜底
                log.warning(
                    "fence.release_failed key_prefix=%s err=%s",
                    key[:8],
                    type(exc).__name__,
                )

    # ---------- 持有校验（心跳用） ----------
    def assert_held(self) -> None:
        """校验全部锁仍由本连接持有；丢失抛 FenceLost（= 租约丢失同路径）。"""
        if self._noop:
            return
        if self._conn is None or not self._held:
            raise FenceLost("会话栅栏未持有")
        conn_id = self._conn.exec_driver_sql("SELECT CONNECTION_ID()").scalar()
        for key in self._held:
            owner = self._conn.exec_driver_sql(
                "SELECT IS_USED_LOCK(%s)", (key,)
            ).scalar()
            if owner is None or int(owner) != int(conn_id):
                raise FenceLost(f"会话栅栏已丢失: {key[:8]}…")

    @property
    def held_keys(self) -> list[str]:
        return list(self._held)

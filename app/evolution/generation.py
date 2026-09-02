"""generation.py：GenerationStore —— 索引代际（generation）指针管理（第10期）。

- 指针文件：settings.kb_generation_path 指定的 JSON（极小，每次读取开销可忽略）。
- 结构：{backend: {generation_id, target, embedding_model, previous_generation_id}}。
- activate 通过 tmp + os.replace 原子写；Windows 上目标被占用（PermissionError）
  指数退避重试 5 次。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.stores.base import StorageUnavailableError


@dataclass
class GenerationInfo:
    """一个索引代际的描述。target：numpy 为索引文件路径，chroma 为 collection 名。"""

    generation_id: str
    target: str
    embedding_model: str
    previous_generation_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationInfo":
        return cls(**data)


def new_generation_id(clock=None) -> str:
    """YYYYMMDDHHMMSS-<uuid4[:8]>，字典序即时间序。"""
    now = (clock.now() if clock else datetime.now()).strftime("%Y%m%d%H%M%S")
    return f"{now}-{uuid.uuid4().hex[:8]}"


class GenerationStore:
    """代际指针的持久化读写（阶段二 2.4：注入 redis 即走共享指针，跨 pod 一致）。

    strict_shared=True（生产，锁后端 mysql/redis 时）：Redis 读写失败**不降级**
    本地文件——多 Pod 下指针分裂比「不可用」更糟；写失败抛
    StorageUnavailableError，调用方保持 ALIAS_ACTIVATED journal 等待恢复补齐。
    strict_shared=True 且 Redis 未配置 → 构造即抛（多 Pod 上传链路强依赖 Redis）。

    strict_shared=False（单机开发）：维持既有「处处降级」哲学。
    """

    def __init__(self, path, redis_client=None, redis_key: str = "",
                 strict_shared: bool = False):
        if strict_shared and redis_client is None:
            raise RuntimeError(
                "GenerationStore(strict_shared=True) 要求 Redis：上传链路的分片状态与"
                "generation 指针都依赖共享 Redis，MySQL GET_LOCK 只替代锁不替代 Redis"
            )
        self._path = Path(path)
        self._redis = redis_client
        self._redis_key = redis_key or f"kb_generations:{self._path.name}"
        self._strict = strict_shared

    @property
    def path(self) -> Path:
        return self._path

    def _read_payload(self) -> dict:
        if self._redis is not None:
            try:
                raw = self._redis.get(self._redis_key)
                if raw is not None:
                    return json.loads(raw)
            except Exception as e:  # noqa: BLE001 —— strict：fail-closed
                if self._strict:
                    raise StorageUnavailableError(
                        f"共享 generation 指针读取失败（strict_shared）: {e}"
                    ) from e
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_payload(self, payload: dict) -> None:
        if self._redis is not None:
            try:
                self._redis.set(
                    self._redis_key, json.dumps(payload, ensure_ascii=False, indent=2)
                )
                return
            except Exception as e:  # noqa: BLE001 —— strict：fail-closed，不降级文件
                if self._strict:
                    raise StorageUnavailableError(
                        f"共享 generation 指针写入失败（strict_shared，保持 journal 等待恢复）: {e}"
                    ) from e
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        last_err: Optional[Exception] = None
        for attempt in range(5):
            try:
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                os.replace(tmp, self._path)
                return
            except PermissionError as e:  # 目标被其他进程短暂占用
                last_err = e
                time.sleep(0.1 * (2**attempt))
        raise last_err or RuntimeError(f"无法写入 generation 指针: {self._path}")

    def read(self) -> dict[str, GenerationInfo]:
        """读取全部后端的当前代际；缺失/损坏返回 {}。"""
        data = self._read_payload()
        out: dict[str, GenerationInfo] = {}
        for backend, value in data.items():
            if isinstance(value, dict) and value.get("generation_id"):
                try:
                    out[backend] = GenerationInfo.from_dict(value)
                except TypeError:
                    continue
        return out

    def active(self, backend: str) -> Optional[GenerationInfo]:
        return self.read().get(backend.lower())

    def activate(self, backend: str, info: GenerationInfo) -> None:
        """原子切换指针；文件后备路径上 Windows PermissionError 指数退避重试 5 次。"""
        backend = backend.lower()
        data = {k: v.to_dict() for k, v in self.read().items()}
        data[backend] = info.to_dict()
        self._write_payload(data)
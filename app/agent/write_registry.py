"""写工具单一注册源（批次 2：注册表合并，P2-b）。

所有「哪些工具是写工具、各自的目标标识 / 状态机」清单一律从本模块派生；
其余模块不得再手写工具名清单。

新增写工具：在 ``_REGISTRY`` 注册一行 spec，并在对应层实现状态机；
漏注册的工具在执行前被拦截（fail-closed），一致性单测（test_write_registry）
会因派生集合漂移直接失败。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WriteToolSpec:
    """一个写工具的注册项。"""

    state_machine: str  # WriteOpTracker 状态机名
    target_arg: str  # 目标标识参数名（order_id / application_id）
    # 写确认两阶段协议（P1-2）：True = 首次调用只登记草稿、不落库，须等用户
    # 确认轮之后的调用才真正执行（工具层强制，不依赖提示词自律）。
    # 撤回类不在此列——撤回是「收回」语义，用户复述申请编号即可放行。
    requires_confirmation: bool = False


# 单一注册表：新写工具只在这里加一行
_REGISTRY: dict[str, WriteToolSpec] = {
    "submit_refund_application": WriteToolSpec(
        state_machine="refund_application",
        target_arg="order_id",
        requires_confirmation=True,
    ),
    "cancel_refund_application": WriteToolSpec(
        state_machine="refund_application",
        target_arg="application_id",
    ),
}


def write_tool_spec(name: str) -> WriteToolSpec | None:
    """未注册工具返回 None（调用方 fail-closed）。"""
    return _REGISTRY.get(name)


def write_tools() -> frozenset[str]:
    """全部写工具名。"""
    return frozenset(_REGISTRY)


def confirmable_tools() -> frozenset[str]:
    """需要两阶段用户确认的写工具名（P1-2）。"""
    return frozenset(
        name for name, spec in _REGISTRY.items() if spec.requires_confirmation
    )


def state_machines() -> dict[str, str]:
    """写工具 → 状态机名（write_ops 拦截未注册工具用）。"""
    return {name: spec.state_machine for name, spec in _REGISTRY.items()}


def target_of(name: str, arguments: dict) -> str:
    """写操作目标标识（去重/拒绝名单/对账锚点用）。"""
    spec = _REGISTRY.get(name)
    key = spec.target_arg if spec is not None else "order_id"
    return str((arguments or {}).get(key, "") or "").strip()

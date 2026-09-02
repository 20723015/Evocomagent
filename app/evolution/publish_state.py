"""发布事务阶段常量与统一恢复决策（2.7）。

阶段（journal.stage，五个提交点，按顺序推进）：
    PREPARED         journal 已写（新文档尚未移动/构建）
    INDEX_BUILT      候选索引已构建并验证（尚未动 alias/指针）
    ALIAS_ACTIVATED  ES alias 已切到候选（真实 alias 确认为准）
    POINTER_UPDATED  generation 指针已切到候选
    LEDGER_COMMITTED ledger 已提交（发布事务完成；之后只差清 journal/报告）

恢复决策（recover_decide）：同时读真实 Alias、pointer 指针、journal 三源
判定，杜绝「只看 pointer 就回滚」造成的已生效知识回退：

    pointer 已指向候选                       → ledger_only（补 ledger、清 journal）
    alias 已指向候选、pointer 未更新（ES）    → forward（只前进，绝不回滚已生效知识）
    alias 指向非旧代、非候选代（ES）          → blocked（写 kb_write_blocked、
                                                保留现场停止全部 KB 写入，人工 reconcile）
    Alias 读取失败/多目标（ES）               → blocked（状态未知，禁止猜测回滚）
    alias/pointer 均未指向候选               → rollback（删候选索引与新文档、恢复旧文档）
    非 ES 后端（无 alias）以 pointer 为提交点 → ledger_only | rollback

revalidate 隔离事务复用同一阶段表：其「回滚」分支的语义是前进式完成隔离
（判定已做出不可回退，见 revalidate.py 说明）。
"""

from __future__ import annotations

from typing import Optional

# 发布事务阶段（2.7 冻结）
PH_PREPARED = "PREPARED"
PH_INDEX_BUILT = "INDEX_BUILT"
PH_ALIAS_ACTIVATED = "ALIAS_ACTIVATED"
PH_POINTER_UPDATED = "POINTER_UPDATED"
PH_LEDGER_COMMITTED = "LEDGER_COMMITTED"

# 提交点之后（alias 已动手）：这些阶段失败只前进，禁止回滚文件
COMMIT_STAGES = (PH_ALIAS_ACTIVATED, PH_POINTER_UPDATED, PH_LEDGER_COMMITTED)

# 恢复动作
REC_ROLLBACK = "rollback"
REC_FORWARD = "forward"
REC_LEDGER_ONLY = "ledger_only"
REC_BLOCKED = "blocked"


def recover_decide(
    backend: str,
    stage: str,
    pointer_target: str,
    alias_target: Optional[str],
    candidate_target: str,
    previous_target: str = "",
) -> str:
    """统一恢复决策（纯函数，便于故障注入单测）。

    参数：
    - backend：numpy | chroma | es（es 才有 alias 真身）
    - stage：journal 里的阶段（PREPARED/INDEX_BUILT/...；空=未写）
    - pointer_target：generation 指针当前 target（未激活为 ""）
    - alias_target：真实 alias 指向（读 ES；非 es 后端传 ""）。ES
      alias 读取失败时传 ``None``；``None`` 与「alias 不存在/尚未切换」
      的空字符串严格区分，前者必须阻断恢复，不能猜测回滚。
    - candidate_target：journal 里的候选索引 target
    - previous_target：journal 记录的发布前活动代 target（判断「非旧代」用）
    """
    backend = (backend or "numpy").lower()
    # stage 为空 = 旧格式 journal（2.7 前）：按 pointer/alias 判定，不依赖 stage

    if pointer_target == candidate_target:
        return REC_LEDGER_ONLY  # 指针已切（提交点 2 已过）：只补 ledger

    if backend != "es":
        # 无 alias 真身：pointer 即提交点；未切 → 回滚
        return REC_ROLLBACK

    if alias_target == candidate_target:
        # alias 已切、指针未切：知识已生效，只前进
        return REC_FORWARD
    if alias_target is None:
        # ES 状态不可确认（连接/权限/响应解析失败）：不能把未知状态
        # 当作 alias 未切换，否则可能删除已经生效的 candidate index/docs。
        return REC_BLOCKED
    if alias_target and alias_target != previous_target:
        # alias 指向既非旧代也非候选代：未知状态，人工 reconcile
        return REC_BLOCKED
    return REC_ROLLBACK


def stage_order() -> dict[str, int]:
    """阶段→顺序（用于判断「是否已过某提交点」）。"""
    return {
        PH_PREPARED: 1,
        PH_INDEX_BUILT: 2,
        PH_ALIAS_ACTIVATED: 3,
        PH_POINTER_UPDATED: 4,
        PH_LEDGER_COMMITTED: 5,
    }

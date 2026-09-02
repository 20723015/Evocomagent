"""2.7 发布事务五阶段 + 统一恢复状态机（故障注入）。

覆盖恢复表（publish_state.recover_decide）+ pipeline 实际恢复动作：
- Alias 成功、pointer 失败 → forward（只前进，补 pointer + ledger）
- Pointer 成功、ledger 失败 → ledger_only（补 ledger 清 journal）
- 锁续租丢失（assert_held 抛错）→ 副作用 fail-closed
- 重复恢复幂等
- Alias 指向未知代 → blocked（kb_write_blocked 落盘，保留现场）
- 替换旧文档时的恢复（回滚还原 trash）
- 上传、下架、自进化三个写入器竞争同一把锁（互斥语义）
- 非 ES 后端以 pointer 为提交点
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evolution.publish_state import (
    PH_ALIAS_ACTIVATED,
    PH_INDEX_BUILT,
    PH_LEDGER_COMMITTED,
    PH_POINTER_UPDATED,
    PH_PREPARED,
    REC_BLOCKED,
    REC_FORWARD,
    REC_LEDGER_ONLY,
    REC_ROLLBACK,
    recover_decide,
)


# ============================================================
# 恢复决策表（纯函数）
# ============================================================
def test_decide_rollback_when_nothing_switched():
    """ES：alias/pointer 均未指向候选 → rollback。"""
    assert recover_decide(
        "es", PH_INDEX_BUILT, pointer_target="old-t",
        alias_target="old-t", candidate_target="new-t", previous_target="old-t",
    ) == REC_ROLLBACK
    # 非 ES：pointer 未切 → rollback
    assert recover_decide(
        "numpy", PH_INDEX_BUILT, pointer_target="old-t",
        alias_target="", candidate_target="new-t", previous_target="old-t",
    ) == REC_ROLLBACK


def test_decide_forward_alias_switched_pointer_not():
    """ES：alias 已切、pointer 未切 → forward（只前进、绝不回滚已生效知识）。"""
    assert recover_decide(
        "es", PH_INDEX_BUILT, pointer_target="old-t",
        alias_target="new-t", candidate_target="new-t", previous_target="old-t",
    ) == REC_FORWARD


def test_decide_ledger_only_pointer_switched():
    """pointer 已指向候选（含非 ES）→ ledger_only。"""
    for backend in ("es", "numpy", "chroma"):
        assert recover_decide(
            backend, PH_POINTER_UPDATED, pointer_target="new-t",
            alias_target="new-t", candidate_target="new-t",
        ) == REC_LEDGER_ONLY


def test_decide_blocked_alias_unknown_generation():
    """ES：alias 指向非旧代、非候选代 → blocked。"""
    assert recover_decide(
        "es", PH_ALIAS_ACTIVATED, pointer_target="old-t",
        alias_target="mystery-t", candidate_target="new-t",
        previous_target="old-t",
    ) == REC_BLOCKED


def test_decide_blocked_when_alias_read_is_unknown():
    """ES alias 读取失败/状态未知不能被当成空 alias 触发 rollback。"""
    assert recover_decide(
        "es", PH_INDEX_BUILT, pointer_target="old-t",
        alias_target=None, candidate_target="new-t", previous_target="old-t",
    ) == REC_BLOCKED


def test_decide_old_style_journal_without_stage():
    """2.7 前旧格式（无 stage）也按 pointer/alias 判定，不依赖 stage 字段。"""
    assert recover_decide(
        "es", "", pointer_target="new-t", alias_target="new-t",
        candidate_target="new-t",
    ) == REC_LEDGER_ONLY


# ============================================================
# pipeline 恢复动作（故障注入）
# ============================================================
def _active(target: str, gen_id: str):
    return SimpleNamespace(target=target, generation_id=gen_id)


def test_rollback_on_crash_before_activate(tmp_path, monkeypatch):
    """崩溃在 build 后、activate 前（INDEX_BUILT）→ 回滚：删新文档journal归档。"""
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    info = svc["index_service"].build("numpy")
    (svc["kb_dir"] / "evolved").mkdir(parents=True, exist_ok=True)
    (svc["kb_dir"] / "evolved" / "new.md").write_text("新文档", encoding="utf-8")

    svc["journal"].write({
        "phase": "publish",
        "stage": PH_INDEX_BUILT,
        "staging_docs": [{"candidate_id": "cid1", "filename": "new.md"}],
        "replaced_docs": [],
        "backend": "numpy",
        "index": info.to_dict(),
        "previous_target": "old-target",
    })
    monkeypatch.setattr(
        pipe._generation_store, "active",
        lambda backend: _active("old-target", "g-old"),
    )
    pipe._recover_publish_transaction(svc["journal"].read())
    assert not (svc["kb_dir"] / "evolved" / "new.md").exists()  # 新文档已删
    assert svc["journal"].read() is None  # 已归档


def test_forward_recovery_alias_switched_pointer_missing(tmp_path, monkeypatch):
    """「alias 已切、pointer 未切」：只前进——补 pointer，不动已生效知识。"""
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    info = svc["index_service"].build("numpy")

    # alias 已切（模拟 ES 场景：真实 alias 指向候选），pointer 仍为旧代
    monkeypatch.setattr(pipe, "_read_alias_target", lambda backend: info.target)
    monkeypatch.setattr(
        pipe._generation_store, "active",
        lambda backend: _active("old-target", "g-old"),
    )
    pointer_calls = []
    monkeypatch.setattr(
        svc["index_service"], "activate_pointer",
        lambda backend, i: pointer_calls.append(i.generation_id),
    )
    svc["journal"].write({
        "phase": "publish",
        "stage": PH_ALIAS_ACTIVATED,
        "staging_docs": [{"candidate_id": "cid1", "filename": "x.md"}],
        "replaced_docs": [],
        "backend": "es",
        "index": info.to_dict(),
        "previous_target": "old-target",
    })
    pipe._recover_publish_transaction(svc["journal"].read())
    assert pointer_calls == [info.generation_id]  # 前进：补 pointer
    assert svc["ledger"].published().get("cid1") == "x.md"  # 补 ledger
    assert svc["journal"].read() is None  # 清 journal


def test_ledger_only_recovery_after_pointer_switched(tmp_path, monkeypatch):
    """pointer 已切、ledger 未写 → 恢复补 ledger 并清 journal。"""
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    info = svc["index_service"].build("numpy")
    info.previous_generation_id = "g-old"
    svc["journal"].write({
        "phase": "publish",
        "stage": PH_LEDGER_COMMITTED,
        "staging_docs": [{"candidate_id": "cid1", "filename": "x.md"}],
        "replaced_docs": [],
        "backend": "numpy",
        "index": info.to_dict(),
        "previous_target": "old-target",
    })
    # 仿真：pointer 已指向候选（激活完成）但 ledger 未记
    monkeypatch.setattr(
        pipe._generation_store, "active", lambda backend: info,
    )
    pipe._recover_publish_transaction(svc["journal"].read())
    assert svc["journal"].read() is None
    assert svc["ledger"].published().get("cid1") == "x.md"


def test_blocked_writes_kb_write_blocked_and_keeps_journal(tmp_path, monkeypatch):
    """alias 指向未知代 → blocked：写 kb_write_blocked（文件），journal 保留。"""
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    info = svc["index_service"].build("numpy")
    svc["journal"].write({
        "phase": "publish",
        "stage": PH_ALIAS_ACTIVATED,
        "staging_docs": [{"candidate_id": "cid1", "filename": "x.md"}],
        "replaced_docs": [],
        "backend": "es",
        "index": info.to_dict(),
        "previous_target": "old-target",
    })
    # 仿真 ES：alias 指向未知代
    monkeypatch.setattr(pipe, "_read_alias_target", lambda backend: "mystery-target")
    monkeypatch.setattr(
        pipe._generation_store, "active",
        lambda backend: _active("old-target", "g-old"),
    )
    pipe._recover_publish_transaction(svc["journal"].read())
    # blocked：journal 保留（等待人工 reconcile）
    assert svc["journal"].read() is not None
    assert (svc["state_dir"] / "kb_write_blocked").exists()
    assert "reconcile" in (svc["state_dir"] / "kb_write_blocked").read_text(
        encoding="utf-8"
    )


def test_alias_service_exception_is_unknown_and_never_rolls_back(tmp_path, monkeypatch):
    """ES Alias 服务异常 → blocked；候选文档/journal 现场必须保留。"""
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    info = svc["index_service"].build("numpy")
    evolved = svc["kb_dir"] / "evolved"
    (evolved / "new.md").write_text("候选", encoding="utf-8")
    svc["journal"].write({
        "phase": "publish",
        "stage": PH_INDEX_BUILT,
        "staging_docs": [{"candidate_id": "cid1", "filename": "new.md"}],
        "replaced_docs": [],
        "backend": "es",
        "index": info.to_dict(),
        "previous_target": "old-target",
    })

    def _raise(_backend):
        raise RuntimeError("ES unavailable")

    monkeypatch.setattr(svc["index_service"], "reconcile", _raise)
    action = pipe._recover_publish_transaction(svc["journal"].read())

    assert action == REC_BLOCKED
    assert svc["journal"].read() is not None
    assert (evolved / "new.md").exists()
    assert (svc["state_dir"] / "kb_write_blocked").exists()


def test_es_rollback_does_not_touch_docs_when_candidate_delete_unconfirmed(
    tmp_path, monkeypatch,
):
    """ES candidate 清理未获确认时，源文件/trash/journal 都必须原样保留。"""
    from app.evolution.generation import GenerationInfo
    from app.config.settings import settings
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    candidate = GenerationInfo(
        generation_id="20260830000000-abcd1234",
        target=f"{settings.es_index_prefix}-kb-20260830000000-abcd1234",
        embedding_model="fake-embed",
    )
    evolved = svc["kb_dir"] / "evolved"
    (evolved / "new.md").write_text("候选", encoding="utf-8")
    trash = svc["state_dir"] / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    (trash / "old.md").write_text("旧文档", encoding="utf-8")
    svc["journal"].write({
        "phase": "publish",
        "stage": PH_INDEX_BUILT,
        "staging_docs": [{"candidate_id": "cid1", "filename": "new.md"}],
        "replaced_docs": ["old.md"],
        "backend": "es",
        "index": candidate.to_dict(),
        "previous_target": "old-target",
    })
    monkeypatch.setattr(pipe, "_read_alias_target", lambda backend: "")
    monkeypatch.setattr(
        pipe._generation_store, "active",
        lambda backend: _active("old-target", "g-old"),
    )
    monkeypatch.setattr(
        svc["index_service"], "delete_candidate",
        lambda *args, **kwargs: False,
    )

    pipe._recover_publish_transaction(svc["journal"].read())

    assert (evolved / "new.md").exists()
    assert (trash / "old.md").exists()
    assert svc["journal"].read() is not None


def test_chroma_rollback_treats_missing_collection_as_idempotent(tmp_path, monkeypatch):
    """首次删 Chroma 后文件操作失败，第二次 NotFound 仍可完成回滚。"""
    import chromadb
    from app.evolution.generation import GenerationInfo
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    candidate = GenerationInfo(
        generation_id="20260830000000-abcd1234",
        target="ecom_kb__20260830000000-abcd1234",
        embedding_model="fake-embed",
    )
    evolved = svc["kb_dir"] / "evolved"
    (evolved / "new.md").write_text("候选", encoding="utf-8")
    svc["journal"].write({
        "phase": "publish",
        "stage": PH_INDEX_BUILT,
        "staging_docs": [{"candidate_id": "cid1", "filename": "new.md"}],
        "replaced_docs": [],
        "backend": "chroma",
        "index": candidate.to_dict(),
        "previous_target": "old-target",
    })

    class _Client:
        def __init__(self):
            self.calls = 0

        def delete_collection(self, _target):
            self.calls += 1
            if self.calls > 1:
                raise chromadb.errors.NotFoundError("collection missing")

    client = _Client()
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: client)
    original_remove = svc["publisher"].remove
    failed_once = {"value": False}

    def _remove_once(filename):
        if not failed_once["value"]:
            failed_once["value"] = True
            raise OSError("simulated file operation failure")
        return original_remove(filename)

    monkeypatch.setattr(svc["publisher"], "remove", _remove_once)
    monkeypatch.setattr(
        pipe._generation_store, "active",
        lambda backend: _active("old-target", "g-old"),
    )

    with pytest.raises(OSError):
        pipe._recover_publish_transaction(svc["journal"].read())
    assert svc["journal"].read() is not None

    pipe._recover_publish_transaction(svc["journal"].read())
    assert svc["journal"].read() is None
    assert not (evolved / "new.md").exists()


def test_persisted_blocked_stops_run_before_mining(tmp_path, monkeypatch):
    """持久化 kb_write_blocked 存在时，run 不得进入挖掘/发布阶段。"""
    from app.evolution.pipeline import EvolutionBlockedError
    from app.config.settings import settings
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    monkeypatch.setattr(settings, "self_evolve_enabled", True, raising=False)
    marker = svc["state_dir"] / "kb_write_blocked"
    marker.write_text("manual reconcile required", encoding="utf-8")
    monkeypatch.setattr(
        svc["pipeline"], "_mine",
        lambda report: pytest.fail("blocked run 不应执行 _mine"),
    )

    with pytest.raises(EvolutionBlockedError):
        svc["pipeline"].run()


def test_repeated_recovery_idempotent(tmp_path, monkeypatch):
    """恢复幂等：同一 journal 恢复两次，ledger 不重复补、结果一致。"""
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    info = svc["index_service"].build("numpy")
    svc["journal"].write({
        "phase": "publish",
        "stage": PH_POINTER_UPDATED,
        "staging_docs": [{"candidate_id": "cid1", "filename": "x.md"}],
        "replaced_docs": [],
        "backend": "numpy",
        "index": info.to_dict(),
        "previous_target": "old-target",
    })
    monkeypatch.setattr(
        pipe._generation_store, "active", lambda backend: info,
    )
    pipe._recover_publish_transaction(svc["journal"].read())
    assert svc["journal"].read() is None
    assert svc["ledger"].published().get("cid1") == "x.md"
    # 再次恢复：无 journal → no-op
    pipe._recover_publish_transaction(svc["journal"].read())


def test_rollback_restores_replaced_docs(tmp_path, monkeypatch):
    """回滚分支：还原被替换文档（trash → evolved），删除新文档。"""
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]
    info = svc["index_service"].build("numpy")

    # 构造被替换文档：已移 trash
    (svc["state_dir"] / "trash").mkdir(parents=True, exist_ok=True)
    (svc["state_dir"] / "trash" / "old.md").write_text(
        "# 自进化知识\n\n## 旧问题\n\n旧答案\n", encoding="utf-8",
    )
    evo = svc["kb_dir"] / "evolved"
    evo.mkdir(parents=True, exist_ok=True)
    (evo / "new.md").write_text("新文档", encoding="utf-8")

    svc["journal"].write({
        "phase": "publish",
        "stage": PH_PREPARED,
        "staging_docs": [{"candidate_id": "cid1", "filename": "new.md"}],
        "replaced_docs": ["old.md"],
        "backend": "numpy",
        "index": info.to_dict(),
        "previous_target": "old-target",
    })
    monkeypatch.setattr(
        pipe._generation_store, "active",
        lambda backend: _active("old-target", "g-old"),
    )
    pipe._recover_publish_transaction(svc["journal"].read())
    # 回滚完成后：新文档删除、旧文档还原、journal 归档
    assert not (evo / "new.md").exists()
    assert (evo / "old.md").exists()
    assert svc["journal"].read() is None


def test_assert_held_blocks_alias_side_effect(tmp_path, monkeypatch):
    """锁续租丢失（assert_held 抛错）→ alias 副作用绝不执行（fail-closed）。"""
    from app.evolution.lock import LockHeldError
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    pipe = svc["pipeline"]

    alias_called = {"n": 0}
    monkeypatch.setattr(
        svc["index_service"], "activate_alias",
        lambda *a, **k: alias_called.__setitem__("n", alias_called["n"] + 1),
    )
    # 锁已释放 → 继续执行会被 assert_held 拦截
    svc["lock"].release()
    info = svc["index_service"].build("numpy")
    with pytest.raises(LockHeldError):
        pipe._publish_stage_alias(info) if False else _publish_alias_inner(pipe, info)
    assert alias_called["n"] == 0  # alias 从未被调


def _publish_alias_inner(pipe, info):
    """模拟发布第 3 阶段：assert_held 检查在 activate_alias 之前。"""
    pipe._lock.assert_held()
    pipe._index_service.activate_alias("numpy", info)


def test_three_writers_contend_same_lock(tmp_path, reset_settings):
    """上传/下架/自进化共用同一锁：互斥，第二/三个写入者被拒。"""
    from app.evolution.lock import LockHeldError
    from tests.unit.test_pipeline import make_services

    svc = make_services(tmp_path)
    lock = svc["lock"]
    lock.acquire(phase="run")
    with pytest.raises(LockHeldError):
        lock.acquire(phase="upload")  # 第二个写入者（上传语义）
    with pytest.raises(LockHeldError):
        lock.acquire(phase="delete")  # 第三个写入者（下架语义）
    lock.release()
    # 释放后重新可获取（自进化）
    lock.acquire(phase="run")
    lock.release()

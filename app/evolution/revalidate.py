"""revalidate.py：人工知识变更 → 存量自进化文档重接地（淘汰半边，P1-2）。

背景：沉淀时的接地 Judge 只证明「当时」断言有证据；人工后来上传/下架/修订了
知识库（generation 变化），旧沉淀可能已失去依据——「淘汰」半边无人核对。

- parse_evolved_doc：按 publisher 固定模板（frontmatter + # 自进化知识 /
  ## <问题> / 正文）解析；frontmatter 复用 loader 解析。
- revalidate：缺 effective_date 的存量文档视为最旧，最旧优先取
  ≤ EVOLVE_MAX_REGROUND_PER_RUN；活动检索器检索（top_k=8）→ hit.chunk 映射
  SourceRef（同 recorder）→ GroundingJudge（1 次 LLM 调用/条）。
  通过 → 原地刷新 last_validated（frontmatter 不进索引，无需重建）；
  不通过 → 隔离到 pending：**两遍式**——先全部判定，再以 journal 事务条目
  （phase=revalidate + trashed_docs + 目标索引）开启隔离（移 trash + 清账 +
  pending），一次 build+activate 后清除 journal。
- recover_revalidate：隔离事务的**前进式**恢复——隔离判定已做出，恢复只前进
  不回退：清单内仍在 evolved/ 的完成隔离，已在 trash 的补账，然后一次重建；
  检测到目标代已激活（崩溃在 activate 与 journal.clear 之间）则无事可做。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from app.agent.rag.loader import parse_frontmatter as parse_doc_frontmatter
from app.config.settings import settings
from app.evolution.generation import GenerationInfo, new_generation_id
from app.evolution.models import CandidateQA, SourceRef
from app.evolution.publish_state import (
    PH_INDEX_BUILT,
    PH_PREPARED,
    REC_BLOCKED,
    REC_FORWARD,
    REC_LEDGER_ONLY,
    recover_decide,
)

EFFECTIVE_DATE_MISSING = ""  # 缺 effective_date 视为最旧（排序靠前）


def _today(clock) -> date:
    return (clock.now() if clock else datetime.now()).date()


def parse_evolved_doc(path) -> Optional[dict]:
    """按 publisher 模板解析 evolved 文档；不可解析返回 None。

    返回 {"filename", "question", "answer", "meta"}：
    meta 为 frontmatter dict（loader 解析，key 小写）。
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = parse_doc_frontmatter(text)
    lines = [ln.rstrip() for ln in body.strip().splitlines()]
    q_idx = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), None)
    if q_idx is None:
        return None
    question = lines[q_idx][3:].strip()
    answer = "\n".join(lines[q_idx + 1:]).strip()
    if not question or not answer:
        return None
    return {
        "filename": Path(path).name,
        "question": question,
        "answer": answer,
        "meta": meta,
    }


def refresh_last_validated(path, today: Optional[date] = None) -> None:
    """原地刷新 frontmatter 的 last_validated（frontmatter 不进索引，无需重建）。"""
    text = Path(path).read_text(encoding="utf-8")
    meta, body = parse_doc_frontmatter(text)
    if not meta:
        return
    meta["last_validated"] = (today or date.today()).isoformat()
    block = "---\n" + "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n---\n"
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text(block + body, encoding="utf-8")
    os.replace(tmp, path)


def _list_evolved_docs(kb_dir: Path) -> list[tuple[Path, str]]:
    """列出 evolved/*.md 及其 effective_date（缺省的为空串，视为最旧）。"""
    evolved = Path(kb_dir) / "evolved"
    if not evolved.is_dir():
        return []
    out: list[tuple[Path, str]] = []
    for p in evolved.glob("*.md"):
        try:
            meta, _ = parse_doc_frontmatter(p.read_text(encoding="utf-8"))
        except OSError:
            continue
        out.append((p, meta.get("effective_date", EFFECTIVE_DATE_MISSING)))
    return out


def _to_sourceref(hit) -> SourceRef:
    """检索命中 → SourceRef（映射同 recorder.parse_turn_slice）。"""
    chunk = hit.chunk
    return SourceRef(
        source_path=getattr(chunk, "source_path", "") or "",
        doc=getattr(chunk, "doc", "") or "",
        section=getattr(chunk, "section", "") or "",
        score=float(getattr(hit, "score", 0.0) or 0.0),
        text=getattr(chunk, "text", "") or "",
    )


def _resolve_candidate_id(ledger, filename: str) -> str:
    """从 published/trash 反查 candidate_id；缺失（历史损坏）再派生稳定 id。

    隔离事务在 ``batch_cleanup_published`` 后、索引激活前崩溃时，published 已清，
    但 trash 账本已持久化原 candidate_id；恢复必须复用它，不能生成第二个 pending。
    """
    for cid, fname in ledger.published().items():
        if fname == filename:
            return cid
    trash_entry = ledger.trash_entry(filename)
    if trash_entry and trash_entry.get("candidate_id"):
        return str(trash_entry["candidate_id"])
    return hashlib.sha256(f"revalidate:{filename}".encode("utf-8")).hexdigest()


def _pending_candidate(cid: str, parsed: dict) -> CandidateQA:
    """隔离到 pending 的候选（ledger 反查不到 → 文件名派生稳定 id 也走人工审核通道）。"""
    return CandidateQA(
        candidate_id=cid,
        turn_id=parsed.get("meta", {}).get("provenance", ""),
        question=parsed["question"],
        answer=parsed["answer"],
        confidence=1.0,
        filter_state="pending",
    )


def _write_kb_write_blocked(reason: str, trash_dir) -> None:
    """revalidate 恢复写入全局阻塞标记（SQL 正本，文件降级）。"""
    from app.stores.sql.document_store import KbControlStore
    from app.stores.sql.engine import get_engine

    written = False
    try:
        engine = get_engine()
    except Exception:  # noqa: BLE001
        engine = None
    if engine is not None:
        try:
            KbControlStore(engine).set("kb_write_blocked", reason)
            written = True
        except Exception:  # noqa: BLE001 —— 降级文件标记
            pass
    if not written:
        marker = Path(trash_dir).parent / "kb_write_blocked"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(reason, encoding="utf-8")
    try:
        from app.observability.metrics import set_kb_write_blocked

        set_kb_write_blocked(True)
    except Exception:  # noqa: BLE001 —— 指标不可用不影响阻塞落盘
        pass


def _confirm_revalidate_activation(index_service, backend: str, target: str) -> tuple[bool, str]:
    """激活后再次确认提交点；ES alias 读取失败也视为未确认。"""
    if backend != "es":
        return True, ""
    try:
        state = index_service.reconcile(backend)
    except Exception as e:  # noqa: BLE001
        return False, f"revalidate 激活后 alias 读取失败: {e}"
    actual = state.get("alias_target")
    if actual != target:
        if actual is None:
            return False, "revalidate 激活后 alias 状态未知"
        return False, f"revalidate 激活后 alias 未指向候选（实际 {actual!r}）"
    return True, ""


def revalidate(
    *,
    kb_dir,
    trash_dir,
    ledger,
    publisher,
    index_service,
    retriever,
    grounding_judge,
    journal=None,
    max_docs: Optional[int] = None,
    exclude_docs: Optional[set[str]] = None,
    clock=None,
    backend: str = "",
) -> dict:
    """存量自进化文档重接地；返回本批结果、已访问文件和剩余数量。

    - 顺序：缺 effective_date 的存量文档视为最旧，最旧优先取 ≤ max_docs。
      ``exclude_docs`` 用于同一 generation 的后续批次跳过已访问文件。
    - 通过 → 只刷 frontmatter（不重建索引）。
    - 不通过 → 两遍式隔离：先全部判定（纯读 + 通过者刷 frontmatter），
      再对失败清单以 journal 事务条目开启隔离（移 trash + pending + 清账），
      一次 build+activate 后清除 journal；崩溃由 recover_revalidate 前进式补齐。
    - failed/failed_docs 只统计实际移入 trash 的文档（文件已不在 → 整条跳过，
      也不重建索引）。
    """
    kb_dir = Path(kb_dir)
    limit = max_docs or settings.evolve_max_reground_per_run
    excluded = set(exclude_docs or ())
    eligible = [
        item for item in _list_evolved_docs(kb_dir)
        if item[0].name not in excluded
    ]
    docs = sorted(
        eligible,
        key=lambda t: (t[1] != EFFECTIVE_DATE_MISSING, t[1]),
    )[:limit]
    processed_docs = [path.name for path, _ in docs]

    # 第一遍：判定（通过者原地刷新 last_validated；失败者只收集不动文件）
    checked = passed = 0
    to_retire: list[dict] = []
    for path, _ in docs:
        parsed = parse_evolved_doc(path)
        if parsed is None:
            continue
        checked += 1
        hits = retriever.search(parsed["question"], top_k=8)
        sources = [_to_sourceref(h) for h in hits]
        verdict = grounding_judge.judge(parsed["answer"], sources)
        if verdict.get("grounded"):
            refresh_last_validated(path, _today(clock))
            passed += 1
        else:
            to_retire.append(parsed)

    # 第二遍：隔离事务（有失败才开启）
    retired: list[tuple[str, str]] = []  # (filename, candidate_id)
    rebuilt = False
    if to_retire:
        backend = backend or settings.rag_backend.lower()
        generation_id = new_generation_id(clock)
        previous = None
        if index_service is not None:
            from app.evolution.generation import GenerationStore

            # 从 index_service 拿不到 store 时 previous_target 留空（旧代未知）
            try:
                prev = getattr(index_service, "_store", None)
                if prev is not None:
                    previous = prev.active(backend)
            except Exception:  # noqa: BLE001 —— 旧代读取失败不阻断隔离
                previous = None
        previous_target = previous.target if previous is not None else ""
        if journal is not None:
            # 前进式恢复凭据：隔离清单 + 目标索引（ES 清理保护同 publish journal）
            journal.write({
                "phase": "revalidate",
                "stage": PH_PREPARED,
                "trashed_docs": [p["filename"] for p in to_retire],
                "backend": backend,
                "index": {
                    "generation_id": generation_id,
                    "target": index_service.target_for(backend, generation_id),
                },
                "previous_target": previous_target,
            })
        for parsed in to_retire:
            # 先移文件：文件已不在 evolved/（可能已被人工处理）→ 整条跳过
            if publisher.unpublish(parsed["filename"], trash_dir) is None:
                continue
            cid = _resolve_candidate_id(ledger, parsed["filename"])
            ledger.add_pending(_pending_candidate(cid, parsed), reason="revalidation_failed")
            retired.append((parsed["filename"], cid))
        if retired:
            ledger.batch_cleanup_published(retired)
            info = index_service.build(backend, generation_id=generation_id)
            if journal is not None:
                journal.write({
                    "phase": "revalidate",
                    "stage": PH_INDEX_BUILT,
                    "trashed_docs": [p["filename"] for p in to_retire],
                    "backend": backend,
                    "index": {
                        "generation_id": generation_id,
                        "target": info.target,
                    },
                    "previous_target": previous_target,
                })
            index_service.activate(backend, info)
            confirmed, reason = _confirm_revalidate_activation(
                index_service, backend, info.target,
            )
            if not confirmed:
                # activate 可能已经切了 alias/pointer，但无法确认就不能清 journal；
                # 让下次恢复按三源状态继续前进或阻断。
                raise RuntimeError(reason)
            rebuilt = True
        if journal is not None:
            journal.clear()

    return {
        "scanned": len(docs),
        "checked": checked,
        "passed": passed,
        "failed": len(retired),
        "failed_docs": [f for f, _ in retired],
        "rebuilt": rebuilt,
        "processed_docs": processed_docs,
        "remaining": max(0, len(eligible) - len(docs)),
        "has_more": len(eligible) > len(docs),
    }


def recover_revalidate(
    entry: dict,
    *,
    kb_dir,
    trash_dir,
    ledger,
    publisher,
    index_service,
    generation_store=None,
    backend: str = "",
) -> dict:
    """revalidate 隔离事务的恢复：与 publish 同一张恢复表（2.7）。

    崩溃点与动作（recover_decide 三源判定：真实 alias + pointer + journal）：
    - pointer 已指向候选（激活已完成）→ already_done（ledger 在 activate 前已写）；
    - ES alias 已切、pointer 未切 → forward：只补 pointer，ledger 无需补；
    - alias 指向非旧代、非候选代 → blocked：写 kb_write_blocked，保留现场；
    - 其余（未激活）→ 前进式完成隔离：仍在 evolved/ 的完成隔离（移 trash +
      pending + 清账），已在 trash 的只补账；随后一次 build+activate。
    返回值带 ``success``；只有明确成功时 pipeline 才能归档 journal。
    """
    kb_dir = Path(kb_dir)
    backend = backend or entry.get("backend") or settings.rag_backend.lower()
    index_info = entry.get("index") or {}
    candidate_target = str(index_info.get("target", ""))
    previous_target = str(entry.get("previous_target", ""))
    stage = str(entry.get("stage", ""))

    pointer_target = ""
    if generation_store is not None and index_info.get("generation_id"):
        active = generation_store.active(backend)
        pointer_target = active.target if active is not None else ""

    # ``""`` = 明确 alias 不存在；``None`` = 读取失败/状态未知。
    alias_target = ""
    if backend == "es":
        try:
            st = index_service.reconcile(backend)
            raw_alias = st.get("alias_target")
            alias_target = None if raw_alias is None else str(raw_alias or "")
        except Exception:  # noqa: BLE001 —— alias 不可读必须 fail-closed
            alias_target = None

    action = recover_decide(
        backend, stage, pointer_target, alias_target,
        candidate_target, previous_target,
    )
    if action == REC_LEDGER_ONLY:
        return {
            "already_done": True, "retired": [], "rebuilt": False,
            "success": True,
        }
    if action == REC_FORWARD:
        # alias 已生效：只补 pointer（幂等），ledger 在 activate 前已写完
        info = GenerationInfo.from_dict(index_info)
        try:
            index_service.activate_pointer(backend, info)
        except Exception as e:  # noqa: BLE001 —— pointer 补写失败保留 journal 待下次
            return {
                "already_done": False, "retired": [], "rebuilt": False,
                "success": False, "pending": True,
                "reason": f"revalidate pointer 补写失败，journal 保留: {e}",
            }
        return {
            "already_done": True, "retired": [], "rebuilt": False,
            "success": True,
        }
    if action == REC_BLOCKED:
        reason = (
            f"revalidate 恢复：alias 状态不可确认/指向非旧代非候选代 "
            f"（alias={alias_target!r}, stage={stage}），需人工 reconcile"
        )
        _write_kb_write_blocked(reason, trash_dir)
        return {"already_done": False, "retired": [], "rebuilt": False,
                "blocked": True, "success": False, "reason": reason}

    # rollback 分支在 revalidate 语义下 = 前进式完成隔离（判定已做出不可回退）
    retired: list[tuple[str, str]] = []
    found_any = False
    for name in entry.get("trashed_docs", []):
        evolved_path = kb_dir / "evolved" / name
        trash_path = Path(trash_dir) / name
        in_evolved, in_trash = evolved_path.exists(), trash_path.exists()
        if not (in_evolved or in_trash):
            continue  # 两处皆无：从未隔离或已被人工清理
        found_any = True
        parsed = parse_evolved_doc(evolved_path if in_evolved else trash_path)
        publisher.unpublish(name, trash_dir)  # 已在 trash → 返回 None，幂等
        cid = _resolve_candidate_id(ledger, name)
        if parsed is not None:
            ledger.add_pending(_pending_candidate(cid, parsed), reason="revalidation_failed")
        retired.append((name, cid))
    if retired:
        ledger.batch_cleanup_published(retired)
    if found_any or retired:
        info = index_service.build(backend)
        index_service.activate(backend, info)
        confirmed, reason = _confirm_revalidate_activation(
            index_service, backend, info.target,
        )
        if not confirmed:
            # 现场已部分前进但提交点不可确认：保留 journal，写阻塞标记，
            # 下次只能由恢复机继续/人工 reconcile，不能归档当前证据。
            _write_kb_write_blocked(reason, trash_dir)
            return {
                "already_done": False,
                "retired": retired,
                "rebuilt": False,
                "blocked": True,
                "success": False,
                "reason": reason,
            }
    return {
        "already_done": False,
        "retired": retired,
        "rebuilt": bool(found_any or retired),
        "success": True,
    }

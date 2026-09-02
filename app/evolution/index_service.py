"""index_service.py：IndexBuildService —— 唯一索引写入口（第10期）。

两阶段激活（v7 冻结，提交点语义）：
  build(backend)：chunk → encode → 写版本化目标 → 验证 → 返回 GenerationInfo。
     不切换任何东西；验证失败抛异常，当前 generation 不受影响，旧索引始终可用。
  activate_alias(backend, info)：提交点 1 —— 切后端真身（ES alias 原子切换，幂等）。
  activate_pointer(backend, info)：提交点 2 —— 切 generation 指针 + 清理旧代。
     指针写失败（strict_shared）→ 抛错，调用方保持 ALIAS_ACTIVATED journal
     等待恢复补齐——**alias 已生效，绝不因此回滚文件**。
  activate(backend, info)：合体版（alias → 指针），build_kb_index / run_evolution 用。

路径统一用 ROOT 拼接（与 build_kb_index.py 一致，避免 cwd 依赖）；
测试可注入 backend_settings 覆盖 kb_index_path / chroma_persist_dir / chroma_collection。
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from pathlib import Path

from app.agent.rag.backends import create_backend
from app.config.settings import settings

from app.evolution.generation import GenerationInfo, GenerationStore, new_generation_id

ROOT = Path(__file__).resolve().parent.parent.parent


def _alias_not_found(exc: BaseException) -> bool:
    """判断 ES 异常是否明确表示 alias 不存在。

    只有明确的 404/NotFound 才能映射成「尚未切换」；连接、权限、超时、
    响应解析等其它异常必须保留为未知状态，供恢复机 fail-closed 阻断。
    测试替身常用 ``Exception('alias not found')``，也保留这一窄消息匹配。
    """
    name = type(exc).__name__.lower().replace("_", "")
    if "notfound" in name:
        return True
    for attr in ("status_code", "status"):
        try:
            if int(getattr(exc, attr)) == 404:
                return True
        except (AttributeError, TypeError, ValueError):
            pass
    message = str(exc).lower()
    return "alias" in message and "not found" in message


def _journal_es_targets() -> set[str]:
    """未完成 journal 中的索引 target（ES 清理保护 + reconcile 判定用）。

    覆盖两类：KB 上传/下架的 .staging/journal/*.json（每 op 一个）与
    自进化 pipeline 的单 journal.json（{phase: publish, index: {...}}）。
    """
    out: set[str] = set()
    jdir = ROOT / settings.kb_upload_staging_dir / "journal"
    if jdir.is_dir():
        for p in jdir.glob("*.json"):
            data = _read_journal(p)
            out.add(str(data.get("index", {}).get("target") or data.get("target") or ""))
    evo = ROOT / settings.evolve_state_dir / "journal.json"
    data = _read_journal(evo)
    out.add(str(data.get("index", {}).get("target") or ""))
    out.discard("")
    return out


def _read_journal(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _chunk_strict(chunk_fn, kb_dir: Path) -> list:
    """strict 构建调度：向 chunker 传 strict=True（如 chunk_kb_dir）。

    chunker 不支持 strict 参数 → 显式报错，绝不静默降级为「坏文件跳过」
    （strict 是所有会切 alias 的构建的硬约束，见 v7 冻结）。
    """
    if "strict" not in inspect.signature(chunk_fn).parameters:
        raise ValueError(
            "strict_build=True 要求 chunker 支持 strict 参数（如 chunk_kb_dir），"
            f"当前: {getattr(chunk_fn, '__qualname__', chunk_fn)!r}"
        )
    return chunk_fn(kb_dir, strict=True)


class IndexBuildService:
    """唯一索引写入口：构建版本化索引 + 激活 + 清理旧代。"""

    def __init__(
        self,
        embedder,
        kb_dir,
        generation_store: GenerationStore,
        backend_settings: dict | None = None,
        clock=None,
        chunker=None,
        strict_build: bool = False,
    ):
        self._embedder = embedder
        self._kb_dir = Path(kb_dir)
        self._store = generation_store
        # 可覆盖的后端路径配置（测试注入 tmp 目录）；缺省取 settings
        self._b = dict(backend_settings or {})
        self._clock = clock
        # chunker 可注入（7.1：默认 md/txt；传 parsers.chunk_kb_dir 支持 pdf/docx/html）
        self._chunker = chunker
        # v7 冻结：会切 alias 的构建（上传/下架/自进化/手工 CLI）一律 strict——
        # 任一源文件解析失败即中止，杜绝「一次重建静默丢失已有知识」
        self._strict_build = strict_build
        self.last_built_size = 0

    # ---------- 路径 ----------
    def _numpy_index_dir(self) -> Path:
        rel = self._b.get("kb_index_path", settings.kb_index_path)
        return ROOT / Path(rel).parent

    def _chroma_persist_dir(self) -> Path:
        rel = self._b.get("chroma_persist_dir", settings.chroma_persist_dir)
        return ROOT / rel

    def _chroma_base_collection(self) -> str:
        return self._b.get("chroma_collection", settings.chroma_collection)

    def target_for(self, backend: str, generation_id: str) -> str:
        """版本化索引目标：numpy 为索引文件路径，chroma 为 collection 名。"""
        backend = backend.lower()
        if backend == "numpy":
            return str(self._numpy_index_dir() / f"kb_index.{generation_id}.json")
        if backend == "chroma":
            return f"{self._chroma_base_collection()}__{generation_id}"
        if backend == "es":
            return f"{settings.es_index_prefix}-kb-{generation_id}"
        raise ValueError(f"未知的 RAG 后端: {backend}（可选: numpy / chroma / es）")

    def _backend_impl(self, backend: str, target: str):
        backend = backend.lower()
        if backend == "numpy":
            return create_backend("numpy", index_path=Path(target))
        if backend == "chroma":
            return create_backend(
                "chroma",
                persist_dir=self._chroma_persist_dir(),
                collection_name=target,
            )
        if backend == "es":
            return create_backend("es", index_name=target)
        raise ValueError(f"未知的 RAG 后端: {backend}（可选: numpy / chroma / es）")

    # ---------- 构建 ----------
    def build(self, backend: str, generation_id: str | None = None) -> GenerationInfo:
        """构建版本化索引并验证；不切换指针。验证失败抛异常。

        generation_id 可显式指定（pipeline 需要先写 journal 时），缺省生成。
        """
        backend = backend.lower()
        chunk_fn = self._chunker
        if chunk_fn is None:
            from app.agent.rag.chunker import chunk_markdown_dir

            chunk_fn = chunk_markdown_dir
        if self._strict_build:
            chunks = _chunk_strict(chunk_fn, self._kb_dir)
        else:
            chunks = chunk_fn(self._kb_dir)
        if not chunks:
            raise ValueError(f"知识库目录未发现任何文档: {self._kb_dir}")

        vectors = self._embedder.encode([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(f"向量数({len(vectors)}) 与 chunk 数({len(chunks)})不一致")

        generation_id = generation_id or new_generation_id(self._clock)
        info = GenerationInfo(
            generation_id=generation_id,
            target=self.target_for(backend, generation_id),
            embedding_model=self._embedder.model,
        )

        impl = self._backend_impl(backend, info.target)
        impl.upsert(chunks=chunks, vectors=vectors, embedding_model=info.embedding_model)
        self._verify(backend, info, chunks)
        # 两阶段激活（评审 R3 / v7 冻结）：build 只创建+验证目标（ES 验证按
        # index_name 直查、不依赖 alias），alias 切换统一在 activate_alias()——
        # 两步之间旧索引始终可用，alias 是「提交点」。
        self.last_built_size = len(chunks)
        return info

    def _verify(self, backend: str, info: GenerationInfo, chunks) -> None:
        """全新 backend 实例 load + 模型一致 + chunk 数一致 + chunk_id 唯一 + 探针 top1。"""
        impl = self._backend_impl(backend, info.target)
        impl.load()

        if impl.expected_embedding_model() != info.embedding_model:
            raise RuntimeError(
                f"验证失败：索引 embedding 模型 {impl.expected_embedding_model()} 与 "
                f"{info.embedding_model} 不一致"
            )
        if impl.size() != len(chunks):
            raise RuntimeError(
                f"验证失败：索引 chunk 数 {impl.size()} 与源 {len(chunks)} 不一致"
            )

        if backend == "numpy":
            data = json.loads(Path(info.target).read_text(encoding="utf-8"))
            ids = [c["chunk_id"] for c in data["chunks"]]
            if len(set(ids)) != len(ids):
                raise RuntimeError("验证失败：chunk_id 存在重复")

        probe = self._embedder.encode_one(chunks[0].text)
        hits = impl.search(probe, top_k=1)
        if not hits or hits[0].chunk.chunk_id != chunks[0].chunk_id:
            raise RuntimeError("验证失败：向量探针 top1 未命中首个 chunk")

    # ---------- 候选检索器（2.6：评测探针与线上共用装配路径）----------
    def open_retriever(self, info: GenerationInfo, retrieval_config=None):
        """按候选索引打开检索器：ES 直查 index_name=info.target（不经过活动 alias）。

        retrieval_config：app.agent.rag.retriever_factory.RetrievalConfig；
        缺省跟随 settings 冻结（与线上 knowledge 单例同一条装配路径——
        hybrid/reranker/recall_k/相关性门控一致，保证评的就是将来要切的配置）。
        """
        from app.agent.rag.retriever_factory import (
            open_retriever,
            retrieval_config_from_settings,
        )

        config = retrieval_config or retrieval_config_from_settings()
        if not config.backend_settings:
            config.backend_settings = dict(self._b)
        return open_retriever(
            config,
            embedder=self._embedder,
            generation_target=info.target,
        )

    # ---------- 激活与清理 ----------
    def activate_alias(self, backend: str, info: GenerationInfo) -> None:
        """提交点 1：切换后端真身（ES alias 原子切换，幂等可安全重跑）。

        只做 alias；调用方（上传/下架）随后 assert_held 再调 activate_pointer
        （v7 约束 2 时序）。numpy/chroma 无 alias 概念：no-op。
        """
        backend = backend.lower()
        impl = self._backend_impl(backend, info.target)
        activate = getattr(impl, "activate", None)
        if callable(activate):
            activate()  # ES：update_aliases 原子切换（目标已是 alias 指向时 no-op）

    def activate_pointer(self, backend: str, info: GenerationInfo) -> None:
        """提交点 2：切换 generation 指针 + 清理旧代（每 backend 只留当前和上一代）。

        指针写失败（strict_shared）→ 抛 StorageUnavailableError：调用方保持
        ALIAS_ACTIVATED journal 等待恢复补齐（alias 已生效，绝不回滚文件）。
        """
        backend = backend.lower()
        current = self._store.active(backend)
        if current and current.generation_id != info.generation_id:
            info.previous_generation_id = current.generation_id
        self._store.activate(backend, info)
        self._cleanup_old(backend)

    def activate(self, backend: str, info: GenerationInfo) -> None:
        """合体版（build_kb_index / run_evolution / 简单调用方）：alias → 指针。"""
        self.activate_alias(backend, info)
        self.activate_pointer(backend, info)

    def _cleanup_old(self, backend: str) -> None:
        """按 generation_id 排序删除超出保留量（当前+上一代）的旧目标。"""
        info = self._store.active(backend)
        if info is None:
            return
        keep_ids = {info.generation_id}
        if info.previous_generation_id:
            keep_ids.add(info.previous_generation_id)

        if backend == "numpy":
            for f in self._numpy_index_dir().glob("kb_index.*.json"):
                gen = f.name[len("kb_index."):-len(".json")]
                if gen not in keep_ids:
                    f.unlink(missing_ok=True)
            return

        if backend == "chroma":
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(self._chroma_persist_dir()))
                base = self._chroma_base_collection()
                prefix = f"{base}__"
                keep = {f"{base}__{g}" for g in keep_ids}
                for c in client.list_collections():
                    name = getattr(c, "name", c)
                    if name.startswith(prefix) and name not in keep:
                        client.delete_collection(name)
            except Exception:  # noqa: BLE001 —— chroma 不可用时清理失败不影响主流程
                pass
            return

        if backend == "es":
            self._cleanup_old_es(info, keep_ids)
            return

    def _cleanup_old_es(self, info: GenerationInfo, keep_ids: set[str]) -> None:
        """ES 旧代清理（评审）：只删严格匹配 {prefix}-kb-<generation> 的索引。

        无条件保护四类：alias 当前实际指向、当前/上一代 target、
        **未完成 journal 中的 target**（上传/下架事务的 pending 代）；
        清理失败仅告警（不回滚已成功的发布）。
        """
        from app.agent.rag.es_util import get_es_client

        prefix = f"{settings.es_index_prefix}-kb-"
        keep = {info.target}
        if info.previous_generation_id:
            keep.add(self.target_for("es", info.previous_generation_id))
        keep.update(_journal_es_targets())
        es = get_es_client()
        if es is None:
            return
        try:
            alias = f"{settings.es_index_prefix}-kb-active"
            try:
                hits = es.indices.get_alias(name=alias)
                keep.update(k for k in hits.keys())
            except Exception as e:  # noqa: BLE001
                if not _alias_not_found(e):
                    # Alias 状态未知时绝不能继续删除旧索引；即便当前/上一代
                    # 已在 keep 中，也可能误删尚未被指针记录的生效代。
                    logging.getLogger("app.evolution.index_service").warning(
                        "es 旧代清理跳过（alias 状态未知）: %s", e,
                    )
                    return
            for name in es.indices.get(index=prefix + "*").keys():
                if name in keep:
                    continue
                tail = name[len(prefix):] if name.startswith(prefix) else ""
                if not re.fullmatch(r"\d{14}-[0-9a-f]{8}", tail):
                    # 非代际格式（如手工索引）一律不动
                    continue
                try:
                    es.indices.delete(index=name)
                except Exception as e:  # noqa: BLE001 —— 单索引删除失败不影响其余
                    logging.getLogger("app.evolution.index_service").warning(
                        "es 旧代删除失败: %s (%s)", name, e,
                    )
        except Exception as e:  # noqa: BLE001 —— 列举失败：跳过清理（不影响主流程）
            logging.getLogger("app.evolution.index_service").warning(
                "es 旧代清理跳过: %s", e,
            )

    def delete_candidate(
        self,
        backend: str,
        target: str,
        *,
        protected_targets: set[str] | None = None,
    ) -> bool:
        """安全删除未激活的候选目标，返回是否已确认完成。

        发布恢复只会在状态机判定 rollback 后调用此方法。ES 仍需再次核验
        alias：读取失败或 alias 指向 candidate/其它目标时一律不删，并返回
        ``False``，让调用方保留 journal 等待人工/下次 reconcile。
        """
        backend = (backend or "").lower()
        target = str(target or "")
        if not target:
            return True
        if backend != "es":
            return False

        prefix = f"{settings.es_index_prefix}-kb-"
        tail = target[len(prefix):] if target.startswith(prefix) else ""
        if not tail or not re.fullmatch(r"\d{14}-[0-9a-f]{8}", tail):
            # 只允许删除严格版本化目标；手工/未知索引必须人工处理。
            return False

        from app.agent.rag.es_util import get_es_client

        es = get_es_client()
        if es is None:
            return False
        protected = set(protected_targets or ())
        try:
            current = self._store.active("es")
            if current is not None:
                protected.add(current.target)
            alias = f"{settings.es_index_prefix}-kb-active"
            try:
                hits = es.indices.get_alias(name=alias)
            except Exception as e:  # noqa: BLE001
                if _alias_not_found(e):
                    hits = {}
                else:
                    return False
            alias_targets = set(hits.keys())
            if target in alias_targets or target in protected:
                return False
            try:
                exists = es.indices.exists(index=target)
            except Exception:
                # 无法确认是否存在时不要删除，也不要宣称清理完成。
                return False
            if not exists:
                return True
            es.indices.delete(index=target)
            return True
        except Exception as e:  # noqa: BLE001
            logging.getLogger("app.evolution.index_service").warning(
                "es candidate 删除跳过: %s (%s)", target, e,
            )
            return False

    def reconcile(self, backend: str) -> dict:
        """一致性核对（运维/恢复测试）：alias 指向 vs generation 指针指向 vs journal。

        返回 {"alias_target", "pointer_target", "journal_generation_id",
        "consistent"}；consistent=False 时调用方按恢复表决定回滚/补齐/
        （journal 含 pending 代且 alias==pending 代）——「alias 已切但指针未写」
        的恢复入口。
        """
        backend = backend.lower()
        pointer = self._store.active(backend)
        pointer_target = pointer.target if pointer is not None else ""
        # ``""`` = 明确不存在/尚未切换；``None`` = ES 状态未知。
        # 两者必须由恢复状态机区分，未知状态不能触发 rollback。
        alias_target = ""
        alias_known = True
        if backend == "es":
            from app.agent.rag.es_util import get_es_client

            alias = f"{settings.es_index_prefix}-kb-active"
            try:
                es = get_es_client()
            except Exception:  # noqa: BLE001 —— ES 客户端构造失败 = unknown
                es = None
                alias_target = None
                alias_known = False
            if es is None and alias_target is not None:
                alias_target = None
                alias_known = False
            elif es is not None:
                try:
                    hits = es.indices.get_alias(name=alias)
                    targets = list(hits.keys())
                    # 一个活动 alias 应只有一个目标；多目标同样不可安全猜测。
                    if len(targets) > 1:
                        alias_target = None
                        alias_known = False
                    else:
                        alias_target = str(targets[0]) if targets else ""
                except Exception as e:  # noqa: BLE001
                    if _alias_not_found(e):
                        alias_target = ""
                    else:
                        alias_target = None
                        alias_known = False
        journal = _journal_es_targets()
        # consistent：alias 与指针一致（无 alias 的本地后端视为自洽）；
        # alias 落在 journal（未完成上传/下架的 pending 代）→ consistent=False
        # 但属于「可恢复的只前进」状态——由恢复机按恢复表处理。
        consistent = alias_target is not None and ((not alias_target) or (
            alias_target == pointer_target and alias_target not in journal
        ))
        return {
            "alias_target": alias_target,
            "alias_known": bool(alias_known),
            "pointer_target": pointer_target,
            "journal_targets": sorted(journal),
            "consistent": bool(consistent),
        }

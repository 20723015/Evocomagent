"""上传链路索引侧单测：两阶段激活、reconcile、ES 旧代清理（journal 保护）、
strict 构建与隐藏目录跳过、子进程解析守护。全程无网络（迷你 ES 替身 + 临时目录）。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.agent.rag.parse_guard import ParseTimeout, parse_with_timeout
from app.agent.rag.parsers import chunk_kb_dir
from app.config.settings import settings
from app.evolution.generation import GenerationInfo, GenerationStore
from app.evolution.index_service import IndexBuildService


# ============================================================
# 迷你 ES 替身（模仿 test_phase8_sql_es 的形态：只支持本链路用到的接口）
# ============================================================
class MiniES:
    """内存 ES 替身：indices.create/get/delete/refresh、update_aliases/get_alias、search。"""

    def __init__(self):
        self._indices: dict[str, dict] = {}
        self._aliases: dict[str, str] = {}  # alias -> index

    class _Indices:
        def __init__(self, es):
            self._es = es

        def create(self, **kwargs):
            name = kwargs["index"]
            assert name not in self._es._indices, f"index 已存在: {name}"
            self._es._indices[name] = {
                "mappings": kwargs.get("mappings", {}),
                "docs": {},
            }

        def get(self, index="*"):
            out = {}
            for name, data in self._es._indices.items():
                if index == "*" or name.startswith(index.rstrip("*")):
                    out[name] = data
            if not out:
                raise Exception(f"index not found: {index}")
            return out

        def get_mapping(self, index):
            data = self._es._indices.get(index)
            if data is None:
                raise Exception(f"index not found: {index}")
            return {index: {"mappings": data["mappings"]}}

        def delete(self, index):
            if index in self._es._indices:
                del self._es._indices[index]

        def refresh(self, **kwargs):
            return {"_shards": {"successful": 1}}

        def update_aliases(self, actions):
            for a in actions:
                if "remove" in a:
                    alias = a["remove"]["alias"]
                    if self._es._aliases.get(alias) == a["remove"]["index"]:
                        del self._es._aliases[alias]
                elif "add" in a:
                    self._es._aliases[a["add"]["alias"]] = a["add"]["index"]

        def get_alias(self, name):
            target = self._es._aliases.get(name)
            if target is None:
                raise Exception("alias not found")
            return {target: {"aliases": {name: {}}}}

    indices = property(lambda self: self._Indices(self))

    def bulk(self, operations=None, index=None, **kwargs):
        docs = self._indices[index]["docs"]
        for i in range(0, len(operations), 2):
            action = operations[i].get("create", {})
            doc = operations[i + 1]
            docs[action["_id"]] = doc
        return {"errors": False, "items": []}

    def count(self, index=None, **kwargs):
        return {"count": len(self._indices.get(index, {}).get("docs", {}))}

    def search(self, index=None, size=10, **kwargs):
        docs = self._indices.get(index, {}).get("docs", {})
        hits = [
            {"_id": cid, "_source": src, "_score": 1.0}
            for cid, src in list(docs.items())[:size]
        ]
        return {"hits": {"hits": hits}}


class _FakeEmbedder:
    model = "fake-embed"

    def encode(self, texts, timeout=None):
        return [[0.1, 0.2] for _ in texts]

    def encode_one(self, text, timeout=None):
        return [0.1, 0.2]


@pytest.fixture()
def tmp_kb(tmp_path: Path, monkeypatch) -> Path:
    kb = tmp_path / "knowledge"
    kb.mkdir()
    (kb / "根文档.md").write_text("# 根文档\n\n## 第一节\n内容\n", encoding="utf-8")
    (kb / ".staging").mkdir()
    (kb / ".trash").mkdir()
    (kb / "uploads").mkdir()
    # journal target 解析按 settings 相对 ROOT；测试统一指向 tmp
    monkeypatch.setattr(settings, "kb_upload_staging_dir", str(kb / ".staging"))
    monkeypatch.setattr(settings, "kb_upload_trash_dir", str(kb / ".trash"))
    return kb


def _build_service(tmp_kb: Path, es: MiniES | None = None, gen_path=None,
                   strict_build: bool = False):
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
    from app.stores.sql.schema import metadata

    metadata.create_all(eng)
    store = GenerationStore(gen_path or tmp_kb / "sessions" / "kb_generations.json")
    from app.agent.rag.es_util import set_es_for_test

    if es is not None:
        set_es_for_test(es)
    svc = IndexBuildService(
        embedder=_FakeEmbedder(),
        kb_dir=tmp_kb,
        generation_store=store,
        backend_settings={
            "kb_index_path": str(tmp_kb / "sessions" / "kb_index.json"),
            "chroma_persist_dir": str(tmp_kb / "chroma"),
            "chroma_collection": "ecom_kb",
        },
        chunker=chunk_kb_dir,
        strict_build=strict_build,
    )
    return svc


def _new_target(svc: IndexBuildService, gen: str) -> GenerationInfo:
    return GenerationInfo(
        generation_id=gen, target=svc.target_for("numpy", gen),
        embedding_model="fake-embed",
    )


class TestTwoPhaseActivation:
    def test_build_does_not_touch_alias(self, tmp_kb, monkeypatch):
        es = MiniES()
        svc = _build_service(tmp_kb, es=es)
        info = svc.build("es", generation_id="20260830000000-aaaaaaaa")
        assert es.indices.get(index=info.target)  # 索引已建
        with pytest.raises(Exception):
            es.indices.get_alias(f"{settings.es_index_prefix}-kb-active")  # alias 未切

    def test_activate_alias_then_pointer(self, tmp_kb):
        es = MiniES()
        svc = _build_service(tmp_kb, es=es)
        info = svc.build("es", generation_id="20260830000000-aaaaaaaa")
        svc.activate_alias("es", info)
        alias = f"{settings.es_index_prefix}-kb-active"
        assert es.indices.get_alias(alias) == {info.target: {"aliases": {alias: {}}}}
        # 指针未动：reconcile 检出不一致
        st = svc.reconcile("es")
        assert st["alias_target"] == info.target
        assert st["pointer_target"] == ""
        assert st["consistent"] is False
        svc.activate_pointer("es", info)
        st = svc.reconcile("es")
        assert st["consistent"] is True

    def test_activate_idempotent_rerun(self, tmp_kb):
        es = MiniES()
        svc = _build_service(tmp_kb, es=es)
        info = svc.build("es", generation_id="20260830000000-bbbbbbbb")
        svc.activate_alias("es", info)
        svc.activate_alias("es", info)  # 重跑 no-op（目标已是 alias 指向）
        alias = f"{settings.es_index_prefix}-kb-active"
        assert es.indices.get_alias(alias) == {info.target: {"aliases": {alias: {}}}}

    def test_activate_alias_switches_atomically(self, tmp_kb):
        es = MiniES()
        svc = _build_service(tmp_kb, es=es)
        i1 = svc.build("es", generation_id="20260830000000-c1c1c1c1")
        svc.activate_alias("es", i1)
        svc.activate_pointer("es", i1)
        i2 = svc.build("es", generation_id="20260830000000-d2d2d2d2")
        svc.activate_alias("es", i2)
        alias = f"{settings.es_index_prefix}-kb-active"
        assert es.indices.get_alias(alias) == {i2.target: {"aliases": {alias: {}}}}
        assert i2.previous_generation_id == ""  # 上一代信息在 pointer 时记录
        svc.activate_pointer("es", i2)
        assert svc.reconcile("es")["consistent"] is True


class TestReconcileWithJournal:
    def test_alias_service_error_is_unknown_not_absent(self, tmp_kb, monkeypatch):
        """ES alias 服务异常必须返回 unknown，恢复机不能据此回滚。"""
        es = MiniES()
        svc = _build_service(tmp_kb, es=es)

        def _raise(*args, **kwargs):
            raise RuntimeError("ES timeout")

        monkeypatch.setattr(MiniES._Indices, "get_alias", _raise)
        st = svc.reconcile("es")
        assert st["alias_target"] is None
        assert st["alias_known"] is False
        assert st["consistent"] is False

    def test_alias_in_journal_marks_inconsistent(self, tmp_kb, monkeypatch):
        es = MiniES()
        svc = _build_service(tmp_kb, es=es)
        info = svc.build("es", generation_id="20260830000000-e3e3e3e3")
        svc.activate_alias("es", info)
        # 模拟：journal 记录了 pending 代但指针未写（alias 已生效）
        jdir = tmp_kb / ".staging" / "journal"
        jdir.mkdir(parents=True)
        (jdir / "up-1.json").write_text(
            f'{{"phase":"ACTIVATING","index":{{"target":"{info.target}"}}}}',
            encoding="utf-8",
        )
        st = svc.reconcile("es")
        assert st["alias_target"] == info.target
        assert st["journal_targets"] == [info.target]
        assert st["consistent"] is False


class TestEsCleanupGuard:
    def _mk_svc_and_indexes(self, tmp_kb):
        es = MiniES()
        svc = _build_service(tmp_kb, es=es)
        return svc, es, svc.build("es", generation_id="20260830000000-f4f4f4f4")

    def test_cleanup_protects_alias_pointer_and_journal(self, tmp_kb):
        svc, es, info = self._mk_svc_and_indexes(tmp_kb)
        svc.activate_alias("es", info)
        svc.activate_pointer("es", info)

        old = svc.build("es", generation_id="20260829000000-a0a0a0a0")  # 更早的代
        # journal 中的 pending 代也要保护
        jdir = tmp_kb / ".staging" / "journal"
        jdir.mkdir(parents=True)
        (jdir / "up-x.json").write_text(
            f'{{"phase":"PREPARED","index":{{"target":"{old.target}"}}}}',
            encoding="utf-8",
        )

        # 无关索引（严格前缀外）不删
        es.indices.create(index=f"{settings.es_index_prefix}-messages", mappings={})
        # 非代际格式索引不删
        es.indices.create(index=f"{settings.es_index_prefix}-kb-manual", mappings={})

        svc._cleanup_old_es(info, {info.generation_id})
        assert info.target in es._indices  # 当前代保留
        assert old.target in es._indices  # journal 保护
        assert f"{settings.es_index_prefix}-messages" in es._indices
        assert f"{settings.es_index_prefix}-kb-manual" in es._indices

    def test_cleanup_deletes_unprotected_generation(self, tmp_kb):
        svc, es, info = self._mk_svc_and_indexes(tmp_kb)
        svc.activate_alias("es", info)
        svc.activate_pointer("es", info)
        victim = svc.build("es", generation_id="20260827000000-b0b0b0b0")
        svc._cleanup_old_es(info, {info.generation_id})
        assert victim.target not in es._indices


class TestChunkStrictAndHidden:
    def test_hidden_dirs_skipped(self, tmp_kb):
        (tmp_kb / ".staging" / "废.md").write_text("# 废\n", encoding="utf-8")
        (tmp_kb / ".trash" / "废2.md").write_text("# 废2\n", encoding="utf-8")
        chunks = chunk_kb_dir(tmp_kb)
        names = {c.doc for c in chunks}
        assert "根文档" in names
        assert "废" not in names and "废2" not in names

    def test_strict_raises_on_bad_file(self, tmp_kb):
        (tmp_kb / "坏.pdf").write_bytes(b"not a pdf")
        with pytest.raises(ValueError, match="strict 构建中止.*坏.pdf"):
            chunk_kb_dir(tmp_kb, strict=True)

    def test_non_strict_skips_bad_file(self, tmp_kb):
        (tmp_kb / "坏.pdf").write_bytes(b"not a pdf")
        chunks = chunk_kb_dir(tmp_kb, strict=False)
        assert {c.doc for c in chunks} == {"根文档"}

    def test_uploads_subdir_included(self, tmp_kb):
        (tmp_kb / "uploads" / "上传统一.md").write_text("# 上传统一\n\n内容\n", encoding="utf-8")
        chunks = chunk_kb_dir(tmp_kb)
        assert {"根文档", "上传统一"} <= {c.doc for c in chunks}

    def test_strict_build_aborts_on_bad_file(self, tmp_kb):
        """生产链路（strict_build=True，deps/build_kb_index/run_evolution 同款装配）：
        任一源文件解析失败 → build 整体中止，不静默丢知识。"""
        (tmp_kb / "坏.pdf").write_bytes(b"not a pdf")
        svc = _build_service(tmp_kb, strict_build=True)
        with pytest.raises(ValueError, match="strict 构建中止.*坏.pdf"):
            svc.build("numpy", generation_id="20260830000000-s1s1s1s1")

    def test_strict_build_requires_strict_aware_chunker(self, tmp_kb):
        """strict_build=True 但 chunker 不支持 strict 参数 → 显式报错（不静默降级）。"""
        from app.agent.rag.chunker import chunk_markdown_dir

        svc = IndexBuildService(
            embedder=_FakeEmbedder(), kb_dir=tmp_kb,
            generation_store=GenerationStore(tmp_kb / "g.json"),
            backend_settings={"kb_index_path": str(tmp_kb / "i.json")},
            chunker=chunk_markdown_dir,  # 无 strict 参数
            strict_build=True,
        )
        with pytest.raises(ValueError, match="要求 chunker 支持 strict 参数"):
            svc.build("numpy")


class TestParseGuard:
    def test_parse_md_ok(self, tmp_path: Path):
        p = tmp_path / "a.md"
        p.write_text("# 标题\n\n正文内容\n", encoding="utf-8")
        out = parse_with_timeout(p, timeout=10)
        assert "正文内容" in out

    def test_parse_missing_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            parse_with_timeout(tmp_path / "不存在.pdf", timeout=10)

    def test_parse_timeout_terminates(self, tmp_path: Path):
        p = tmp_path / "slow.md"
        p.write_text("x", encoding="utf-8")

        def slow(path):
            time.sleep(60)
            return "never"

        with pytest.raises(ParseTimeout):
            parse_with_timeout(p, timeout=0.2, _impl=slow)

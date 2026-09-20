"""RAG 切分优化（2026-09-15 方案 P0/P1-2）：生成单元、前缀去重、表头继承、索引输入分离。

覆盖点：
- P0-1 检索块/生成块分离：相邻小节合并为生成单元、每个检索块都有 parent_text、
  同单元共用 parent_id、超长单节回退窗口装配、关闭开关逐字节回到旧装配；
- P0-2 前缀去重：根标题与文档名重复时去重（含品牌前缀与空格差异），
  第三方来源文档不做有损删减，关闭开关回到旧格式；
- P0-3 治理元数据（status/authority/effective_date）随块下沉；
- P0-4 长表续块继承表头（含预算不足放弃继承、strip 还原、父块包含性校验）；
- P1-2 evolved 沉淀文档的索引输入让问题居首（块文本不受影响）。
"""

from __future__ import annotations

import pytest

from app.agent.rag.chunker import (
    MAX_CHUNK_CHARS,
    MAX_PARENT_CHARS,
    Chunk,
    _chunk_text,
    _display_label,
    _is_doc_name_variant,
    chunk_body,
    strip_inherited_table_head,
)
from app.agent.rag.parsers import chunk_kb_dir


def _write(kb, name: str, content: str) -> None:
    path = kb / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _section(text: str, size: int) -> str:
    """构造指定长度的正文（避免依赖真实语料）。"""
    filler = "支持七天无理由退货，运费由商家承担。"
    return (filler * (size // len(filler) + 1))[:size]


# ============================================================
# P0-1 生成单元（small-to-big 重定位）
# ============================================================
def test_adjacent_sections_merge_into_one_generation_unit(tmp_path):
    """相邻小节合并：父块包含同单元两个小节的原文，且共用 parent_id。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "policy.md", (
        "# 并夕夕 退款政策\n\n"
        "## 七天无理由\n\n" + _section("甲", 300) + "\n\n"
        "## 生鲜商品\n\n" + _section("乙", 300) + "\n\n"
    ))
    chunks = chunk_kb_dir(kb)
    seven = [c for c in chunks if c.section == "七天无理由"]
    fresh = [c for c in chunks if c.section == "生鲜商品"]
    assert seven and fresh
    # 两个小节落在同一生成单元（文档短于目标长度）
    assert seven[0].parent_id == fresh[0].parent_id
    assert seven[0].parent_text == fresh[0].parent_text
    assert chunk_body(seven[0].text) in seven[0].parent_text
    assert chunk_body(fresh[0].text) in fresh[0].parent_text
    # 父块是「大块」：比任一检索块都大（小块检索、大块生成）
    assert len(seven[0].parent_text) > len(chunk_body(seven[0].text))


def test_every_chunk_has_parent_text(tmp_path):
    """全量装配：不再有「单块章节无父块」的特例（旧实现覆盖率 1%）。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "a.md", "# 政策\n\n## 一节\n\n内容甲。\n\n## 二节\n\n内容乙。\n")
    chunks = chunk_kb_dir(kb)
    assert chunks
    for c in chunks:
        assert c.parent_text
        assert c.parent_id
        assert chunk_body(c.text) in c.parent_text


def test_oversized_section_falls_back_to_window(tmp_path):
    """单节超过父块硬上限：独占单元并按命中位置取窗口（不截断、不丢包含性）。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    big = "\n\n".join(_section(f"段{i}", 900) for i in range(7))  # ~6300 字
    _write(kb, "big.md", f"# 政策\n\n## 长章节\n\n{big}\n\n## 短章节\n\n结尾。\n")
    chunks = chunk_kb_dir(kb)
    long_chunks = [c for c in chunks if c.section == "长章节"]
    assert len(long_chunks) > 1
    for c in long_chunks:
        assert len(c.parent_text) <= MAX_PARENT_CHARS
        assert chunk_body(c.text) in c.parent_text
    # 超长小节独占单元：与短章节不共用 parent_id
    short = [c for c in chunks if c.section == "短章节"]
    assert short
    assert short[0].parent_id != long_chunks[0].parent_id


def test_generation_unit_split_by_target_length(tmp_path):
    """长文档按目标长度切成多个生成单元（同单元内 parent_id 相同、单元间不同）。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    sections = "\n\n".join(
        f"## 小节{i}\n\n{_section(f'第{i}节', 1200)}" for i in range(6)
    )
    _write(kb, "long.md", f"# 长政策\n\n{sections}\n")
    chunks = chunk_kb_dir(kb)
    units = {}
    for c in chunks:
        units.setdefault(c.parent_id, set()).add(c.section)
    assert len(units) >= 2, "6×1200 字应切成多个生成单元"
    # 单元内不混入同一小节的重复归属
    for pid, secs in units.items():
        assert pid.startswith("long#u")
        assert secs


def test_parent_merge_off_restores_legacy_assembly(tmp_path):
    """回退开关：parent_merge=False 回到旧装配（单块章节无父块、parent_id 同章节）。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "a.md", "# 政策\n\n## 一节\n\n内容甲。\n\n## 二节\n\n内容乙。\n")
    chunks = chunk_kb_dir(kb)
    legacy = _chunk_text(
        (kb / "a.md").read_text(encoding="utf-8"), "a.md", kb,
        parent_merge=False, prefix_dedup=False,
    )
    assert len(legacy) == len(chunks)
    for c in legacy:
        assert c.parent_text == ""  # 单块章节不装配父块（旧语义）
        assert c.parent_id.startswith("a#p")
    assert all(c.parent_text for c in chunks)  # 新装配则全量装配


def test_parent_child_disabled_clears_both_fields(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "a.md", "# 政策\n\n## 一节\n\n内容甲。\n\n## 二节\n\n内容乙。\n")
    for c in chunk_kb_dir(kb, parent_child=False):
        assert c.parent_text == ""
        assert c.parent_id == ""


# ============================================================
# P0-2 前缀去重
# ============================================================
def test_display_label_drops_branded_root_title():
    assert _is_doc_name_variant("并夕夕 3C数码类目规则", "3C数码类目规则")
    assert _is_doc_name_variant("常见问题 FAQ", "常见问题FAQ")  # 空格差异归一
    assert _is_doc_name_variant("退换货政策", "退换货政策")
    # 第三方来源文档：根标题是不同表述，不做有损删减
    assert not _is_doc_name_variant("安平保险运费险承保条款", "保险公司运费险承保条款")
    assert not _is_doc_name_variant("自进化知识", "20260901-abc-问题是什么")
    assert _display_label("并夕夕 退换货政策 > 一、七天", "退换货政策") == "一、七天"
    assert _display_label("并夕夕 退换货政策", "退换货政策") == ""
    # 兼容旧调用（不传 doc_name → 不去重）
    assert _display_label("A > B") == "A > B"


def test_prefix_dedup_shrinks_prefix_and_keeps_full_path(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "3C数码类目规则.md",
           "# 并夕夕 3C数码类目规则\n\n## 一、激活与联网\n\n未激活可退。\n")
    chunk = chunk_kb_dir(kb)[0]
    assert chunk.text.split("\n")[0] == "【3C数码类目规则 · 一、激活与联网】"
    # 完整标题路径仍在元数据里（去重只影响展示前缀）
    assert chunk.heading_path == "并夕夕 3C数码类目规则 > 一、激活与联网"

    legacy = _chunk_text(
        (kb / "3C数码类目规则.md").read_text(encoding="utf-8"),
        "3C数码类目规则.md", kb, parent_merge=False, prefix_dedup=False,
    )[0]
    assert legacy.text.split("\n")[0] == (
        "【3C数码类目规则 · 并夕夕 3C数码类目规则 > 一、激活与联网】"
    )
    assert len(chunk.text) < len(legacy.text)


def test_single_h1_equal_doc_name_collapses_prefix(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "常见问题.md", "# 常见问题\n\n## 发货时效\n\n48 小时内发货。\n")
    chunk = chunk_kb_dir(kb)[0]
    assert chunk.text.split("\n")[0] == "【常见问题 · 发货时效】"


# ============================================================
# P0-3 治理元数据下沉
# ============================================================
def test_governance_metadata_lands_on_chunks(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "a.md", (
        "---\nstatus: active\nauthority: platform\n"
        "effective_date: 2026-09-12\nowner: ops\n---\n"
        "# 政策\n\n## 一节\n\n内容。\n"
    ))
    chunk = chunk_kb_dir(kb)[0]
    assert chunk.status == "active"
    assert chunk.authority == "platform"
    assert chunk.effective_date == "2026-09-12"
    assert chunk.owner == "ops"
    # 元数据不进块文本（证据仍是原文）
    assert "effective_date" not in chunk.text


def test_governance_metadata_survives_backend_roundtrip(tmp_path):
    """新字段必须随索引持久化（旧索引缺字段用默认值反序列化兼容）。"""
    from app.agent.rag.backends.numpy_backend import NumpyBackend

    chunk = Chunk(
        chunk_id="a#00-00", doc="a", section="s", text="【a · s】\n正文",
        status="active", authority="platform", effective_date="2026-09-12",
        index_text="上下文\n【a · s】\n正文",
    )
    backend = NumpyBackend(tmp_path / "idx.json")
    backend.upsert([chunk], [[1.0, 0.0]], "m")
    loaded = NumpyBackend(tmp_path / "idx.json")
    loaded.load()
    got = loaded.chunks()[0]
    assert (got.status, got.authority, got.effective_date) == (
        "active", "platform", "2026-09-12",
    )
    assert got.index_input() == chunk.index_text
    # 旧索引（无新字段）反序列化不炸
    old = Chunk(**{"chunk_id": "x#00-00", "doc": "x", "section": "s", "text": "t"})
    assert old.index_input() == "t"
    assert (old.status, old.authority, old.effective_date) == ("", "", "")


# ============================================================
# P0-4 长表续块带表头
# ============================================================
def _big_table(rows: int, cell: str = "数据") -> str:
    head = "| 期数 | 费率 | 说明 |\n| --- | --- | --- |"
    body = "\n".join(f"| {i} | 1.{i}% | {cell * 20} |" for i in range(rows))
    return f"{head}\n{body}"


def test_long_table_continuation_keeps_header(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "fee.md", f"# 费率表\n\n## 分期\n\n{_big_table(24)}\n")
    chunks = [c for c in chunk_kb_dir(kb) if c.section == "分期"]
    assert len(chunks) > 1, "长表应被切开"
    first = chunk_body(chunks[0].text)
    assert first.startswith("| 期数 | 费率 | 说明 |")
    for c in chunks[1:]:
        body = chunk_body(c.text)
        assert body.startswith("| 期数 | 费率 | 说明 |\n| --- | --- | --- |"), (
            "续块必须继承表头，否则列语义断裂"
        )
        # 表头之后的第一行必须是完整表格行（重叠不得把上一行拦腰带进来）
        assert body.split("\n")[2].startswith("| "), "续块以半行开头（重叠未对齐行首）"
        assert len(c.text) <= MAX_CHUNK_CHARS
    # 续块同样保留父块（继承表头不破坏包含性：用原文切片 core 校验）
    for c in chunks[1:]:
        assert strip_inherited_table_head(chunk_body(c.text)) in c.parent_text


def test_strip_inherited_table_head_only_strips_copied_header():
    head = "| a | b |\n| --- | --- |"
    assert strip_inherited_table_head(f"{head}\n| 1 | 2 |") == "| 1 | 2 |"
    # 真实表头（首块）不剥：只有两行时不属于「续块」
    assert strip_inherited_table_head(head) == head
    # 非表格内容原样返回
    assert strip_inherited_table_head("正文\n正文") == "正文\n正文"


def test_parent_window_check_accepts_inherited_header():
    """knowledge._parent_window：续块父块命中仍需通过包含性校验。"""
    from app.agent.tools.knowledge import _parent_window

    head = "| 期数 | 费率 |\n| --- | --- |"
    chunk = Chunk(
        chunk_id="fee#00-01", doc="fee", section="分期",
        text=f"【fee · 分期】\n{head}\n| 3 | 1.5% |",
        parent_text=f"开头。\n{head}\n| 1 | 0.5% |\n| 3 | 1.5% |\n结尾。",
    )
    text, context_type = _parent_window(chunk)
    assert context_type == "parent"
    assert text == chunk.parent_text


# ============================================================
# P1-2 evolved 双路表征
# ============================================================
def test_evolved_index_input_leads_with_question(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "evolved/20260901-abc-钻石会员响应时效是多少.md", (
        "---\nowner: system\n---\n"
        "# 自进化知识\n\n## 钻石会员响应时效是多少？\n\n30 秒内接入。\n"
    ))
    chunk = chunk_kb_dir(kb)[0]
    assert chunk.section == "钻石会员响应时效是多少？"
    # 索引输入：问题居首 + 机械前缀 + 正文
    assert chunk.index_input().startswith("钻石会员响应时效是多少？\n【自进化知识 · ")
    # 块文本与证据文本不含额外生成内容（首行仍是机械前缀）
    assert chunk.text.startswith("【自进化知识 · ")
    assert chunk_body(chunk.text).startswith("30 秒内接入")


def test_non_evolved_chunks_have_no_index_text(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "a.md", "# 政策\n\n## 一节\n\n内容。\n")
    chunk = chunk_kb_dir(kb)[0]
    assert chunk.index_text == ""
    assert chunk.index_input() == chunk.text


def test_bm25_indexes_index_input(monkeypatch):
    """BM25 词法统计吃索引输入（evolved 问题因此可被问题式查询命中）。"""
    from app.agent.rag.bm25 import BM25Index

    chunk = Chunk(
        chunk_id="e#00-00", doc="自进化知识", section="问题",
        text="【自进化知识 · 问题】\n答案正文",
        index_text="钻石会员响应时效是多少？\n【自进化知识 · 问题】\n答案正文",
    )
    hits = BM25Index([chunk]).search("钻石会员响应时效", top_k=1)
    assert hits and hits[0].chunk.chunk_id == "e#00-00"
    # 去掉 index_text（旧索引语义）时同一查询不再命中
    plain = Chunk(chunk_id="e#00-00", doc="自进化知识", section="问题",
                  text=chunk.text)
    assert not BM25Index([plain]).search("钻石会员响应时效", top_k=1)


def test_chunk_text_length_stays_within_limit(tmp_path):
    """回归：前缀去重后正文预算变大，块长上限仍不被突破。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "a.md", "# 政策\n\n## 一节\n\n" + _section("甲", 5000) + "\n")
    for c in chunk_kb_dir(kb):
        assert len(c.text) <= MAX_CHUNK_CHARS


def test_build_to_evidence_uses_generation_unit(tmp_path):
    """端到端：新格式索引 → 候选检索器 → search_knowledge 证据 = 生成单元（父块）。

    锁住 P0-1 的对外效果链路（构建 → 装配 → 证据组装），并顺带验证评测门禁所用
    的 evaluate() 能吃新格式索引（检索质量数字另由 959 例门禁给，此处只核链路）。
    """
    from app.agent.rag.retriever_factory import RetrievalConfig, open_retriever
    from app.agent.tools import knowledge as ktool
    from app.evaluation.retrieval_metrics import evaluate
    from app.evolution.generation import GenerationStore
    from app.evolution.index_service import IndexBuildService
    from tests.unit.conftest import FakeEmbedder

    kb = tmp_path / "kb"
    kb.mkdir()
    _write(kb, "退换货政策.md", (
        "# 并夕夕 退换货政策\n\n"
        "## 一、七天无理由退货\n\n自签收之日起 7 天内可申请七天无理由退货。\n\n"
        "## 二、运费承担\n\n质量问题退货运费由商家承担。\n\n"
    ))
    settings_override = {"kb_index_path": str(tmp_path / "idx.json")}
    service = IndexBuildService(
        embedder=FakeEmbedder(), kb_dir=kb,
        generation_store=GenerationStore(tmp_path / "gen"),
        backend_settings=settings_override, chunker=chunk_kb_dir,
    )
    info = service.build("numpy")

    config = RetrievalConfig(backend="numpy", kb_index_path=settings_override["kb_index_path"],
                             backend_settings=settings_override)
    retriever = open_retriever(
        config, embedder=FakeEmbedder(), generation_target=info.target,
    )
    assert retriever.size == 2

    # 门禁同一函数（evaluate）能吃新索引：调用不抛错且给出正例指标结构
    report = evaluate(
        [{"id": "c1", "query": "七天无理由退货", "expected": ["退换货政策.md"], "k": 3}],
        retriever, top_k=3, min_score=None,
    )
    assert report["summary"]["positive"]["cases"] == 1

    # 证据 = 生成单元：命中「运费承担」小节时上下文里也含同单元的另一小节
    ktool.push_retriever_override(retriever)
    try:
        out = ktool.search_knowledge(query="运费谁承担", top_k=3)
    finally:
        ktool.pop_retriever_override()
    assert out["success"] and out["results"]
    top = out["results"][0]
    assert top["context_type"] == "parent"
    assert len(top["text"]) > len(top["matched_text"])
    assert "七天无理由" in top["text"]  # 同单元的另一小节一并进入上下文


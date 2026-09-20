"""构建期上下文增强（P1-1 Contextual Retrieval）：生成、缓存、降级、装配。

覆盖点：
- 生成上下文进入 index_text（问题式定位内容），块文本/证据文本不被污染；
- content-hash 缓存：未变化块零调用重建（命中缓存不调 LLM）；
- 单块失败/空输出按块降级（回退机械前缀），不阻断构建；
- 已有 index_text 的块（evolved 问题文本）不重复生成；
- 缓存键随模型/prompt 版本变化失效；clean_context 清洗与截断；
- index_service 集成：注入增强器后 encode 吃 index_input，构建报告落 last_context_report。
"""

from __future__ import annotations

from app.agent.rag.chunker import Chunk
from app.agent.rag.contextual import (
    ContextCache,
    ContextualEnricher,
    build_context_messages,
    clean_context,
    context_key,
)
from app.evolution.index_service import IndexBuildService
from app.evolution.generation import GenerationStore
from tests.unit.conftest import FakeEmbedder


def _chunk(cid: str, body: str = "钻石会员 30 秒内接入。") -> Chunk:
    return Chunk(chunk_id=cid, doc="自进化知识", section="响应时效",
                 text=f"【自进化知识 · 响应时效】\n{body}")


class _Recorder:
    """脚本化 chat 完成函数：按调用次序返回结果或抛错，记录每次 messages。"""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls: list[list[dict]] = []

    def __call__(self, messages, max_tokens):
        self.calls.append(messages)
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


# ============================================================
# 生成与装配
# ============================================================
def test_generated_context_goes_to_index_input_only():
    recorder = _Recorder(["客服响应时效章节，说明钻石会员的接入时限。"])
    chunk = _chunk("a#00-00")
    original_text = chunk.text
    report = ContextualEnricher(
        recorder, model="m", cache=None, max_chars=80,
    ).enrich([chunk])

    assert report.generated == 1 and report.failed == 0
    assert chunk.index_text.startswith("客服响应时效章节，说明钻石会员的接入时限。")
    # 索引输入 = 生成上下文 + 机械前缀 + 原文；块文本与证据文本不含生成内容
    assert chunk.index_text.endswith(original_text)
    assert chunk.text == original_text
    # prompt 带上文档名/标题路径/正文（定位上下文三要素）
    prompt = recorder.calls[0][-1]["content"]
    assert "自进化知识" in prompt and "响应时效" in prompt


def test_evolved_index_text_is_not_regenerated():
    """已有 index_text（evolved 问题文本）→ 跳过 LLM，保留原文问题优先语义。"""
    recorder = _Recorder([])  # 不应被调用
    chunk = _chunk("e#00-00")
    chunk.index_text = "响应时效是多少？\n" + chunk.text
    report = ContextualEnricher(recorder, model="m").enrich([chunk])
    assert report.cached == 1 and report.generated == 0
    assert not recorder.calls
    assert chunk.index_text.startswith("响应时效是多少？")


def test_failure_degrades_per_chunk_without_aborting():
    recorder = _Recorder([RuntimeError("boom"), "第二块的上下文"])
    chunks = [_chunk("a#00-00"), _chunk("a#00-01")]
    report = ContextualEnricher(recorder, model="m").enrich(chunks)
    assert report.total == 2
    assert report.failed == 1 and report.generated == 1
    assert report.errors and "boom" not in report.errors[0]  # 只记异常类型
    assert chunks[0].index_text == ""  # 降级：回退机械前缀
    assert chunks[0].index_input() == chunks[0].text
    assert chunks[1].index_text.endswith(chunks[1].text)
    assert report.as_dict()["degrade_ratio"] == 0.5


def test_empty_generation_counts_as_failure():
    chunk = _chunk("a#00-00")
    report = ContextualEnricher(_Recorder(["   "]), model="m").enrich([chunk])
    assert report.failed == 1 and chunk.index_text == ""


# ============================================================
# 缓存
# ============================================================
def test_cache_avoids_second_llm_call(tmp_path):
    path = tmp_path / "ctx.json"
    chunk_text = "【政策 · 一节】\n七天无理由退货。"
    first = ContextualEnricher(
        _Recorder(["退换货政策的适用范围。"]), model="m",
        cache=ContextCache(path),
    )
    c1 = Chunk(chunk_id="a#00-00", doc="政策", section="一节", text=chunk_text)
    assert first.enrich([c1]).generated == 1
    assert path.exists()

    recorder = _Recorder([])  # 不应被调用
    second = ContextualEnricher(recorder, model="m", cache=ContextCache(path))
    c2 = Chunk(chunk_id="a#00-00", doc="政策", section="一节", text=chunk_text)
    report = second.enrich([c2])
    assert report.cache_hits == 1 and report.generated == 0
    assert not recorder.calls
    assert c2.index_text.startswith("退换货政策的适用范围。")


def test_cache_key_varies_with_model_and_prompt_version():
    base = context_key("text", "m", "v1")
    assert base != context_key("text", "m2", "v1")  # 换模型
    assert base != context_key("text", "m", "v2")  # 改 prompt
    assert base != context_key("text2", "m", "v1")  # 换内容
    assert base == context_key("text", "m", "v1")


def test_corrupt_cache_is_treated_as_empty(tmp_path):
    path = tmp_path / "ctx.json"
    path.write_text("{not json", encoding="utf-8")
    cache = ContextCache(path)
    assert cache.get("k") is None
    cache.put("k", "v")
    assert cache.get("k") == "v"


# ============================================================
# prompt / 清洗
# ============================================================
def test_build_context_messages_truncates_body():
    messages = build_context_messages(
        doc="d", heading_path="h", body="字" * 5000, max_chars=60,
    )
    assert messages[0]["role"] == "system"
    assert len(messages[1]["content"]) < 3000  # 正文截断，不整篇塞进 prompt


def test_clean_context_strips_quotes_and_prefix():
    assert clean_context('"退换货政策 适用范围。"', 80) == "退换货政策 适用范围。"
    assert clean_context("定位上下文：会员权益说明", 80) == "会员权益说明"
    assert clean_context("  多  空白\n 折叠  ", 80) == "多 空白 折叠"


def test_clean_context_truncates_at_punctuation():
    raw = "第一句说明定位。第二句继续补充说明内容并且很长很长很长"
    out = clean_context(raw, 12)
    assert out == "第一句说明定位。"  # 在句读处收口


# ============================================================
# index_service 集成
# ============================================================
class _SpyEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    def encode(self, texts, timeout=None):
        self.seen.extend(texts)
        return super().encode(texts)


def _service(tmp_path, embedder, enricher):
    return IndexBuildService(
        embedder=embedder,
        kb_dir=tmp_path / "kb",
        generation_store=GenerationStore(tmp_path / "gen"),
        backend_settings={"kb_index_path": str(tmp_path / "idx.json")},
        chunker=_chunker,
        contextual_enricher=enricher,
    )


def _chunker(kb_dir, strict=False):
    from app.agent.rag.parsers import chunk_kb_dir

    return chunk_kb_dir(kb_dir, strict=strict)


def test_build_indexes_contextual_text_and_reports(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 政策\n\n## 一节\n\n七天无理由退货。\n", encoding="utf-8")
    embedder = _SpyEmbedder()
    enricher = ContextualEnricher(
        _Recorder(["退换货政策适用范围。"]), model="m", cache=None,
    )
    service = _service(tmp_path, embedder, enricher)
    service.build("numpy")

    assert service.last_context_report["generated"] == 1
    assert any(t.startswith("退换货政策适用范围。") for t in embedder.seen)
    # 索引里持久化的是 index_text，证据文本仍是原文
    chunk = service.last_chunks[0]
    assert chunk.index_input().startswith("退换货政策适用范围。")
    assert chunk.text.startswith("【a · 政策 > 一节】")


def test_build_skips_enricher_when_disabled(tmp_path):
    """未注入且 settings 关闭 → 不增强（默认 no-op，索引内容与旧版一致）。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# 政策\n\n## 一节\n\n七天无理由退货。\n", encoding="utf-8")
    embedder = _SpyEmbedder()
    service = _service(tmp_path, embedder, None)
    service.build("numpy")
    assert service.last_context_report == {}
    assert all(t == c.text for t, c in zip(embedder.seen, service.last_chunks))

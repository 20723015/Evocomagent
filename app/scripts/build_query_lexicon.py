"""构建 query 表层规范化词表（检索 hard 侧工作流 A1）。

动机（门禁轮 hard 76.6% 失分解剖）：334 条 hard 失分中 11 条 typo（同音错字，
如「积份/运废险/手续废」）+ 9 条 pinyin（拼音首字母，如「YFX/店铺q/积f」）。
两类噪声的共同点是**领域词的表面形式被写坏**，而向量/BM25 都对表面形式敏感。
修复方式不是改检索算法，而是给 query 做一次表层规范化：把已知领域词的各种
写坏形式还原成词表中的规范词。

产出的词表由运行时规范化器（app/agent/rag/query_normalizer.py）加载，**运行时
零依赖**：pypinyin 只在构建期用（requirements-dev.txt），构建结果落盘为
app/agent/rag/query_lexicon.json。

词源（全部来自 KB 语料，不引入外部词表）：
1. 文档名 stem（含 frontmatter 之外的 source_path 文件名）；
2. heading_path 的各级标题，按 与/和/、/及/ 切分后的片段；
3. parent_text 中词频 ≥ --min-freq 的 2~8 字 n-gram（Apriori 式逐层增长，
   只保留所有子片段都高频的候选），用于补齐「只在正文出现、不在标题出现」的
   领域词（如 订单/税费/微信）。

隔离：F 类跨界近域文档（银行/保险/快递公司/品牌等**平台外机构条款**，作为
负例压力源入库）与 evolved/（自进化沉淀）一律不参与词表构建——规范化只能
把 query 拉向**平台权威文档**的词。F 类的 frontmatter 目前仍是
`authority: platform`（v3 冻结口径：改成 external_reference 会把它们移出
回答索引、破坏 959 冻结评测，frontmatter 治理留待 v3.5 轮），因此词表侧
按下方 F_CLASS_SOURCE_PATHS 显式排除；待元数据修正后该清单与
`authority == "external_reference"` 判定二选一收敛。

用法：
    python app/scripts/build_query_lexicon.py                  # 写默认产物路径
    python app/scripts/build_query_lexicon.py --min-freq 5 --out /tmp/lex.json
    python app/scripts/build_query_lexicon.py --report         # 只打印统计，不写盘
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings

DEFAULT_OUT = "app/agent/rag/query_lexicon.json"
MIN_LEN = 2
MAX_LEN = 8
# 标题/文档名里承接语义的连接词：切出片段后各段都是独立领域词
SEGMENT_SEPARATORS = ("与", "和", "、", "/", "及")
# n-gram 边界停用字：以「的了和与吗呢」等虚词开头/结尾的 n-gram 不是领域词。
# 刻意只收极小集合：像「到账」这类以常见虚字起头但确属领域词的形式要保留
# （「到」不在集合内）；而「为」在集合内，「为限」这类形式会被过滤——
# review 修正：原注释举「为限要保留」与实现矛盾（为 ∈ 集合必被滤除）。
STOP_BOUNDARY = set("的了和与吗呢吧啊嘛哦呀么之其及或则即如若为在着过把被")
BRAND_PREFIXES = ("并夕夕",)
# F 类跨界近域文档（v3 计划 §7.6，负例压力源——平台外机构条款）：词表不得
# 把 query 拉向这些文档的词汇。frontmatter 仍是 platform（见模块 docstring），
# 在此按 source_path 显式排除；清单改动需同步 v3 计划。
F_CLASS_SOURCE_PATHS = frozenset({
    "银行信用卡分期业务条款.md",
    "快递公司延误与丢件赔偿标准.md",
    "厂家全国联保三包政策.md",
    "航空公司行李损坏赔偿指引.md",
    "电信运营商话费退费规则.md",
    "支付平台账户安全险条款.md",
    "保险公司运费险承保条款.md",
    "品牌官方延保服务条款.md",
    "第三方鉴定机构流程说明.md",
    "银行储蓄卡盗刷赔付规则.md",
})


def _pinyin_tables():
    """构建期依赖 pypinyin；缺依赖时给出可操作的报错（仅构建期需要）。

    返回 (首字母表, 同音字表) 的**构造器**：两者都只覆盖语料里真实出现的汉字，
    这样词表体积由语料词汇量决定，而不是由汉字全集（2 万字）决定。
    """
    try:
        from pypinyin import Style, pinyin
    except ImportError as e:  # pragma: no cover - 构建期环境问题
        raise SystemExit(
            "构建词表需要 pypinyin（仅构建期依赖）："
            "pip install -r requirements-dev.txt"
        ) from e

    def _syllable(ch: str) -> str:
        """单字拼音（去声调、去数字），如 份 → fen。"""
        try:
            raw = pinyin(ch, style=Style.TONE3, heteronym=False, errors="ignore")
        except Exception:  # noqa: BLE001 - 生僻字/异常字符按无拼音处理
            return ""
        if not raw or not raw[0]:
            return ""
        return "".join(c for c in raw[0][0] if c.isalpha()).lower()

    return _syllable


def _is_cjk(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def _is_word_char(ch: str) -> bool:
    return _is_cjk(ch) or ch.isalnum()


def _iter_texts(chunks) -> list[str]:
    """参与词频统计与 n-gram 挖掘的正文（parent_text 优先）。"""
    return [c.parent_text or c.text for c in chunks]


def _count_ngrams(texts: list[str], length: int, prev_survivors: set[str]):
    """统计长度为 length 且所有 (length-1) 子片段均高频的 n-gram。

    Apriori 剪枝：子片段不高频 → 整词必然不高频（词频单调性），
    于是先用「前 2 字是否高频」做廉价剪枝，再校验其余子片段。
    """
    if length <= 2:
        cnt: Counter = Counter()
        for t in texts:
            for i in range(len(t) - 1):
                gram = t[i:i + 2]
                if _is_word_char(gram[0]) and _is_word_char(gram[1]):
                    cnt[gram] += 1
        return cnt
    heads = {g[:2] for g in prev_survivors}
    cnt: Counter = Counter()
    for t in texts:
        n = len(t)
        for i in range(n - length + 1):
            if t[i:i + 2] not in heads:
                continue
            gram = t[i:i + length]
            if any(gram[j:j + length - 1] not in prev_survivors
                   for j in (0, 1)):
                continue
            cnt[gram] += 1
    return cnt


def _harvest_frequent_ngrams(texts: list[str], min_freq: int,
                             max_len: int = MAX_LEN) -> dict[int, dict[str, int]]:
    """逐层挖掘高频 n-gram（2..max_len），返回 {长度: {n-gram: 词频}}。"""
    per_len: dict[int, dict[str, int]] = {}
    prev: set[str] = set()
    for length in range(2, max_len + 1):
        cnt = _count_ngrams(texts, length, prev)
        prev = {
            gram for gram, freq in cnt.items()
            if freq >= min_freq and _is_vocab_shaped(gram)
        }
        per_len[length] = {gram: cnt[gram] for gram in prev}
        if not prev:
            break
    return per_len


def _maximal_terms(terms: dict[str, int], per_len: dict[int, dict[str, int]],
                   *, ratio: float = 0.6) -> set[str]:
    """剔除「片段词」：只在更长短语内部高频出现的伪词。

    语料里最高频的 n-gram 有大量跨词边界片段（「时后」来自「半小时后」、
    「要部」来自「需要补缴」、「用机」来自「使用机器」）。把这类片段当改写
    目标，会自信地把写对的 query 改坏（时候→时后、要补→要部）。

    判据：若存在单字扩展 T'（左右各一格，仍为领域词形状）且
    freq(T') ≥ ratio × freq(T)，说明 T 的多数出现都在更长词组内部 →
    T 只是片段，丢弃。真词的出现分散在多种上下文里，单个扩展的频率远低于
    整词频率（若某扩展真的接近整词频率，丢弃它也无害——那个更长的高频词
    本身在词表里，会以更长的匹配覆盖同一片段）。
    """
    keep: set[str] = set()
    for length in sorted({len(t) for t in terms}):
        longer = per_len.get(length + 1) or {}
        if not longer:
            keep.update(t for t in terms if len(t) == length)
            continue
        # 单字扩展的最大词频：右扩 T+Y 与左扩 X+T 各取最大
        prefix_max: dict[str, int] = {}
        suffix_max: dict[str, int] = {}
        for gram, gram_freq in longer.items():
            prefix, suffix = gram[:-1], gram[1:]
            if gram_freq > prefix_max.get(prefix, 0):
                prefix_max[prefix] = gram_freq
            if gram_freq > suffix_max.get(suffix, 0):
                suffix_max[suffix] = gram_freq
        for term in (t for t in terms if len(t) == length):
            if max(prefix_max.get(term, 0), suffix_max.get(term, 0)) >= (
                ratio * terms[term]
            ):
                continue
            keep.add(term)
    return keep


def _dict_guard(lexicon_payload: dict) -> list[str]:
    """通用词典里的真词，凡会被改写的一律进护栏。

    词表只来自 KB 语料，挡不住「span 是通用中文词、且恰好与某个词表词同音」
    的情况（已经→一经、超市→超时、累积→累计）：这类词在 KB 里可能根本不出现，
    语料护栏（freq≥5）覆盖不到。用通用词典（jieba，仅构建期依赖）把真词
    预查一遍：会被改写的真词全部进护栏，改写器对它们恒等。

    只保留「真的会被改写」的词条，护栏体积由碰撞量决定（几百条），不是
    全量词典。
    """
    try:
        import jieba
    except ImportError as e:  # pragma: no cover - 构建期环境问题
        raise SystemExit(
            "构建真词护栏需要 jieba（仅构建期依赖）："
            "pip install -r requirements-dev.txt"
        ) from e
    from app.agent.rag.query_normalizer import QueryNormalizer

    jieba.initialize()
    normalizer = QueryNormalizer(_lexicon_from_payload(lexicon_payload))
    existing_guard = set(lexicon_payload["guard_ngrams"])
    guard: list[str] = []
    for word, freq in jieba.dt.FREQ.items():
        if not (2 <= len(word) <= 8) or freq < 1:
            continue
        if not _is_vocab_shaped(word):
            continue
        if word in existing_guard or normalizer.would_rewrite(word):
            guard.append(word)
    return guard


def _lexicon_from_payload(payload: dict):
    """把产物载荷还原成运行时词表对象（复用规范化器的判定逻辑）。"""
    from app.agent.rag.query_normalizer import _Lexicon

    return _Lexicon(
        terms=frozenset(payload.get("terms") or ()),
        guard=frozenset(payload.get("guard_ngrams") or ()),
        syllables=dict(payload.get("char_syllables") or {}),
        freq={k: int(v) for k, v in (payload.get("term_freq") or {}).items()},
        meta=dict(payload.get("source_generation") or {}),
    )


def _is_vocab_shaped(gram: str) -> bool:
    """领域词形状过滤：全词字，首尾非虚词，且至少含一个汉字。"""
    if any(not _is_word_char(ch) for ch in gram):
        return False
    if gram[0] in STOP_BOUNDARY or gram[-1] in STOP_BOUNDARY:
        return False
    return any(_is_cjk(ch) for ch in gram)


def _split_segment(segment: str) -> list[str]:
    """标题片段 → 领域词：整段与按连接词切分后的各片段都作为候选。"""
    out = [segment]
    for sep in SEGMENT_SEPARATORS:
        if sep in segment:
            out.extend(segment.split(sep))
    return out


def _collect_from_metadata(chunks) -> Counter:
    """文档名 + 标题路径片段 → 候选词（计数为该词在语料中出现的次数）。"""
    counts: Counter = Counter()
    for c in chunks:
        names = []
        if c.doc:
            names.append(c.doc)
        if c.source_path:
            names.append(Path(c.source_path).stem)
        for name in names:
            for prefix in BRAND_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):].strip()
            counts[name] += 1
        for segment in (c.heading_path or "").split(">"):
            segment = segment.strip()
            if not segment:
                continue
            for prefix in BRAND_PREFIXES:
                if segment.startswith(prefix):
                    segment = segment[len(prefix):].strip()
            for piece in _split_segment(segment):
                piece = piece.strip()
                if piece:
                    counts[piece] += 1
    return counts


def _normalize_candidate(term: str) -> str:
    """候选词整形：只保留字，折叠内部空白；长度不足/超限返回空串。"""
    term = "".join(ch for ch in term if _is_word_char(ch))
    if not (MIN_LEN <= len(term) <= MAX_LEN):
        return ""
    return term


def _isolation_reason(chunk) -> str:
    """不可参与词表构建的 chunk（返回原因，空串表示可用）。

    archive/ 不在此统计——它在 chunk_kb_dir 遍历层就被 governance.is_indexable
    黑名单跳过，根本不会出现在 all_chunks 里（excluded_chunks 只含到达本层
    后被排除的 chunk）。
    """
    if (chunk.source_path or "").startswith("evolved/"):
        return "evolved"
    if (chunk.source_path or "") in F_CLASS_SOURCE_PATHS:
        return "f_class"
    if (chunk.authority or "") == "external_reference":
        return "external_reference"
    if (chunk.status or "") == "archived":
        return "archived"
    return ""


def _syllable_table_for(charset: set[str], syllable_of) -> dict[str, str]:
    """给定字符集 → {字: 无声调拼音}（如 份→fen）。

    覆盖语料字集 + 通用词典词汇字集：错字常常是**语料里从不出现**的字
    （富/迭/帖/飘），只建语料字集会让这类错字无法被拼音规则命中。
    同音判定在运行时按「音节相同」直接比较，不再预存同音字列表。
    """
    table: dict[str, str] = {}
    for ch in sorted(charset):
        if len(ch) != 1 or not _is_cjk(ch):
            continue
        syl = syllable_of(ch)
        if syl:
            table[ch] = syl
    return table


def _vocab_charset() -> set[str]:
    """通用词典（jieba，仅构建期依赖）里出现过的汉字集合。"""
    try:
        import jieba
    except ImportError as e:  # pragma: no cover - 构建期环境问题
        raise SystemExit(
            "构建词表需要 jieba（仅构建期依赖）：pip install -r requirements-dev.txt"
        ) from e
    jieba.initialize()
    chars: set[str] = set()
    for word, freq in jieba.dt.FREQ.items():
        if len(word) < 2 or freq < 1:
            continue
        for ch in word:
            if _is_cjk(ch):
                chars.add(ch)
    return chars


def build_lexicon(*, min_freq: int = 80, guard_min_freq: int = 5,
                  guard_max_len: int = 6, gather_max_len: int = MAX_LEN) -> dict:
    """从 KB 语料构建词表载荷（同名产物即本函数返回值的 JSON 序列化）。

    两套集合刻意分开：
    - ``terms``（改写目标）：文档名/标题片段 ∪ 词频 ≥ min_freq 的 n-gram，
      再经「片段词」过滤（只作更长短语内部片段出现的 n-gram 一律剔除）。
      阈值偏高，宁可漏改也不把 query 拉向泛化中文词（「一个」「可以」这类高频
      通用词在语料里同样高频，只能靠阈值挡）。
    - ``guard_ngrams``（真词护栏）：词频 ≥ guard_min_freq 的 n-gram，**比词表宽**。
      命中片段本身就在语料里出现过（如「时候」）时不改写——否则错字纠正会反过来
      破坏本来写对的 query。
    """
    from app.agent.rag.parsers import chunk_kb_dir

    kb_dir = Path(settings.kb_dir)
    if not kb_dir.is_absolute():
        kb_dir = ROOT / kb_dir
    all_chunks = chunk_kb_dir(kb_dir, strict=False)
    chunks = [c for c in all_chunks if not _isolation_reason(c)]
    skipped = Counter(
        _isolation_reason(c) for c in all_chunks if _isolation_reason(c)
    )
    texts = _iter_texts(chunks)
    corpus = "\n".join(texts)
    corpus_chars = {ch for ch in corpus if _is_cjk(ch)}

    meta_counts = _collect_from_metadata(chunks)
    harvest_floor = min(min_freq, guard_min_freq)
    per_len = _harvest_frequent_ngrams(texts, harvest_floor, gather_max_len)

    # 词频校验：n-gram 候选必须 ≥ min_freq；文档名/标题候选只需在语料中出现过
    # （它们是语料自身的规范命名，可能只出现一次，如「上门取件服务说明」，
    # 用频次门槛反而会丢掉最典型的领域词）。
    raw_terms: dict[str, int] = {}
    meta_terms: set[str] = set()
    for term in meta_counts:
        term = _normalize_candidate(term)
        if not term or term not in corpus:
            continue
        if term not in raw_terms:
            raw_terms[term] = corpus.count(term)
            meta_terms.add(term)
    for level in per_len.values():
        for term, freq in level.items():
            if freq >= min_freq:
                raw_terms[term] = max(raw_terms.get(term, 0), freq)
    # 片段词过滤只作用于 n-gram 来源：文档名/标题是语料自身的规范命名，
    # 即使高频出现在更长标题里（如「积分」在「积分兑换与抵扣规则」里）也要保留。
    ngram_terms = {
        t: f for t, f in raw_terms.items() if t not in meta_terms
    }
    terms = {t: raw_terms[t] for t in _maximal_terms(ngram_terms, per_len)}
    terms.update({t: raw_terms[t] for t in meta_terms})

    syllable_of = _pinyin_tables()
    guard = sorted({
        gram
        for level in per_len.values()
        for gram, freq in level.items()
        if freq >= guard_min_freq and len(gram) <= guard_max_len
    })

    char_syllables = _syllable_table_for(corpus_chars | _vocab_charset(), syllable_of)
    # 通用词典真词护栏：需要先有拼音表与词表才能判定「真词会不会被改写」，
    # 因此在拼音表之后构建，并与语料护栏合并。
    guard = sorted(set(guard) | set(_dict_guard({
        "terms": sorted(terms),
        "guard_ngrams": guard,
        "char_syllables": char_syllables,
        "term_freq": terms,
        "source_generation": {},
    })))

    corpus_sha = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    return {
        "protocol": "rag-query-lexicon-v1",
        "source_generation": {
            "kb_dir": str(settings.kb_dir),
            "corpus_sha256": corpus_sha,
            "documents": len({c.source_path for c in chunks}),
            "chunks": len(chunks),
            "excluded_chunks": dict(skipped),
            "terms_total": len(terms),
            "terms_from_metadata": len(meta_terms),
            "terms_from_ngram": len(terms) - len(meta_terms),
            "terms_dropped_fragment": len(raw_terms) - len(terms),
            "min_freq": min_freq,
            "guard_min_freq": guard_min_freq,
            "guard_max_len": guard_max_len,
            "min_len": MIN_LEN,
            "max_len": MAX_LEN,
        },
        "terms": sorted(terms),
        "term_freq": {t: terms[t] for t in sorted(terms)},
        "char_syllables": char_syllables,
        "guard_ngrams": guard,
        # 预留：全拼（拼音整词）匹配尚未启用，字段先占位以免产物协议再次变更
        "term_full_pinyin": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 query 表层规范化词表（工作流 A1）")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"产物路径（默认 {DEFAULT_OUT}）")
    parser.add_argument("--min-freq", type=int, default=80,
                        help="正文 n-gram 候选（改写目标）词频下限（默认 80："
                             "领域词需稳定出现才作改写目标）；文档名/标题词不受此限")
    parser.add_argument("--guard-min-freq", type=int, default=5,
                        help="真词护栏 n-gram 词频下限（默认 5：语料里出现过就护）")
    parser.add_argument("--guard-max-len", type=int, default=6,
                        help="真词护栏 n-gram 词长上限（默认 6，控制产物体积）")
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="词长上限（默认 8）")
    parser.add_argument("--report", action="store_true", help="只打印统计，不写产物")
    args = parser.parse_args()

    payload = build_lexicon(
        min_freq=args.min_freq, guard_min_freq=args.guard_min_freq,
        guard_max_len=args.guard_max_len, gather_max_len=args.max_len,
    )
    stats = payload["source_generation"]
    print("=" * 78)
    print("  query 词表构建（工作流 A1）")
    print("=" * 78)
    print(f"  语料      : {stats['kb_dir']}（{stats['documents']} 文档 / "
          f"{stats['chunks']} 块，排除 {stats['excluded_chunks'] or '无'}）")
    print(f"  词表      : {len(payload['terms'])} 词")
    print(f"  真词护栏  : {len(payload['guard_ngrams'])} n-gram")
    print(f"  拼音表    : {len(payload['char_syllables'])} 字音节")
    if args.report:
        return
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    # 紧凑序列化：产物含数万条护栏 n-gram，缩进会让体积翻倍；它是运行时资源
    # 而非人工审阅文件（统计与来源都落在 source_generation 字段里）。
    out.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  产物      : {out}（{out.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()

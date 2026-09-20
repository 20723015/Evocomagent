"""query 表层规范化：把「写坏的领域词」还原回词表规范形式（工作流 A2）。

动机（门禁轮 hard 76.6% 失分解剖）：领域词被同音错字（积份→积分、运废险→
运费险、手续废→手续费）或拼音首字母（YFX→运费险、店铺q→店铺券、运f→运费）
写坏后，向量检索与 BM25 都拿不到应有的召回。噪声出在 query 表面，就在 query
表面修：拿 KB 自身的高频领域词做一张词表（app/scripts/build_query_lexicon.py
构建），把 query 里能确认是「某词表词的写坏形式」的片段替换回规范词。

匹配规则（最长优先，每个位置只改一次）：
- 逐字接受三种形态：完全相同 / 同音字 / **query 侧 ASCII 字母 ↔ 词表汉字首字母**
  （YFX、店铺q、运f —— 即缩写与半缩写写法）；
- 同音替换数 ≤ max(1, len//2)（两字词只允许一个字错，避免整词漂移）；
- ASCII 缩写不限替换数（YFX 三个字母全缩写是常态）。
- 真词护栏：命中片段本身就在语料高频 n-gram 集里 → 整段跳过不改写
  （否则会把本来写对的 query 改坏）。

与计划的刻意偏差：**不启用「汉字↔汉字仅首字母相同」的替换**。同音字已经覆盖
错字场景（错字几乎都是同音/近音），而首字母相同但音节不同的汉字对（有房 ↔
运费）会把正常 query 改坏——ASCII 字母才是无歧义的缩写形态。单字词不参与同音。

工程边界：
- 运行时**零依赖**（词表已含首字母表/同音字表，不用 pypinyin）；
- 词表缺失/损坏 → 恒等返回 + 打点（fail-open，检索绝不因此中断）；
- 纯字符串处理，微秒级；不命中任何词表词时原样返回。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL = "rag-query-lexicon-v1"
MIN_LEN = 2
MAX_LEN = 8
KIND_ABBR = "abbr"
KIND_HOMOPHONE = "homophone"
KIND_MIXED = "mixed"


def _is_cjk(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def _is_abbr_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _ascii_runs(text: str) -> list[tuple[int, int]]:
    """query 中连续拉丁字母段的 [start, end) 区间（缩写只能整段消费）。"""
    runs: list[tuple[int, int]] = []
    start = -1
    for i, ch in enumerate(text):
        if _is_abbr_letter(ch):
            if start < 0:
                start = i
        elif start >= 0:
            runs.append((start, i))
            start = -1
    if start >= 0:
        runs.append((start, len(text)))
    return runs


def _run_atomic(runs: list[tuple[int, int]], start: int, end: int) -> bool:
    """[start, end) 是否与每个字母段「要么完全不相交、要么完整包含」。"""
    for run_start, run_end in runs:
        if end <= run_start or start >= run_end:
            continue
        if start <= run_start and end >= run_end:
            continue
        return False
    return True


@dataclass(frozen=True)
class Rewrite:
    """一次表层还原：query 中的 span → 词表 term。"""

    span: str
    term: str
    kind: str  # abbr | homophone | mixed


@dataclass(frozen=True)
class NormalizedQuery:
    """规范化结果：text 供检索使用，rewrites 供观测/测试。"""

    original: str
    text: str
    rewrites: tuple[Rewrite, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.rewrites)

    def kinds(self) -> list[str]:
        return [r.kind for r in self.rewrites]


@dataclass
class _Lexicon:
    """词表的内存形态：改写目标 + 护栏 + 拼音表（均由构建期产物提供）。"""

    terms: frozenset[str]
    guard: frozenset[str]
    # 字 → 无声调拼音（如 份→fen）。同音判定=音节相同；首字母=音节首字母。
    syllables: dict[str, str]
    freq: dict[str, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class QueryNormalizer:
    """词表驱动的 query 表层规范化器（无状态、线程安全、微秒级）。"""

    def __init__(self, lexicon: _Lexicon, *, max_len: int = MAX_LEN,
                 min_len: int = MIN_LEN):
        self._lex = lexicon
        self._min_len = min_len
        self._max_len = max_len
        # 候选索引：(长度, 每字 token 元组) → 词表词（按语料词频降序）。
        # token：汉字取其拼音首字母（同音字必然同首字母，故错字也能命中候选桶），
        # 无拼音的字符（生僻字/数字/字母）取自身——只支持完全相同。
        # 同一桶里可能有多个候选（首字母 x ↔ 险/效/胁/性），必须靠证据排序：
        # 语料词频最高的词才最可能是 query 想写的词（微信 94 > 威胁 64）。
        index: dict[tuple, list[str]] = {}
        for term in lexicon.terms:
            if not (min_len <= len(term) <= max_len):
                continue
            index.setdefault((len(term), self._tokens(term)), []).append(term)
        self._index = {
            key: tuple(sorted(vals, key=lambda t: (-lexicon.freq.get(t, 0), t)))
            for key, vals in index.items()
        }

    # ---------------- 装配 ----------------
    @classmethod
    def load(cls, path: str | Path) -> QueryNormalizer | None:
        """从产物文件加载；缺失/损坏返回 None（调用方 fail-open + 打点）。"""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("protocol") != PROTOCOL:
                return None
            lexicon = _Lexicon(
                terms=frozenset(payload.get("terms") or ()),
                guard=frozenset(payload.get("guard_ngrams") or ()),
                syllables=dict(payload.get("char_syllables") or {}),
                freq={k: int(v) for k, v in (payload.get("term_freq") or {}).items()},
                meta=dict(payload.get("source_generation") or {}),
            )
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        if not lexicon.terms:
            return None
        return cls(lexicon)

    @property
    def size(self) -> int:
        return len(self._lex.terms)

    @property
    def meta(self) -> dict:
        """构建期审计信息（corpus 指纹/词数/阈值），供 manifest 快照。"""
        return dict(self._lex.meta)

    def snapshot(self) -> dict:
        """进报告/manifest 的快照字段（不可泄漏语料内容，只有计数与指纹）。"""
        return {
            "enabled": True,
            "protocol": PROTOCOL,
            "num_terms": len(self._lex.terms),
            "num_guard_ngrams": len(self._lex.guard),
            "num_syllables": len(self._lex.syllables),
            "kb_dir": self._lex.meta.get("kb_dir", ""),
            "corpus_sha256": self._lex.meta.get("corpus_sha256", ""),
            "min_freq": self._lex.meta.get("min_freq"),
            "guard_min_freq": self._lex.meta.get("guard_min_freq"),
        }

    # ---------------- 规范化 ----------------
    def would_rewrite(self, span: str) -> bool:
        """该片段按当前匹配规则是否会被改写成词表词（≠原样）。

        供构建期用：把「通用词典里的真词」预查一遍，凡是会被改写的真词都进
        护栏集——真词不该被错字纠正改坏。也供单测直接断言改写判定。
        """
        for length in range(min(self._max_len, len(span)), self._min_len - 1, -1):
            if length != len(span):
                continue
            if span in self._lex.guard:
                return False
            for term in self._index.get((length, self._tokens(span)), ()):
                if term == span:
                    continue
                if self._verify(span, term) is not None:
                    return True
        return False

    def normalize(self, query: str) -> NormalizedQuery:
        """返回规范化后的 query（无改写时原样返回，零分配快路径）。"""
        text = query or ""
        if len(text) < self._min_len:
            return NormalizedQuery(original=query, text=query)
        runs = _ascii_runs(text)
        out: list[str] = []
        rewrites: list[Rewrite] = []
        i = 0
        n = len(text)
        prot_end = 0  # 已确认为规范词/真词的区间 [.., prot_end)，其内部不再匹配
        while i < n:
            length, term, kind, prot = self._match_at(text, i, runs, prot_end)
            if term is None:
                out.append(text[i:i + length])
                if prot:
                    prot_end = max(prot_end, i + prot)
            else:
                out.append(term)
                rewrites.append(Rewrite(span=text[i:i + length], term=term, kind=kind))
            i += length
        if not rewrites:
            return NormalizedQuery(original=query, text=query)
        result = NormalizedQuery(
            original=query, text="".join(out), rewrites=tuple(rewrites),
        )
        _record_hits(result.kinds())
        return result

    def _token(self, ch: str) -> str:
        """单字 token：有音节表的字取拼音首字母，其余取小写自身。

        同音字与首字母写法都必然落在同一个 token 上，于是「桶命中 → 逐字精校」
        两步走：桶把候选压到常数级，精校负责真正的判定。
        """
        if _is_cjk(ch):
            syl = self._lex.syllables.get(ch)
            return syl[0] if syl else ch
        return ch.lower()

    def _tokens(self, text: str) -> tuple:
        return tuple(self._token(ch) for ch in text)

    def _match_at(self, text: str, i: int, runs: list[tuple[int, int]],
                  prot_end: int = 0,
                  ) -> tuple[int, str | None, str, int]:
        """位置 i 上的最长匹配 → (前进长度, 词表词或 None, 改写类型, 真词区长度)。

        term=None 且 prot>0：命中真词/规范词，只前进 1 字并声明保护区
        [i, i+prot)——保护区内不再匹配，但跨出保护区边界的更短候选仍可尝试
        （「大促价」是真词拦住 3 字，不影响后面的「价b→价保」）。
        同长度多候选取**替换数最少**者（微信 1 个缩写位优于 维修 的
        同音+缩写两位），再按语料词频（微信 94 > 威胁 64）。
        """
        for length in range(min(self._max_len, len(text) - i), self._min_len - 1, -1):
            if i + length <= prot_end:
                continue  # 完全落在已确认真词区间内：不改写
            span = text[i:i + length]
            if span in self._lex.guard or span in self._lex.terms:
                return 1, None, "", length
            if not _run_atomic(runs, i, i + length):
                # 缩写字母串是一个整体，不允许被两个改写切开（「商家送YFX能」里
                # YF 与 X 分属两个词必然是错的）：退到更短的候选，让
                # 「YFX→运费险」这类完整消费字母串的匹配胜出。
                continue
            best_key: tuple | None = None
            best_term: str | None = None
            best_kind = ""
            # 已确认真词的覆盖位（换算成 span 内偏移）：只许原样匹配
            frozen_to = max(0, min(prot_end, i + length) - i)
            for term in self._index.get((length, self._tokens(span)), ()):
                verdict = self._verify(span, term, frozen_to=frozen_to)
                if verdict is None:
                    continue
                kind, subs = verdict
                key = (subs, -self._lex.freq.get(term, 0), term)
                if best_key is None or key < best_key:
                    best_key, best_term, best_kind = key, term, kind
            if best_term is not None:
                return length, best_term, best_kind, 0
        return 1, None, "", 0

    def _verify(self, span: str, term: str, *, frozen_from: int = 0,
                frozen_to: int = 0) -> tuple[str, int] | None:
        """逐字校验 span 是否为 term 的可接受写坏形式。

        返回 (改写类型, 替换数) 或 None；替换数用于同长度多候选时择优。
        [frozen_from, frozen_to) 是已确认为真词/规范词的字符区间：这些位置
        只接受原样匹配，不许替换——保护区的字符不该被本轮改写动到。
        """
        homophones = 0
        abbrs = 0
        for offset, (got, want) in enumerate(zip(span, term)):
            if frozen_from <= offset < frozen_to:
                if got != want:
                    return None
                continue
            if got == want or got.lower() == want.lower():
                continue
            if _is_abbr_letter(got) and _is_cjk(want):
                # 拼音首字母/半缩写：YFX→运费险、店铺q→店铺券、运f→运费
                want_syl = self._lex.syllables.get(want)
                if want_syl and want_syl[0] == got.lower():
                    abbrs += 1
                    continue
                return None
            if _is_cjk(got) and _is_cjk(want):
                # 同音：无声调音节相同（积份→积分、退宽→退款、输错误次里的
                # 误/五同音）。任一字无音节记录（生僻字）不参与同音。
                got_syl = self._lex.syllables.get(got)
                if got_syl and got_syl == self._lex.syllables.get(want):
                    homophones += 1
                    continue
                return None
            return None
        if homophones == 0 and abbrs == 0:
            return None
        if len(term) > 1 and homophones > max(1, len(term) // 2):
            return None
        if homophones and abbrs:
            return KIND_MIXED, homophones + abbrs
        return (KIND_ABBR if abbrs else KIND_HOMOPHONE), homophones + abbrs


def normalize_query(query: str,
                    normalizer: QueryNormalizer | None = None) -> str:
    """便捷入口：未装配规范化器时恒等返回（线上默认路径零开销）。"""
    if normalizer is None:
        return query
    return normalizer.normalize(query).text


def _record_hits(kinds: Iterable[str]) -> None:
    try:
        from app.observability.metrics import record_query_normalize_hits

        record_query_normalize_hits(kinds)
    except Exception:  # noqa: BLE001 - 观测失败绝不影响检索
        pass


def load_normalizer(path: str | Path,
                    *, enabled: bool = True) -> QueryNormalizer | None:
    """装配入口：enabled=False / 词表缺失/损坏 → None + fail-open 打点。"""
    if not enabled:
        return None
    normalizer = QueryNormalizer.load(path)
    if normalizer is None:
        try:
            from app.observability.metrics import record_query_normalize_missing

            record_query_normalize_missing(str(path))
        except Exception:  # noqa: BLE001
            pass
    return normalizer

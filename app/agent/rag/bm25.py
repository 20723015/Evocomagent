"""手写 BM25 稀疏检索：零第三方依赖，中文按二元组（bigram）切分。

为什么有了向量检索还要 BM25（7.2 混合检索的动机）：
- 向量检索擅长"语义相似"，但对"精确术语"（SKU 编号、法条原文、专有名词）
  经常漏召回——"七天无理由退货"按词面匹配时，向量可能只给中等分数。
- BM25 对词面重合极其敏感：含查询原词的段落必然排在前面，恰好补上向量盲区。
- 与向量路做 RRF 融合后两路互补，这正是 7.2 的落地方式（见 hybrid.py）。

中文分词策略（不引入 jieba，保持零依赖）：
- ASCII 字母/数字：连续串切一个词，统一转小写（聚合时大小写不敏感）。
- 连续 CJK 字符：按二元组切分；落单的单个 CJK 字符保留为单字 token
  （避免"退"这种单字查询完全无 token 可匹配）。
- 标点与空白：只作分隔符，不产生 token，二元组不跨分隔符。

注音示例（方便教学时口头讲解）：
- "退货政策"（tuì huò zhèng cè）→ ["退货", "货政", "政策"]
- "七天无理由"（qī tiān wú lǐ yóu）→ ["七天", "天无", "无理", "理由"]
- "7天无理由" → ["7", "天无", "无理", "理由"]
- "Apple发货" → ["apple", "发货"]
- "" → []
"""

from __future__ import annotations

import math
import re

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk

# 一个 token：ASCII 字母/数字连续串，或连续 CJK 字符块
#（交替捕获保证 CJK 不会被 ASCII 分支吞掉）
_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[\u4e00-\u9fff]+")

# Okapi BM25 标准超参
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """把文本切成检索 token：ASCII 词 + CJK 二元组（含注音示例见模块 docstring）。"""
    tokens: list[str] = []
    for block in _TOKEN_RE.findall(text):
        if block.isascii():
            tokens.append(block.lower())
        elif len(block) >= 2:
            # 连续 CJK → 二元组：如 "退货政策" → 退货/货政/政策
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
        else:
            # 落单的单个 CJK 字符 → 单字 token
            tokens.append(block)
    return tokens


class BM25Index:
    """零依赖 BM25 索引：构造时统计词频/文档频率，search 时在线打分。

    用法：HybridRetriever 建索引时从向量后端取回全部 chunk 一次性喂入；
    本类只依赖传入 chunks 的 text 字段做词法统计，与向量后端完全解耦。
    """

    def __init__(self, chunks: list[Chunk]):
        self._chunks = list(chunks)
        self._doc_tfs: list[dict[str, int]] = []  # 每个 chunk 的词频表
        self._doc_lens: list[int] = []            # 每个 chunk 的 token 数
        self._df: dict[str, int] = {}             # 文档频率（含该词的 chunk 数）
        self._avg_dl = 0.0
        self._build()

    @property
    def chunks(self) -> list[Chunk]:
        """建索引用到的全部 chunk（与向量后端 chunks() 对应）。"""
        return self._chunks

    def _build(self) -> None:
        df: dict[str, int] = {}
        total_len = 0
        for chunk in self._chunks:
            terms = tokenize(chunk.text)
            tf: dict[str, int] = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            self._doc_tfs.append(tf)
            self._doc_lens.append(len(terms))
            total_len += len(terms)
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._avg_dl = total_len / len(self._chunks) if self._chunks else 0.0

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """按 BM25 打分返回 Top-K 命中；空 query 或空语料返回 []。

        分数公式（k1=1.5, b=0.75）：
            idf = ln((N - n + 0.5) / (n + 0.5) + 1)   # 加 1 保证非负
            score = Σ idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avg_dl))

        词面零重合（score=0）的文档不进结果列表——与 ES 的 BM25 查询语义一致，
        否则 RRF 会把无关文档误算成"单路命中"，污染融合排名。
        """
        if not query or not query.strip() or not self._chunks:
            return []
        # 查询词去重：同一 bigram 重复出现只贡献一次分数
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms:
            return []

        n_docs = len(self._chunks)
        avg_dl = self._avg_dl or 1.0  # 空文档集兜底，避免除零
        scored: list[tuple[int, float]] = []
        for idx, (tf, dl) in enumerate(zip(self._doc_tfs, self._doc_lens)):
            score = 0.0
            for term in q_terms:
                n = self._df.get(term, 0)
                doc_tf = tf.get(term, 0)
                if n == 0 or doc_tf == 0:
                    continue
                idf = math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0)
                denom = doc_tf + K1 * (1 - B + B * dl / avg_dl)
                score += idf * (doc_tf * (K1 + 1)) / denom
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(chunk=self._chunks[idx], score=score)
            for idx, score in scored[:top_k]
        ]
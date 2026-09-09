# retrieval-a7c9749 — 递归分块 + 统一最终检索口径重新评测

## 结论

a7c9749（RAG 递归分块与多格式解析补全）下，40 份业务文档 + 1 篇自进化知识重建为 **183 个 chunk**
（旧实现 164+1=165）。统一最终检索口径（3 倍候选 → 阈值门控 → parent_id 去重 → Top-K）下，
dev 535 条整体指标较旧版（96.1% / 90.3% / 91.6%）**小幅上升**：

| 指标 | 旧版 c5bfd0e | a7c9749（本轮） | 变化 |
| --- | --- | --- | --- |
| 正例 Recall@5（520 条） | 96.1% | **96.4%** | +0.3pp |
| 正例 MRR | 90.3% | **91.3%** | +1.0pp |
| 正例 nDCG@5 | 91.6% | **92.4%** | +0.8pp |
| easy Recall@5（499 条） | 98.1% | **98.2%** | +0.1pp |
| hard Recall@5（21 条） | 47.6% | **54.8%** | +7.2pp |
| 负例拒绝率（15 条） | 93.3% | **86.7%** | -6.6pp |

门禁仍有两项未达标（如实登记，不改门槛）：
- 困难正例 recall@k 54.8% < 80.0%
- 负例拒绝率 86.7% < 90.0%

（hard 与负例样本量小，波动大；整体正例三项均达标且高于旧版。）

## 阈值

- 本轮校准阈值：**RAG_MIN_RELEVANCE_SCORE = 0.052327625**（旧阈值 0.091220066 不复用）。
  校准口径：easy 正例 recall@k 98.2%、负例拒绝率 86.7%（正例召回约束 ≥98%、负例拒绝 ≥80%
  中取满足约束的最高阈值，见 `calibrate_threshold`）。
- dev-final 报告 `online_threshold_match=true`：applied == online == hard 同值 0.052327625，
  hard 未显式覆盖。
- holdout 报告 `online_threshold_match=true`。

## 环境与索引

- 提交：`a7c9749`（worktree `tmp/wt-a7c9749`，manifest 内 `git.commit=a7c974901c2f24...`）
- ES：8.11.4（单节点，trial license RRF），`http://localhost:19200`
- 索引：`ecom-kb-20260903142911-a12df5e1`（generation `20260903142911-a12df5e1`，
  alias `ecom-kb-active` 已切换；构建时 ES/Reranker 相关单测全绿）
- 验收：41 个 source_path（40 业务文档 + 1 evolved）、183 chunk、chunk_id 去重无误
  （183 hits / 183 unique）、evolved 知识 1 chunk 在场、无解析跳过
- Reranker：bge-reranker-v2-m3（TEI，本地容器 `http://127.0.0.1:8001`，GPU）；全程无 fallback
- Embedding：bge-m3（SophNet EasyLLM，1024 维）
- 检索装配：ES 原生 hybrid（BM25 + kNN，RRF，recall_k=60）+ bge-reranker-v2-m3 精排 +
  统一 `final_search`（3 倍候选≤15 → 阈值门控 → parent_id 去重 → Top-K）

## 数据集（未变化，SHA 校验通过）

- dev：`app/evaluation/retrieval_cases.json`，535 条（easy 499 / hard 21 / 负例 15），
  SHA-256 `ec4a977e82ca9eff0adac55b77d43f45c7c43c793316272e7ae5034f72b9c949`
- holdout：`app/evaluation/holdout_cases.json`，120 条（easy 80 / hard 20(含3负例) / 负例 20，
  实际分层：正例 97 / 负例 23）

## 结果（三份报告）

| 运行 | 数据集 | Recall@5 | MRR | nDCG@5 | easy Recall | hard Recall | 负例拒绝 | online_threshold_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hybrid-rerank-dev-cal` | dev 535 | 96.4% | 91.3% | 92.4% | 98.2% | 54.8% | 86.7% | false（校准实验，未部署） |
| `hybrid-rerank-dev-final` | dev 535 | 96.4% | 91.3% | 92.4% | 98.2% | 54.8% | 86.7% | **true** |
| `hybrid-rerank-holdout` | holdout 120 | 87.6% | 81.0% | 82.7% | 93.8% | 58.8% | 82.6% | **true** |

- dev-final 与 dev-cal 数值一致（同一候选/分数，阈值同值），dev-final 记为线上口径唯一依据。
- holdout 为配置冻结后一次性运行，未依据 holdout 调参。

## 自进化知识 Top-1 复测（同一最终检索入口）

查询「钻石会员专属客服的响应时效SLO是多少」→ `final_search`（hybrid-rerank + 阈值门控）：

1. 0.999591 — `evolved/20260901-938835580ee4-钻石会员专属客服的响应时效SLO是多少.md` ← Top-1 ✓
2. 0.083890 — 会员权益.md
3. 0.012241 — 价保与赔付规则.md

自进化幂等发布逻辑未变更，本轮未重跑发布链路，仅复测检索命中。

## 运行补丁说明

- `app/agent/rag/embedder.py`（a7c9749 工作树本地补丁，未并入报告所引提交）：
  `SophnetEmbedder._post` 增加 5xx/429/连接异常自动重试（最多 6 次，指数退避），
  用于抵抗 SophNet 上游偶发 503（ConnectionReset）。检索与指标语义不受影响（幂等重发），
  单测 `tests/unit/test_sophnet_embedder.py` 16 例全绿。
- 其余代码与 a7c9749 完全一致；单测：chunker/retriever/metrics 相关（test_chunker_headings、
  test_chunker_evolved、test_retriever_factory、test_retrieval_metrics、test_bm25_hybrid、
  test_phase7_ingest_rerank、test_sophnet_embedder）全部通过。

## 复现命令

```bash
# 1) 重建索引（a7c9749 worktree）
RAG_BACKEND=es ES_URL=http://localhost:19200 KB_WRITE_LOCK_BACKEND=file \
  python -m app.scripts.build_kb_index --backend es

# 2) 校准（dev）
RAG_BACKEND=es ES_URL=http://localhost:19200 RERANK_ENDPOINT_URL=http://127.0.0.1:8001 \
  RAG_HYBRID=true \
  python -m app.scripts.run_retrieval_eval --dataset app/evaluation/retrieval_cases.json \
  --variant hybrid-rerank --calibrate \
  --json-out artifacts/eval/v2/retrieval-a7c9749/hybrid-rerank-dev-cal/report.json

# 3) 固定阈值 dev（沉淀为线上配置 RAG_MIN_RELEVANCE_SCORE=0.052327625）
RAG_BACKEND=es ES_URL=http://localhost:19200 RERANK_ENDPOINT_URL=http://127.0.0.1:8001 \
  RAG_HYBRID=true RAG_MIN_RELEVANCE_SCORE=0.052327625 \
  python -m app.scripts.run_retrieval_eval --dataset app/evaluation/retrieval_cases.json \
  --variant hybrid-rerank --min-score 0.052327625 \
  --json-out artifacts/eval/v2/retrieval-a7c9749/hybrid-rerank-dev-final/report.json

# 4) holdout（配置冻结后仅一次）
RAG_BACKEND=es ES_URL=http://localhost:19200 RERANK_ENDPOINT_URL=http://127.0.0.1:8001 \
  RAG_HYBRID=true RAG_MIN_RELEVANCE_SCORE=0.052327625 \
  python -m app.scripts.run_retrieval_eval --dataset app/evaluation/holdout_cases.json \
  --variant hybrid-rerank --min-score 0.052327625 \
  --json-out artifacts/eval/v2/retrieval-a7c9749/hybrid-rerank-holdout/report.json
```

## 文件清单

- `hybrid-rerank-dev-cal/report.json` + `manifest.json`（校准实验，非线上口径）
- `hybrid-rerank-dev-final/report.json` + `manifest.json`（线上口径唯一依据）
- `hybrid-rerank-holdout/report.json` + `manifest.json`（一次性 holdout）
# retrieval C 批 — Docker 评测口径（ES + bge-reranker-v2-m3）归档报告

- 日期：2026-09-18
- 依据：本次实测（C1~C5），可复现命令见文末；本报告数字逐项取自下列归档产物，未做二次四舍五入改写。
- 范围：候选索引构建 → dev 959 release 门禁 → 激活 → 拒绝参数重校准 → holdout 250 一次性评测。
- 相关文档：`docs/v3.5-负例语料补强-立项.md`（负例侧结论立项）、`artifacts/eval/v3/RESULT-gate.md`（2026-09-10 归档基线）、`docs/P1-5-治理元数据消费侧-立项评估.md`（§5 四臂协议）。

## 0. 结论摘要

- **候选索引 `20260918080753-74c8a721` 已构建并激活**（781 chunk，`config_fingerprint=64b1c95282fd1250`，`PARSER_CHUNKER_VERSION=2026-09-15.1`）；激活前 ES 中只有候选索引，代际指针指向的旧索引已随容器重建丢失（检索路径 404，证据见 §2）。
- **C2 dev 959（候选索引）**：hard 92.07%（≥80% PASS）、P95 484.6ms（<500ms PASS）、degraded 0；正例 94.69%（−0.31pp）、easy 96.43%、MRR 79.25%、nDCG 82.70% 未达 release 下限；负例拒绝 0% 为评测路径口径（见 §6 口径说明 1）。
- **C5 holdout 250（一次性）**：正例 97.0%（+3.0pp vs 归档）、hard 94.0%（+8.0pp）、nDCG 90.62%（+2.75pp）、P95 475.6ms；MRR 88.42% 仍低于 90% 下限（见 §6 口径说明 2）。
- **C4 拒绝参数重校准**：冻结 `min_top1=0.206 / min_gap=0.0 / min_coverage=0.0`；dev 784 负例拒绝 58.33% / 正例误拒 3.63%，holdout 175 首次真跑 60.0% / 5.33%；双门禁（负例 ≥90% 且正例误拒 ≤5%）不可达 → `feasible=false` / `fallback=best_effort` 如实冻结。

## 1. 口径与 env 清单

构建与评测逐键一致（来源：`artifacts/eval/v3/es-hybrid-rerank/env-snapshot.txt`）：

| 键 | 值 |
| --- | --- |
| `RAG_BACKEND` | `es` |
| `RAG_HYBRID` | `true` |
| `RAG_HYBRID_RECALL_K` | `60` |
| `RAG_RERANK` | `bge-reranker-v2-m3` |
| `RERANK_ENDPOINT_URL` | `http://localhost:8001/rerank`（TEI，GPU） |
| embedding | SophNet `bge-m3`（1024 维） |
| `ES_URL` | `http://localhost:19200` |
| `ES_INDEX_PREFIX` | `ecom` |
| `PARSER_CHUNKER_VERSION` | `2026-09-15.1` |
| `config_fingerprint` | `64b1c95282fd1250` |
| `rag_query_normalize` | `true`（运行时默认；评测变体行为见 §3 注） |
| 数据集 | dev `app/evaluation/retrieval_cases_v3.json`（959：838 正例 + 121 负例）；holdout `app/evaluation/holdout_cases_v3.json`（250：200 正例 + 50 负例） |

> `env-snapshot.txt` 采于 C1/C2 时点，其中 `rag_rejection_min_top1 = 0.549` 为旧值；C4 重校准后 settings 默认值已同步为 `0.206`（`app/config/settings.py:121`）。

金丝雀 rerank 留档（验收复测，2026-09-18）：TEI GPU（`x-compute-type: gpu+optimized`），
query「七天无理由退货的政策是什么」× [退货政策原文, 会员权益, 配送时效] →
**0.9622 / 0.0008 / 0.0001**——强匹配 0.9+、弱匹配 <0.01，`min_top1=0.206` 落在空档。

## 2. 构建与激活记录（C1 / C3）

**C1 候选索引构建**（`--no-activate`，不切指针）：

- generation：`20260918080753-74c8a721`，target：`ecom-kb-20260918080753-74c8a721`
- 781 chunk；`config_fingerprint=64b1c95282fd1250`；`PARSER_CHUNKER_VERSION=2026-09-15.1`
- 候选描述 `candidate.json` 中 `previous_generation_id=""`、`expected_active_generation_id=20260908094339-b0c865f2`
- 构建日志留痕：`build.log`（2026-09-18T00:07:35Z 起）；PDF 表格抽取回退纯文本 1 次（`平台治理与处罚总则.pdf`，`ModuleNotFoundError`，走 pypdf 回退）

**重要发现（激活前检索路径 404）**：激活前 ES 中只有候选索引——代际指针指向的 `ecom-kb-20260908094339-b0c865f2` 在 ES 中不存在（容器重建导致旧索引丢失）。证据：`artifacts/eval/v3/es-hybrid-rerank-live/eval.log`（2026-09-18T00:26:54Z，`index_not_found_exception: no such index [ecom-kb-20260908094339-b0c865f2]`）。即激活前线上 ES 检索路径是 404；C2/C5 走 `--candidate` 直接评测候选索引，不依赖指针。

**C3 激活**：Redis 代际指针 `es → 20260918080753-74c8a721`，`previous_generation_id=20260908094339-b0c865f2`（可回滚）。

**运维注意**：代际指针存在 Redis；激活后文件回落副本 `app/sessions/kb_generations.json` **未同步更新**（其中 `es.generation_id` 仍为旧值 `20260908094339-b0c865f2`）。Redis 不可用时读到的会是陈旧指针，指向已不存在的索引。后续重建/激活后应同步核对 Redis 指针与文件副本，或修复回落写回路径。

## 3. C2 dev 959 门禁（候选索引，`--variant hybrid-rerank --release-profile`）

| 指标 | 结果 | release 门禁 | 判定 |
| --- | --- | --- | --- |
| 正例 recall@5（n=838） | 94.69% | ≥95% | FAIL（−0.31pp） |
| easy（n=504） | 96.43% | ≥98% | FAIL |
| hard（n=334） | **92.07%** | ≥80% | **PASS** |
| MRR | 79.25% | ≥90% | FAIL |
| nDCG@5 | 82.70% | ≥90% | FAIL |
| 负例拒绝（n=121） | 0%（评测路径未施加门控） | ≥90% | FAIL |
| P95 延迟 | 484.6ms | <500ms | PASS |
| degraded | 0 | — | — |

分层 recall@5（按 tags 的逐例 recall 均值；与 `report.json` 的 `cases[].recall_at_k` 一致）：

| tag | recall@5 | n |
| --- | --- | --- |
| multi_intent | 83.3% | 51 |
| pinyin | 85.0% | 20 |
| noisy | 88.2% | 51 |
| timing | 90.5% | 42 |
| hard | 92.1% | 334 |
| indirect | 92.5% | 40 |
| condition | 94.6% | 56 |
| multihop | 95.5% | 33 |
| easy | 96.4% | 504 |
| direct | 97.0% | 489 |
| typo | 100% | 30 |
| ellipsis | 100% | 20 |

**全对用例 786/838 = 93.8%**。判读：失分集中在 multi_intent / pinyin / noisy —— 查询侧歧义，不是索引侧。

> 注：本次 `--variant hybrid-rerank` 在评测脚本内**显式关闭** `rag_query_normalize`（`app/scripts/run_retrieval_eval.py::_apply_variant`；manifest 记录 `query_normalize.enabled=false`、`query_normalize_rewrites=0`、逐例 `normalized_query=null`）。因此 typo 100% 是 hybrid（BM25+kNN+RRF）+ bge-reranker-v2-m3 本身的能力，**不是**查询规范化带来的；规范化的增量需用 `--variant hybrid-rerank-norm` 单独测量，本批未跑该臂。

## 4. C4 拒绝参数重校准（激活后线上口径）

命令：`python -m app.scripts.calibrate_rejection --dev`（holdout 首次真跑见下）。

- 冻结参数：`min_top1=0.206` / `min_gap=0.0` / `min_coverage=0.0`（`min_rerank=None`，4 号信号未纳入网格）
- 旧值 `0.549` 是 numpy 纯向量余弦尺度；挂精排后量纲不可比，已废弃
- dev 784 例：负例拒绝 **58.33%** / 正例误拒 **3.63%**；`feasible=false`，`fallback=best_effort`
- holdout 175 例（**首次真跑**，`holdout_used=true`）：负例拒绝 **60.0%** / 正例误拒 **5.33%**（与 dev 一致 → 未过拟合）
- 产物：`artifacts/eval/v2/retrieval-rejection/params.json`（`dataset_hash=98c0fecbeffd461e`）
- settings 默认值已同步为 `0.206`（`app/config/settings.py:121`）

## 5. C5 holdout 250（一次性，`--variant hybrid-rerank --release-profile`）

| 指标 | 本次 | 归档（2026-09-10） | Δ |
| --- | --- | --- | --- |
| 正例 recall@5（n=200） | **97.0%** | 94.0% | +3.0pp |
| easy（n=150） | **98.0%** | 96.67% | +1.33pp |
| hard（n=50） | **94.0%** | 86.0% | +8.0pp |
| MRR | **88.42%** | 85.75% | +2.67pp |
| nDCG@5 | **90.62%** | 87.87% | +2.75pp |
| 负例拒绝（n=50） | 0%（评测路径未施加门控） | 80.0% | 口径不同，见 §6 |
| P95 延迟 | 475.6ms | — | — |

holdout 门禁判定：正例 ✓（≥95%）、easy ✓（恰好 ≥98%）、hard ✓（≥80%）、nDCG ✓（≥90%）、**MRR 88.42% < 90% FAIL**、负例拒绝 FAIL（未配阈值）。

## 6. 口径说明（诚实边界）

**说明 1：负例拒绝在标准评测 CLI 里恒为 0%。** 联合拒绝（4 信号）接在工具路径 `app/agent/tools/knowledge.py` 的 `search_knowledge`，而检索评测走 `retriever_factory.final_search`，两条路径不同 → 评测报告测不到联合门控；当前配置下 `RAG_MIN_RELEVANCE_SCORE` 未配置（联合门控取代单阈值），因此 C2/C5 的负例拒绝均为 0%。生产侧的负例拒绝数字来自 `calibrate_rejection.py`（C4：dev 58.33% / holdout 60.0%）。归档的 80.0% 是**单阈值**（`--min-score 0.261897`）口径，与联合门控不是同一测量，**不可直接相减比较**。

**说明 2：MRR ≥90% / nDCG ≥90% 是 release profile 的固定下限（只升不降），仓库历史归档从未达到。** v3 归档为 MRR 71.4% / nDCG 74.1%（`artifacts/eval/v3/RESULT-gate.md`）。本次 79.25%（dev）/ 88.42%（holdout）是历史最好，但仍未达该下限 → 属**语料/查询侧上界**，不是索引回归；门禁判定按 FAIL 如实留档，不降门槛。

## 7. 与归档基线的对比

- **holdout 250（同数据集、同变体）**：正例 94.0% → 97.0%、hard 86.0% → 94.0%、MRR 85.75% → 88.42%、nDCG 87.87% → 90.62%。但归档轮是单阈值 `0.261897` 口径（含阈值过滤），本次未配 `--min-score`；负例侧不可比（说明 1）。
- **dev 959**：归档门禁轮（阈值 0.261897，2026-09-10）为 84.6% / 89.9% / 76.6% / 71.4% / 74.1% / 66.9%；本次候选索引无阈值口径为 94.69% / 96.43% / 92.07% / 79.25% / 82.70% / 0%。两轮索引代际与阈值口径均不同（本次 781 chunk、`20260918080753-74c8a721`；归档 836 chunk、`20260910012731-65a69d8e`），只能作为方向性对照，不构成同口径提升结论。
- **分层失分结构**：归档轮（严格口径）最弱为 confusion 33.3%（n=6）、pinyin 55.0%、noisy 62.7%、typo 63.3%；本次（均值口径）最弱为 multi_intent 83.3%、pinyin 85.0%、noisy 88.2%，typo 已到 100%。查询侧噪声/歧义仍是主要失分层，与归档结论同源。

## 8. 产物清单（C1~C5）

| 步骤 | 内容 | 产物 |
| --- | --- | --- |
| C1 | 候选索引构建（`--no-activate`） | `artifacts/eval/v3/es-hybrid-rerank/candidate.json`、`build.log`、`env-snapshot.txt` |
| C2 | dev 959 门禁评测 | `artifacts/eval/v3/es-hybrid-rerank/report.json`、`eval.log`、`manifest.json` |
| C3 | 激活 + 激活前 404 留痕 | `artifacts/eval/v3/es-hybrid-rerank-live/eval.log`；Redis 指针记录（本报告 §2） |
| C4 | 拒绝参数重校准 | `artifacts/eval/v2/retrieval-rejection/params.json`（含 `dev_signals.json` 为更早 numpy 轮遗留，非本批 ES 口径） |
| C5 | holdout 250 一次性评测 | `artifacts/eval/v3/holdout-250-es/report.json`、`eval.log`、`manifest.json` |

## 9. 可复现命令链

```bash
# 0) 依赖（MySQL/Redis/ES 8.11.4/TEI reranker）——本机 Python 为 .venv/Scripts/python.exe（3.12）
docker compose -f deploy/compose/docker-compose.yml up -d

# 1) 迁移
python -m app.scripts.migrate_db

# 2) C1 候选索引构建（不激活；需 RAG_BACKEND=es / RAG_HYBRID=true /
#    RAG_HYBRID_RECALL_K=60 / RAG_RERANK=bge-reranker-v2-m3 / RERANK_ENDPOINT_URL=http://localhost:8001/rerank）
python -m app.scripts.build_kb_index --backend es --no-activate \
  --json-out artifacts/eval/v3/es-hybrid-rerank/candidate.json

# 3) C2 dev 959 门禁（候选索引）
python -m app.scripts.run_retrieval_eval \
  --dataset app/evaluation/retrieval_cases_v3.json \
  --variant hybrid-rerank --release-profile \
  --candidate artifacts/eval/v3/es-hybrid-rerank/candidate.json \
  --json-out artifacts/eval/v3/es-hybrid-rerank/report.json

# 4) C3 激活候选（切 Redis 代际指针；旧代保留可回滚）
python -m app.scripts.build_kb_index \
  --activate-candidate artifacts/eval/v3/es-hybrid-rerank/candidate.json

# 5) C4 拒绝参数重校准（dev 冻结 → holdout 一次性验证）
python -m app.scripts.calibrate_rejection --dev
python -m app.scripts.calibrate_rejection --holdout

# 6) C5 holdout 250 一次性评测
python -m app.scripts.run_retrieval_eval \
  --dataset app/evaluation/holdout_cases_v3.json \
  --variant hybrid-rerank --release-profile \
  --candidate artifacts/eval/v3/es-hybrid-rerank/candidate.json \
  --json-out artifacts/eval/v3/holdout-250-es/report.json
```

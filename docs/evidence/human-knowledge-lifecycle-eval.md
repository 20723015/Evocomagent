# 人工客服知识链路（010）真实环境验收报告

- 日期：2026-09-08
- 协议：MySQL（compose, 001→010 全迁移）+ Elasticsearch 8.11（compose, localhost:19200）
  + 真实 LLM（OPENAI_BASE_URL 抽取/评审）+ 真实 embedding（SophNet bge-m3, 1024 维）
  + 真实检索链路（ES hybrid BM25+kNN+RRF，rerank=none，与 .env 线上配置一致）
- 驱动：`docs/evidence/human-knowledge-lifecycle/real_run_human.py`
  （接入→评审→批准→发布→命中验证）；
  `docs/evidence/human-knowledge-lifecycle/run_retrieval_eval_resilient.py`（536 例检索评测，embedder 层 8 次指数退避重试）。

## 1. 检索质量：发布前后对比（536 例，同一口径）

发布物：人工链路批准的 1 篇新文档
`evolved/20260908-human-1-全屋定制衣柜的下单后流程和时间节点是怎样的.md`
（题材"全屋定制量尺/安装流程"为 KB 零覆盖新知识；基线索引 183 chunks → 发布后 185 chunks）。

| 指标 | 基线 | 发布后 | Δ |
| --- | --- | --- | --- |
| 正例 recall@5 | 98.7% | 98.7% | +0.0pp |
| 正例 MRR | 90.5% | 90.6% | +0.1pp |
| 正例 nDCG@5 | 92.4% | 92.5% | +0.1pp |
| easy recall@5（500 例） | 98.9% | 98.9% | +0.0pp |
| hard recall@5（21 例） | 92.9% | 92.9% | +0.0pp |
| 逐例回退用例数 | — | **0** | — |

- 口径说明：本机未配置 RAG_MIN_RELEVANCE_SCORE，负例拒绝率两轮同为 0%（不做门控）；
  前后两轮口径完全一致，Δ 值有效。负例拒绝率指标需先下发校准阈值（eval-v2 口径 0.0912）。
- 结论：经新链路发布 1 篇人工知识，536 例**零回退**，MRR/nDCG 微升。

## 2. 新知识线上命中（"批准后 Top-5 排名第一"在新链路复现）

发布后 generation：`ecom-kb-20260908103320-85cd67e9`（alias 原子切换自
`ecom-kb-20260908094339-b0c865f2`），ES 直查验证：

- 「全屋定制什么时候上门量尺？多久能安装？」→ 新文档 **第 1 名**
- 「上门量尺收费吗？量完不下单呢？」→ 新文档 **第 1 名**

旧验收（eval-v2，2026-09-01，0.9075 命中第 1）走的是 ledger 审批通道；
本报告为 **010 MySQL 链路**（审批快照 → Worker → 探针 → alias → CAS 结算）的复现。

## 3. 全链路运行记录（真实数据）

- **接入**：会话 `acceptance/measure-install-002`（4 条消息）写 MySQL 正本，created；
- **评审**（真实 LLM 抽取 + 真实 embedding 双侧去重）：2 条候选，证据绑定
  `["a0"]` / `["a0","a1"]`，evidence_state=ok；双侧 top-1 相似度
  q ∈ [0.57, 0.60]、a ∈ [0.65, 0.67]（均低于 0.90 阈值，无误判重复）；
  价值 0.95 / 0.9，分类 new → pending_review；
- **批准**：候选 #1 审批快照冻结（digest `258e3e6f59d9…`，approved_by=acceptance-runner）；
- **发布**：单批次 completed；strict ES 重建 → 存在性探针通过 → 激活前复核通过
  → alias/pointer 原子切换 → CAS 结算（lifecycle_revision 0→1，
  published_at / published_generation 回写）；
- 会话栅栏（MySQL GET_LOCK）在该链路真实执行（sqlite 单测为 no-op，本验收为真锁）。

## 4. 实测发现与修复（真实环境暴露，单测环境不触发）

### 4.1 冻结 composite 门槛在真实 embedding 空间不可达（阈值校准实证）

4 条真实候选（两个题材）一致呈现：bge-m3 下**任何域内像样答案**与既有文档的
答案侧 top-1 相似度 ∈ [0.65, 0.75]，新颖度 ≤ 0.35，
composite = 0.6×价值 + 0.4×新颖 在价值 0.85-0.95 时上限 ≈ 0.67，
**冻结门槛 0.70 不可达 → 真实新知识全部被 low_value 自动拒绝**（见 §5 附录）。

处置：`COMPOSITE_THRESHOLD` 从代码常量改为 settings
（`HUMAN_COMPOSITE_THRESHOLD`，默认仍 0.70 行为不变），验收以 0.62 下发
（0.62 仍拒绝价值 <0.7 的噪声：0.6×0.7+0.4×0.35=0.56 < 0.62）。
后续校准建议：以真实通过/拒绝样本回填后再定生产值。

### 4.2 MySQL DATETIME(0) 四舍五入导致审批快照 digest_mismatch（P0，已修复）

现象：发布端对 item 复核报 `digest_mismatch`；单测（sqlite）从不触发。
根因：创建侧 digest 用 Python 截断到秒的 `approved_at` 字符串（…:50），
MySQL DATETIME(0) 对小数秒**四舍五入**（…:50.6 → …:51），列值读回与摘要串差
1 秒 → 逐字段复核必失败。
修复：生成审批时间时 `microsecond=0`（human_store.py），摘要串与列值天然一致；
并用 spy 复验（CREATION digest == stored digest）。

### 4.3 build_kb_index 指针分裂（eval-v2 ⚠️ 观察项的根因，已修复）

`build_kb_index.py` 构造 GenerationStore 未传 redis_client → 只写本地文件；
线上检索器经 Redis 读到 09-03 旧代 → 535 评测报 `index_not_found`。
修复：build 工具同样走 Redis 共享指针（strict_shared）。

## 5. 附录：阈值校准原始数据（4 条真实候选被冻结 0.70 拒绝）

| # | 题材 | 价值 | 答案侧 top-1 | 新颖度 | composite | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 预约配送流程 | 0.90 | 0.707 | 0.293 | 0.657 | low_value |
| 2 | 错过预约时段 | 0.90 | 0.747 | 0.253 | 0.641 | low_value |
| 3 | 量尺安装流程 | 0.90 | 0.676 | 0.324 | 0.669 | low_value |
| 4 | 量尺收费规则 | 0.85 | 0.651 | 0.349 | 0.650 | low_value |

## 6. 数据集门禁联动（发布后规定动作）

发布新文档后 `check_eval_dataset` 文档覆盖门禁按设计拦截（新文档无检索用例）；
已在 `retrieval_cases.json` 补直接命中用例 `retrieval_evolved_measure_install_01`
（536 → 537，变更依据记入 _change_log），门禁恢复通过。此联动与 eval-v2 时期
为 SLO 文档补用例的做法一致。

## 7. 局限

- 检索对比为单文档发布（与 eval-v2 协议对齐）；批量发布（≤100 篇）的扰动需另行验收。
- 评审质量依赖真实 LLM 单次抽取；未做多采样/多评审员一致性评估。
- 负例拒绝率、端到端延迟/吞吐指标需配置阈值与压测环境，未纳入本轮。

# retrieval-v3 gate — v3 测试集门禁轮（2026-09-10）

## 结论

v3 测试集（959 条冻结：easy 504 / hard 334 / 负例 121 / 时序 42 / multi 51）首次让
门禁判定**具备统计效力**（v2 hard n=21 ±21.3pp → v3 hard n=334 ±4.6pp；负例 n=15
±17.2pp → n=121 ±8.3pp）。阈值 0.261897（正例约束 90% 下可行最高）下：

| 指标 | 实测 | 门禁 | 判定 |
| --- | --- | --- | --- |
| 正例 Recall@5（838 条） | 84.6% | ≥95% | FAIL |
| easy Recall@5（504 条） | 89.9% | ≥98% | FAIL |
| hard Recall@5（334 条） | **76.6%** | ≥80% | **FAIL（-3.4pp）** |
| MRR | 71.4% | ≥90% | FAIL |
| nDCG@5 | 74.1% | ≥90% | FAIL |
| 负例拒绝率（121 条） | **66.9%** | ≥90% | **FAIL（-23.1pp）** |

**门禁 FAIL 如实留档，不降门槛**（沿用 v2 纪律）。与 v2 的根本差异：本次 FAIL 有
统计效力背书（CI 已收窄到 ±4.6pp / ±8.3pp），"为什么 FAIL"可被量化回答，而不是
样本太少"没资格判"。

## 阈值可行带（不降门槛的依据）

校准只用 easy 正例 + 负例，hard 不参与（避免困难题泄漏进阈值选择）。
**口径说明（review 修正）**：下表正例侧基于**修正前 960 条冻结版**（easy 501），
负例侧拒绝率取自门禁轮（修正后 121 条）——门禁轮前人工复核修正 4 条
（3 负例转正 + 1 删除）发生在校准之后，当时未重扫；3 条转正用例原为
rerank 0.99+ 的强命中负例，对阈值候选集影响可忽略，但两轮口径不同，如实注明。
dev easy 89.9%（453/504）略低于校准约束 90% 的原因即此：校准在修正前 501 条
easy 上达到 ≥90%，门禁的 504 条 = 501 + 3 条转正（按 453/504 反推该阈值下
约 2 中 1 漏）——口径差而非检索回归。
约束扫描：

| 正例 recall 约束 | 可行最高阈值 | 该阈值下负例拒绝率 |
| --- | --- | --- |
| 0.98 / 0.95 | — | 原始 easy recall < 约束，不可达 |
| 0.90 | 0.261897 | 66.9% |
| 0.85 | 0.344995 | 65.6% |
| 0.80 | 0.494187 | 72.0% |
| 0.70 | 0.669745 | 76.0% |
| 0.50 | 0.872782 | 84.8% |

**正例 recall ≥80% 与负例拒绝 ≥80% 不可兼得**（可行带为空）。v2 在 15 条负例上的
"hard≥80% 与负例≥90% 不可兼得"结论，在 v3 的 121 条负例上依然成立且带更清晰。
降门槛（如负例 ≥80%）会让线上把 1/3 的平台外问题当答案返回，不可接受。

> **扫描产物（review 修复）**：上表此前无归档产物，且 CLI `--calibrate` 在 0.90
> 约束下因负例拒绝 <80% 抛错退出、不留 report。已补两件事：
> ① `run_retrieval_eval.archive_calibration_failure`——校准失败也写最小失败记录
> （约束/数据集 SHA/原因）；② `app/scripts/scan_threshold_band.py`——对冻结数据集
> （修正后 959：easy 504 / 负例 121）live probe 一次、多档约束离线扫描，归档于
> `threshold-band/scan.json`，扫描语义与 `calibrate_threshold` 一致
> （门控→折叠→截断；仅不施加负例 ≥80% 约束、负例基不含黄金集 rag_no_hit_* 扩展）。
>
> **复扫结果（959 口径，easy 504 / 负例 121，金丝雀 0.9996）与上表（960 冻结口径）对照**：
>
> | 正例 recall 约束 | 上表（960 口径）阈值 / 拒绝率 | 复扫（959 口径）阈值 / 拒绝率 |
> | --- | --- | --- |
> | 0.98 / 0.95 | 不可达 | 不可达（原始 easy recall 94.4%） |
> | 0.90 | 0.261897 / 66.9% | **0.26170810 / 66.9%** |
> | 0.85 | 0.344995 / 65.6% | 0.35086610 / 68.6% |
> | 0.80 | 0.494187 / 72.0% | 0.49507920 / 74.4% |
> | 0.70 | 0.669745 / 76.0% | 0.67640615 / 78.5% |
> | 0.50 | 0.872782 / 84.8% | 0.87245613 / 87.6% |
>
> 0.90 约束档（门禁实际采用的档位）两口径阈值差 0.0002、拒绝率完全一致
> （66.9%）——**证实 4 条人工修正对阈值选取无实质影响，门禁阈值 0.261897
> 的选取过程自此有归档产物背书**。其余档位差 ≤0.007 量级、方向一致
> （959 口径拒绝率略高：3 条转正用例不再计入负例分母）。另：原始 easy recall
> 94.4% 与 `hybrid-rerank-uncal` 轮 easy 94.4% 一致，交叉印证扫描链路正确。
> 阈值数字存在 ±0.0003 级的 rerank GPU 浮点波动（两次复扫同档位对比），
> 不影响档位结论；复扫产物含金丝雀自检记录（`retrieval.canary_top_score`）。

## 逐例分数分析（负例拒绝率低的原因）

负例 top score 分布：P50 **0.097**（大多数负例分数很低，正常拒绝）/ P90 0.920 /
max 0.997。**高分负例集中在两类**：

1. **F 类外部机构文档（设计内张力，暴露真实缺陷）**：负例问"光驰官方延保
   能退吗/星海空调压缩机保修几年/汇付宝安全险多少钱"，检索器以 0.99+ 分命中
   库中对应外部条款文档（品牌官方延保服务条款.md 等）。这些文档作为负例压力源
   入库，但检索器**没有"平台规则 vs 外部机构条款"的区分能力**——生产上会把
   别家机构的条款当平台答案返回给用户。这是本轮 FAIL 的核心发现，也是
   v3.5 商家侧文档/时序过滤方向的直接依据。
2. **数据集修正**：门禁轮前人工复核修正 4 条负例——3 条改正例（商家入驻保证金
   → 商家入驻与保证金管理规范.docx；洗衣机电机保修 → 家用电器类目规则.md 明确
   "主要部件 3 年"；天猫运费险能否叠加 → 运费险细则明确"不重复赔付"）、1 条歧义
   删除（顺丰私人件索赔）。修正披露于 retrieval_cases_v3.json 的 _change_log 与
   _meta.corrections。新增 423 条用例的 20% 抽检（84 条）已补逐条 verdict
   （`sample_20pct.json`：expected 文件存在性 84/84、负例全库词面重叠扫描均 <0.7、
   异档信号最强的 10 条逐条语义裁决均合理，84/84 pass）。

## 计划偏差披露（review 补记）

- **真实问法仅 2 条**（`_meta.human_real_queries: 2`）：计划 §3 强调从人工会话
  挖掘真实问法，实际 010 人工链路已接入会话量有限（2026-09-08 首批），仅 2 条
  适合转 hard 用例；其余 hard/负例为 LLM 起草 + 20% 抽检 + 全量负例人工复核。
- **单文档用例占比 max 4.69%**（会员权益.md 45/959）：计划目标 ≤2% 未达——
  v2 冻结 535 原样保留（纵向对照锚点）是主要成因（其中会员权益 45 条全部承自
  v2），新增 423 条已按新文档分摊。
- commit 92d82d6 message 写"960 条/负例 125"与最终冻结 959/121 不一致：
  人工修正（3 转正 + 1 删除）在冻结 commit 之后执行，`_change_log` 已披露。

## 新维度（计划 §9）

| 维度 | 结果 | 判读 |
| --- | --- | --- |
| current-version hit rate（42 条） | **21.4%**（9/42） | 门禁 ≥95% 严重 FAIL：archive 旧版大量混入 Top-5，检索器无时序意识——最弱一环 |
| 多跳双口径（59 条） | 严格 49.2% / 宽松 93.2% | 双文档全命中难；单文档命中易 |
| 条件叠加 Recall@5（56 条） | 89.3% | 隐藏前提类问题表现好 |
| 分层 Recall | A-short 86.1% / A-mid 80.0% / C 区域 84.3% / **D 数值易混 73.6%** / E 长文档 91.7% | D 类最低——数值易混对的判别极限被压出（设计意图达成）；E 长文档标题路径生效 |
| rerank score 分布 / AUC | 正例 P50 0.907 / 负例 P50 0.092，**AUC 0.840** | 整体可分；19 条高分负例（F 类）拖低 AUC，与可行带为空互为印证 |
| 检索延迟（全 dev 959 条） | **P50 296ms / P95 407ms** / max 1017ms | 语料 ×4 后仍保持亚秒级 |

## A/B：检索层双臂（v3 dev 959）

| 指标 | kNN（无阈值） | hybrid-rerank（无阈值） | hybrid-rerank（阈值 0.261897） |
| --- | --- | --- | --- |
| 正例 Recall@5 | 83.8% | **91.8%** | 84.6% |
| MRR | 68.4% | **76.1%** | 71.4% |
| nDCG@5 | 71.8% | **79.5%** | 74.1% |
| easy | 82.6% | **94.4%** | 89.9% |
| hard | 85.6% | **87.9%** | 76.6% |

同口径（均无阈值）下 hybrid-rerank 全面优于 kNN：正例 +8.0pp、MRR +7.7pp、
nDCG +7.7pp、easy +11.8pp、hard +2.3pp——BM25+kNN+RRF 与 bge-reranker 的
增益在 v3 语料规模上成立（与 v2 c5bfd0e A/B 方向一致，本次为检索层重新量化）。

## holdout 250（冻结一次性，阈值 0.261897）

| 指标 | 值 | 门禁 |
| --- | --- | --- |
| 正例 Recall@5（200 条） | **94.0%** | ≥95% 边缘接近 |
| MRR | 85.8% | — |
| nDCG@5 | 87.9% | — |
| easy Recall@5（150 条） | 96.7% | — |
| hard Recall@5（50 条） | **86.0%** | **≥80% 达标 ✓** |
| 负例拒绝率（50 条） | 80.0% | ≥90% FAIL（-10pp） |

holdout 为正例泛化提供独立佐证（94.0% vs dev 84.6%）：dev 的 hard 极端干扰
（时序/多跳/条件叠加 334 条）压低了 dev 数字，holdout 的常规 hard（50 条）
86.0% 达标。两组集合约定了"压测集 vs 泛化抽查"的双层口径。负例拒绝 80% 为
holdout 单次值（n=50 ±11.1pp），与 dev 的 66.9% 同向：**F 类外部文档压制
负例拒绝是当前检索器的主要短板**。dev hard 76.6%（±4.6pp）与 holdout hard
86.0%（±9.7pp）按区间并集判读：差距可能全部或部分是 holdout 小样本噪声；
`analyze_retrieval_report.py` [6] 子标签分层（严格口径，expected 全命中）进一步
定位 dev hard 失分来源：condition 89.3% / timing 81.0% 表现较好，最弱为
confusion 33.3%（n=6）、pinyin 55.0%、noisy 62.7%、typo 63.3%——**噪声类改写
（错字/拼音/口语）是主要失分层**，与时序/多跳关系不大。

## 配置落地与会话

- `RAG_MIN_RELEVANCE_SCORE=0.261897` 已写入 .env（第 192 行），
  dev-final 与 holdout 报告 `online_threshold_match=true`
- evolved 知识 Top-1 复测：钻石会员 SLO 问题仍为 Top-1（score 0.9996）✓
- 服务：ES 8.11.4 / TEI bge-reranker-v2-m3（GPU）/ SophNet bge-m3（1024 维）

## 运行目录

`artifacts/eval/v3/gate/`：dev-final（门禁正式轮）、ab-knn、hybrid-rerank-uncal、
holdout-250、threshold-band（review 修复补归档的可行带扫描）；
离线分析脚本 `app/scripts/analyze_retrieval_report.py`（含 hard 子标签分层与 95% CI，
dev vs holdout 差异的显著性判读见其 [6] 输出）。
早期版本本文档曾引用"gate/dev-cal（校准失败留档）"目录——该目录实际不存在：
CLI 校准失败当时直接 sys.exit(1) 不留产物，此为 review 发现的缺口，已由
`archive_calibration_failure`（失败留最小记录）与 `threshold-band/`（可行带扫描）
双双补上；特此订正。

## 后续方向（v3.5 预埋）

1. **F 类文档区分**：检索结果标注"外部机构条款"来源或引入平台域问答闭包，
   负例拒绝率是它的直接度量
2. **时序过滤**：archive 文档降权/过期标记，current-version hit 21.4% 是
   当前最弱指标
3. 门禁重跑时机：上述任一项落地后，dev 959 不变、阈值重校准、门禁重判
4. **rerank 降级与阈值门控的相互作用**（本次扫描排障发现）：ES 原生 RRF
   融合不返回 `_score`（es_backend 记 0.0），reranker 端点不可用时静默
   降级原序但分数全 0——排序看似正常，而线上阈值门控（0.261897）会把
   全部结果滤空（用户拿到空检索而非降级检索）。建议：降级路径显式
   跳过阈值门控或返回降级标记；`scan_threshold_band.py` 已内置金丝雀
   自检（已知高分查询探分数量级）防此类无效扫描
---

# 检索 hard 侧双工作流实施记录（A. query 表层规范化 + B. 多子查询补缺）

> 本节为实施留档（追加不改写上文）；分支 `feat/retrieval-norm-and-multiquery`。
> 门禁复测需 live 环境（ES/bge-reranker/bge-m3），本轮交付代码+单测+离线
> 产物，**门禁数字未重测**——执行序列见下文，验收标准不变。

## 失分解剖（本轮依据）

334 条 hard 失分 246 过 + 38 阈值截断 + 50 未召回；两个可治层：
- 噪声层：typo 11 / pinyin 9 miss——领域词同音错字（积份→积分）与拼音
  首字母（YFX→运费险）→ 工作流 A
- 多文档层：59 条双期望用例严格口径 49.2%（宽松 93.2%），44pp 全是
  「漏掉一份」；multi_intent 36 条 hard 中 10 条未召回 → 工作流 B

## 工作流 A：query 表层规范化

| 件 | 说明 |
| --- | --- |
| A1 词表构建 | `app/scripts/build_query_lexicon.py`：词源 = 文档名/标题片段 ∪ 正文高频 n-gram（Apriori 逐层 + 片段词过滤）；**构建期依赖 pypinyin + jieba（requirements-dev.txt），运行时零依赖**；产物 `app/agent/rag/query_lexicon.json`（928 词 / 68,153 护栏 n-gram / 6,945 字音节表，1,073 KB；corpus_sha256 与阈值全落 source_generation） |
| A2 规范化器 | `app/agent/rag/query_normalizer.py`：最长匹配，逐字接受 精确 / 同音（音节相同，≤max(1,len//2) 处）/ ASCII 首字母（不限数）；真词护栏（语料 n-gram freq≥5 ∪ 会被改写的通用词典真词）；**刻意不启用汉字↔汉字仅首字母替换**（有房↔运费类误改），缩写仅认 ASCII 字母；字母串原子性（YFX 不得被两个词切开）；同长度多候选择优 = 替换数最少 → 语料词频 |
| A3 接线 | settings `RAG_QUERY_NORMALIZE`（默认 **false**）+ `RAG_QUERY_NORMALIZE_LEXICON_PATH`；RetrievalConfig 增 `query_normalize`；open_retriever 构造 normalizer 注入 ESHybridRetriever/HybridRetriever（kwarg 默认 None 向后兼容）；`search_with_status` 顶部规范化一次同喂 hybrid/embedder/reranker；metrics `rag_query_normalize_hits_total`（kind 标签）+ `rag_query_normalize_missing_total`；词表缺失 fail-open 恒等 + 打点；manifest/scan_threshold_band 增 `query_normalize` 快照字段 |
| A4 评测臂 | 变体 `hybrid-rerank-norm`（hybrid+rerank+规范化）；既有三变体**显式置 False** 防臂间泄漏；norm 臂词表缺失 fail-loud（exit 2）；scan_threshold_band choices 同步 |
| A5 单测 | `tests/unit/test_query_normalizer.py` 20 条：同音/缩写/混合、最长匹配、同音预算、护栏、保护区跨界改写、字母串原子性、词频择优、fail-open、装配注入、A/B 臂隔离 |

**离线改写实测（v3 959 例全量，规范词表）**：typo 23/30、pinyin 19/20 改写
命中；easy 仅 2/504 条被改写（均为近义改写）、near_domain / out_of_domain 0 条。
残余 10 条 typo 未命中主因：家保/定单/发飘 被真词护栏有意拦下（precision 优先）、
倦（quàn vs juàn 非同音）、申家保（无对应词表词）。

## 工作流 B：多子查询补缺

| 件 | 说明 |
| --- | --- |
| B1 口径提取 | `retriever_factory.final_multi_search`：单子查询**逐字段等价 final_search**（回归保护）；多子查询 = 并行召回 → 逐路门控（RRF/降级跳过）→ chunk 级 RRF（与 evidence.rrf_merge 同秩融合语义）→ 父块折叠 → Top-K；`FinalMultiSearchOutcome`（含 hit_queries 子查询归因）；**顺手统一口径**：min_score 变参数（线上传 settings、评测传阈值），检索层不再自读 settings |
| B2 评测接入 | overlay 方案不动冻结 959：`app/evaluation/retrieval_cases_v3_multihop_queries.json`（59 例，每例 ≤2 条补充子查询，原 query 居首 ≤3 条与线上 `_normalize_subqueries` 同规）；`run_retrieval_eval --multi-query-overlay <path>`（加载 fail-loud exit 2；报告 payload 记 path + sha256 + protocol）；`evaluate()` 增 `multi`/`subqueries` 逐例标注 + summary `multi_hop` 双口径（严格=recall==1.0 / 宽松=recall>0）；每路门控用现有校准阈值，RRF 合并分永不门控——**B 与 A 的阈值互不干扰，无需重校准** |
| B3 提示词教学 | `customer_service.py`：search_knowledge 条目下新增拆解指示 + 2 个 few-shot（mixed_1 型：钻石会员退换货要运费吗 → queries=[钻石会员运费特权, 退换货运费承担方]）；使用原则 4 改写为「先拆解、后改写」；新增防回退断言测试 |
| B4 单测 | `tests/unit/test_multi_query_search.py` 15 条：单子查询与 final_search 等价、多路合并/归因/跨路父块去重、逐路门控（RRF 跳过/精排分过滤）、RRF 数学、降级聚合、**迷你双文档库拆分召回 e2e**（单 query 漏一份 → 拆解后两份都进 Top-K）、search_knowledge 全链路（subqueries 回显/截断/evidence）、overlay 同规、多跳双口径 |

## 执行序列（Step 6 扩展，待 live 环境执行）

1. **A 轮**：`hybrid-rerank-norm` vs `hybrid-rerank`（baseline 阈值 0.261897）
   uncal A/B → 扫可行带新阈值 → dev-final 门禁（hard ≥80%、负例 ≥66.9% 守门、
   easy 回退 ≤1pp）→ 新鲜 holdout 250 一次性 → 延迟对照（P50 增幅 ≤10ms）
2. **B 轮**（独立单变量，baseline 阈值）：`--variant hybrid-rerank
   --multi-query-overlay …` vs 单 query 臂 → 59 条双口径对照 + multi_intent
   子标签 → 3 臂并行延迟（59 条 P50/P95）
3. 两轮通过后可选组合臂（norm+multi）仅留档，不参与门禁
4. 文档：本文件各追加章节；`简历指标来源.md` 视结果补口径

## 诚实边界

- B 轮测的是**理想拆解上限**：overlay 是人工审核的拆解，线上实际收益取决于
  模型经提示词教学后是否真的拆；两者差距需 agent 侧（run_eval/会话抽样）
  另行验证，不在本轮门禁
- A 轮词表与规范化规则在 v3 dev 集（959）上调试（阈值 80 / 护栏 5 / 比例
  0.6 等参数），holdout 250 为**新鲜一次性**集合未参与任何调参——两轮通过
  后以 holdout 数字为准
- overlay 起草纪律：只读 query 文本与 KB 文档命名空间；例外披露——前期数据
  集探查曾以统计口径打印过 4 条样例（hard_confusion_03/04/05、hard_multi_01）
  的完整用例（含 expected），该 4 例子查询仍按 query 文本独立起草；20% 人工
  抽检（12 例）以 query-only 口径复核后方可用于门禁
- 残余失分：LLM 语义改写（noisy/indirect）、F 类区分、时序过滤——仍为后续
  轮次（见上文「后续方向」）

## review 修复（2026-09-16，实施 commit 之上追加）

> 本节为对 7812494 的评审修复留档（追加不改写上文）。评审人复核：全量单测
> 1490 绿、ruff（项目事实口径）已清、main 未被触碰。以下按编号修复，1 条
> 评审发现撤回并披露。

1. **P1 overlay 文件名词面泄漏（全局口径 45/59 例，52/116 条）**：复核将口径
   从 per-case expected 扩到「任意 KB 文档 stem ∪ 全数据集 expected stem」后
   共 45 例含与文档名逐字相同的子查询——虽符合「可读文档命名空间」纪律，
   但 B 轮会测成「标题字面命中」而非真实拆解增益。已全部按 query 主题改写
   为 query 词汇表述（改写不看 expected），改写后 116 条中 **0 条**与任何
   文档 stem / 本例 query 逐字相同；顺带修正 1 例主题错配（c_078 价保用例
   原子查询是 3C 数码类目）。`drafting_discipline` 字段同步披露本次改写。
   20% 人工抽检（12 例）仍需按 query-only 口径复核后方可用于门禁。
2. **P2 hit_queries 平行性**：`final_multi_search` 此前返回折叠前全长归因
   列表（靠 zip 截断才没出事）。父块折叠改为与 `collapse_by_parent` 同语义
   的手写循环并同步截断归因，`hit_queries` 与 `hits` 严格平行；新增回归
   测试（含截断无悬空归因、双路命中归先提交子查询）。
3. **P2 F 类词表隔离空转**：语料中 10 份 F 类文档 frontmatter 仍是
   `authority: platform`（v3 冻结口径，改 external_reference 会把它们移出
   回答索引、破坏冻结评测；frontmatter 治理留 v3.5），`excluded_chunks` 因此
   无 external_reference，原隔离未生效。builder 按显式清单
   `F_CLASS_SOURCE_PATHS`（=《语料补足与评测v3计划》§7.6 十份）排除；
   重建词表 928→**884 词**（排除 F 类 50 chunk；guard 68,153→62,473）。
   **离线改写复测与原版完全一致**：typo 23/30、pinyin 19/20、easy 2/504、
   负例三组（near_domain/out_of_domain/keyword_overlap）全 0——排除零代价。
   延保/保修仍在词表（来源为平台侧《售后维修与延保》等文档，属平台词汇，
   符合设计意图）；F 类专属词（汇付宝/光驰/星海/安全险/压缩机）确认不入表。
4. **P2 提示词 few-shot 发明主题**：旧例句「黄金会员发偏远地区包邮吗多久到」
   给出 query 里不存在的「快递丢件赔偿标准」子查询（教模型发明无关主题）。
   换成三主题真实例句（大促价保+退货运费+券退回，三个子查询全部来自
   query），并加「不要为了凑数发明顾客没问的主题」指示；防回退断言同步
   （新例句在场 + 旧发明式例句退场）。
5. **P3 norm 臂报告改写留痕**：`evaluate(query_normalizer=…)` per-case 记录
   `normalized_query`（仅变化时非 None）+ summary `query_normalize_rewrites`
   计数，CLI norm 臂加载同词表传入并打印改写条数——「typo 23/30」这类
   离线实测自此可从评测产物直接复核。normalizer 只记录不参与检索
   （检索路径规范化在检索器内部，两者同词表幂等）。另清 C401×2、
   测试夹具死键（"退款": None）、`_is_vocab_shaped` 注释与实现矛盾
   （「为限要保留」→「为限必被滤除」）。
6. **评审发现撤回**：原 P2-3「corpus_sha256 依赖 rglob 遍历序、跨机器不可
   复现」不成立——`parsers.py:1045` 为 `sorted(kb_dir.rglob("*"))`，遍历序
   确定，指纹可复现。评审人判断有误，特此更正。

> 修复后全量单测（1490 既有 + 35 工作流 + 2 新增回归：hit_queries 平行性、
> normalized_query 留痕）全绿；ruff 新文件零告警。词表产物（884 词）已随本次
> 修复重建入盘。

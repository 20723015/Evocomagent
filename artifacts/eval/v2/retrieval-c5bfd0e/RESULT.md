# 正式检索实验（c5bfd0e 冻结数据集）结果记录

协议：dev 535 条（520 正 / 15 负；easy 499 / hard 21）+ holdout 120 条（冻结，仅对最终选定配置跑一次）。
环境：ES 8.11.4（127.0.0.1:19200，active alias ecom-kb-active → ecom-kb-20260901044839-eb5bea47，bge-m3 164 chunks）、
TEI reranker（bge-reranker-v2-m3，127.0.0.1:8001）、SophNet bge-m3 embedding（外部 API）。

## 环境问题（已修复，不影响结论）
1. 系统代理（127.0.0.1:7890）注入 HTTP(S)_PROXY 环境变量，httpx 默认把 localhost 请求转发给代理 → 502。
   reranker 客户端已有 trust_env=False（c5bfd0e 已带）；其余本地服务（ES/Redis）走非 httpx 客户端不受影响。
2. wslrelay 抢占 [::1]:8001，localhost 优先解析 ::1 → rerank 请求被劫持 502。
   修复：RERANK_ENDPOINT_URL 必须用 127.0.0.1（IPv4 直达 docker-proxy）。
   影响：早期 hybrid-rerank 校准的"分数"实为精排失败后的降级原序原分，结论作废，重跑。
3. embedding 外部 API 周期性 503（Connection reset）——评估运行需传输层重试（本次以进程外 wrapper 提供，未改代码）。

## 链路缺陷（已修复，本轮 dev 校准基于修复后链路）
ESBackend.hybrid_search 原先 size=top_k：reranker 只看到 RRF 融合 top-5，排在 6-60 位的期望文档永远不可达。
修复：返回 recall_k 候选窗口（size=recall_k），ESHybridRetriever 精排后截断 top_k；
与 Python 侧 HybridRetriever（先取 recall_k 候选再精排）语义对齐。单测已更新并过。

## 三变体 dev 校准结果（修复后链路）
- knn（ES 纯向量）：**失败**——原始正例 recall@k 无法达到 98%（easy 召回不达标，阈值无关）。
- hybrid（BM25+kNN+RRF，无精排）：**失败**——RRF 融合分正负例不可分：保持正例 recall 99.1% 时负例拒绝率 0.0%。
- hybrid-rerank（hybrid+bge-reranker-v2-m3）：**校准通过（30 负例：正例 recall 98.1% / 拒绝 70% → 校准搜索给出阈值 0.091220066）**，
  dev 评估 6 项门禁中 5 项达标，hard Recall@5 不达标（见下）。

hybrid-rerank dev 评估（阈值 0.091220066，hard 与线上同阈值）：
| 指标 | 门禁 | 实测 |
|---|---|---|
| Recall@5 | ≥95% | 96.1% ✅ |
| easy Recall@5 | ≥98% | 98.1% ✅ |
| hard Recall@5 | ≥80% | 47.6% ❌ |
| MRR | ≥90% | 90.3% ✅ |
| nDCG@5 | ≥90% | 91.6% ✅ |
| 负例拒绝率 | ≥90% | 93.3% ✅ |

## 结论：单一阈值不可行（证据）
hard 21 条期望文档精排分（阈值过滤前 top-5 原始召回 20/21，其中 11 条 ≤0.06）：
hard ≥80% 要求阈值 ≤ 0.0252（第 5 低分）。
dev 15 条负例 top1 分：0.1537(连麦互动)、0.0728(旧家具回收)、0.0603(代购材料)、0.0309、0.0204、≤0.0033×11；
负例拒绝 ≥90%（14/15）要求阈值 > 0.0728。
可行带 (0.0728, 0.0252] = ∅。→ **无单一阈值可同时满足硬门禁**，按协议不冻结配置、不跑 holdout。

## 数据质量核查（供回 dev 参考）
- golden 负例 rag_no_hit_02"运费险怎么买"（精排分 0.86）：运费险细则.md 有直接答案（"购买商品时可随单勾选"）→ **标签错误**。
- golden 负例 rag_no_hit_04"会员分期付款有利息吗"（0.26）：先用后付与分期付款.md 覆盖分期手续费/免息/逾期利息 → **疑似误标**。
- dev 负例"直播间可以连麦互动吗"（0.15）：文档无连麦 → 标签正确，属真实难负例。
- dev 负例"家里旧家具能上门回收吗"（0.07）：文档为旧机（手机）回收，非家具 → 标签正确，near-domain 难负例。

## 未执行步骤（按协议）
- 阈值未写入生产配置（RAG_MIN_RELEVANCE_SCORE 保持未配置；RERANK_ENDPOINT_URL=127.0.0.1 系环境修复项，已建议写入 .env）。
- holdout 120 条未运行（仅对最终选定配置跑一次）。
- 回 dev 修改的候选方向：修正 golden 误标负例（还原基准真实性）；hard 用例的口语化/间接查询规范化后再精排；
  或按业务实际重议 hard≥80% 与拒绝≥90% 的联合门禁（两者经数据证实不可同阈值兼得）。
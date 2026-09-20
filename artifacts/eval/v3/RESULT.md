# retrieval-v3 — 语料补足与评测（传统指标轮，含 review 修复）

## 结论

按《语料补足与评测v3计划》完成语料扩充：40 份 / 183 chunk → **147 source_path / 836 chunk**
（135 现行 + 10 archive 历史版本 + 2 evolved 知识；含 .docx/.pdf 原生格式 2 份）。
传统指标以 **v2 冻结 535 口径**（`--exclude-id` 排除 2 条 evolved 增补用例）重跑
hybrid-rerank 线上同链路，结果较 v2 明显回落并落入计划预期区间，**干扰有效性检验通过**：

| 指标 | v2（a7c9749，535 冻结） | v3（535 口径，阈值 0.22832851） | 变化 |
| --- | --- | --- | --- |
| 正例 Recall@5（520 条） | 96.4% | **88.3%** | -8.1pp |
| 正例 MRR | 91.3% | **72.9%** | -18.4pp |
| 正例 nDCG@5 | 92.4% | **76.4%** | -16.0pp |
| easy Recall@5（499 条） | 98.2% | **90.2%** | -8.0pp |
| hard Recall@5（21 条） | 54.8% | **42.9%** | -11.9pp（n=21，无门禁效力） |
| 负例拒绝率（15 条） | 86.7% | **86.7%** | 0pp（n=15，无门禁效力） |

- **干扰有效性检验（§2.4）通过**：Recall@5 降幅 8.1pp（> 1pp 阈值），88.3% 落入计划
  预期 85–92% 区间；"仍 ≥96% 则判定干扰失败回炉"的反向门禁同样通过
- hard（n=21）/负例（n=15）样本量下不做门禁判定（CI ±21pp/±17pp），留待 v3 测试集
  （hard ≥150 / 负例 ≥80）——这是测试集扩充阶段的直接依据
- 537 口径对照轮（未排除 evolved 用例，`hybrid-rerank-dev-cal/dev-final` 目录）：正例
  Recall@5 88.1%、MRR 72.9%、nDCG 76.4%、easy 90.0%、hard 42.9%、负例拒绝 86.7%，
  校准阈值为 0.26095408（数字与 535 口径差异 <0.3pp，但阈值差 1.3pp——535 口径为准）

## review 修复（2026-09-10）

1. **数据集口径**：数据集文件实际 537 条（`b42c590`，SHA-256 `...`）= v2 冻结 535
   （`ec4a977e...`）+ 2 条 evolved 覆盖用例（`retrieval_evolved_slo_01`、
   `retrieval_evolved_measure_install_01`，由 9e1a4c5/a6336df 各补 1 条）。
   新增 `--exclude-id`（可重复）复现 v2 冻结 535 口径；纵向对照以 535 口径为主。
2. **校准约束留档**：report `thresholds` 与 manifest 新增 `calibrate_min_positive_recall`
   （CLI 值）与 `calibrate_min_negative_rejection`（0.80 常量）字段。
3. **语料份数口径**：147 source_path = 135 现行 + 10 archive + 2 evolved（此前"145 份
   含 evolved 2 份"计数有 2 份出入，已修正）。
4. **bulk 错误静默**：`es_backend.upsert` 逐项检查 bulk 响应，errors=true 即 raise
   （对照 outbox 同步判定）；构建期发现的 48 个 chunk_id 冲突（同名 .md/.docx）在
   修复前已按 §7.5 原生格式意图删除同名 .md 规避，修复后此类错误将显式失败。
5. **holdout 执行披露**：holdout 120 共执行 3 次——v2 于 09-03（a7c9749 时代）、
   v3 于 09-10 两次（阈值 0.26095408 轮 71.1%、535 口径重校准 0.22832851 轮 73.2%）。
   v3 门禁轮将按冻结协议换新鲜 holdout 250 条后仅执行一次。

## 阈值

- v2 阈值 0.052327625 在 v3 语料上复测：负例拒绝率仅 66.7%（< 80%），确认不复用
- 535 口径校准：**RAG_MIN_RELEVANCE_SCORE = 0.22832851**（正例 recall 90.2%、
  负例拒绝 86.7%）；校准正例约束 0.98→0.90（v3 语料下 easy 原始 recall 94.4% < 98%，
  0.98 不可达），约束值随报告留档（`calibrate_min_positive_recall=0.90`、
  `calibrate_min_negative_rejection=0.80`）
- 537 口径校准轮阈值为 0.26095408（被排除的 2 条 evolved 用例 score≈0.9996 改变了
  阈值扫描候选集）；两轮均 `online_threshold_match=true`（进程配置 RAG_MIN_RELEVANCE_SCORE
  与评测阈值同值；.env 落地留待门禁轮）

## 索引验收（§2.4）

- 语料：**147 source_path / 836 chunk**（135 现行 + 10 archive + 2 evolved；chunk_id 无重复）
- 多格式：.docx（商家入驻与保证金管理规范）/.pdf（平台治理与处罚总则）原生入库，
  a7c9749 多格式解析链路首次真实验收通过（检索 Top-1 命中）
- generation：`20260910012731-65a69d8e`，alias `ecom-kb-active` 已切换
- Top-5 覆盖率：5/836 = **0.60%** < 0.8% 目标

## 新增维度（§9 首批实测）

| 维度 | 结果 |
| --- | --- |
| 时序有效性（current-version hit） | 4/4 现行规则查询命中现行文档；2 条旧版混入 Top-5（严格口径 50% < 95% 门禁，样本 4 条，正式判定待 v3 测试集） |
| evolved 知识 Top-1 | 「钻石会员专属客服的响应时效SLO是多少」仍为 Top-1（score 0.9996）✓ |
| 多格式解析验收 | docx（0.9754）/ pdf（0.9915）查询均 Top-1 命中 ✓ |
| 检索延迟分位 | 30 条混合查询：P50 282ms / P95 333ms |

## 报告与复现

| 运行 | 数据集 | 阈值 | 正例 Recall@5 | MRR | nDCG@5 | easy | hard | 负例拒绝 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hybrid-rerank-dev-cal-v2frozen535` | dev 535 | 0.22832851 | 88.3% | 72.9% | 76.4% | 90.2% | 42.9% | 86.7% |
| `hybrid-rerank-dev-final-v2frozen535` | dev 535 | 0.22832851 | 88.3% | 72.9% | 76.4% | 90.2% | 42.9% | 86.7% |
| `hybrid-rerank-holdout-v2frozen535` | holdout 120 | 0.22832851 | 73.2% | 58.1% | 61.9% | 83.8% | 23.5% | 91.3% |
| `hybrid-rerank-dev-cal` | dev 537 | 0.26095408 | 88.1% | 72.9% | 76.4% | 90.0% | 42.9% | 86.7% |
| `hybrid-rerank-dev-final` | dev 537 | 0.26095408 | 88.1% | 72.9% | 76.4% | 90.0% | 42.9% | 86.7% |
| `hybrid-rerank-holdout` | holdout 120 | 0.26095408 | 71.1% | 57.3% | 60.8% | 81.2% | 23.5% | 95.7% |
| `hybrid-rerank-dev-uncal` | dev 537 | 0（无阈值） | 93.0% | 76.1% | 80.0% | 94.4% | 59.5% | 0%（口径无意义） |
| `hybrid-rerank-dev-v2thr` | dev 537 | 0.052327625 | 91.3% | 75.1% | 78.9% | 93.2% | 45.2% | 66.7% |

复现（535 口径）：
```bash
RAG_BACKEND=es ES_URL=http://localhost:19200 RERANK_ENDPOINT_URL=http://127.0.0.1:8001 \
  RAG_MIN_RELEVANCE_SCORE=0.22832851 \
  python -m app.scripts.run_retrieval_eval --dataset app/evaluation/retrieval_cases.json \
  --variant hybrid-rerank \
  --exclude-id retrieval_evolved_slo_01 --exclude-id retrieval_evolved_measure_install_01 \
  --json-out artifacts/eval/v3/hybrid-rerank-dev-final-v2frozen535/report.json
```

工具：语料一致性校验 `python app/scripts/check_corpus_v3.py`（0 错误）；
多格式转换 `python app/scripts/convert_kb_docx_pdf.py`

## 环境

- ES 8.11.4（localhost:19200，trial license RRF）/ bge-reranker-v2-m3 TEI（localhost:8001，
  GPU 直通 RTX 4070，容器日志确认 CudaDevice）/ bge-m3 embedding（SophNet 线上 API，1024 维）
- 数据集：dev 537（535 冻结 + 2 evolved 增补）、holdout 120（v2 冻结）
- 评测脚本变更：`--exclude-id`、`--calibrate-min-positive-recall`、校准约束留档
  （`build_retrieval_thresholds` / `filter_cases_by_exclude` 纯函数，单测见
  `tests/unit/test_retrieval_eval_cli.py`）；`es_backend.upsert` bulk 错误显式失败
  （单测见 `tests/unit/test_phase8_sql_es.py`）
- 门禁轮（hard ≥80%、负例 ≥90%、current-version ≥95%）待 v3 测试集（hard ≥150/
  负例 ≥80，dev ~900）与门禁轮阶段执行；v2 dev 535 仅作干扰检验与纵向对照
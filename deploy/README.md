# 部署（阶段二 2.5/2.6）

本目录是部署物。注意：**开发/测试机器不需要 Docker**——本地直接
`python -m uvicorn app.server.main:app` 即可（Redis 缺失自动降级本地文件）。

## 构建（需 Docker 的机器上执行）

```bash
docker build -f deploy/Dockerfile -t registry.example.com/ecom-agent:0.2.0 .
docker push registry.example.com/ecom-agent:0.2.0
```

镜像特性：multi-stage（构建层缓存依赖）、非 root（10001）、不含 .env、
内置 healthcheck。所有配置来自环境变量（pydantic-settings 自动映射）。

## 安装 Helm Chart

```bash
# 1) 密钥不落 Chart/镜像
kubectl create secret generic ecom-agent-secrets \
  --from-literal=OPENAI_API_KEY=sk-xxx \
  --from-literal=OPENAI_BASE_URL=https://api.openai.com/v1

# 2) 安装
helm install ecom-agent deploy/helm/ecom-agent -n production \
  --set image.tag=0.2.0
```

灰度/回滚（阶段五 5.4）：

```bash
helm upgrade ecom-agent deploy/helm/ecom-agent -n production --set image.tag=0.2.1
helm rollback ecom-agent 1   # 秒级回滚（RollingUpdate maxUnavailable=0）
```

## 就绪与优雅退出

- `readiness` 探 `/readyz`：返回 `status`（ready/degraded/not_ready）+ `components`
  依赖明细（not_configured/ok/unavailable）+ `capabilities` 能力分级
  （chat/message_search/kb_upload/turn_archive）。聊天核心依赖不可用 → 503
  not_ready；仅非聊天能力不可用 → 200 degraded；
- `REDIS_REQUIRED=true` 时 Redis 为聊天核心依赖（fail-closed，锁与预算均不降级）；
- `liveness` 探 `/healthz`：进程存活（不依赖外部组件）；
- Pod 驱逐时收到 SIGTERM → uvicorn 停接新请求 → drain 在途请求（含断连收尾任务：
  其持有会话租约，必须等 Agent 真正结束才释放）→ 退出；
  `terminationGracePeriodSeconds = max(TURN_BUDGET_SECONDS, 30) × 1.5 + 30`
  （默认 120 → 210 秒）；
- 会话/记忆/锁都在 Redis/MySQL，Pod 漂移零丢失（2.2/2.6 验收）。

## Reset 删除事件两阶段上线（修复计划·二）

Reset 会写 `message_delete_outbox`（ES 旧消息删除事件）；上线分两步：

1. 先部署迁移 011、UUID 写入与兼容新事件的消费者，并将
   `MESSAGE_DELETE_OUTBOX_ENABLED=false`（Helm 默认已是第一阶段值）关闭删除事件生产；
2. 全部 Pod 升级后置 `MESSAGE_DELETE_OUTBOX_ENABLED=true` 启用 Reset 删除事件。

`MESSAGE_DELETE_OUTBOX_ENABLED` **只是生产开关**：消费者始终处理/重试/清理已存在
的删除事件，关闭生产不会让存量事件积压。

回滚前必须先关闭生产并排空新事件。删除事件未完成期间，运营搜索按 tombstone
过滤旧 `session_uuid`，保证「重置后立即不可搜索」。死信可用
`python -m app.scripts.outbox_admin list|replay` 查询/重放。

上线顺序（迁移先行、应用后发）：Reset/Outbox 状态机不允许新旧 Worker 长时间
混跑——切流前排空旧 Worker，至少等待一个 Outbox lease 周期。旧会话消息在 Reset
时被标记 `obsolete`（终态，不再写 ES）；ES 文档 ID 含 session UUID，Reset 后
seq 重新从 1 开始也不会覆盖新消息。Schema 是前向兼容资产，应用回滚不回滚迁移。

## 混沌验收（对照计划 2.6 验收）

```bash
# 对话中途删 pod，下一轮在新 pod 无缝继续：
kubectl delete pod -l app=ecom-agent
# 滚动更新中持续压测（优雅退出尾部不丢数据）
```

## 定时沉淀（阶段六 6.2 开启）

`values.yaml` 的 `evolutionJob.enabled: true` 开启 CronJob（默认
`--dry-run`，人工审核后才发布；6.2 落地后在 6.3 审核后台齐备再放开）。

## 灰度发布（阶段五 5.4）

```bash
# 5% 灰度：独立 values.gray.yaml（见 templates/values.gray.yaml 说明）
helm upgrade ecom-gray deploy/helm/ecom-agent -n production \
  -f deploy/helm/ecom-agent/values.yaml \
  -f deploy/helm/ecom-agent/values.gray.yaml \
  --set image.tag=0.3.0-rc1

# 观测灰度：按 version 标签过滤体验/成本/质量看板；对比：
#   promql: sum by (version) (rate(chat_latency_seconds_bucket[5m]))
# 回滚：
helm rollback ecom-gray 1
```

## KB 异步建库 Worker（多实例异步改造）

- Chart 自带 `{{ .Release.Name }}-kb-worker` Deployment（默认 2 副本、**无
  Service**）：`python -m app.agent.rag.kb_worker`，复用 Web 的 MySQL/ES/S3/RWX/
  模型凭据；**Web Pod 不启动 Worker**。
- 两阶段上线：先部署本版（含迁移 007，由 migration Job 自动执行）并排空旧同步
  请求 → 再用**幂等 upsert**置共享开关，一次性启用异步 API 与 Worker。不能使用
  可能影响 0 行的裸 `UPDATE`（旧库可能尚未有该 key）：
  ```sql
  INSERT INTO kb_control (`key`, value, version)
  VALUES ('kb_async_enabled', '1', 0)
  ON DUPLICATE KEY UPDATE value = VALUES(value), updated_at = CURRENT_TIMESTAMP;
  ```
  回退异步也必须使用同一 upsert（将值改为 `'0'`），回到同步语义；排队任务仍
  由 Worker 收尾。
- 优雅退出：SIGTERM → Worker 停止领取；当前任务在
  `KB_WORKER_GRACE_SECONDS`（默认 120s ≥ 租约）内收尾，超时让出租约由其他
  实例接管。
- 告警基线：`kb_job_backlog`（最老任务年龄 >10min）、`kb_job_blocked > 0`
  （等待人工 reconcile）、`kb_job_dead_total` 增长（重试耗尽死信）。

### 异步版本的安全回滚顺序

应用回滚前必须按以下固定顺序执行，数据库不得回滚到 v6 或更低版本：

1. 关闭新入队：用上面的 upsert 将 `kb_async_enabled` 设为 `'0'`，停止新
   `complete/delete` 任务进入队列。
2. 排空 active jobs：保持 Worker 运行，等待 `queued`、`running`、
   `retry_wait`、`blocked` 全部处理完（必要时先修复依赖或人工 reconcile）。
3. 确认没有 `queued`/`indexing`/`deleting` 文档，再执行 `helm rollback`；
   例如：
   ```sql
   SELECT status, COUNT(*) FROM kb_documents
   WHERE status IN ('queued', 'indexing', 'deleting') GROUP BY status;
   ```
4. 回滚应用后保留并继续使用 schema v7（`kb_index_jobs`、租约字段和
   `error NOT NULL` 不删除），恢复时仍先执行迁移/启动 gate 校验。

灰度前强制走一轮「离线评估 → 影子回放」：
```bash
python -m app.scripts.run_eval --no-judge                 # 黄金集门禁
python -m app.scripts.shadow_replay --turns app/sessions/evolution/turns --limit 200
```

## RAG 检索发布顺序（RAG 修复计划·6）

严格按序执行，前一步未通过不进入下一步：

1. **修复降级语义与文档过滤**：`rerank=none` → 真 None；RRF 分不做阈值门控；
   reranker 不可用 → 生产 fail-closed（`RAG_RERANK_FAIL_CLOSED=true`）；
   索引仅含 根目录 + `evolved/` + `uploads/`（排除 `archive/`、`.trash/`、
   `.staging/`、隐藏/临时文件）。
2. **补齐文档元数据**（`status` / `authority` / `effective_date`）并开启
   `RAG_DOC_METADATA_REQUIRED=true`（strict 构建缺字段直接失败）。
3. **在生产 ES 构建未激活候选**：`python -m app.scripts.build_kb_index --backend es
   --no-activate --json-out rag_release_candidate.json`
   （生产同构：`RAG_BACKEND=es RAG_HYBRID=true RAG_HYBRID_RECALL_K=60
   RAG_RERANK=bge-reranker-v2-m3 EMBEDDING_PROVIDER=sophnet EMBEDDING_MODEL=bge-m3`）。
4. **用 v3 dev 集重新校准**：`run_retrieval_eval --dataset
   app/evaluation/retrieval_cases_v3.json --calibrate ...`；门槛：正例 Recall@5
   ≥95%、Easy ≥98%、Hard ≥80%、MRR ≥90%、nDCG@5 ≥90%、负例拒绝率 ≥90%、
   P95 延迟 ≤500ms。
5. **dev 门禁通过**后，`rag-release` 从校准报告提取阈值，后续 dev/holdout
   均以同一个 `RAG_MIN_RELEVANCE_SCORE` 重跑，并生成发布覆盖文件
   `rag_release_values.yaml`。部署时必须同时传入该文件，例如
   `helm upgrade ... -f values-production.yaml -f rag_release_values.yaml`；基础生产
   values 不保存未经本轮验收的旧阈值。
6. **执行一次 holdout**（`holdout_cases_v3.json`，只跑一次，不得反向调参）；
   要求 Recall@5 ≥95%、Hard ≥80%、负例拒绝率 ≥90%。
7. **应用生产 embedding/reranker 配置和本轮阈值覆盖文件**，然后在能访问同一
   ES/Redis 的发布环境执行 `python -m app.scripts.build_kb_index --backend es
   --activate-candidate rag_release_candidate.json`；命令会校验配置指纹和构建前的
   活动 generation，期间若已有其他发布则拒绝覆盖。随后
   确认 `/readyz` 的 `rag` 能力为 ok（ES 可连接 / generation 存在 /
   embedding 模型与索引一致 / reranker 可用 / 阈值有有效打分器）。
8. **灰度 5%**，观察错误率、空结果率（`rag_retrieve_empty_total`）、
   reranker fallback（`reranker_fallback_total`）与延迟；无异常后逐步
   扩大到 25% 与 100%。

> 注：`rag-release` 使用 `production-rag` 受保护环境中的
> `RAG_RELEASE_ES_URL`、`RAG_RELEASE_REDIS_URL`、SophNet 与 reranker secrets，
> 候选直接构建在待发布 ES 中，但门禁期间不切 alias/共享指针。工作流产出的
> candidate JSON、三份评测报告和 values 覆盖必须作为同一个发布包使用。

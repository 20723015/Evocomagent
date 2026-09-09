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

- `readiness` 探 `/readyz`：组件就绪 + `REDIS_REQUIRED=true` 时 Redis 必须在线；
- `liveness` 探 `/healthz`：进程存活（不依赖外部组件）；
- Pod 驱逐时收到 SIGTERM → uvicorn 停接新请求 → drain 在途请求（Agent 线程池
  内任务完成后写回 Redis 会话/记忆）→ 退出；`terminationGracePeriodSeconds=45`
  给单轮（≤ 30s）留足余量；
- 会话/记忆/锁都在 Redis，Pod 漂移零丢失（2.2/2.6 验收）。

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

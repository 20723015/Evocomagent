# KB 文档上传链路（人工上传 → 分片断点续传 → 版本化重建入库）

> 状态：已实现（工作区）。本页记录架构、部署要求与运维操作。
> 设计评审历史（8 轮）要点全部落在代码注释与本页「不可回退的表白」节。
> **多实例异步改造**（迁移 007）：complete/下架默认改为 MySQL 持久化异步任务，
> API 立即返回 202，由独立 KB Worker Deployment 执行——见「异步建库（迁移 007）」节。

## 一句话

运营把 **md/txt/pdf/docx/html** 文档上传到知识库：Redis 记断点状态（分片级续传），
MySQL `kb_documents` 为元数据正本（状态机 + CAS），解析/消毒后规范化为
**Markdown 入库**（frontmatter 携带 `provenance: upload:{id}` / `owner: ops`），
随后复用 `IndexBuildService` 全量重建版本化索引（**strict**）——验证通过后先切
ES Alias，再更新 Redis generation pointer；线上 retriever 由 generation pointer
驱动热刷新，无需重启。Alias 与 pointer 是两个需要一致性核对的阶段，不把 Alias
单独当作线上 retriever 的最终入口。

## API（8 端点 + 4 任务端点，全部 `SCOPE_OPS`）

| 端点 | 说明 |
|---|---|
| `POST /v1/kb/uploads` | body：`filename/size_bytes/content_type?/chunk_size?/sha256?/upload_id?`；`upload_id` 为幂等锚点（同 id 参数不一致 → 409）；返回 `upload_id/total_chunks/received[]` |
| `PUT /v1/kb/uploads/{id}/chunks/{seq}` | 二进制分片（octet-stream）；重传同内容幂等、异内容 409（**原分片不破坏**）；超发 413 |
| `GET /v1/kb/uploads/{id}` | 断点状态（`received[]` 供续传）；异步模式下以 SQL 文档+任务为正本——Redis 会话/分片已清理也能返回终态（`job` 字段） |
| `POST /v1/kb/uploads/{id}/complete` | 异步：**202** + `job_id/status/status_url` + `Location` + `Retry-After: 2`（重复请求返回同一任务；已 indexed 幂等 **200**；永久失败 **409** 附任务地址）。同步模式（开关关）：原 200 终态语义 |
| `DELETE /v1/kb/uploads/{id}` | 取消（异步：委托任务取消，queued 才可取消、运行中 409；文档/会话/分片清为 cancelled 终态） |
| `GET /v1/kb/documents` | 列表（ops 全量） |
| `GET /v1/kb/documents/{doc_id}` | 详情 |
| `DELETE /v1/kb/documents/{doc_id}` | 下架（异步：**202** + 任务地址，已 deleted **200**；同步模式：原 200 终态语义） |
| `GET /v1/kb/jobs/{job_id}` | 任务详情：`status/stage/progress/attempts/error/created_at/...` |
| `GET /v1/kb/jobs` | 任务分页查询（`status`/`operation` 过滤，`limit/offset`） |
| `POST /v1/kb/jobs/{job_id}/retry` | 人工重试：仅**可重试**的 `failed/blocked`；重置本轮 attempts，`manual_retry_count` 审计数只增 |
| `DELETE /v1/kb/jobs/{job_id}` | 取消任务：仅文档仍为 `queued/delete_queued` 时允许（上传同事务转 `cancelled`，下架同事务恢复 `indexed`）；运行中及提交点后的 `retry_wait` 均 **409** |

`uploader` 语义：认证身份（`AUTH_ENABLED` 关闭时回退 body/query `uploader`）；
complete/delete 校验行归属（不一致 403），create 幂等比对（不一致 409）。

## 状态机（全部 CAS：status + version [+ operation_id]）

```
uploading ──→ queued ──→ validating ──→ indexing ──→ indexed   （异步：complete 入队）
   │  │          │  │         │  │          │  │
   │  ├→ failed  │  ├→ cancelled          │  ├───────→ queued（异步回退：锁等待/构建异常，不计失败）
   │             │                        │
   └─────────────┴────────────────────────┘（同步路径/分片未就绪/校验重试 → uploading 可重试）
indexed ──→ delete_queued ──→ deleting ──→ deleted        （异步：下架入队）
indexed ──→ deleting ──→ deleted                          （同步路径）
   deleting / delete_queued 仅提交点前可回滚 indexed
```

- **输入错误**（hash/magic/解析/消毒/超限）→ `failed`（终态，重新上传）；
- **分片未就绪**（缺片/存在 publishing）→ `UPLOAD_INCOMPLETE` 409 + 回退 `uploading`（续传语义，**不**进 failed）；ready 对象缺失 → Lua 原子解封（sealed→accepting + 删除该片记录），重传走全新声明；异步执行中发现缺片 → 解封 + 文档回 `uploading` + 任务回 `queued`（**不计失败次数**），补传后重复 complete 补封口续跑同一任务；
- `operation_id`：服务端内部处理令牌（CAS 三条件），异步模式下 = `job_id`（接管后按同一 operation_id 安全重跑）；幂等锚点是 `upload_id`。

## 异步建库（迁移 007：`kb_index_jobs` 任务队列）

**两阶段上线**：先部署兼容代码 + 迁移 007（新旧语义共存，默认同步）→ 排空旧
同步请求并执行 journal reconciliation → 置共享开关 `kb_control.kb_async_enabled=1`
一次性启用异步 API 与 Worker（避免滚动期新旧语义混用）。开关未设置时回退
`KB_ASYNC_API`（默认 false = 同步）。

- **入队**：`INSERT kb_index_jobs` 与文档状态 CAS（`uploading→queued` /
  `indexed→delete_queued`）**同一事务**；唯一键 `(operation, doc_id)` 保证重复
  请求返回同一任务；SQL 失败整体回滚，重复 complete 可补入队；分片先幂等封口；
  已取消的上传会话为终态；若需重新上传须创建新会话。
- **任务状态**：`queued → running → succeeded | failed | blocked`；
  `retry_wait`（退避中，到点可再领取）；仅文档仍为 `queued/delete_queued`
  时可转 `cancelled`，提交点后的 `retry_wait` 禁止取消。
- **领取**：`SELECT … FOR UPDATE SKIP LOCKED`，单实例并发 1（每实例单任务）；
  领取即生成新 `lease_token` 并 `attempts+1`；租约默认 120s（**MySQL 服务端时间**
  计算），heartbeat 每 30s 续租；租约过期 = 原 Worker 崩溃，其他实例接管。
- **提交安全**：每个副作用（journal 写/原件/文件移动/构建/alias 切换/指针/元数据
  CAS）前同时校验**写锁 assert_held + lease_token 所有权**——失去所有权的旧
  Worker 不得继续提交（提交点前后同样拦截）。
- **阶段/进度**：`validating|parsing|waiting_for_lock|chunking|embedding|
  writing_index|activating|finalizing`；embedding 每批（64 chunk）推进 progress
  并顺带续租；阶段耗时打 `kb_job_stage_seconds`。
- **失败分流**（`job_store.fail`）：锁等待 → 回 `queued` 且 attempts 回退（**不计
  失败**，短延迟重试）；分片未就绪 → 同上（30s 延迟等补传）；提交点后可确定故障
  → `retry_wait` **持续重试，不进入普通死信**；暂态故障 → 指数退避（基数 5s，
  上限 300s，±30% 抖动），超 5 次 → `failed`（可人工重试）；永久输入错误 →
  `failed(retryable=0)`。
- **Worker 接管语义**（journal 相位表）：`validating` 且无 journal → 按同一
  `job_id/operation_id` 从解析阶段安全重跑；`PREPARED/INDEX_BUILT` → 回滚
  （uploading/indexed）后重跑；`ACTIVATING/ALIAS_ACTIVATED/POINTER_UPDATED` →
  只前进；alias 指向未知代 → 任务 `blocked` + 维持 `kb_write_blocked` 全局写
  阻塞，等待人工 reconcile（人工修复后 `POST /jobs/{id}/retry` 重排）。
- **优雅停机**：SIGTERM → 停止领取新任务；当前任务在宽限期（默认 120s）内收尾，
  超时主动让出租约（接管立即发生）；期间 heartbeat 照常续租。
- **保留期清理**：`succeeded/cancelled` 90 天、`failed/blocked` 180 天
  （`finished_at` 起算）。
- **观测**：`kb_job_backlog{operation}`、`kb_job_oldest_age_seconds`（积压 >10min
  告警日志）、`kb_job_blocked`、`kb_job_retries/dead/takeover/lease_lost/
  lock_wait_total`、`kb_job_stage_seconds`、`kb_job_e2e_seconds`。

## 提交点与恢复（ES Alias + Redis pointer 两阶段）

```
PREPARED → INDEX_BUILT → ACTIVATING(fsync) → ALIAS_ACTIVATED → POINTER_UPDATED
  ↑ journal 先于 CAS 写（行 indexing 时 journal 必然存在）      ↑ 提交点
```

- 相位 < ALIAS_ACTIVATED：**可回滚**（文件移出知识目录/trash 移回 + 状态回退）；
- 相位 == ACTIVATING：**查真实 alias**——== pending target → 只前进；== 旧 target → 回滚；**指向其它代 → `kb_write_blocked` 阻塞全部后续 KB 写**（`reconcile` 人工三态确认后清除）；
- 相位 ≥ ALIAS_ACTIVATED：**只前进**（activate 幂等重跑 + CAS 前进），绝不回滚文件；
- Alias、pointer 或 journal 读取状态未知时 **fail-closed**，保持 `kb_write_blocked`，等待人工 reconcile；不得把读取异常当作「未切换」而回滚。
- 恢复在**取得统一写锁后、接受任何新写前**执行（先恢复全部残留 journal）。

## 部署要求（关键！）

1. **共享指针/断点依赖 Redis**：`GenerationStore(strict_shared=True)`（生产）——
   Redis 写失败 **fail-closed**（不降级本地文件）；`strict_shared` 且无 Redis →
   构造即失败。
2. **锁是部署期选择，运行时不降级**（`KB_WRITE_LOCK_BACKEND`）：
   - `auto`：方言 mysql → `MysqlAdvisoryLock`（`GET_LOCK`，连接租约，无 TTL 抢占）；
     否则 Redis → 可续租租约锁；双不可用 → 报错（要求显式 `file`）；
   - `mysql/redis` 后端故障一律 **503**（绝不降级文件锁——杜绝双锁域双写者）；
   - `file` 仅显式配置用于单机开发。
   - 语义边界（如实记录）：**不是严格 fencing**——接受「assert_held 与 ES
     alias 更新之间」的极小窗口；Redis 租约另存「暂停越过租约期」极小竞态，
     由 assert_held 在移动文件/build/alias 切换前后拦截。
3. **分片/知识目录**：
   - 分片存储 `local`（默认，单 Pod 或共享卷）| `s3`（对象存储，跨 Pod 共享；
     临时对象 `{seq}.{token}.tmp` 与正式对象 `{seq}-{sha256}` 均为内容寻址）；
   - `knowledge/uploads/`、`knowledge/.staging/`、`knowledge/.trash/` 与知识库
     **同挂载点**（同卷 rename 原子；跨卷退化为复制+fsync）；`.staging/.trash`
     为隐藏目录，`chunk_kb_dir` 扫描时跳过（**不会**被误索引）；
   - 多 Pod：必须 Redis 锁 + S3（或共享卷）；`knowledge/` 目录需共享卷
     （build 是本地文件扫描，helm PVC 后续接入）。
4. **子进程解析**（`parse_guard`）：`python -m app.agent.rag.parse_worker` +
   subprocess（**不用 multiprocessing.spawn**——Windows 上 spawn 会对
   `sys.argv[0]` 做 runpy，uvicorn/pytest 入口不可用）；超时终止可杀；
   zip 解压字节/压缩比、PDF 页数、文本长度（解析炸弹防护）。
5. **KB Worker Deployment**（异步模式）：独立 Deployment（`kbWorker`，默认 2 副本、
   **无 Service**），命令 `python -m app.agent.rag.kb_worker`，进程门禁
   `KB_WORKER_ENABLED=1`；复用 Web 的 MySQL/ES/S3/RWX 共享卷/模型凭据；
   **Web Pod 不启动 Worker**。副本数只提高容灾与解析并行度——索引提交由
   kb_write 全局写锁 + 任务租约串行化（仍是全库重建，非增量）；
   `terminationGracePeriodSeconds ≥ KB_JOB_LEASE_SECONDS`。

## 保留期与 GC

- `indexed`：原件**永留**；`deleted`：原件 90 天、trash 文件 30 天；
  `failed`：现场 30 天；`cancelled`/过期会话：审计行保留，文件/对象清理
  （时间一律以 `status_changed_at` 计算，不用文件 mtime）；
- 定时 GC（`_kb_gc_loop`，默认 6h，上限 100 项）+ complete/delete 机会式 GC；
  GC 须持统一写锁（持锁内复用当前 guard，不重入）。

## 运维命令

```bash
# 全量构建/上传触发构建（锁：auto 解析）
python -m app.scripts.build_kb_index
# 一致性核对（alias vs pointer vs journal）
#   —— 由 reconcile（IndexBuildService）在恢复机内自动处理；
#      人工恢复入口：修复 alias 后重跑 complete/delete（幂等）或将
#      kb_control.kb_write_blocked 清空（先确认 alias/pointer/MySQL 三态一致）
```

## 不可回退的表白（评审确认的取舍）

- 上传文档规范化 Markdown 入库（二进制不写 frontmatter）；原件存
  `originals/{doc_id}/original.{ext}`（**用户 filename 绝不进路径**，仅展示）；
- 全部会改变正式 alias 的构建（上传/下架/自进化/手工）一律 **strict=True**
  （任一源文件解析失败即中止——杜绝「一次重建静默丢知识」）；
- 文件锁残留：`evolution.lock` 的接管/强制解锁沿用 `run_evolution --force-unlock`
  语义（跨主机锁的唯一解法）；Redis 锁靠 TTL 自动过期。

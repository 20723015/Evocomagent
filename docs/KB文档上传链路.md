# KB 文档上传链路（人工上传 → 分片断点续传 → 版本化重建入库）

> 状态：已实现（工作区）。本页记录架构、部署要求与运维操作。
> 设计评审历史（8 轮）要点全部落在代码注释与本页「不可回退的表白」节。

## 一句话

运营把 **md/txt/pdf/docx/html** 文档上传到知识库：Redis 记断点状态（分片级续传），
MySQL `kb_documents` 为元数据正本（状态机 + CAS），解析/消毒后规范化为
**Markdown 入库**（frontmatter 携带 `provenance: upload:{id}` / `owner: ops`），
随后复用 `IndexBuildService` 全量重建版本化索引（**strict**）——验证通过后先切
ES Alias，再更新 Redis generation pointer；线上 retriever 由 generation pointer
驱动热刷新，无需重启。Alias 与 pointer 是两个需要一致性核对的阶段，不把 Alias
单独当作线上 retriever 的最终入口。

## API（8 端点，全部 `SCOPE_OPS`）

| 端点 | 说明 |
|---|---|
| `POST /v1/kb/uploads` | body：`filename/size_bytes/content_type?/chunk_size?/sha256?/upload_id?`；`upload_id` 为幂等锚点（同 id 参数不一致 → 409）；返回 `upload_id/total_chunks/received[]` |
| `PUT /v1/kb/uploads/{id}/chunks/{seq}` | 二进制分片（octet-stream）；重传同内容幂等、异内容 409（**原分片不破坏**）；超发 413 |
| `GET /v1/kb/uploads/{id}` | 断点状态（`received[]` 供续传） |
| `POST /v1/kb/uploads/{id}/complete` | 同步入库（幂等：已 indexed 直接返回；`uploader` 必须=创建者，403 否则） |
| `DELETE /v1/kb/uploads/{id}` | 取消（封口→清对象→cancelled） |
| `GET /v1/kb/documents` | 列表（ops 全量） |
| `GET /v1/kb/documents/{doc_id}` | 详情 |
| `DELETE /v1/kb/documents/{doc_id}` | 下架（trash 隔离→重建成功→deleted；失败自动回滚 indexed） |

`uploader` 语义：认证身份（`AUTH_ENABLED` 关闭时回退 body/query `uploader`）；
complete/delete 校验行归属（不一致 403），create 幂等比对（不一致 409）。

## 状态机（全部 CAS：status + version [+ operation_id]）

```
uploading ──→ validating ──→ indexing ──→ indexed
   │  │           │  │          │  │
   │  ├→ failed   │  ├→ cancelled│  ├───────→ uploading（系统错误回退：锁冲突/构建异常）
   │              │              │
   └──────────────┘（锁冲突/分片未就绪/校验重试 → uploading 可重试）
indexed ──→ deleting ──→ deleted；deleting 仅提交点前可回滚 indexed
```

- **输入错误**（hash/magic/解析/消毒/超限）→ `failed`（终态，重新上传）；
- **分片未就绪**（缺片/存在 publishing）→ `UPLOAD_INCOMPLETE` 409 + 回退 `uploading`（续传语义，**不**进 failed）；ready 对象缺失 → Lua 原子解封（sealed→accepting + 删除该片记录），重传走全新声明；
- `operation_id`：服务端内部处理令牌（CAS 三条件），GET 返回仅供诊断；幂等锚点是 `upload_id`。

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

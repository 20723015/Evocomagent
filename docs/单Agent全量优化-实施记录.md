# 单 Agent 全量优化 · 实施记录（2026-09-05）

按《单 Agent 全量优化计划》完成一次性切换，并按《单 Agent Review 问题全量
修复计划》完成 8 个问题的单批修复（迁移 006 + 退款确认闸门 + 终答协议收紧 +
SQL 正本归一化 + 异步记忆可靠性 + 事实冲突按主题锚点）。本文记录实施内容、
兼容性边界与发布前仍需在线上评测环境完成的门禁项（LLM 实测类）。

## 0. Review 问题全量修复（2026-09-05 追加）

### 会话正本与终答协议
- SQL `load` 按 `turn_id` 分组归一化（`normalize_model_history`）：只返回
  user + 最后一条非 tool-call assistant；缺失终答轮次只保留 user（不产生
  孤儿 tool 消息）；完整审计仍在 `chat_messages` 行正本中。
- 最终 assistant 同时写入审计增量（`ctx.full_turn_messages`）——SQL、outbox、
  重载历史均包含最终答复。
- 混合业务工具 + final_response：只写一条 assistant tool-call 消息（含全部
  call），每个 call ID 恰好一个 tool 结果（混合 → PREMATURE；纯多 final →
  协议错误）；被接受的 final call 也补记接受结果（审计消息序完整）。
- `PlainTextFinal` 成功路径删除：纯文本输出 → 追加协议纠错（system 说明）并
  重试（指标 `final_protocol_corrections_total`）；强制收尾显式
  `tool_choice=final_response`，失败 → 确定性转人工 fallback
  （指标 `turns_missing_final_total`）。

### 服务端退款确认闸门（`app/agent/refund_gate.py`）
- 程序层三态判定：`confirm / cancel / ambiguous`（否定优先；明确确认词或
  唯一待确认的简短肯定回复才 confirm；多笔待确认未带订单号 → 确定性澄清
  问题，零 LLM）；ambiguous 本轮禁止一切写工具。
- 模型可见工具参数仅 order_id/reason；`confirmation_token/idempotency_key/
  refund_id` 为保留字段，模型提交一律 `RESERVED_TOOL_ARGUMENT` 拒绝。
- token 只存确认存储（`user_id+session_id+refund_id` 反向解析），不再写入
  消息 metadata、outbox、日志或模型上下文（fold/上下文构建层剔除保留字段；
  工具结果写入前统一脱敏 token/凭证）；执行器仅在服务端判定 confirm 时经
  **内部参数通道**（校验后合并）注入 token 与幂等键；cancel 删除待确认状态
  并作废 token。
- token 载荷绑定用户/会话/订单/原因/refund_id，提交时全量校验；绑定失败
  尝试同样烧掉 token（单次语义）。旧会话 metadata 中未过期 token 仅供服务端
  内部迁移进确认存储。
- reset 清理待确认退款状态（按重置前 session 标识）并清理遗留 memory job。

### 异步记忆可靠性
- 迁移 `006`：`memory_jobs.session_uuid`（从 sessions 回填），
  `EXPECTED_SCHEMA_VERSION=6`；worker 比对 job 与当前 session UUID，不匹配
  → `obsolete`（不读取新会话消息，且绝不确认完成）。
- reset 在同一事务删除该 session 的 memory jobs（SQL delete）；文件模式
  `purge_session`。
- SQL 水位原子单调（UPDATE 内 `case` clamp，数据库现值为准；
  `watermark_regression_blocked_total` 观测回退阻断）；worker 推进水位后
  失效 Redis 会话热缓存；水位语义 = `chat_messages.seq`，历史压缩不再重置。
- 文件队列负载自包含（任务条目携带待巩固消息，不再依赖 marker 文件与可能
  回退的消息长度）；`leased_by/lease_until` + 过期接管 + 原子重写
  （tmp+os.replace，Windows 兼容）+ 完成记录（done log）；enqueue 按
  (session_key, through_seq) 幂等。

### 事实冲突与可靠度
- 数字声明按「回复子句首词主题锚点 + 单位」匹配证据；同一声明键在证据间
  存在多个不同值且回复选取其一 → conflict：删除该句并转核实/转人工话术，
  可靠度上限 0.2；不同主题同单位（7天退货 vs 15天到货）不误报。

### Review 新增监控
`final_protocol_corrections_total`、`turns_missing_final_total`、
`refund_confirmation_blocked_total{reason}`、`memory_job_obsolete_total`、
`watermark_regression_blocked_total`、文件任务复用 `memory_job_retries_total`。

---

## 1. 已落地内容（代码级）

### 阶段A：执行链路收敛
- `EcomAgent` 收敛为统一流水线：`输入策略 → 上下文构建 → ReAct/工具执行 →
  结构化终答 → 安全与事实校验 → 持久化 → 异步记忆`。
- 拆分组件（均显式接收 `AgentTurnContext`，替代实例临时字段）：
  - `app/agent/input_policy.py`：guardrail + 范围闸门 + 业务升级规则的唯一实现；
    API（`server/main._input_prefilter`）与 Agent 内共用同一核心。
  - `app/agent/context_builder.py`：system+Skill、记忆片段、摘要、历史窗口
    的唯一组装点。
  - `app/agent/react_runner.py`：ReAct 循环（见阶段A2）。
  - `app/agent/turn_finalizer.py`：固定顺序收尾管线。
  - `app/agent/turn_repository.py`：轮次持久化唯一出口。
- `TurnFinalizer` 固定顺序：写操作状态校验 → 引用与事实接地 → 业务升级 →
  输出安全与脱敏 → 可靠度计算 → 保存 → Handoff 事件。
- 普通与 SSE 复用 `server/_complete_handoff` 完成器：**SSE 此前漏建 Handoff
  工单，已修复**；两接口语义一致。
- SSE `thought` 事件改为受控状态说明（步数），不再输出/记录模型原始思考。

### 阶段A2：final_response 终止协议
- 虚拟终止工具 `final_response`（`app/agent/final_response.py`）：不进
  ToolManager，只在发给模型的工具列表末尾追加；参数 Pydantic 严格校验
  （`extra="forbid"`）；不合法 → 稳定错误码 JSON 回模型，可在剩余步数内修复。
- 达到最大步数：最后一轮只挂 `final_response` 强制收尾；失败/预算耗尽走
  确定性 fallback（可靠度 0.0）。
- **正常路径的第二次 `_extract_structured_response` 调用已删除**；纯文本
  （未走协议）兼容视为终答。模型自评 confidence 不再采纳。

### 阶段B：上下文与成本治理
- 每轮历史只保存一个 assistant 消息：`content`=用户可见回复，结构化字段
  （intent/confidence/requires_human/follow_up/tools 摘要/pending_writes/
  sources）进 `metadata`（schema=2）。
- 旧会话的「JSON 副本」助手消息仅在构建模型上下文时折叠（`fold_history`），
  历史正本不修改。
- 历史压缩从消息条数改为 token 水位（`app/agent/token_budget.py`）：
  system+Skill 20% / memory 10% / 对话+工具 50% / 输出预留 20%
  （`CONTEXT_WINDOW_TOKENS`，默认 32768）。
- `llm_max_tokens` 全量接线：ReAct、强制终答、摘要、STM/LTM 提取。
- 工具完整结果只进审计存储（append_log→SQL 行式正本 / evolution turn 记录）；
  后续上下文只保留 digest 摘要（metadata.tools）。跨轮退款确认所需的
  `confirmation_token/idempotency_key` 以受控系统说明回注（唯一例外）。

### 阶段C：写操作状态机
- `app/agent/write_ops.py`：`collecting → ownership_verified →
  awaiting_confirmation → executing → committed|rejected|indeterminate`。
- 执行器前置拦截（`ToolBatchExecutor`，预算检查优先于写闸门）：缺订单号/
  原因只追问；越权拒绝单本轮禁重试；未注册状态机的写工具默认禁止执行。
- 归属校验由工具侧两段式强制（签发前过网关 get_order，fail-closed），
  状态机不重复设门以保跨轮确认。
- 无本轮 `committed` 证据时「退款成功」宣称被程序改写 + 强制转人工
  （`TurnFinalizer._apply_write_op_state`）。

### 阶段D：RAG 与可信回答
- `search_knowledge` 保持原参数兼容，新增可选 `queries`（≤3 子查询）：
  并行召回 → RRF 合并 → 父块去重 → Top-K；返回保持原字段 +
  `subqueries`/`evidence` 诊断。
- `app/agent/rag/evidence.py`：`EvidencePack`（doc/section/chunk/query/分数/
  tainted）， tainted 片段不进证据与合法来源。
- `app/agent/fact_guard.py`：声明级校验（金额/时效/比例/计数），无证据数字
  的句子删除，全部被删转核实/人工提示；证据冲突检测。
- 拒答校准：`app/scripts/calibrate_rejection.py` 联合 top1 分数、分数间隔、
  词面覆盖率、reranker 分数；dev 网格搜索 → 冻结（含数据集哈希/切分哈希/
  版本记录）→ holdout 仅可运行一次（重跑强制 --force 且记档不作为发布依据）。
- 评测数据修正：补充自进化文档（钻石会员 SLO）检索覆盖用例，
  `retrieval_cases.json` 附 `_change_log`（变更依据留痕），与
  `generate_eval_data.py` 生成器保持同构（门禁通过）。

### 阶段E：确定性可靠度
- `app/agent/reliability.py`：1.0（规则响应/确定性拒绝）、0.9（工具直接
  支持）、0.8（知识声明全接地）、0.6（非事实/轻微不确定）、上限 0.4（检索
  不足/参数缺失/可重试工具错误）、上限 0.2（引用缺失/事实冲突/写结果未知）、
  0.0（预算耗尽/无法生成有效终答）。硬风险只降不升，模型不可自评抬高。
- 外部 `confidence` 语义切换为该可靠度（仍 0–1，接口字段不变）。

### 阶段F：异步记忆
- `close()` 只释放本地资源（tool_manager/执行器），**不再调用 LLM**。
- STM 改为零 LLM 确定性槽位提取（`app/agent/memory/stm_rules.py`：
  显式句式 + PII 脱敏 + 每键最新覆盖）。
- 新增持久化 `memory_jobs`（`app/stores/sql/schema.py` + `deploy/sql/
  005_memory_jobs.sql`，`EXPECTED_SCHEMA_VERSION=5`）：
  - 唯一键 `session_key+through_seq`；状态 pending/processing/done/failed；
    重试退避、租约接管、脱敏错误、死信。
  - SQL 模式：消息保存与 job 入队同一事务（`SqlSessionStore.save`）。
  - Worker（`app/agent/memory/jobs.py`）：按水位读增量 → LLM 提取复杂事实 →
    原子推进 `sessions.consolidated_len`；重复投递按水位直接确认（幂等）。
    服务端 `_memory_job_worker_loop` 每 5s 批处理。
  - 文件/Redis 开发模式：轻量文件队列（`memory_dir/jobs`）；idle scanner
    仍为漏单修复器。
- 记忆注入：相关性阈值（dice ≥ 0.02）+ 有效期（365 天）+ 上限不变；
  身份/偏好不再无条件保底注入无关请求。

### 阶段G：可观测性与错误治理
- 嵌套 span：`agent.turn` 已接（其余点位沿用既有埋点：llm.call 等）。
- 新指标（`app/observability/metrics.py`）：llm_calls_total、
  tool_result_codes_total、tool_side_effect_total、context_truncated_total、
  citation_coverage_ratio、fact_guard_failures_total、write_claim_blocked_total、
  write_blocked_total、memory_job_backlog/latency/retries/dead。
- 修正 `{"error": ...}` 被统计为成功的问题（ToolManager 判定含 error 键即失败）。
- 修正成本 70/30 猜测：`record_llm_usage` 按真实 usage 的 prompt/completion
  拆分计价（`ResilientLLM._record` 归集）。
- 生产日志：用户原文、模型原始思考、完整工具结果均不落日志（摘要/状态码）。

## 2. 测试与质量

- 测试配置不再读取工作目录 `.env`（`tests/conftest.py` + `ECOM_ENV_FILE`）；
  修复了本地 `.env` 泄漏导致的 15 个隐性失败。
- 修复两个预存失败：检索评测集文档覆盖缺口（补用例+生成器同步）、
  AB 配置冻结形状断言（min_score 为有意冻结的校准产物）。
- 新增契约测试 `tests/unit/test_single_agent_contract.py`（20 条）：无工具
  单调用、工具决策+终答、纠错修复、强制终答、预算 fallback、单 assistant
  消息+metadata、旧格式折叠、跨轮确认令牌回注、不安全输出不落会话、非法
  参数不执行工具、退款成功宣称拦截、越权禁重试、未注册写工具拦截、close
  零 LLM、可靠度/水位/接地/RRF 单元。
- Ruff 接入（新增文件与改动文件清零 F/E 类问题；仓库历史风格问题不属本轮）。

## 3. 兼容性边界

- HTTP/SSE 字段、错误状态码、工具名称全部不变；`metadata`/`subqueries`/
  `evidence`/`deprecation` 均为可选新增。
- `confidence` 数值语义切换（模型自评 → 程序可靠度），字段与取值域不变；
  下游（演进挖掘、评估报告）按数值消费无需改动，但阈值解读需知悉新口径。
- 会话文档新增 `metadata` 字段为向后兼容新增；旧格式读取折叠、正本不动。
- 多 Agent 代码未纳入本轮（`multi_agent/` 未改动）。

## 4. 发布前待执行门禁（需真实模型/检索环境）

- 离线回放 + 全量评测：规则轮 ≥80%、Critical 13/13、API error=0、
  注入/越权/退款/敏感信息类别不低于正式基线。
- RAG 门禁：dev Recall@5 ≥95%、easy ≥98%、hard ≥80%、MRR/nDCG ≥90%、
  负例拒绝率 ≥90%；`calibrate_rejection --dev` 冻结后 `--holdout` 仅跑一次。
- 成本/延迟对拍：同等评测集同步 LLM 调用数 -30%、平均 token 成本 -25%、
  P95 ≤ 基线（目标 15s）；连续三轮波动 ≤2pp。
- 预生产压测 + 发布观察项：LLM 调用数、P95、工具错误、Handoff、memory
  backlog；Critical 失败/写异常/持久化错误 → 整体回滚。

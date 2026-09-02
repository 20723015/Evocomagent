# Agent 能力强化计划（v3.1 定稿）

> 目标：解除 Agent 本体的四个能力天花板——推理预算、工具并发、引用来源真实性、记忆利用率。
> 状态：**定稿，可直接实施**。经三轮评审修订（v1 方向 → v2 阻塞项修复 → v3 设计钉死 → v3.1 四处贯通补丁）。
> 边界：不动存储协议与 API 契约（SSE 事件新增字段为 additive 例外）；ES 记忆检索另立项目，不在本期。
> 基线：372 条无网络单测全绿；全部新增用例走 FakeChatClient/FakeEmbedder 无网络模式。

---

## 0. 实施形态：ToolBatchExecutor + ToolTurnState

主 Agent（chat.py）与 SubAgent（agents.py）的循环内机制统一收敛，不再各复制一套。

```
ToolBatchExecutor（pod 单例，无状态；app.state 持有；CLI/评估自建并负责 shutdown）
├── execute(tool_calls, state: ToolTurnState, ctx) -> list[ToolOutcome]
├── 参数解析（json.loads 失败 → 错误 JSON，错误回模型自愈）
├── 重复调用检测（读 state.signature 历史）
├── 并行安全分类（PARALLEL_SAFE 白名单）
├── 双层限流（见下）
├── 原序分段调度（写工具 = 串行屏障）
├── 稳定回填（按模型给定顺序返回 ToolOutcome，与执行顺序解耦）
└── trace hook：state.on_outcomes(list[ToolOutcome])（评审·三轮3）

ToolTurnState（请求级，每轮 chat() 新建——pod 单例绝不持有任何请求状态）
├── 签名历史、来源集合、事件回调句柄、sequence 计数器
└── on_outcomes：按稳定顺序一次性写入 RunTrace（供评估轨迹）
```

**双层限流**：`tool_parallelism=2`（单批次并行度）+ `tool_max_concurrent=16`（pod 全局信号量）。
pod 全局工具并发上限**严格等于 16**；单批次再受 `tool_parallelism=2` 约束。

**并发安全前置**：MCP client 补两项测试——并发请求测试、close 与在途调用竞争测试
（`run_coroutine_threadsafe` 只证明提交线程安全，不证明 ClientSession 生命周期安全）。

---

## 1. 改造一：预算扩容 + 重复检测 + TurnBudget 贯通（1–1.25 天）

- `max_react_steps` 5→8（env 可覆盖）；`SubAgent.handle` 的 `max_steps=5` 硬编码删除，统一读 settings。
- 重复检测：签名 = `name + json.dumps(args, sort_keys=True)`，同签名连续 ≥2 次 → 拦截并注入
  `{"error": "重复调用被拦截：请换参数或基于已有结果回答"}`；签名历史仅在本轮内（跨轮合法重查不受影响）。
- **TurnBudget 对象**（非裸 ContextVar）：`chat()` 起点 `deadline = monotonic() + turn_budget_seconds`
  同时存两处——Agent 实例属性 + ContextVar。**turn 与 close 两个 worker 线程各自显式绑定
  ContextVar 并在 finally reset**（close 在另一线程执行，无法继承 chat() 内的 ContextVar——
  评审·三轮1）。
- **ResilientLLM**：每次尝试 timeout 取 `min(llm_timeout, remaining)`，`remaining ≤ 0` 直接放弃
  （含重试与降级分支）；覆盖 ReAct 每步、强制收尾、结构化提取、STM/摘要/LTM 巩固。
- **工具提交规则**（ContextVar 不能中断已运行的调用——评审·三轮2）：
  - 提交前检查剩余预算，不足则不发起新工具调用；
  - MCP/HTTP 工具调用显式传递 `remaining` 作为超时；
  - 到期后停止等待**只读**任务（结果丢弃）；
  - **写工具必须同步等待其幂等结果**——绝不允许响应 fallback 后后台遗留副作用。
  以上规则落实后方可称"全链路 deadline"，否则只能叫 "LLM deadline"。
- **预算耗尽路径**：不调 LLM 强制收尾，返回确定性结构化 fallback（固定话术 + requires_human=true
  + intent=other）。**STM/摘要/LTM 辅助任务直接跳过或延后**——辅助任务不得毁掉已生成的回复。
- 口径：`turn_budget_seconds=120` 是安全熔断值，与 15s P95 SLO 无关；断连等待上限改为基于
  turn_budget 推导。
- orchestrator 回填真实 step count（SubAgent 返回步数并累计；现在恒为 1，分布指标失真）。

---

## 2. 改造二：原序分段并行（1.5–2 天）

- **原序分段**：按模型给定的 tool_calls 顺序切分——连续只读段内并行；写工具（apply_refund）
  作为**串行屏障**单独执行；其后新只读段再并行。执行顺序永远尊重模型原序，段内提速。
  （不采用"写工具批内最后"——那会改变模型语义。）
- **并行白名单（首批）**：query_order / query_product / query_logistics / list_user_orders /
  search_knowledge / recall_user_memory。
  **移出**：load_skill（loader.py `load_body` 无锁延迟缓存写入，加锁前不并行）、apply_refund（写）。
- **SSE 事件**：`tool_call` 增加 `tool_call_id + sequence`；`tool_result` 增加 `tool_call_id +
  tool_name + sequence`——additive，老客户端忽略新字段，更新 SSE 事件文档。
- **评估轨迹保序**（评审·三轮3）：sandbox 现有 `_wrap_execute_tool(name, args)` 拿不到 sequence，
  **保序方案改为 trace hook 移到 Executor 协调层**——`ToolTurnState.on_outcomes(list[ToolOutcome])`
  按稳定顺序一次性写入 RunTrace；旧 ToolManager wrapper 仅保留兼容或移除。
  RunTrace 扩展：`retrieved_sources`（实际来源集合）、`citation_verdict`（评估 citation_check 输入）。
- **测试**：`max_active == 2`（单批次观测峰值，受双层限流共同约束）+ 回填顺序稳定断言 +
  写屏障语义断言（屏障前段完成先于写工具执行）；不做严格墙钟断言（CI 抖动）。

---

## 3. 改造三：引用来源真实性（1–1.5 天）

> 命名口径：本能力证明**引用名称来自本轮召回集合**，不证明回答内容被该文档支持
> （后者是演进管线接地 Judge 的领域）。

**引用提取**（宁漏勿误杀）：
- 仅认两种：① 引用语境中的《书名》——书名号前 6 字符内含"根据/依据/来源/参考/政策/文档"之一；
  ② 显式文件名（含 `.md/.txt/.pdf/.docx/.doc/.html` 后缀）。
- 孤立书名号（商品名、书名）不触发。

**来源归一**：NFKC → basename → 去扩展名 → casefold；命中集合同时收 `doc` 与 `source_path` 两路别名。

**tainted 排除**：结果字段为 `tainted=true` 的检索块（注意：`kb_chunk_tainted` 是函数名不是字段名）
不进合法来源集合——被污染的召回不能为引用背书。

**Verdict**：`{cited: [...], matched: [...], missing: [...]}` 三态全量保留，进日志、trace、RunTrace；
比较用规范化值，报告保留原文。

**校验时机**：Agent 内部（单/多同约定），`_extract_structured_response` 之后、`_record_turn`/
`_save_session` 之前——confidence/requires_human 的修改先于落库与演进采集，三处一致。
来源采集经 ToolBatchExecutor 统一返回，协调线程合并（工作线程不写共享集合）；
`_turn_sources` 每次 `chat()` 开头重置（防 CLI 多轮串线）。

**分级处置**：有引用不匹配 → confidence 压低 + 告警；零检索却有引用 → requires_human=true；
无引用（纯闲聊/纯工具数据）→ 放行。

**评估接线**：
- EvalCase 增加 `expected_citations: list[str]`、`forbid_unretrieved_citations: bool`；
- **citation_check 公式（定死）**：
  - 有 `expected_citations`：得分 = `|expected ∩ matched| / |expected|`（规范化值比对）；
  - `forbid_unretrieved_citations=true` 且 `missing` 非空：直接 0；
  - 两项均未配置：返回 `None`（不计入通过判定）；
- 无网络验收：脚本化 FakeChatClient 用例（预置回复与召回集合，断言分级处置）；
  CI 数据集门禁校验新字段格式；63 条真实模型通过率仅作联网回归指标。

---

## 4. 改造四：记忆相关性 naive 版（1–1.25 天）

- **接口**：`MemoryManager.build_memory_prompt_sections(query: str)`；query = 本轮原始
  `user_input` 在 `chat()` 签名处显式传入（单/多 Agent 同），不从消息尾部猜测。
- **最终注入 = 严格 ≤8 条**（含保底；保底插入后淘汰最低分非保底项）：
  - 保底：identity、preference 各 1 条（取该类得分最高者；已进 top-8 不重复占位）；
  - 得分 = **词面分（Dice 系数）** + 类别权重 + 新近度衰减：
    - Dice：中文字符 bigram 集合 `2|A∩B| / (|A|+|B|)`；**任一集合为空返回 0**；
      ASCII 段先 casefold 再按空格分词、整词进集合；
    - 类别权重：identity +0.15 / preference +0.10 / behavior +0.05 / issue +0.05 / other +0；
    - 新近度：`0.1 × exp(−age_days/180)`；**时间统一转 UTC 后计算**（避免 naive/aware 混算），
      `age = max(0, now_utc − created_at_utc)`（未来时间 clamp 到 0），created_at 缺失按 90 天；
  - 同分排序：得分降序 → created_at 降序 → content 字典序。
- **交互摘要保持注入最近 3 条不变**（long_term.py:202 现状；此前版本误写 2 条）。
- `recall_user_memory(query="")` 全量（兼容旧行为）；非空按同公式返回 top-10。
- `max_ltm_facts=50` 本期不放宽——SQL merge 为普通 SELECT 后删写、无 `SELECT FOR UPDATE`/
  用户级锁，REPEATABLE READ 下并发事务可能读同一旧快照；放宽前先补真实 MySQL 并发验证。

---

## 明确移出本期

**ES 记忆检索**（另立项目，估 2–3 人天）：需 memory 索引 mapping、写入同步、存量回填、
user_id 隔离过滤、ES 不可用降级五件套。触发条件：facts >500 条，或 naive 命中率可量化不足。

---

## 工期与验收

| 改造 | 工期 | 关键验收 |
|------|------|----------|
| 一：预算+检测+TurnBudget | 1–1.25 天 | 重复被拦且自愈；预算耗尽返回零 LLM fallback；close 线程同受预算；写工具同步等幂等结果 |
| 二：分段并行+Executor | 1.5–2 天 | max_active==2；回填顺序稳定；写屏障语义；trace 按 sequence 保序；SSE 新字段 |
| 三：引用真实性 | 1–1.5 天 | 分级三态；校验先于落库；tainted 排除；citation_check 公式单测；FakeChat 用例 |
| 四：记忆 naive | 1–1.25 天 | 严格 ≤8；Dice 空集=0；UTC age；摘要仍 3 条；空 query 兼容 |

**合计 4.5–6 人天**。基线 372 条无网络单测保持全绿。

## 风险清单

| 风险 | 应对 |
|------|------|
| 步数 8 拉长最坏墙钟 | turn_budget 熔断 + 工具提交规则 + 断连上限推导式 |
| 并行改变事件/轨迹顺序 | sequence 字段 + on_outcomes 协调层保序 + 专项断言 |
| 引用校验误杀 | 只认语境引用；处置默认降置信不硬拦；黄金集观察误杀率 |
| 记忆漏注入关键事实 | identity/preference 保底 + 头部注明"已按相关性筛选" |
| MCP 并发生命周期未知 | 并发/close 竞争两项测试前置，不过则白名单只留本地工具 |
| SQL LTM merge 并发覆盖 | max_ltm_facts 不放宽；MySQL 并发验证列为放宽前置 |
| 两套 ReAct 实现改漂移 | 本期 Executor 收敛；长期抽公共循环（记账） |

## 评审修订记录

- **一轮（v1→v2）**：pod 保护声明错误、引用校验时机、多 Agent 来源覆盖、评估数据模型缺口、
  8 步与 SLO 冲突、记忆公式空白、ES 就绪度修正。
- **二轮（v2→v3）**：双层限流拆分、ToolTurnState 请求级隔离、原序分段替代"写工具批内最后"、
  全链路 deadline 概念、引用匹配规则、Dice/UTC/严格 ≤8、评估轨迹保序、工期 4.5–6。
- **三轮（v3→v3.1）**：TurnBudget 跨线程贯通（close 显式绑定）、辅助任务跳过策略、
  工具提交规则（remaining 传递/只读弃等/写工具同步等幂等）、trace hook 迁 on_outcomes、
  citation_check 公式定死、tainted 字段名纠正、全局并发上限口径、Dice 空集与 UTC 细则。

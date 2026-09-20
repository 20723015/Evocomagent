# ReAct 步数余量感知 · 实施记录（2026-09-16）

按《ReAct 步数余量感知补全》计划完成 6 项修改：runner 注入、指标、SSE 文案、
单元测试、评估可观测性、文档。这是**主动加固**而非缺陷修复——当前线上
`ForcedFinalizeFailed = 0`，但实测缺口是「模型看不到自己还剩几步，最后
一两步还在开新查询方向」，只能在步数耗尽后被动收到最后通牒。

## 1. 动机与设计边界

- 模型窗口里没有任何「剩余步数」信号；唯一的步数相关干预是耗尽后的
  `FORCED_FINALIZE_SOFT_PROMPT`（最后通牒，只挂 final_response）。缺一个
  「预告」层：在耗尽**之前**告知模型尽快收尾。
- **零额外 LLM 调用**：预告通过 `_push_window_message` 注入一条 system 消息，
  不新增任何模型请求；单测断言 `len(client.calls)` 与无预告剧本完全一致。
- **当步即生效**：注入发生在本步 `build()` 之前（`_step` 每步经
  `context_builder.build` 重建窗口），模型在同一步就能看到。
- **只注一次**：`margin_hint_sent` 标志防重；后续步沿用窗口中同一条消息。
- **边界**：`max_react_steps ≤ 2` 时首步即满足阈值（remaining 含当前步），
  预告会在首步注入——默认 8 步不受影响；行为事实正确（确实只剩 1-2 步），
  刻意不加分支特判，避免阈值与步数上限耦合出新特例。
- **不泄漏进历史**：消息虽进 `raw_messages`（窗口正本），但轮末
  `turn_repository._collapse_turn` 清空 user 消息之后的全部中间消息，与
  `PLAIN_TEXT_CORRECTION` / 软强制提示的既有先例一致。
- **语义互补**：预告（`STEPS_MARGIN_PROMPT`）是建议——仍可不调
  final_response 继续干活；最后通牒是协议强制。两者不重叠、不互替。
- 明确排除（维持现状）：不做教科书式 Thought 模板、不引入规划器/反思、
  不放开纯文本步、不改 `ContextBuilder`、不改预算/挂钟时间语义、
  不改 `max_react_steps` 默认值（8）。

## 2. 修改清单

### 2.1 注入（`app/agent/react_runner.py`）
- 新常量：`STEPS_MARGIN_REMAINING = 2`（阈值，含当前步）与
  `STEPS_MARGIN_PROMPT`（预告文案，含 `{remaining}` 插值）。
- `run()` 循环：`remaining = max_steps - step`；`remaining <= 2` 且未注入过 →
  置 `ctx.steps_margin_hint = True`，推送 system 预告，调用
  `record_steps_margin_hint()`（本地导入，与相邻指标调用同范式）。
- 触发步 SSE 状态语改为收尾话术「正在整理您的请求（第 X/N 步）」，其余步
  保持「正在处理您的请求…」；`emit_reasoning` 不动。
- 纯文本纠错路径补记 `ctx.protocol_corrections += 1`（此前只有指标没有
  轮次上下文计数，评估层无法按轮归因）。

### 2.2 轮次上下文（`app/agent/turn_context.py`）
- 新字段：`steps_margin_hint: bool`、`protocol_corrections: int`。
- `to_usage_dict()` 新增输出 `steps_margin_hint` / `forced_finalize` /
  `protocol_corrections`（协议级观测随 usage 进日志与审计切片）。

### 2.3 指标（`app/observability/metrics.py`）
- `STEPS_MARGIN_HINTS = Counter("steps_margin_hints_total", ...)` +
  `record_steps_margin_hint()`。触发率 = 该计数 / 轮数，即「早收尾提示」
  的影响面上限（零额外 LLM 调用）。

### 2.4 评估可观测性（`app/evaluation/{trace,sandbox,evaluator}.py`）
- `RunTrace` 新增 4 个协议级字段：`react_steps`（各轮步数之和）、
  `steps_margin_hint` / `forced_finalize`（任一轮触发即真）、
  `protocol_corrections`（各轮之和）；`to_dict()` 透传进 case 级 trace 快照。
- `Sandbox.run()` 每轮 `agent.chat()` 返回后从 `agent._last_turn_ctx` 采集
  （与来源集合 / 引用 verdict 的采集同一位置）。
- `Evaluator._aggregate()` summary 新增 `react_protocol` 块：
  `steps_margin_hint_rate`（预告触达率）、`forced_finalize_rate`（最后通牒
  触发率）、`avg_react_steps`、`protocol_corrections_total`。预告触达率对照
  forced_finalize 率，可评估预告是否前置消化了步数耗尽。

## 3. 测试

- `tests/unit/test_single_agent_contract.py`（修改4）：
  - `test_steps_margin_hint_injected_once_before_last_two_steps`：
    max_steps=4 的耗尽剧本——零额外 LLM 调用（恰好 5 次调用）、时机
    （第 1/2 步无预告、第 3 步起可见且含「还剩 2 步」）、单次注入
    （后续窗口出现次数恰为 1、计数器增量=1）、触发步 SSE 文案、
    `ctx.steps_margin_hint` 与 usage dict 可观测、轮末历史折叠后不残留
    （roles == [user, assistant]）。
  - `test_short_turn_no_margin_hint`：1-2 步即完成的短轮次零注入
    （计数器零增量、ctx 标记 False、任何窗口无预告）。
  - 既有强制终答 / 纯文本纠错契约测试全部保持通过。
- `tests/unit/test_sandbox_eval.py`（修改5）：`fake_chat` 注入
  `_last_turn_ctx` 桩，断言沙箱按轮累计（2 轮 × 3 步 = 6、纠错和=2）
  且 `to_dict()` 透传。
- `tests/unit/test_eval_v2_protocol.py`（修改5）：离线 `_aggregate` 测试，
  断言 `summary.react_protocol` 四项口径与 case 级 trace 透传。
- 全量回归：失败清单与基线（改动前 stash 复跑）**双向 diff 为空**——
  35 个失败全部为缺 `OPENAI_API_KEY` 的既有网络/配置类测试
  （test_react_agent / test_skills / test_multi_agent / test_memory /
  test_mcp / test_evaluation / test_conversation_management / test_agent），
  本次改动零新增失败。

## 4. 遗留与发布前门禁

- **线上前后评测待补**：本环境未配置 `OPENAI_API_KEY`、`eval_judge_model`
  为空，计划中的 before/after 评测（`--run-id react-hint-before` /
  `react-hint-after`，同一 `cases_large.json` + 同一 Judge，manifest 记录
  git commit）无法在本机执行。预期口径：pass_rate / keyword_coverage /
  tool_calls 分布 / avg_tokens 基本持平（不追求提升，验证无回归），
  `ForcedFinalize_FALLBACK` 保持 0，`react_protocol.steps_margin_hint_rate`
  给出预告实际触达面。
- 若触达率显著高于 forced_finalize 率且早收尾行为改善，可考虑后续把阈值
  （`STEPS_MARGIN_REMAINING=2`）做成 settings 可配；当前为常量，避免
  过早参数化。

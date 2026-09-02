# Ecom-Service-Agent: 从0到1实战企业级电商客服Agent系统

> **项目作者：** 雷腾（独立开发与维护）
>
> 本仓库为完整可运行的工程实现：Agent 主链路、RAG、记忆、评测体系、自进化闭环、生产化改造与多实例验证均由项目作者独立完成。

---

## 快速开始（Quick Start）

跑起来只需要一个 OpenAI API Key，5 分钟即可看到「小夕」上线对话。

```bash
# 1. 进入项目并创建虚拟环境（Python 3.11+）
cd ecom-service-agent
python3.11 -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，至少填入：
#   OPENAI_API_KEY=sk-你的key
#   （可选）OPENAI_BASE_URL=https://... 如用中转/代理
#   （可选）MODEL_NAME=gpt-4o-mini

# 4. 构建知识库索引（RAG 检索需要，首次运行一次即可）
python -m app.scripts.build_kb_index

# 5. 启动对话
python main.py
```

启动后直接输入问题即可，试试这些：

- `我的订单还没发货，怎么回事？` —— 触发订单 + 物流查询
- `有没有宽松透气的裤子推荐？` —— 触发商品推荐技能
- `这件衣服质量有问题，我要退货` —— 触发退货退款流程

每条回复底部会显示 `[意图 | 置信度 | 是否转人工]`。对话中还支持这些命令：

| 命令 | 作用 |
|------|------|
| `skills` | 查看已加载的技能模块 |
| `memory` | 查看短期 / 长期记忆 |
| `reset` | 清空当前会话 |
| `quit` / `exit` | 退出 |

**开启进阶能力**（可选，改 `.env` 后重启即可）：

- `MULTI_AGENT_ENABLED=true` —— 多 Agent 协作（售前/售后/投诉分流）
- `MCP_ENABLED=true` —— 通过 MCP 协议调用工具（需另起 `python mcp_server/server.py`）
- `RAG_BACKEND=chroma` —— 换用 Chroma 向量数据库（需 `pip install chromadb`）
- `RAG_HYBRID=true` —— 混合检索（向量 + BM25 + RRF，阶段七：补精确术语盲区）

**跑评估 & 测试**：

```bash
python -m app.scripts.run_eval --judge --judge-model <different-model>
                                    # 离线评估；Judge 必须是不同模型
pytest                                # 运行全部单元测试
```

正式评测的 `report.json` 与 `manifest.json` 会记录实际 CLI 的 model、Judge、
`mode`、`use_judge`、模型端点、RAG/guardrail/tool guard 配置和阈值；断点续跑
会拒绝任何配置漂移。Judge 与被测模型使用同一 `OPENAI_BASE_URL` 时，代码只
声明“不同模型”，不宣称不同提供方或不同服务来源。

---

## 背景

很多同学对 Agent 技术感兴趣，但缺少一个完整的、可跟着动手的实战项目。本项目以电商客服为场景——业务逻辑清晰（查订单、退换货、推荐商品、售后处理），是 Agent 最经典的落地场景之一。

本仓库按由浅入深的技术路线逐步演进（第 1-10 期，每期对应一个 tag），包含完整可运行的工程实现：Agent 主链路、RAG 混合检索、三层记忆、评测体系、自进化闭环、生产化改造与多实例验证，由项目作者独立完成，并以此作为 Agent 方向的求职项目实践。

### 为什么选电商客服？

电商客服是 Agent 最经典的落地场景之一：业务逻辑清晰（查订单、退换货、推荐商品、售后处理），大家容易理解，面试中也经常被问到。

### 技术演进节奏

本仓库按由浅入深的技术路线逐期演进（第 1-10 期，每期对应一个 tag），与技术参考系列同步；你可以在本仓库通过标签逐步跟进每一期的实现。

### 技术演进路线（更新预告）

本项目会按照由浅入深的节奏，逐步叠加 Agent 相关技术：

**基础篇**
- 纯 Prompt 实现客服对话
- 结构化输出（Structured Output）
- 多轮对话管理

**进阶篇**
- ReAct 范式的 Agent（思考-行动交替，最经典的 Agent 范式）
- 工具调用 / Function Calling（查订单、查库存等）
- MCP（Model Context Protocol）集成
- RAG 检索增强生成（接入商品库、FAQ、退换货政策等）

**高级篇**
- Multi-Agent 协作（客服路由、售前售后分流）
- Memory：短期记忆 & 长期记忆
- Skill：可复用的能力模块（退货处理、订单跟踪等标准化流程）✅
- Agent 评估体系 ✅

**生产篇**
- Guardrails 安全护栏（Prompt Injection 检测、输出幻觉校验、敏感信息过滤、意图越界拦截）
- Human-in-the-Loop 人机协作（置信度评估与自动转人工、Agent↔真人客服交接协议、上下文传递）
- Agent Observability 可观测性（调用链 Trace、Token/延迟指标采集、工具成功率看板、异常告警）

> 以上为初步规划，实际更新可能会根据大家的反馈进行调整。

---

## 项目架构 & 更新历史

> 这是本项目最核心的部分，会随着每一期的更新持续完善。

### 当前架构

```
ecom-service-agent/
├── main.py                        # CLI 入口（支持单 Agent / Multi-Agent 模式切换 + memory/skills 命令）
├── requirements.txt
├── .env.example
│
├── app/                           # 主 Bot 全部代码 + 数据
│   ├── config/
│   │   └── settings.py            # 配置管理（从 .env 读取，含 MCP / RAG / Multi-Agent / Memory / Skill / Evaluation 配置）
│   ├── prompts/
│   │   ├── customer_service.py    # 电商客服 system prompt（含工具使用指南 + 记忆能力）
│   │   ├── summarizer.py          # 历史摘要 prompt
│   │   ├── agents.py              # Multi-Agent 子 Agent prompt（售前/售后/投诉 + Router）
│   │   ├── memory.py              # 记忆提取 prompt（短期 STM / 长期 LTM 事实抽取）
│   │   └── evaluation.py          # LLM-as-judge prompt（回答质量 / 幻觉 / 过程合理性）
│   ├── schemas/
│   │   └── response.py            # 结构化输出 schema（Pydantic）
│   ├── agent/                     # Agent 核心实现 + 全部 Agent 技术栈（tools / rag / skills）
│   │   ├── chat.py                # 核心 ReAct 循环（集成 MemoryManager + SkillManager）
│   │   ├── summarizer.py          # LLM 自我压缩老对话（支持工具消息）
│   │   ├── storage.py             # 会话 JSON 持久化（含短期记忆）
│   │   ├── memory/                # 记忆系统（第7期）
│   │   │   ├── __init__.py        # 导出 MemoryManager / ShortTermMemory / LongTermMemory
│   │   │   ├── manager.py         # MemoryManager：统一管理短期 + 长期记忆
│   │   │   ├── short_term.py      # 短期记忆：会话内事实提取
│   │   │   ├── long_term.py       # 长期记忆：跨会话持久化（JSON per user）
│   │   │   └── extraction.py      # LLM 事实提取（共用模块）
│   │   ├── skills/                # Skill 模块（第8期）：代码 + 技能内容分层
│   │   │   ├── __init__.py        # 导出 SkillManager / SkillMeta
│   │   │   ├── loader.py          # SkillManager：扫描、发现、加载 SKILL.md（渐进式披露）
│   │   │   └── definitions/       # 技能内容（遵循 Agent Skills 开放标准，每个一个 SKILL.md）
│   │   │       ├── process-return/
│   │   │       │   └── SKILL.md   # 退货退款处理技能（确认订单→校验资格→退款→告知进度）
│   │   │       ├── track-order/
│   │   │       │   └── SKILL.md   # 订单物流跟踪技能（查单→查物流→综合建议）
│   │   │       └── product-recommend/
│   │   │           └── SKILL.md   # 商品推荐技能（了解需求→查偏好→搜索→推荐）
│   │   ├── strategies/            # (upcoming) Agent 执行策略
│   │   ├── tools/                 # 电商工具集（Function Calling）
│   │   │   ├── mock_data.py       # Mock 数据：订单、商品、物流
│   │   │   ├── registry.py        # 本地工具注册表 + OpenAI schema + 分发执行
│   │   │   ├── manager.py         # ToolManager：统一管理本地 + MCP 工具（支持 allowed_tools 过滤）
│   │   │   ├── order.py           # 查询订单详情
│   │   │   ├── product.py         # 搜索商品信息
│   │   │   ├── logistics.py       # 查询物流轨迹
│   │   │   ├── refund.py          # 申请退款
│   │   │   ├── knowledge.py       # search_knowledge：RAG 政策/FAQ 检索
│   │   │   ├── memory_tool.py     # recall_user_memory：查询用户记忆
│   │   │   └── skill_tool.py      # load_skill：按需加载技能指令
│   │   └── rag/                   # RAG 模块
│   │       ├── chunker.py         # Markdown → Chunk（按二级标题切分，frontmatter 元数据 + .txt 接入）
│   │       ├── loader.py          # 文档接入：frontmatter 解析 + 文本规范化（阶段七 7.1）
│   │       ├── embedder.py        # OpenAI Embeddings 封装
│   │       ├── retriever.py       # KnowledgeRetriever：query → 向量检索
│   │       ├── bm25.py            # 手写 BM25（中文 bigram 分词，零依赖）
│   │       ├── hybrid.py          # HybridRetriever：向量 + BM25 双路召回 + RRF 融合（7.2）
│   │       ├── rerank.py          # Reranker 抽象 + 注入点（7.3，默认 none 不启用）
│   │       ├── backends/          # 向量后端（可切换）
│   │       │   ├── base.py        # VectorBackend 抽象接口
│   │       │   ├── numpy_backend.py   # 手写余弦 + JSON（教学透明，零依赖）
│   │       │   └── chroma_backend.py  # Chroma 嵌入式向量数据库（生产代表）
│   │       └── knowledge/         # 知识库源文档（markdown，RAG 数据源）
│   │           ├── 退换货政策.md
│   │           ├── 配送说明.md
│   │           ├── 会员权益.md
│   │           └── 常见问题FAQ.md
│   ├── mcp_client/                # MCP Client（同步封装）
│   │   ├── client.py              # MCPClient：后台线程管理异步连接
│   │   └── converter.py           # MCP Tool schema → OpenAI function calling 格式
│   ├── evaluation/                # Agent 评估体系（第9期）
│   │   ├── __init__.py            # 导出 EvalCase / Sandbox / Evaluator / RunTrace 等
│   │   ├── dataset.py             # EvalCase 数据结构 + load_dataset
│   │   ├── trace.py               # RunTrace：沙箱采集的过程+结果载体
│   │   ├── sandbox.py             # Sandbox：隔离环境 + 共享 client 插桩 + 采集（第10期：配置快照与恢复）
│   │   ├── metrics.py             # 过程/结果双层指标（代码规则 + LLM judge）
│   │   ├── evaluator.py           # Evaluator：跑用例 → 双层评分 → 聚合报告
│   │   ├── retrieval_metrics.py   # 检索质量指标：recall@k / MRR / nDCG（阶段七 7.6）
│   │   ├── retrieval_cases.json   # 检索回归/dev 集（535 条，easy/hard/no-hit 分层）
│   │   ├── holdout_cases.json     # 独立 holdout 检索集（120 条，冻结后仅运行一次，3.4）
│   │   ├── manifest.py            # eval-v2 清单：数据集/commit/prompt/模型/阈值指纹（3.1）
│   │   └── cases_large.json       # 端到端黄金集（317 条，v2 冻结，含安全用例）
│   ├── evolution/                 # QA 自动沉淀（第10期）：轮次捕获 → 挖掘 → 审核 → 发布
│   │   ├── models.py              # TurnRecord / SourceRef / CandidateQA / EvolutionReport
│   │   ├── sanitizer.py           # PII / 注入检测 + 规范化 + slug + candidate ID
│   │   ├── recorder.py            # TurnRecorder：轮次切片解析 + 原子落盘（挂载于两类 Agent）
│   │   ├── miner.py               # 未处理 turn 读取 + legacy session 状态机 + 规则过滤
│   │   ├── judges.py              # ValueJudge / GroundingJudge（结构化输出 + 文本降级）
│   │   ├── dedup.py               # 精确 / 预去重 0.95 / 最终去重 0.9 / 本轮互查
│   │   ├── ledger.py              # processed（cursor）/ published / pending / trash / 报告
│   │   ├── lock.py                # 单写者锁（PID/hostname/stale）+ 事务 journal
│   │   ├── generation.py          # GenerationStore：索引代际指针（kb_generations.json）
│   │   ├── index_service.py       # IndexBuildService：唯一索引写入口（版本化 + 验证）
│   │   ├── publisher.py           # Markdown 渲染 + 最终复扫 + 文件名 + unpublish
│   │   └── pipeline.py            # EvolutionPipeline：12 步编排（dry-run / with-eval）
│   ├── multi_agent/               # Multi-Agent 协作（第6期）
│   │   ├── router.py              # 意图路由器（LLM 分类 → 子 Agent）
│   │   ├── agents.py              # SubAgent 子 Agent 类 + 配置
│   │   └── orchestrator.py        # 编排器：路由 → 执行 → 结构化提取（集成 MemoryManager + SkillManager）
│   ├── scripts/
│   │   ├── __init__.py            # 支持 python -m app.scripts.*
│   │   ├── build_kb_index.py      # 离线构建知识库索引（版本化 generation + 单写者锁）
│   │   ├── run_eval.py            # 离线运行评估（--mode single/multi · --judge/--no-judge · --output）
│   │   ├── run_retrieval_eval.py  # 检索质量评估（recall@k / MRR / nDCG，阶段七 7.6）
│   │   └── run_evolution.py       # QA 自动沉淀 CLI（dry-run / with-eval / pending 审核 / 运维）
│   └── sessions/                  # 运行时生成，已 .gitignore
│       ├── session.json           # 当前会话快照（v2：含 session_id）
│       ├── kb_index.json          # NumpyBackend 固定路径索引（回退用）
│       ├── kb_generations.json    # 索引代际指针（第10期）
│       ├── chroma/                # ChromaBackend 持久化目录
│       ├── evolution/             # QA 自动沉淀运行时状态（第10期）
│       │   ├── turns/             # 轮次原始记录（YYYYMMDD/turn_id.json）
│       │   ├── state/             # ledger.json / evolution.lock / journal.json
│       │   ├── output/            # 运行报告 reports/
│       │   └── staging/           # 发布前暂存目录
│       └── memory/                # 长期记忆存储（按 user_id 分文件）
│           └── {user_id}.json
│
├── mcp_server/                    # MCP Server（独立微服务）
│   └── server.py                  # FastMCP + Streamable HTTP，暴露电商工具
│
└── tests/                         # 全部测试
    ├── test_agent.py              # 结构化输出 + 多轮 + reset（独立脚本，需 API Key）
    ├── test_conversation_management.py  # 多轮对话管理
    ├── test_react_agent.py        # ReAct Agent + Function Calling
    ├── test_mcp.py                # MCP 集成
    ├── test_rag.py                # RAG 知识库检索
    ├── test_multi_agent.py        # Multi-Agent 协作
    ├── test_memory.py             # Memory 短期记忆 & 长期记忆
    ├── test_skills.py             # Skill 可复用能力模块
    ├── test_evaluation.py         # Agent 评估体系（沙箱 + 双层测评）
    └── unit/                      # 第10期 pytest 单测（CI 只收集这里，全程无网络）
        ├── conftest.py            # FakeEmbedder / FakeChatClient / FakeBackend / fixtures
        ├── test_storage_v2.py     # session v2（session_id 贯通）
        ├── test_recorder.py       # 轮次切片解析 + 原子落盘 + 挂载点 + 压缩不丢
        ├── test_legacy_miner.py   # legacy 状态机 + 规则过滤
        ├── test_sanitizer.py      # PII / 注入 / 长度闸门 / slug
        ├── test_candidate_id.py   # 候选 ID 跨 LLM 稳定
        ├── test_chunker_evolved.py  # rglob + source_path + evolved 显示名
        ├── test_dedup.py          # 三类去重 + 容差 + 本轮互查
        ├── test_judges.py         # 价值/接地 Judge 成功与降级
        ├── test_lock_journal.py   # 锁语义 + journal 恢复素材
        ├── test_generation_index.py  # 版本化索引 + 验证 + 两代清理
        ├── test_knowledge_singleton.py  # generation 热刷新 + last-known-good
        ├── test_pipeline.py       # dry-run 零写入 / 幂等 / 复扫 / 上限
        ├── test_pending_ops.py    # pending 审核 + CLI 子命令
        ├── test_build_kb_index.py # 手动构建切 generation + 热刷新
        └── test_sandbox_eval.py   # 配置恢复 + with-eval 通过/阻断
├── pytest.ini                     # testpaths = tests/unit
├── requirements-dev.txt           # -r requirements.txt + pytest
└── .github/workflows/ci.yml       # Python 3.11 → pytest（仅 tests/unit）
```

### 更新日志

| 期数 | 主题 | Tag | 日期 |
|------|------|-----|------|
| 第 1 期 | 项目框架 + 纯 Prompt 客服 + 结构化输出 | v1-prompt-and-structured-output | 2025-04-14 |
| 第 2 期 | 多轮对话管理：Summary 压缩 + JSON 持久化 | v2-conversation-management | 2026-04-18 |
| 第 3 期 | ReAct Agent + 工具调用 (Function Calling) | v3-react-and-function-calling | 2026-04-27 |
| 第 4 期 | MCP 集成 (Streamable HTTP) | v4-mcp-integration | 2026-05-01 |
| 第 5 期 | RAG 检索增强生成（FAQ + 政策知识库） | v5-rag | 2026-05-13 |
| 第 6 期 | Multi-Agent 协作（客服路由 + 售前/售后/投诉分流） | v7-multi-agent | 2026-05-17 |
| 第 7 期 | Memory：短期记忆 & 长期记忆 | v8-memory | 2026-05-23 |
| 第 8 期 | Skill：可复用能力模块（基于 Agent Skills 开放标准） | v9-skills | 2026-05-31 |
| 第 9 期 | Agent 评估体系（沙箱重跑测试集 + 过程/结果双层指标 + LLM judge） | v10-evaluation | 2026-06-06 |
| 第 10 期 | QA 自动沉淀（轮次捕获 → 五层闸门 → 索引代际 → 自进化知识库） | v11-evolution | 2026-08-28 |

> 每期更新后，这里会同步更新架构图和更新日志。

---

## 第 10 期：QA 自动沉淀（自进化知识库）

把「真实对话里的优质问答」自动沉淀进知识库：Agent 每轮对话自动落盘 → 离线挖掘 → 五层闸门过滤 → LLM 双重 Judge → 去重 → 人工审核（pending）→ 版本化发布到 `knowledge/evolved/`。发布采用 **ES Alias + Redis generation pointer 两阶段提交**；线上 retriever 以 Redis generation pointer 驱动热刷新，Alias 用于 ES 索引切换与一致性校验。

### 五层闸门（从对话到知识库）

```
轮次捕获 → ①规则闸门 → ②预去重0.95 → ③价值Judge → ④接地Judge → ⑤最终去重0.9+复扫
                   置信度/转人工      原问题向量      是否值得沉淀    断言有无证据    发布前全文复扫
                   无来源/长度/注入
```

- **可信来源边界**：只有检索命中**非自进化人工文档**（根目录 + `uploads/` 上传文档，`source_path` 不以 `evolved/` 开头）的候选才能通过接地 Judge；`evolved/` 自进化内容一律不作为证据（防止「自证自答」）。
- **PII / 注入**：捕获时脱敏、发布前对**渲染后全文**再复扫，命中即整条拒绝。PII 覆盖：手机/邮箱/身份证/银行卡、邻近字段词的订单/快递/**运单**单号、**≥12 位裸长数字**（无字段词的运单号/流水号）、关键字邻近的联系方式（微信/vx/wx/qq 后缀 5-20 位账号）；政策数字（7天/12元）不受影响。
- **Judge 失败降级**：结构化输出失败 → 文本 JSON 降级 → 全部失败只能进 pending 人工审核。

### 配置（.env，全部有默认值）

| 配置 | 默认 | 说明 |
|------|------|------|
| `EVOLVE_CAPTURE_ENABLED` | `true` | 每轮对话是否落盘 turn 原始记录 |
| `SELF_EVOLVE_ENABLED` | `false` | 是否允许自动写入知识库（**安全开关，默认关闭**） |
| `EVOLVE_MIN_CONFIDENCE` | `0.8` | 规则闸门的置信度门槛 |
| `EVOLVE_MIN_QUALITY` | `0.8` | 价值 Judge 质量分闸门（4/5 分以下不自动发布） |
| `EVOLVE_PRE_DEDUP_THRESHOLD` | `0.95` | 预去重相似度（原问题） |
| `EVOLVE_DEDUP_THRESHOLD` | `0.9` | 最终去重相似度（规范问题+答案） |
| `EVOLVE_MAX_JUDGE_PER_RUN` | `50` | 单次运行最多送审候选数（成本上限） |
| `EVOLVE_MAX_PER_RUN` | `20` | 单次运行最多发布文档数 |
| `EVOLVE_PENDING_AGING_DAYS` | `30` | pending 超过该天数标记 aging |
| `EVOLVE_TURN_RETENTION_DAYS` | `90` | turn 原始记录保留天数 |
| `EVOLVE_LOCK_STALE_SECONDS` | `7200` | 运行锁超时接管阈值 |
| `EVOLVE_EFFECTIVE_DAYS` | `180` | 自进化文档 `effective_date` = 发布日 + 该天数（时效治理） |
| `EVOLVE_MAX_REGROUND_PER_RUN` | `20` | 单次重接地最多核对的自进化文档数（成本上限） |
| `EVOLVE_REVALIDATE_ENABLED` | `true` | 人工知识变更后是否自动触发存量沉淀重接地 |

### 命令

```bash
# 预览（知识库/索引/ledger 零写入、不取锁；S3 后端会做 turns 幂等缓存下载）
python -m app.scripts.run_evolution --dry-run

# 正式运行（需 SELF_EVOLVE_ENABLED=true；S3 后端先同步 turns 再挖，见下）
python -m app.scripts.run_evolution

# 带评测灰度：staging 索引跑沙箱 before/after + 候选探针，阻断退出码 2
python -m app.scripts.run_evolution --with-eval

# pending 人工审核（reviewer 工作台）
python -m app.scripts.run_evolution --list-pending
python -m app.scripts.run_evolution --approve <cid> [cid...]   # 通过 → 走同一发布链路
python -m app.scripts.run_evolution --reject  <cid> [cid...]   # 拒绝 → 永久跳过

# 运维
python -m app.scripts.run_evolution --unpublish <cid>        # 下架：移 trash + 重建索引
python -m app.scripts.run_evolution --revalidate             # 强制重接地存量沉淀（受单次上限约束）
python -m app.scripts.run_evolution --prune-turns --older-than 90
python -m app.scripts.run_evolution --prune-pending
python -m app.scripts.run_evolution --force-unlock           # 跨主机锁的唯一解法

# 手动重建索引（含 evolved/ 子目录；更新 ES Alias + Redis generation pointer，运行中 Agent 热刷新）
python -m app.scripts.build_kb_index
```

### 运行报告样例

```
  挖掘 turn 数        : 128
  规则过滤后候选      : 43
  进入 Judge 的候选   : 43
  Judge API 调用      : 86
  跳过明细            : 低置信 31 / 转人工 12 / 无来源 26 / 过短 9 / 敏感 0 / 重复 5 / Judge 拒 2
  pending（待审核）    : 4
  发布文档数          : 12
```

### 事务与恢复

- 单写者锁（PID/hostname/stale）：`build_kb_index` 与 pipeline 互斥。
- 发布事务：**文档移动前写 journal** → 移入 `evolved/` → 构建并验证 staging 索引 → 原子切换 ES Alias → 更新 Redis generation pointer；线上 retriever 只认 pointer 指向的 generation。替换文档（见下）记录在 journal 的 `replaced_docs`，事务回滚时从 trash 还原。
- 中断恢复：下次运行联合读取 ES Alias、Redis generation pointer 与 journal；明确未切换才回滚候选索引和文档，明确已切换则只前进补 pointer/ledger，任何状态未知均 fail-closed、保留 journal 并设置 `kb_write_blocked` 等待人工 reconcile。
- 任一阶段失败旧索引始终可用：不激活就不切换，失败保留 last-known-good。

### 存量沉淀治理（淘汰半边 + 替换与运营）

- **重接地**：人工知识变更（上传/下架/手工 CLI 切了 generation）后，下次运行自动重接地存量沉淀——活动检索器检索 → 接地 Judge（1 次 LLM/条），缺 `effective_date` 的存量文档视为最旧、最旧优先，单次 ≤ `EVOLVE_MAX_REGROUND_PER_RUN`。通过 → 原地刷新 `last_validated`（frontmatter 不进索引，无需重建）；不通过 → 隔离到 pending（`revalidation_failed`）+ 一次重建激活。隔离是**带 journal 的事务**（`phase=revalidate` + 隔离清单 + 目标索引）：中断后按**前进式恢复**补齐——清单内仍在 `evolved/` 的完成隔离、已在 trash 的补账，再一次重建；重建已生效则无事可做。以 `state/last_human_generation.json` 记录的运行末活动代为触发基准（run/approve/`--revalidate` 及异常路径都会刷新记录），自身发布/重建不会自我触发；`--revalidate` 可强制全量。
- **近重复替换**：最终去重返回 (命中, 命中侧)，仅**问题侧**命中 `evolved/` 旧沉淀且新候选 `quality_score` ≥ 旧文档 frontmatter 的 `quality_score`（旧文档缺字段视为可替换）→ 保留候选并在发布事务中替换：旧文档移 trash（新代只含新文档）→ ledger 旧条目清账（`unpublish_mark + move_to_trash`）；**答案侧命中（问题不同、回答模板化）与人工文档命中一律按重复丢弃**——前者替换会误删回答另一个问题的旧沉淀，后者永不替换人工知识。
- **with-eval 阻断进 pending**：阻断的候选不再「不 mark processed 下轮重试」（那会持续烧 Judge），直接进 pending（`eval_blocked:<原因>`）；下轮被精确去重零成本跳过，人工凭 `-blocked` 报告的 before/after 在审核后台 approve/reject。
- **S3 turns 同步**：`TURNS_ARCHIVE_BACKEND=s3` 时 recorder 只写对象存储，CLI 运行前从归档同步到本地矿工目录（字典序=时间序，游标 `state/sync_cursor.json` 只拉新增且只前进，本地已存在跳过，tmp + `os.replace` 原子写；最近 3 天的 key 不受游标限制——时钟回拨/迟到写入保护）。**S3 不可用 → fail-closed 拒绝空跑**（提示查配置或改 local），杜绝多 Pod 静默挖空。dry-run 的「零写入」指知识库/索引/ledger——turns 幂等缓存下载是唯一例外。
- **processed 限界**：`--prune-turns` 删除超期 turn 文件时同步从 `ledger.processed` 移除对应 turn_id（pending 关联的记录本就受保护），游标随保留期自然收缩。

### 成本与灰度建议

- Judge 是主要成本：`EVOLVE_MAX_JUDGE_PER_RUN` 是**每次运行**的硬上限，dry-run 会给出预计 API 调用数。
- 上线顺序：先 `--dry-run` 观察命中量 → 开 capture 累积 turns（不动知识库）→ 小步 `--with-eval` 灰度 → 稳定后再放开 `SELF_EVOLVE_ENABLED`。
- 知识被污染时：`--unpublish` 下架 + 重建索引（旧代保留可回滚），文档保留在 `state/trash/`。

---

## RAG 生产化（阶段七，已落地部分）

按 `docs/生产化改造计划.md` 阶段七推进，本仓库已落地的四块：

**混合检索（7.2）**：`RAG_HYBRID=true` 开启。向量路 + 手写 BM25 路（中文 bigram 分词，零依赖）双路召回，RRF（k=60）按排名融合——BM25 补向量对精确术语（"七天无理由"这类词面强匹配）的盲区。每路召回量由 `RAG_HYBRID_RECALL_K` 控制（默认 30）。后端只扩了 `VectorBackend.chunks()` 一个方法，换 Elasticsearch 只动 `backends/`。

**Rerank 精排（7.3，2.1 修复后实测）**：`RAG_RERANK=none` 默认不启用。`rerank.py` 提供 `Reranker` 抽象与 `create_reranker` 工厂；`HTTPReranker` 支持 cohere/jina/bge-reranker-v2-m3（TEI 协议），已用持久化 `httpx.Client(trust_env=False)` + 可注入 client 重构，精排失败退回原序并打 `reranker_fallback_total` 指标。本机真实评测（ES8.11.4 + bge-reranker-v2-m3）结果见「检索门禁」段。

**检索质量评估（7.6 / 3.4）**：`app/evaluation/retrieval_cases.json` 为回归/dev 集（535 条：easy 499、hard 21、no-hit 15），`holdout_cases.json` 为独立冻结集（120 条，仅运行一次）。正例计算 `recall@k / MRR / nDCG`，负例计算拒绝率；门槛不达标即非零退出。命令带 `--json-out` 时，报告同时保存逐例 `cases`（含 raw `top_score`、应用阈值与命中键）和独立 `retrieval-eval-v1` manifest（数据集 SHA-256、git、检索配置、阈值）；hard 用例默认复用线上 `RAG_MIN_RELEVANCE_SCORE`，显式 `--min-score-hard` 会在报告中醒目标记为非线上口径：

```bash
python -m app.scripts.run_retrieval_eval --variant knn          # ES kNN
python -m app.scripts.run_retrieval_eval --variant hybrid       # ES BM25+kNN+RRF
python -m app.scripts.run_retrieval_eval --variant hybrid-rerank --calibrate  # +bge-reranker
python -m app.scripts.build_holdout --check                     # holdout 冻结校验
```

**历史检索结果（legacy，不能作为当前 release 证据）**：仓库既有 dev/holdout
数字产生于旧逻辑（hard 用例默认不经过相关度阈值过滤），且对应产物只有 summary，
没有当前协议要求的逐例 cases/manifest；因此不在 README 中把它们当作可复现的线上指标。
按新协议重跑后，应分别提交 dev 与 holdout report、manifest 和逐例证据，再回填简历。

生产环境通过 `RAG_MIN_RELEVANCE_SCORE` 应用校准后的阈值；低于阈值的候选会被过滤，全部低于阈值时知识检索返回空结果。阈值依赖 embedding、后端、hybrid 与 reranker 配置，任一项变化都需重校准；**阈值只允许在 dev 集校准，holdout 冻结后不得据其结果调整查询/文档**。简历/报告应同时呈现 dev 与 holdout，不能只挑 dev 数字。

**文档元数据（7.1 + 7.7）**：源文档支持 `---\nprovenance: ...\nowner: system\n---` frontmatter（chunk 携带 provenance/owner 元数据，不进正文）；`*.txt` 与 `*.md` 统一接入。自进化沉淀文档发布时自动写入 `provenance=<turn_id>` / `owner=system`，沉淀内容可溯源到原始对话。

**未做**：多路召回（7.5，计划明确暂不做，触发条件见 `docs/生产化改造计划.md`）；ES/Milvus 迁移（7.4，规模驱动，接口已抽象）。

---

## 生产化部署（4.x，kind 双副本验收通过）

> 完成标准与证据见 `docs/evidence/eval-v2.md`；本段是「代码完成 ≠ 生产完成」的
> 明确界线——部署物与验收结果分开陈述。

### 配置（Helm values）

- `deploy/helm/ecom-agent/values-production.yaml`：生产默认——`RAG_BACKEND=es`、
  `RAG_HYBRID=true`、`RAG_RERANK=bge-reranker-v2-m3`、`REDIS_REQUIRED=true`、
  `AUTH_ENABLED=true`、`ENFORCE_ORDER_OWNERSHIP=true`、
  `REFUND_CONFIRMATION_REQUIRED=true`、`OPS_RBAC_REQUIRED=true`、
  `KB_UPLOAD_STORAGE=s3`、`TURNS_ARCHIVE_BACKEND=s3`、`SELF_EVOLVE_ENABLED=false`。
- `values-kind.yaml`：kind 验收（双副本 + 确定性 fake 服务，无外部密钥）。
- **Secret 不默认创建**（`secret.create=false`，必填 existingSecret——不再有
  REPLACE_ME 占位）；RWX PVC（`kbSharedVolume`）挂载知识源/staging/journal/trash；
  startupProbe/readiness/liveness、Pod 反亲和、版本标签、migration Job
  （pre-install/pre-upgrade hook，成功后才滚动 Deployment）。

### 数据库迁移（2.9）

```bash
python -m app.scripts.migrate_db          # MySQL GET_LOCK 串行；schema_migrations 记录
APP_ENV=prod 时应用启动只校验 schema 版本（落后拒绝启动），绝不自动 DDL。
```

### 本地可复现环境

```bash
docker compose -f deploy/compose/docker-compose.yml up -d   # MySQL8/Redis7/ES8.11.4/MinIO/reranker/fake 服务
kind create cluster --config deploy/kind/kind-config.yaml
kubectl apply -f deploy/kind/deps.yaml                       # 依赖 manifests
helm install ecom deploy/helm/ecom-agent -f deploy/helm/ecom-agent/values-kind.yaml
```

### kind 双副本验收结论（4.3）

跨 Pod 会话一致、同 session 并发 409、跨 Pod 分片上传合并发布（164→165 chunk）、
发布零空窗、Pod 删除会话转移、扩容 2→3 数据不丢、Redis 断连 fail-closed
（readyz 503 / healthz 200）、ES 断连 fail-degraded、helm rollback 数据不丢、
`SELF_EVOLVE_ENABLED=false` 无 CronJob 不自动发布——全部通过，明细见
`docs/evidence/eval-v2.md` §7。

### 监控

Prometheus 指标（`/metrics`）：依赖就绪（dependency_readiness）、evolution
阶段/恢复/阻塞（evolution_phase / evolution_recovery_total / kb_write_blocked）、
alias/pointer 不一致（alias_pointer_mismatch）、对象存储失败
（object_store_failures_total）、精排降级（reranker_fallback_total）、outbox
积压（outbox_backlog）。告警规则 `deploy/grafana/alerts/ecom-agent-alerts.yaml`
（alias 不一致/KB 阻塞/迁移落后/outbox 堆积/reranker 持续降级/对象存储失败）。

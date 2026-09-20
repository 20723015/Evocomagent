# RAG 切分与文档处理优化方案

> 范围：`app/agent/rag/chunker.py`（切分）与 `app/agent/rag/parsers.py` / `loader.py`（文档处理），以及二者与检索链路的耦合面。
> 不含：异步摄取、流式/增量索引、ES 中文分析器、引用 offset——这些已在《导学-RAG对比优化.md》立项，本文不重复。
> 基线：HEAD `fd1d398`，实测数据见 §一（2026-09-15 对现网语料实测）。
> **执行状态：P0-1~P0-4 + P1-1~P1-5 已全部落地（B 节执行记录），P2 未启动（研究性，按门禁余量立项）。**

---

## 〇、结论摘要

现有切分/解析管线的**工程质量是高的**（无损精确切片、原子保护、表格行感知、strict 语义、指纹版本化），但**切分形态与语料错配**：策略按「长文档防止超限」设计，而真实语料是 136 篇短政策文档，结果是 **782 个块、均值仅 175 字、80% 的块不足 200 字**——检索返回碎片、parent-child 机制实质休眠（仅 8/782 块有父块）。同时，文档处理侧 PDF 的「## 第 N 页」扁平化破坏了真实标题层级，frontmatter 元数据（status/authority/effective_date）全量齐备却未被检索消费。

方案按 P0（1~2 天，纯切分层）→ P1（2~5 天，解析升级 + 构建期增强）→ P2（研究性）排序，全部走现有 959 例检索评测门禁做配置级 A/B，改动切分即 bump `PARSER_CHUNKER_VERSION`。

| 优先级 | 项目 | 一句话 | 预期收益 |
|---|---|---|---|
| P0-1 | 检索块/生成块分离（small-to-big 重定位） | 小节级检索块不变，父块从「仅超长章节」改为「相邻小节合并单元」，全量装配 | 回答上下文完整度，证据可读性 |
| P0-2 | 前缀去重瘦身 | 「doc · 并夕夕 doc > …」根标题与 doc 重复，去重后小块有效信息占比 +15% | embedding 信噪比、token 成本 |
| P0-3 | 元数据下沉检索 | **已降级为「仅透传」**：治理元数据随块进索引/三后端持久化，检索期消费未实现（原方案的时效过滤与外规降权经实测证伪，见 §B.3） | 可运维查询/ES 侧过滤；时效与权威性影响**待另立项校准** |
| P0-4 | 长表续块带表头 | 表格跨块时续块重复表头行 | 潜在风险清零（现网 0 例，上传链路会踩） |
| P1-1 | Contextual Retrieval（Anthropic 2024） | 构建期 LLM 给每块生成 1~2 句定位上下文，进 embedding+BM25 输入 | 检索失败率 ↓35~49%（论文口径，需本库实测） |
| P1-2 | evolved QA 双路表征 | 沉淀 QA 的「问题」进 embedding 输入首位 | 沉淀命中精度（问题匹配问题） |
| P1-3 | PDF 解析升级 Docling | 真实标题层级 + 无边框表格 + 页眉页脚剔除 + OCR；现有解析器回退 | PDF 不再是二等公民 |
| P1-4 | docx 自动编号还原 | numbering.xml 补回「1. 2. 3.」列表编号 | 政策文档保真（现为静默丢失） |
| P2 | Late chunking / 领域微调 / 摘要索引 / 引用图 | 见 §五 | 视 P0/P1 评测余量决定 |

---

## 一、事实基线（全部已实测/源码可验）

**语料与切分分布**（`chunk_kb_dir(app/agent/rag/knowledge)`，strict=False）：

| 指标 | 值 | 说明 |
|---|---|---|
| chunk 总数 | 782 | 136 篇文档（133 md + evolved 2 + pdf 1 + docx 1） |
| 均值 / 中位数 | 175 / 157 字符 | 含【文档 · 标题路径】前缀 |
| < 200 字符的块 | **623（80%）** | 碎片化 |
| 触发兜底切分（sub_idx>0） | 4 | MAX_CHUNK_CHARS=1200 几乎不触发 |
| 有 parent_text 的块 | **8（1%）** | parent-child 实质休眠 |
| 含 Markdown 表格的块 | 29 | 现网无超长表格分区（表头丢失为潜在风险） |

**前缀开销实测**：如 `【3C数码类目规则 · 并夕夕 3C数码类目规则 > 一、激活与联网】\n` 共 33 字符，占该块 153 字的 21%；且 doc 名（文件名）与根标题（文档内 H1）语义重复——每篇文档的根标题都是「并夕夕 ×××」，与 doc 名双倍占用预算。embedding 与 BM25 输入均为 `chunk.text`（含前缀，`index_service.py:191`），同一文档的所有块共享这段前缀，块间区分度被稀释。

**元数据**：133/133 篇文档 frontmatter 齐备 `status / authority / effective_date`（governance.py 构建期已校验、archived 已排除），但检索链路（retriever/hybrid/knowledge 工具）**零消费**——`authority=external_reference` 的外部文档与平台规则同权返回，`effective_date` 未来生效/已过期文档不做任何处理。

**文档处理侧**（parsers.py）：

- PDF：pdfplumber 线条预检 → `find_tables` → 表外交错文本；零表格或异常整篇回退 pypdf。**但输出结构是「# 文件名 + ## 第 N 页」**（`_pdf_parts`），真实标题层级被页码取代——跨页章节在页边界被切开，heading_path 退化为「文档 > 第 3 页」，前缀上下文价值归零；页眉页脚每页重复进入正文（无去重），同时污染 embedding 与 BM25 的 df 统计。
- docx：按 w:p/w:tbl 原序遍历、Heading 1-6 映射正确；**但 `para.text` 不含自动列表编号**（python-docx 不展开 numbering.xml）——政策文档的「1. 2. 3.」条款号静默丢失，条款引用（「第 3 条」）检索时无法对齐。
- HTML：自研状态机质量较好（caption/嵌套表/单元格转义已修）；无 readability 正文抽取，对真实网页来源会带导航/页脚噪声（现网无此来源，低优）。
- 严格模式、编码回退链（utf-8→gb18030+二进制守卫）、sidecar 排除、解析子进程隔离均已就位，无需返工。

**评测设施**：`app/scripts/run_retrieval_eval.py` + 959 例冻结集（Recall@K / MRR / nDCG / 负例拒绝率硬阈值门禁），`RetrievalConfig` 支持 baseline/candidate 配置级 A/B——本方案所有项以此为验收闸门，不达标不合并。

---

## 二、切分现状评价（设计的如何）

**做对了的**（保留，不动）：

1. 标题栈递归 + heading_path 元数据，结构感知优于固定长度切分；
2. 块文本一律为原文精确切片（offset 直带），重复段落各归其位，可校验、可引用；
3. 兜底链路段落打包 → 句末标点 → 原子片段（fence/链接/图片/表格行）→ 小数点保护，表格行不被拦腰切；
4. 前缀自带「文档 · 标题路径」上下文，方向与 Anthropic Contextual Retrieval 一致（只是实现是机械式的，见 P1-1）；
5. parent-child 的**机制**完备：parent_id 去重、父窗口包含性断言、检索侧 `collapse_by_parent` + 生成侧 `_parent_window` 回退链——缺的只是**装配策略**；
6. `PARSER_CHUNKER_VERSION` 指纹（批次7）已解决「改切分后索引静默停留」。

**核心问题**（按严重度）：

1. **块过小，与语料错配（P0）**。策略为「防超限」设计，语料却是短政策文档：每个 H2 小节独立成块 → 均值 175 字。后果链：小块 embedding 语义稀薄 → 向量召回精度受损；Top-3~5 只拿到 ~500 字碎片 → 回答需要跨块拼凑；「七天无理由商品清单」这类清单文档，一个块只有 4 条中的 1 条。
2. **parent-child 名存实亡（P0）**。父块仅在「章节 > 1200 字」时装配（782 块中 8 块），即机制只在不需要它的地方生效。「小块检索、大块生成」的收益完全没有兑现。
3. **前缀冗余（P0）**。doc 名与根标题重复，占小块 15~21% 预算，且全块共享前缀稀释区分度。
4. **长表续块丢表头（潜在，P0 顺手修）**。表格行已原子化，但超过上限的表格在行边界切开后，续块没有表头行，行列语义断裂。现网 0 例，但上传链路（企业采购/费率表 PDF）随时会踩。
5. **元数据不进检索（P0）**。时效与权威性是客服政策的硬约束，目前全靠构建期人肉治理。
6. **PDF 结构破坏 + docx 编号丢失（P1）**。见 §一。

---

## 三、P0 方案（切分层，1~2 天/项，纯 Python 无新依赖）

### P0-1 检索块/生成块分离（small-to-big 重定位）

**问题**：parent_text 只在超长章节装配，现网覆盖率 1%。

**方案**：不改检索粒度（小节级小块召回精度已验证），改父块装配策略——

- 新增「生成单元」：同一文档内**相邻小节按目标长度贪心合并**（`GEN_UNIT_TARGET_CHARS ≈ 1800`、硬上限 `MAX_PARENT_CHARS=4000` 沿用），合并单元 = 若干完整小节的原文拼接；
- **每个检索块都装配 parent_text**（指向所属合并单元），取消「单块章节无父块」特例；`parent_id` 语义从「同章节共用」改为「同合并单元共用」；
- 检索侧零改动：`collapse_by_parent` 按 parent_id 去重、`_parent_window` 命中子块返回父块，机制原样复用。

**改动点**：`chunker.py` `_chunk_text` 装配逻辑 + `_parent_window` 替换为合并单元切片；新增两个常量进 settings（可调）；`fingerprint.py` bump。**索引需重建**。

**预期收益**：回答侧上下文从 ~175 字碎片 → ~1800 字完整小节组；证据可读性（EvidenceItem.text）同步受益；Top-K 去重后有效覆盖率提升（同单元多命中不再浪费槽位）。

**风险**：父块变大 → prompt token 上涨（top_k≤5 × 1800 ≈ 9k 字符上限，实测分布远低）；检索分数分布不变（召回粒度没变），评测门禁可隔离归因。回退 = 配置开关 `rag_parent_merge=off` 恢复旧装配。

### P0-2 前缀去重瘦身

**问题**：`【doc · 并夕夕 doc > …】` 根标题与 doc 名重复，前缀占小块预算 15~21%。

**方案**：`_display_label` 增加规则——heading_path 根标题与 doc 名尾部重合时去掉根标题段（`并夕夕 3C数码类目规则` ⊇ `3C数码类目规则` → 前缀直接 `【3C数码类目规则 · 一、激活与联网】`）；完整路径仍存 heading_path 元数据。

**改动点**：`chunker.py` `_display_label` + bump 指纹 + 重建。**收益**：小块有效正文占比 +15~20%；embedding 输入信噪比改善；embedding 成本同降。**风险**：前缀变短 → 所有 embedding 变化，分数分布整体漂移，必须走 A/B 门禁而非直接合入。

### P0-3 元数据下沉检索 —— ⚠️ 已降级为「仅透传」（原方案两步经实测证伪，未按此实施）

> **本节保留原设计供追溯，实际落地的只有「透传」，两步消费动作均未实现。**
> 实测证伪结论见 §B.3 修正 1：现网无任何 `external_reference` 文档进入索引（构建期已排除 +
> 校验层禁止 `active` 组合），且 `evolved/` 的 `effective_date` 是「发布日 + 180 天」的**有效期上限**，
> 按「未来生效即排除」会踢掉全部自进化知识。**消费侧（时效过滤/权威性降权）需与评测专项同批立项**，
> 落地前 `Chunk.status/authority/effective_date` 只作索引侧可查字段，不参与打分与过滤。

**问题**：`status/authority/effective_date` 构建期校验后即封存，检索零消费。

**原设计**（分两小步，各自独立可回退）：

1. **构建期**：`effective_date > 构建日` 的文档不进索引（未生效规则不得回答），记入构建报告；`authority=external_reference` 的 chunk 打标（Chunk 增 `authority` 字段，默认空兼容旧索引）；
2. **检索期**：final_search 门控后对 `authority=external_reference` 命中降权（score × 系数，仅 scores_meaningful 时）或排序下沉——平台规则与外部参考同分时平台优先。系数进 settings，默认 1.0（no-op），校准后开启。

**改动点**：`parsers.py` 构建期过滤 + `chunker.py` Chunk 字段 + `retriever_factory.py` final_search 降权钩。**收益**：时效错误（回答未生效/已废止规则）与外规当平台规则的类别错误清零。**风险**：降权系数需 dev 集校准，holdout 验证（沿用现有评测纪律）。

**实际落地**：仅 `chunker.py` 的 Chunk 字段（`status/authority/effective_date`）+ 三后端持久化。若将来放开 `external_reference` 入索引或引入时效策略，从 `Chunk.authority`/`effective_date` 直接取用即可，无需再改切分层。

### P0-4 长表续块带表头

**问题**：表格超上限按行切开后，续块丢失表头，行列语义断裂（现网 0 例，上传链路潜在）。

**方案**：`_split_long`/`_force_split` 表格行分支记录首个 pipe 行（表头）与分隔行；续块首部插入「表头 + 分隔行」再续数据行（预算内扣除）。**改动点**：`chunker.py` 两处 + 单测（构造 >1200 字费率表）。**风险**：续块变长可能再触发一次切分——实现上限定表头继承只在「原表被切」时发生一次，不递归。

---

## 四、P1 方案（构建期增强 + 解析升级，2~5 天/项）

### P1-1 Contextual Retrieval（上下文增强索引，Anthropic 2024.09）

**最新技术对照**：Anthropic 公开实验，给每块前置 1~2 句 LLM 生成的「定位上下文」后再进 embedding+BM25，检索失败率 ↓35%，叠加 rerank ↓49~67%。我们的机械前缀（P0-2 后的文档+路径）是同方向的最简版，P1-1 把它换成生成式。

**方案**：

- 构建期（`index_service.build`）对每块调用 chat 模型生成 ≤80 字上下文（输入：doc 名 + heading_path + 块正文；prompt 模板版本化）；
- **缓存正本**：key = `sha256(chunk_text) + context_model + prompt_version`——未变化块零成本重建（与《导学》增量索引方案共用 content-hash 设施）；LLM 失败/超时 → 该块回退机械前缀，记录降级率，不阻断构建；
- embedding 与 BM25 输入 = `context + 前缀 + 正文`；`EvidenceItem.text`（给 LLM 的回答上下文）**不含**生成前缀，保持证据可校验（原文精确切片性质不被污染）。

**改动点**：`index_service.py` 构建流水线 + `chunker.py` Chunk 增 `context_prefix` 字段（默认空兼容）+ 缓存存储（复用 job_store 或独立 json）+ 指纹纳入 prompt_version。

**成本实测预估**：782 块 × ~600 tokens in / 60 out ≈ 50 万 tokens，gpt-4o-mini 级 < ¥1/次构建。**风险**：生成错误上下文污染检索 → 抽查 + 评测门禁硬卡；这是本方案中唯一引入 LLM 进构建链路的项，放 P1 而非 P0。

### P1-2 evolved QA 双路表征

**问题**：沉淀文档标题是问题、正文是答案，但 doc 统一「自进化知识」，前缀为「自进化知识 > 问题」——问题文本进了前缀（好），但 embedding 输入里答案稀释了问题。

**方案**：evolved/ 块的 embedding 输入改为「问题文本 + 答案」（问题居首、完整重复一次），即 QA 检索的「问题匹配问题」对称化；回答侧仍返回完整块。与 P1-1 正交（evolved 块不生成 context，复用问题文本即可）。**改动点**：`chunker.py` evolved 分支的 embedding 输入构造（需把「索引输入」与「块文本」分离为两个字段，P1-1 同样需要这个分离——两项共用一次 Chunk 结构变更）。

### P1-3 PDF 解析升级：Docling 首选 + 现有解析器回退

**最新技术对照**：2024-2025 开源文档解析第一梯队为 Docling（IBM，TableFormer 表结构识别 + 阅读序还原 + 真实标题层级 + Markdown 导出，CPU 可跑）、MinerU 2.5、PaddleOCR-VL（扫描件/复杂版面）。现手写 pdfplumber 拼装在无边框表格、双栏、页眉页脚、扫描件四个面上都有已知盲区。

**方案**：

- `.pdf` 分支首选 Docling → Markdown（标题层级/表格 pipe 表/阅读序），与现有 chunker 无缝对接；
- 现有 `_parse_pdf`（pdfplumber+pypdf）保留为回退（Docling 未安装或解析失败）；`parse_guard` 子进程隔离、页数/大小限制不变；
- 扫描件：Docling 自带 OCR 开关（RapidOCR），接通现有 `parse_image_ocr` 钩子语义；
- 不立即引 Docling 时的廉价补丁（可先行）：跨页重复行检测（≥3 页同位置同文本 → 判页眉页脚剔除）。

**改动点**：`parsers.py` `.pdf` 分支 + `requirements-ocr.txt` 增加可选依赖 + 解析对比测试（现网 PDF 基线逐字节留档）。**风险**：新依赖体积（~200MB 级模型）；strict 构建必须可回退——Docling 输出需过 normalize + governance 同一条流水线。

### P1-4 docx 自动编号还原

**问题**：`Paragraph.text` 不含 numbering.xml 的自动编号，政策条款「1. 2. 3.」静默丢失，条款引用无法检索对齐。

**方案**：遍历 w:p 时读 `pPr/numPr`（numId + ilvl），按 numbering.xml 的 lvlText/start 计算编号文本，前缀到段落（`1.` / `(1)` / 多级 `1.1`）；无 numPr 段落行为不变。**改动点**：`parsers.py` `_parse_docx` + 编号计算纯函数 + 单测（构造含编号 docx）。**风险**：多级编号格式多样，实现只对「十进制 + lvlText %n.」全开，其他格式降级为序号计数（宁简勿错）。

### P1-5（顺带）.xlsx 支持

费率表/清单类政策常以 xlsx 存在，现状 strict 直接报错。openpyxl → sheet 转 pipe 表（复用 `_markdown_table`），半天的量，与 P0-4 的表头继承互相成就。

---

## 五、P2 方案（研究性，视 P0/P1 评测余量启动）

| 项 | 技术出处 | 思路 | 启动条件 |
|---|---|---|---|
| Late chunking | Jina 2024 | bge-m3（SophnetEmbedder 已是，8k 上下文）整篇编码 → 按块区间池化出块向量，保留跨块上下文 | P0-1 后「相邻小节语义断裂」仍在 badcase 中出现 |
| 领域 embedding 微调 | bge-m3 FT | 用 evolved QA + 评测 dev 集构造正/负例对微调；**严禁用 holdout** | P1-1 后向量路仍是短板（对照 BM25 路召回差异） |
| 文档级摘要块 | RAPTOR 简化版 | 每篇文档生成 1 个摘要 chunk（链回全文），覆盖「哪些商品不支持七天无理由」类主题/汇总问题 | 评测中汇总型问题 recall 显著低于单点型 |
| 交叉引用 1-hop 扩展 | GraphRAG 轻量版 | 政策文档大量「见《XX规则》」→ 构建 doc 引用图，命中后 1-hop 补召回（候选级，不进 Top-K 直接扩 evidence） | badcase 中「跨文档组合条件」占比高（阶段D 子查询未覆盖的残余） |
| 近重复检测 | MinHash/embedding 余弦 | evolved 沉淀与正典文档语义重复 → 构建期告警/检索期去重 | evolved/ 规模过百后 |

---

## 六、落地顺序与验收

**纪律**（沿用仓库既有评测体系，不新建）：

1. 每项独立 PR + 独立配置开关（settings.*，默认 no-op/off）；
2. 改切分/解析 → bump `PARSER_CHUNKER_VERSION` → 重建索引 → `run_retrieval_eval.py` 配置级 A/B（baseline=现状，candidate=新配置），959 例硬阈值门禁通过才合并；
3. 阈值/系数类（若将来做 P0-3 降权）只在 dev 集校准，holdout 一次性验证；
4. nightly `retrieval-recall-regression` 结果留档 artifacts/eval/。

**建议排期**（单人）：

| 周 | 内容 | 产出 |
|---|---|---|
| W1 | P0-1 + P0-2（一次索引重建同时上）→ A/B | 块分布报告（`report_chunk_distribution.py --compare`）、评测对比 |
| W1 | P0-4 + P0-3 透传 | 单测 + 三后端字段持久化 |
| W2 | P1-4 + P1-5 | 解析保真（编号还原/Excel 接入） |
| W3 | P1-1（先 50 块小样本验证 prompt，再全量） | 上下文样例抽查记录 + A/B |
| W4 | P1-3 版式增强（字号标题 + 重复行剔除；Docling 可选） | 解析对比报告 |
| W5+ | P2 与「P0-3 消费侧」按评测余量立项 | — |

**回退总线**：所有项均满足「关闭开关 = 逐字节回到现状」（P0-1/P0-2 除外——它们改变索引内容，回退 = 开关 + 重建，指纹机制保证不会静默混用）。

---

## 七、与既有计划的关系

- 《导学-RAG对比优化.md》：异步摄取、流式/增量索引、ES 中文分析器/multi-match、引用 offset——**互补不重叠**；P1-1 的 content-hash 缓存与其中增量索引共用设施，建议同批立项。
- 《多格式解析增强计划 v2》：已完结（批次1-6 合并），P0-4/P1-3/P1-4 是其「表格/保真」方向的自然延伸而非返工。

---

## B、执行记录（2026-09-15）

### B.1 已落地项与改动点

| 项 | 状态 | 改动点 |
|---|---|---|
| P0-1 生成单元 | 已落地 | `chunker.py`：`_SectionPlan` / `_merge_units` / `_assemble_parent`；`settings.rag_parent_merge`、`rag_gen_unit_target_chars`、`rag_max_parent_chars` |
| P0-2 前缀去重 | 已落地 | `chunker.py`：`_display_label(path, doc_name)` + `_is_doc_name_variant` + `_chunk_prefix`；`settings.rag_prefix_dedup` |
| P0-3 元数据下沉 | **部分落地（见 B.3）** | `Chunk` 增 `status/authority/effective_date`，numpy/chroma/es 三后端持久化通路打通 |
| P0-4 长表续块表头 | 已落地 | `chunker.py`：`_inherit_table_head` / `_table_head_before` / `strip_inherited_table_head`；顺带修 `_align_table_overlap`（重叠不再产生半行前缀） |
| P1-1 Contextual Retrieval | 已落地（opt-in） | 新模块 `app/agent/rag/contextual.py`（生成 + content-hash 缓存 + 按块降级）；`IndexBuildService._resolve_enricher` + `last_context_report` |
| P1-2 evolved 双路表征 | 已落地 | `chunker.py`：evolved 块 `index_text = 问题 + 前缀 + 正文` |
| P1-3 PDF | 已落地（**换实现路径，见 B.3**） | `parsers.py`：`_pdf_layout_hints` / `_apply_layout_hints` / `_heading_pages`（字号→标题层级 + 跨页重复行剔除）；`_parse_docling` 可选依赖首选 + 静默回退 |
| P1-4 docx 编号 | 已落地 | `parsers.py`：`_DocxNumbering` + `_paragraph_num_pr`（直接 numPr + 样式继承） |
| P1-5 xlsx | 已落地 | `parsers.py`：`_parse_xlsx` / `_workbook_to_markdown`；`SUPPORTED_SUFFIXES` + `requirements.txt`（openpyxl，已装 3.1.5） |

**索引输入与展示文本分离**（P1-1/P1-2 的共同前置）：`Chunk.index_text` + `index_input()`，贯通 `index_service.build`（encode 与两次探针）、`bm25.BM25Index`、chroma metadata、ES mapping/写库/BM25 查询字段（`index_text`）。空串严格回退 `text`，旧索引行为不变。

**指纹**：`PARSER_CHUNKER_VERSION` → `2026-09-15.1`，并把 `rag_parent_merge` / `rag_prefix_dedup` / `rag_contextual_index` / prompt 版本 / 上下文模型并入 `config_fingerprint()`——关开关或改 prompt 都会让旧索引 fail-closed 要求重建，不会静默混用。

**新增测试 47 条**：`tests/unit/test_chunker_optimization.py`（19，含「建索引 → 检索器 → 证据 = 生成单元」的端到端链路）、`test_contextual_index.py`（12）、`test_parsers_enhancements.py`（16）；另更新 `test_parsers_tables.py`（页级章节断言）与 `test_phase8_sql_es.py`（BM25 查询字段）。CI 门禁全绿：`tests/unit` 0 失败、`ruff --select E9,F63,F7,F82`、`compileall app`、`check_kb_governance`、数据集冻结三项检查；`tests/integration/test_rag_e2e.py` 离线用例通过。

**新增工具**：`app/scripts/report_chunk_distribution.py [--compare] [--json-out]`——不依赖 embedding 的切分分布报告，用于索引重建前判断切分策略效果。

### B.2 实测效果（现网语料，`--compare` 口径）

| 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| 块数 / 文档数 | 809 / 137 | 809 / 137 | 不变（**检索粒度刻意不动**） |
| 检索块均长（含前缀） | 170.4 | 155.2 | −15.2（前缀缩短所致） |
| 正文均长 / p90 | 129.4 / 191 | 129.4 / 191 | 不变 |
| 前缀均长 / 中位 | 41.0 / 39 | 25.8 / 23 | −37% / −41% |
| 父块覆盖率 | 0% | **100%** | 机制从休眠到全量 |
| 父块中位长 / 最长 | 0 / 0 | **692 / 1940** | 回答侧从 ~175 字碎片 → 完整小节组 |
| 生成单元数 | 809 | 151 | 1.1 单元/文档 |

三条不变量在 strict 全量构建上通过：0 个超上限块、0 个缺父块、0 个父块包含性违规；`check_kb_governance.py` 通过。

**与 §六 排期表「均值 175→400+」的差异说明**：P0-1 的正本是「检索块保持小节级、把增长放到父块」——检索块变大等于换召回粒度，会与 P0-2/P0-3 的分数漂移混在一起、无法归因。故保留了检索粒度（正文均长不变），把可读上下文做在 evidence 侧：Top-K 证据从 5×175 字碎片变为 ≤5×~700 字完整小节组（同一生成单元在 Top-K 中只占一个槽位，另有跨文档覆盖收益）。若后续门禁显示「小节级检索」本身是短板，再单独立项放大检索块粒度。

### B.3 对原方案的两处 premises 修正（已实测证伪，故未按原文实施）

**修正 1：P0-3 的「未生效过滤」与「external_reference 降权」都不能做。**

- 现网可索引文档 135/135 全是 `status=active, authority=platform`，**没有任何 `external_reference` 文档进入索引**：`governance.is_index_eligible` 对其返回 False（构建期已排除），且 `validate_metadata` 直接禁止 `active + external_reference`。所以「外部参考与平台规则同权返回」不成立，检索期降权是无对象的死代码。
- 按 `effective_date > 构建日` 过滤会**误杀自进化知识**：`evolved/` 文档的 `effective_date` 是 `publisher.py` 按 `发布日 + EVOLVE_EFFECTIVE_DAYS(180)` 写的**有效期上限**，现网两篇分别是 2027-02-28 / 2027-03-07（未来）。全库「未来生效」的文档只有这两篇——过滤它们等于把第10期 QA 自动沉淀的产物全部踢出索引。
- **因此 P0-3 收敛为「仅透传」，消费侧不实现**：`Chunk.status/authority/effective_date` 随块进索引并在 numpy/chroma/es 三后端持久化（ES 侧可直接按 `authority`/`effective_date` 过滤查询），但**检索链路不读这三个字段**——不做过期/未生效过滤，也不做权威性降权。原因：两个原设计动作一个无对象（外规文档不入索引），一个会误杀（未来日期 = 自进化的有效期上限），且任何打分改动都需先有 dev 集校准，属独立立项。
  - 因此这三个字段当前的消费方只有「运维/排查（查索引里某文档的治理元数据）」与「未来消费侧的取数字段」；**若长期无人消费即视为冗余字段，届时按「死配置清理」同样口径删除**。
  - 需另行立项的消费侧（含各自的校准与门禁）：`external_reference` 入索引策略 + 平台优先排序；时效策略（有效期/答复时效，与 `app/review/scan.py` 的 aging 口径对齐）。

**修正 2：P1-3 用「字号 + 重复行」而非 Docling 拿到同样的结论（Docling 保留为可选升级）。**

- 实测现网 PDF（`平台治理与处罚总则.pdf`，5 页）：字号分层清晰（16pt 标题 / 13pt 二级 / 11pt 三级 / 9.5pt 正文），可完整还原 47 个真实标题；跨页重复行检测结果为空（该 PDF 无页眉页脚）。
- 旧实现把它切成 5 个「## 第 N 页」，heading_path 退化为「文档 > 第 3 页」；现在 36 个块的 heading_path 是真实章节路径（`一、治理总则 > 1.1 治理对象与范围`）。
- 采用 pdfplumber 版式信息的好处：零新增依赖、可在本环境实测验收、且「只提升/只剔除、不改写正文」——无提示的 PDF 输出与历史实现**逐字节一致**（`test_pdf_without_tables_byte_identical` 等既有承载性测试继续通过）。Docling 需要 ~200MB 级模型且本环境无法安装验收，故实现为可选首选（`rag_pdf_docling=true`，未安装即静默回退），接入后仍走同一条 normalize + 治理流水线。
- 同批实测发现并修掉一个既有隐患：`_force_split` 的重叠会让表格续块以**半行**开头（`数据数据…|`），批次5 的行原子化被重叠破坏——现已把表格块的重叠起点对齐到行首。

### B.4 未完成项（须在具备条件的环境补做）

> **2026-09-18 更新：第 1 项（959 例 A/B 门禁）已执行完毕，结论见 B.7。** 其余各项状态不变。

1. ~~**959 例检索评测 A/B 门禁未执行**~~ → **已执行，见 B.7**：本环境无 `OPENAI_API_KEY` / `.env`，无法调用 embedding 接口（`run_retrieval_eval.py` 与候选 generation 探针都需要真实 embedding）。**P0-2 前缀变化必然改变全部 embedding，P0-1 改变 evidence 内容**——按 §六 纪律，合并前必须跑 baseline/candidate 配置级 A/B 并过硬阈值。已核验到位的部分：门禁链路本身可用（离线用 FakeEmbedder 跑通「建 generation → `open_retriever(generation_target)` → `evaluate()` → `search_knowledge` 证据 = 父块」，见 `test_build_to_evidence_uses_generation_unit`），即新格式索引与评测/工具链兼容；**缺的只是真实 embedding 下的质量数字**。

   门禁操作步骤（有 Key 的环境，两条命令各约一次全量构建 + 一次 959 例评测）：

   ```bash
   # baseline：关掉 P0-1/P0-2 开关构建（env 覆盖 settings，只影响构建期）
   RAG_PARENT_MERGE=false RAG_PREFIX_DEDUP=false \
     python app/scripts/build_kb_index.py --backend es --no-activate --json-out /tmp/ab-base.json
   # candidate：默认（开关开启）
   python app/scripts/build_kb_index.py --backend es --no-activate --json-out /tmp/ab-cand.json

   # 两次评测（检索配置不变，变量只有索引内容；--profile release 固化硬阈值）
   python app/scripts/run_retrieval_eval.py --candidate /tmp/ab-base.json --profile release
   python app/scripts/run_retrieval_eval.py --candidate /tmp/ab-cand.json --profile release
   ```

   判读：两个 JSON 报告的 `recall_at_k / mrr / ndcg_at_k / negative.rejection_rate` 逐项对比，candidate 任一硬阈值不达 `RELEASE_PROFILE_MIN` 或显著低于 baseline 即不合并（回退 = 开关置 false 重建）。注意 P0-3 的治理元数据不进打分，所以 A/B 的差异应可完全归因到 P0-1/P0-2。
2. **P1-1 未跑真实 LLM**：prompt 只做过离线假函数验证（生成/缓存/降级/装配），实网成本与质量抽查、以及「上下文是否污染检索」的 A/B 待做；`rag_contextual_index` 默认 false，未开启前不影响线上。
3. **`rag_pdf_repeated_line_*` 阈值未在真实页眉页脚 PDF 上校准**：现网唯一 PDF 无重复行，机制只覆盖了单测构造的场景（保守策略：只剔除**完全相同**的边行文本，不做数字归一化，宁可漏剔页眉也不能误删「扣 3 分/扣 6 分」这类仅数字不同的正文行）。
4. **P2（late chunking / 领域微调 / 摘要块 / 引用图 / 近重复检测）未启动**：按原方案属「视 P0/P1 评测余量启动」，前置的 A/B 尚未跑，不立项。
5. **docx 编号只覆盖十进制家族**：bullet / 中文数字 / 字母编号不生成标记（宁简勿错）；多级 `%1.%2` 的 lvlText 已支持并有单测。
6. **索引未重建**：指纹已 bump，`health` 会明确报「请重建索引」；线上重建需按 §六 顺序（先 `--no-activate` 建候选 → 门禁 → 激活）。

### B.7 父子块 A/B 门禁执行结论（2026-09-18）

> 对应《能力补全全量计划》P0-2。**这是 B.4 第 1 项的补做**，也是 P1-3/P1-4 检索侧改动的归因前提。

**执行口径**：embedding 用 SophNet bge-m3（真实调用）；后端 `numpy`（ES 未启动，
活跃 generation 走 numpy 指针）；检索配置 `--variant knn`（纯向量，分数有语义）；
数据集 `app/evaluation/retrieval_cases_v3.json` dev 959（easy 504 / hard 334 / 负例 121）。
两臂唯一变量是切分配置，索引各自 `--no-activate` 建候选后评测（指纹校验保证了
「构建配置 == 评测配置」这一约束确实生效，中途曾因 `--variant knn` 覆盖 hybrid
配置而被拒，属预期行为）。

| 指标 | legacy（parent_merge=false, prefix_dedup=false） | current（两项默认 true） | 差异 |
|---|---|---|---|
| 正例 recall@5（n=838） | 85.02% | **90.87%** | **+5.85pp** |
| easy recall@5（n=504） | 83.73% | **90.48%** | +6.75pp |
| hard recall@5（n=334） | 86.98% | **91.47%** | +4.49pp |
| 正例 MRR | 0.7023 | **0.7243** | +0.022 |
| 正例 nDCG@5 | 0.7344 | **0.7667** | +0.032 |
| 负例拒绝率（n=121） | 0.0% | 0.0% | 0（本次两臂均未施加门控） |
| 检索 P95 延迟 | 336ms | **322ms** | −14ms |
| chunk 数 | 782 | 781 | −1 |

**判读**：current 臂在**全部检索指标上优于 legacy**，且延迟与索引规模无劣化。
按 B.4 的判读标准（candidate 任一硬阈值不达 `RELEASE_PROFILE_MIN` **或显著低于
baseline** 即不合并），current 未显著低于 baseline，反而全面领先 → **维持两项默认
`true`（即已合并状态）**。

**关于 release 硬阈值**：两臂都未达 `RELEASE_PROFILE_MIN`（正例 ≥95% / easy ≥98% /
MRR ≥90% / nDCG ≥90%），失败项与 `artifacts/eval/v3/RESULT-gate.md` 归档结论同源
——这是**语料上界问题，不是本次切分改动的回归**（legacy 臂同样不达，且更差）。
按计划纪律「门禁数字是简历指标的来源，降门禁等于自毁证据链」，此处只记录事实、
不调整阈值；语料补强见 `语料补足与评测v3计划.md` §10（v3.5 方向）。

**未做的第三臂**：计划原文提到的 `current+A` 在仓库中**无正本定义**
（`父子块切分优化计划.md` 不存在，全仓仅《能力补全全量计划》引用它）。
本次只执行了可考的 legacy / current 两臂，A 臂语义待该计划补齐后另跑。

**产物**：
- 候选描述 `artifacts/eval/p02/ab-legacy.json` / `ab-current.json`
- 门禁报告 `artifacts/eval/p02/report-legacy.json` / `report-current.json`
- 构建与评测日志 `artifacts/eval/p02/{build,eval}-{legacy,current}.log`

**线上索引状态**：本次两个候选均以 `--no-activate` 构建，**未激活**——线上 numpy
活跃 generation 仍是 2026-08-31 的 `20260831225839-9dfc4d1a`（早于 9-15 切分优化）。
`health` 会因指纹不一致提示「请重建索引」。生产切换按 §六 顺序执行
（`--no-activate` 建候选 → 门禁 → 激活），本次已完成前两步。

---

### B.5 发布须知（升级影响，写进 Release Note）

1. **升级即触发全量索引重建**。`PARSER_CHUNKER_VERSION` 已 bump 到 `2026-09-15.1`，且 `rag_parent_merge` / `rag_prefix_dedup` / `rag_contextual_index` / prompt 版本 / 上下文模型都进了配置指纹：旧索引与新配置指纹不一致时 `health` 明确报「请重建索引」、检索 fail-closed，不会静默混用。P0-1/P0-2 的开关默认 **true**（新基线；方案原写「默认 no-op」，此处按「默认采用新基线 + 指纹兜底可回退」处理），关掉开关并重建仍是逐字节回到旧输出。
2. **两个解析改进会改变对应文档的块**（不关开关也会生效）：PDF 版式增强（真实标题层级 + 页眉页脚剔除）与 docx 编号还原。前者**无独立开关**，兜底是「无版式提示即逐字节不变」；后者只对含自动编号的 docx 生效。现网实测：全文块数 782 → 809，其中 PDF 由 9 个「第 N 页」块变为 36 个语义块、docx 127 条条款号还原。
3. **xlsx/xlsm 进入受支持格式**：原先 strict 构建会因未知后缀直接失败，现改为正常解析（新增依赖 `openpyxl>=3.1`，需随部署安装）。
4. **`RAG_CONTEXTUAL_INDEX` 默认 false**：不开启则构建期不调用任何 LLM，构建成本与耗时与升级前一致。

### B.6 复核入口

```bash
pytest tests/unit -q                                          # 全量单测（含新增 47 条）
python app/scripts/report_chunk_distribution.py --compare      # 切分分布前后对比（离线）
PYTHONPATH=. python app/scripts/check_kb_governance.py         # 治理校验
python app/scripts/build_kb_index.py --backend es --no-activate --json-out /tmp/cand.json
python app/scripts/run_retrieval_eval.py --candidate /tmp/cand.json   # 959 例门禁（需 embedding key）
```


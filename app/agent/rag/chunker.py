"""按标题层级递归切分知识库文档（.md / .txt，多格式接入）。

切分策略（目录扫描已递归，这里补全「文档内部标题层级递归」）：
- 识别 Markdown `#` 至 `######` 全部六级标题（fenced code 内的形似标题忽略），
  标题栈维护层级，heading_path 形如 "退款政策 > 特殊商品 > 生鲜商品"；
- 每个标题节点的直属正文独立成块，子标题正文不再混入父标题正文；
- 文档开头到第一个标题之间的内容作为 "概览" 块；
- 章节正文超过硬上限（MAX_CHUNK_CHARS）时段落优先打包到目标长度
  （TARGET_CHUNK_CHARS）；单段仍超限时按句末标点兜底切分（Markdown
  图片/链接/代码块为原子片段，不从语法内部截断），相邻块保留
  ~OVERLAP_CHARS 字符重叠。
- 批次5：兜底切分对 **Markdown 表格行**（行 strip 后以 ``|`` 开头）做
  原子化——行内禁用标点断点、只在行边界分段，避免费率行在 ``0.5%`` 的
  小数点处被切开；``.`` 两侧均为数字时不作断点（编号列表 ``1. `` 不变）。

parent-child（2026-09-15 方案 P0-1：检索块/生成块分离）：
命中子块时返回其所属**生成单元**原文——同一文档内相邻小节按
GEN_UNIT_TARGET_CHARS 贪心合并（硬上限 MAX_PARENT_CHARS），单元内每个检索块
都装配单元原文，同单元子块共用稳定 parent_id（检索侧按其去重）。单节超过硬
上限时该节独占单元，其子块回退旧式「命中位置附近的章节窗口」。窗口无法容纳
子块或包含性校验失败时清空 parent_text，最终回退命中子块（context_type=self）。
rag_parent_merge=False 回到旧装配（父块 = 命中子块附近的章节窗口，单块章节无父块）。

MAX_CHUNK_CHARS 是**最终 Chunk.text（含前缀）**的硬上限：展示标签超长时
截断（根标题…末级标题，完整路径存 heading_path 元数据），正文按
body_limit = MAX_CHUNK_CHARS - len(prefix) 的动态预算切分，绝不事后截断
字符串丢正文；前缀过长留不出正文空间时拒绝该文档。

索引输入与展示文本分离（P1-1/P1-2）：Chunk.text 始终是「机械前缀 + 原文精确
切片」（证据可校验、可引用），Chunk.index_text 只进 embedding/BM25 输入
（构建期生成的定位上下文、evolved 问题文本等），默认空 → index_input() 回退 text。

每个 chunk 保留：
- doc：文档名（不含扩展名），如 "退换货政策"；evolved/ 子目录下统一为 "自进化知识"
- section：章节叶子标题，如 "生鲜商品"
- heading_path：完整标题路径（元数据），如 "退款政策 > 特殊商品 > 生鲜商品"
- text：chunk 全文，带 "【文档 · 标题路径】" 前缀（检索文本自带上下文）
- chunk_id：稳定的字符串 id；parent_id：同生成单元检索块共用的父标识
- source_path：相对 kb_dir 的 posix 路径；provenance / owner：来源溯源（7.7）
- status / authority / effective_date：文档治理元数据透传（构建期已校验；随索引
  落盘供运维查询，**检索链路不消费**——消费侧需按方案 §B.3 另立项）

文档接入流程：raw → normalize_document → parse_frontmatter（frontmatter
不进入 chunk text）→ 其余为正文走标题递归切分。
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.agent.rag.loader import normalize_document, parse_frontmatter, read_text

MAX_CHUNK_CHARS = 1200  # 最终 Chunk.text（含【文档 · 标题路径】前缀）硬上限
TARGET_CHUNK_CHARS = 900  # 段落打包目标长度（硬上限前的优先策略）
OVERLAP_CHARS = 120  # 兜底切分时相邻块重叠字符数
MAX_PARENT_CHARS = 4000  # 生成单元/父块上限（settings.rag_max_parent_chars 可覆盖）
GEN_UNIT_TARGET_CHARS = 1800  # 生成单元目标长度（settings.rag_gen_unit_target_chars）
MAX_LABEL_CHARS = 240  # 检索文本前缀里展示标签的上限（完整路径存元数据）
MIN_BODY_CHARS = 100  # 前缀之后必须给正文保留的最小空间，否则拒绝该文档
# 根标题比文档名多出的「品牌前缀」长度上限（超过即认为不是同一名称的变体）
_DOC_NAME_TAIL_MAX = 8

# evolved/ 子目录（第10期 QA 自动沉淀）的统一展示名
EVOLVED_DOC_NAME = "自进化知识"

# ATX 标题：1-6 个 # + 空白 + 标题文本（7 个及以上 # 不是标题）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# 行首围栏标记（``` 或 ~~~）
_FENCE_CHARS = ("```", "~~~")
# 行内原子片段：图片 / 链接（不从语法内部截断）
_INLINE_ATOMIC_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)")
# 句末标点（超长段落兜底切分的优先断点）
_SENTENCE_END = "。！？；.!?;"

@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str
    source_path: str = ""  # 相对 kb_dir 的 posix 路径；默认空保证旧索引反序列化兼容
    provenance: str = ""  # 来源溯源（如沉淀文档的 turn_id）；默认空兼容旧索引
    owner: str = ""  # 归属（system 表示自进化生成）；默认空兼容旧索引
    parent_text: str = ""  # 生成单元原文（≤ MAX_PARENT_CHARS）；默认空兼容旧索引
    heading_path: str = ""  # 完整标题路径（如 "退款政策 > 特殊商品"）；默认空兼容旧索引
    parent_id: str = ""  # 同生成单元检索块共用的稳定父标识；默认空兼容旧索引
    # P0-3 文档治理元数据透传（构建期已校验；默认空兼容旧索引）。
    # 只落索引、不参与检索打分/过滤：消费侧（时效策略、权威性排序）须先校准再立项
    status: str = ""  # active | archived
    authority: str = ""  # platform | external_reference
    effective_date: str = ""  # ISO 日期（YYYY-MM-DD）
    # P1-1/P1-2 索引侧输入（构建期生成上下文 / evolved 问题文本）；
    # 空串 = 未启用，index_input() 回退 text（旧索引行为不变）
    index_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def index_input(self) -> str:
        """embedding / BM25 的输入文本（与展示文本 text 分离）。

        index_text 承载「只该影响检索、不该污染证据」的内容（P1-1 生成式
        定位上下文、P1-2 evolved 问题文本）；为空时严格回退 text。
        """
        return self.index_text or self.text


def chunk_markdown_dir(kb_dir: Path) -> list[Chunk]:
    """递归扫描目录下所有 .md 与 .txt 文档（含 evolved/ 子目录），逐一切分并汇总。

    7.1 多格式接入第一步：在 Markdown 基础上叠加 .txt，按后缀合并排序扫描。
    """
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.rglob("*.md")) + sorted(kb_dir.rglob("*.txt")):
        if path.is_dir():
            continue
        chunks.extend(_chunk_one_file(path, kb_dir))
    return chunks


def _chunk_one_file(path: Path, kb_dir: Path) -> list[Chunk]:
    """切分单个文档文件：normalize → 解析 frontmatter → 正文切分。

    frontmatter 中的 provenance / owner 透传到每个 chunk（缺省 ""），
    frontmatter 本身不进入 chunk text。
    """
    raw = read_text(path)
    rel = path.relative_to(kb_dir).as_posix()
    return _chunk_text(raw, rel, kb_dir)


@dataclass
class _SectionPlan:
    """单小节的切分计划（前缀/预算/正文一次算清，供合并与装配复用）。"""

    idx: int  # 文档内小节序号（chunk_id 与旧模式 parent_id 用）
    section: str
    heading_path: str
    prefix: str  # 检索文本前缀行（含换行）
    text: str  # 小节正文（原文精确切片源）
    body_limit: int  # 正文预算 = MAX_CHUNK_CHARS - len(prefix)
    index_head: str = ""  # 索引输入前置（P1-2 evolved 问题文本），空则无


def _setting(name: str, default):
    """读 settings 开关（导入失败或字段缺失时回落默认值，切分不因配置炸）。"""
    try:
        from app.config.settings import settings
    except Exception:  # noqa: BLE001 —— 单独使用 chunker 时不依赖 settings
        return default
    value = getattr(settings, name, None)
    return default if value is None else value


def _chunk_text(raw: str, rel: str, kb_dir: Path, *,
                parent_child: bool = True,
                parent_merge: bool | None = None,
                prefix_dedup: bool | None = None) -> list[Chunk]:
    """对已读入文本做 frontmatter 解析与正文切分（7.1 解析流水线复用）。

    parent_child=False 时清空 parent_id / parent_text（父子块关闭）。
    parent_merge / prefix_dedup 缺省跟随 settings（rag_parent_merge /
    rag_prefix_dedup）；显式传参供灰度与测试覆盖，False 即逐字节回到旧行为。
    """
    path = Path(rel)
    is_evolved = rel.startswith("evolved/") or rel == "evolved.md"
    doc_name = EVOLVED_DOC_NAME if is_evolved else path.stem
    # id 前缀统一用无扩展名相对路径：根文档与旧格式一致（stem），
    # 子目录同名文档之间不再互相碰撞
    id_prefix = path.with_suffix("").as_posix()
    meta, body = parse_frontmatter(normalize_document(raw))
    provenance = str(meta.get("provenance", "") or "")
    owner = str(meta.get("owner", "") or "")
    # P0-3：治理元数据（构建期已校验）随块下沉到索引，供运维/排查与未来消费侧取数
    status = str(meta.get("status", "") or "")
    authority = str(meta.get("authority", "") or "")
    effective_date = str(meta.get("effective_date", "") or "")

    merge = _setting("rag_parent_merge", True) if parent_merge is None else parent_merge
    dedup = _setting("rag_prefix_dedup", True) if prefix_dedup is None else prefix_dedup
    max_parent = int(
        _setting("rag_max_parent_chars", MAX_PARENT_CHARS) or MAX_PARENT_CHARS
    )
    unit_target = int(
        _setting("rag_gen_unit_target_chars", GEN_UNIT_TARGET_CHARS)
        or GEN_UNIT_TARGET_CHARS
    )

    plan: list[_SectionPlan] = []
    for idx, (section, heading_path, text) in enumerate(_split_by_headings(body)):
        text = text.strip()
        if not text:
            continue
        # 展示标签超长时截断（根标题…末级标题）；完整路径保留在 heading_path 元数据。
        # 预算 = 最终文本上限 - 前缀：正文按动态预算切分，绝不事后截断字符串
        label = _display_label(heading_path or section, doc_name, dedup=dedup)
        prefix = _chunk_prefix(doc_name, label, dedup=dedup)
        body_limit = MAX_CHUNK_CHARS - len(prefix)
        if body_limit < MIN_BODY_CHARS:
            raise ValueError(
                f"{rel}: 标题/文档名过长（前缀 {len(prefix)} 字符），"
                f"无法在 {MAX_CHUNK_CHARS} 字符上限内保留正文，已拒绝该文档"
            )
        # P1-2：evolved 沉淀文档是「问题=标题 / 答案=正文」的 QA，索引输入让问题
        # 居首（问题匹配问题）；块文本与证据文本仍是原文，不被生成内容污染
        index_head = f"{section}\n" if is_evolved else ""
        plan.append(
            _SectionPlan(idx, section, heading_path, prefix, text, body_limit,
                         index_head)
        )
    if not plan:
        return []

    out: list[Chunk] = []
    # 生成单元：相邻小节贪心合并（检索块粒度不变，只是父块按单元装配）；
    # 关闭合并或不装配父子块时每节独占一单元，等价于旧的「同章节」语义
    units = (
        _merge_units(plan, unit_target, max_parent)
        if parent_child and merge else [[i] for i in range(len(plan))]
    )
    for unit_idx, unit in enumerate(units):
        unit_text = "\n\n".join(plan[i].text for i in unit)
        for i in unit:
            item = plan[i]
            if not parent_child:
                parent_id = ""
            elif merge:
                parent_id = f"{id_prefix}#u{unit_idx:02d}"
            else:
                parent_id = f"{id_prefix}#p{item.idx:02d}"
            if len(item.text) <= item.body_limit:
                pieces = [(item.text, 0, len(item.text))]
            else:
                pieces = _inherit_table_head(
                    item.text,
                    _split_long(item.text, limit=item.body_limit),
                    limit=item.body_limit,
                )
            for sub_idx, (piece, piece_start, piece_end) in enumerate(pieces):
                # 包含性断言用原文切片 core：续块表头（P0-4）是同一小节更早位置
                # 的复制行，与 core 拼接后不再是父块的连续子串
                core = item.text[piece_start:piece_end]
                parent_text = (
                    _assemble_parent(item.text, piece_start, piece_end,
                                     unit_text=unit_text, merge=merge,
                                     max_parent=max_parent)
                    if parent_child else ""
                )
                if parent_text and core not in parent_text:
                    parent_text = ""  # 构建期包含性断言（防御）
                out.append(_make_chunk(
                    doc_name, item.section, item.heading_path, item.prefix, piece,
                    item.idx, sub_idx, id_prefix, rel, provenance, owner,
                    parent_text=parent_text, parent_id=parent_id,
                    status=status, authority=authority,
                    effective_date=effective_date,
                    index_text=(
                        f"{item.index_head}{item.prefix}{piece}"
                        if item.index_head else ""
                    ),
                ))
    return out


def _merge_units(plan: list[_SectionPlan], target: int,
                 max_parent: int) -> list[list[int]]:
    """相邻小节贪心合并为生成单元（返回 plan 下标分组，保持文档顺序）。

    - 累计长度达到 target 即收口（避免父块无谓膨胀）；
    - 加入下一节会超过硬上限 max_parent 时先收口——单元原文必须完整装下
      才谈「大块生成」，绝不截断；
    - 单节自身超上限时独占一单元，其子块由 ``_assemble_parent`` 回退窗口装配。
    """
    units: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for i, item in enumerate(plan):
        length = len(item.text)
        if length > max_parent:
            if cur:
                units.append(cur)
                cur, size = [], 0
            units.append([i])
            continue
        extra = length + (2 if cur else 0)  # "\n\n" 连接开销
        if cur and size + extra > max_parent:
            units.append(cur)
            cur, size = [], 0
            extra = length
        cur.append(i)
        size += extra
        if size >= target:
            units.append(cur)
            cur, size = [], 0
    if cur:
        units.append(cur)
    return units


def _assemble_parent(text: str, start: int, end: int, *, unit_text: str,
                     merge: bool, max_parent: int) -> str:
    """父块装配：优先整个生成单元原文；单节超限或旧模式回退位置窗口。

    旧装配（merge=False）保持原语义：单块章节（切片覆盖整章）不装配父块。
    """
    if not merge:
        if start == 0 and end == len(text):
            return ""
        return _parent_window(text, start, end, max_parent)
    if len(unit_text) <= max_parent:
        return unit_text
    return _parent_window(text, start, end, max_parent)


def _split_by_headings(raw: str) -> list[tuple[str, str, str]]:
    """按 H1-H6 递归切分，返回 [(section, heading_path, body), ...]（文档顺序）。

    - 标题栈维护层级：遇到 level L 的标题时先弹出所有 ≥L 的祖先再入栈；
    - 每个标题节点的直属正文（到下一个任意级别标题之前）独立成块；
    - fenced code（``` / ~~~）内部的形似标题行不作为标题；
    - 第一个标题之前的内容为 "概览" 块（heading_path 也是 "概览"）。
    """
    sections: list[tuple[str, str, str]] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    title = ""
    body: list[str] = []
    in_fence = False
    fence_char = ""

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            heading_path = " > ".join(t for _, t in stack) or "概览"
            sections.append((title or "概览", heading_path, text))
        body.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if not in_fence and stripped[:3] in _FENCE_CHARS:
            in_fence, fence_char = True, stripped[0]
        elif in_fence and stripped and stripped[0] == fence_char \
                and len(stripped) >= 3 and set(stripped) == {fence_char}:
            in_fence = False
        elif not in_fence:
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                continue
        body.append(line)

    flush()
    return sections


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[tuple[str, int, int]]:
    """超长正文的兜底切分，返回 [(piece, start, end)]。

    - 段落（空行分隔）贪心打包到 target = min(TARGET_CHUNK_CHARS, limit)；
    - 超过 limit 的单段交由 _force_split（句末标点切分，原子片段内不切）；
    - 仅兜底切分的相邻块保留 ~OVERLAP_CHARS 字符重叠（重叠计入预算）；
    - (start, end) 在**切片阶段直接携带**：块文本一律取章节原文的精确切片
      （piece == text[start:end]，含 overlap 前缀），不做事后 find 定位——
      重复内容的段落/句子也各归其位，不会全部落到第一次出现的位置。
    """
    if limit <= 0:
        raise ValueError(f"切块预算必须为正数: {limit}")
    pieces: list[tuple[str, int, int]] = []
    buf: list[tuple[int, int]] = []  # 段落区间（原文偏移）
    target = min(TARGET_CHUNK_CHARS, limit)

    def flush() -> None:
        if buf:
            pieces.append((text[buf[0][0]:buf[-1][1]], buf[0][0], buf[-1][1]))
            buf.clear()

    for p_start, p_end in _paragraph_spans(text):
        if p_end - p_start <= limit:
            # 打包长度按原文区间精确计算（含段间空白的真实间隔）
            if buf and (p_end - buf[0][0]) > target:
                flush()
            buf.append((p_start, p_end))
        else:
            flush()
            pieces.extend(_force_split(text, p_start, p_end, limit))
    flush()
    return pieces


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """段落区间 [(start, end)]：指向首/末个非空白字符（原文精确切片边界）。

    与旧实现 re.split(r"\\n\\s*\\n") 的分段语义一致，但携带 offset。
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    start = -1
    end = 0
    for line in text.split("\n"):
        if line.strip():
            if start < 0:
                start = offset + (len(line) - len(line.lstrip()))
            end = offset + len(line.rstrip())
        elif start >= 0:
            spans.append((start, end))
            start = -1
        offset += len(line) + 1  # +1 为换行符（末行多计无影响）
    if start >= 0:
        spans.append((start, end))
    return spans


def _parent_window(text: str, start: int, end: int,
                   max_chars: int = MAX_PARENT_CHARS) -> str:
    """围绕子块区间 [start, end) 的父上下文窗口（≤ max_chars）。

    必须完整包含子块；剩余空间优先平均分配到子块前后，一侧不足时由另一侧
    补齐（不超过章节边界）。窗口取自章节原文，保证 piece 是其精确子串。
    """
    span = end - start
    if span <= 0 or span > max_chars:
        return ""
    budget = max_chars - span
    before = min(start, budget // 2)
    after = min(len(text) - end, budget - before)
    before += min(start - before, budget - before - after)
    after += min((len(text) - end) - after, budget - before - after)
    return text[start - before:end + after]


_WHITESPACE_RE = re.compile(r"\s+")


def _normalized(text: str) -> str:
    """去空白归一（含全角空格与中英文之间的空格差异）。"""
    return _WHITESPACE_RE.sub("", str(text or ""))


def _is_doc_name_variant(root: str, doc_name: str) -> bool:
    """根标题是否只是「文档名 + 品牌/来源前缀」（如「并夕夕 3C数码类目规则」）。

    归一后完全相等，或根标题以文档名结尾且多出的部分很短（品牌前缀），判为同一
    名称的变体——前缀里再展示一次纯属重复占预算；第三方来源文档（安平保险/华顺
    快递等，根标题与文件名是不同表述）不在此列，保持原样不做有损删减。
    """
    a, b = _normalized(root), _normalized(doc_name)
    if not a or not b:
        return False
    if a == b:
        return True
    return a.endswith(b) and 0 < len(a) - len(b) <= _DOC_NAME_TAIL_MAX


def _display_label(heading_path: str, doc_name: str = "", *,
                   dedup: bool = True) -> str:
    """检索文本前缀里的展示标签：超长时保留根标题与末级标题，中间省略。

    完整标题路径始终保存在 Chunk.heading_path 元数据中，这里只约束前缀展示
    长度（为正文预算让路）。dedup=True 时去掉与文档名重复的根标题段
    （P0-2：小块预算 15~21% 被「并夕夕 ×××」这类重复前缀占掉）。

    返回空串表示该路径除文档名外无可展示信息（前缀退化为「【文档】」）。
    """
    path = str(heading_path or "")
    if dedup and doc_name:
        root, sep, rest = path.partition(" > ")
        if _is_doc_name_variant(root, doc_name):
            path = rest if sep else ""
    if len(path) <= MAX_LABEL_CHARS:
        return path
    if " > " not in path:
        return path[:MAX_LABEL_CHARS]
    root, leaf = path.split(" > ", 1)[0], path.rsplit(" > ", 1)[-1]
    keep = max(MAX_LABEL_CHARS - len(root) - len(" > … > "), 1)
    return f"{root} > … > {leaf[:keep]}"[:MAX_LABEL_CHARS]


def _chunk_prefix(doc_name: str, label: str, *, dedup: bool = True) -> str:
    """检索文本前缀行「【文档 · 标签】」。

    dedup=True 且标签为空或与文档名重复时退化为「【文档】」；dedup=False 保持
    旧格式（逐字节回退）。
    """
    if dedup and (not label or _normalized(label) == _normalized(doc_name)):
        return f"【{doc_name}】\n"
    return f"【{doc_name} · {label}】\n"


def chunk_body(text: str) -> str:
    """去掉检索文本的「【文档 · 标题路径】」前缀行，返回正文。

    供检索侧校验父块窗口是否完整包含命中子块正文（前缀行不含换行）。
    """
    if text.startswith("【") and "\n" in text:
        return text.split("\n", 1)[1]
    return text


def _force_split(text: str, start: int, end: int,
                 limit: int) -> list[tuple[str, int, int]]:
    """超长段落兜底切分：句末标点优先 → 打包到目标长度 → 相邻块重叠。

    返回 [(piece, abs_start, abs_end)]，piece == text[abs_start:abs_end]。
    句段连续无空洞，块文本是原文精确切片；overlap 前缀取上一块尾部并回退
    起点，且必须是原文对应区间的精确内容（校验失败则放弃重叠，宁缺勿错）。
    表格行块的重叠起点对齐到行首（批次5 的行原子化不被重叠破坏——否则续块
    以「数据数据…|」这类半行开头，表头继承也就失去了干净的落点）。
    """
    core = [(start + s, start + e)
            for s, e in _sentence_spans(text[start:end], limit)]
    groups: list[tuple[int, int]] = []
    buf: list[tuple[int, int]] = []
    buf_len = 0
    for s, e in core:
        if buf and buf_len + (e - s) > min(TARGET_CHUNK_CHARS, limit):
            groups.append((buf[0][0], buf[-1][1]))
            buf, buf_len = [(s, e)], e - s
        else:
            buf.append((s, e))
            buf_len += e - s
    if buf:
        groups.append((buf[0][0], buf[-1][1]))

    out: list[tuple[str, int, int]] = []
    for bs, be in groups:
        piece = text[bs:be]
        if out:
            overlap = _safe_tail(out[-1][0], OVERLAP_CHARS)
            if (overlap and len(overlap) + len(piece) <= limit
                    and text[bs - len(overlap):bs] == overlap):
                start = bs - len(overlap)
                if _has_table_row(piece):
                    start, overlap = _align_table_overlap(text, start, bs)
                if overlap and len(overlap) + len(piece) <= limit:
                    piece = overlap + piece
                    bs = start
        if piece.strip():
            out.append((piece, bs, be))
    return out


def _align_table_overlap(text: str, start: int, bs: int) -> tuple[int, str]:
    """表格块的重叠区间对齐到行首：返回 (新起点, 新重叠文本)。

    重叠区内没有换行（整段落在同一行内）时放弃重叠——半行前缀对检索只有噪声，
    宁可少一点上下文也不把表格行拦腰带进下一块。
    """
    break_at = text.find("\n", start, bs)
    if break_at < 0 or break_at + 1 >= bs:
        return bs, ""
    return break_at + 1, text[break_at + 1:bs]


def _sentence_spans(p: str, limit: int) -> list[tuple[int, int]]:
    """句段区间（p 内相对区间）：标点后可断、原子片段内不可断。

    保留空白句段（保证区间连续、块文本与原文切片严格一致），空白内容由
    上层按块粒度丢弃。单句段仍超限则硬切。

    批次5（表格行感知）：
    - **表格行原子化**：行 strip 后以 ``|`` 开头者视为表格行；行内禁用一切
      标点断点，只在行首/行尾换行处分段——否则含 ``0.5%`` 的费率行会被 ``.``
      命中、无标点的表格行会落到 ``_hard_spans`` 被拦腰切开；
    - **小数点保护**：``.`` 两侧均为 ASCII 数字时不作断点（``3.9``/``0.5%``）；
      「数字 + ``.`` + 空格」（编号列表 ``1. ``）不受影响；
    - 非表格文本的切分结果与批次5 前逐字节一致（唯一例外是小数点保护）。
    """
    atomic = _atomic_spans(p)
    rows = _table_row_bounds(p)

    def _in_table_row(pos: int) -> bool:
        return any(start < pos < cut for start, cut in rows)

    cuts: set[int] = set()
    for i, ch in enumerate(p):
        if ch not in _SENTENCE_END:
            continue
        pos = i + 1
        if _in_atomic(pos, atomic) or _in_table_row(pos):
            continue
        if ch == "." and _is_decimal_dot(p, i):
            continue
        cuts.add(pos)
    for start, cut in rows:
        if start > 0:
            cuts.add(start)  # 表格行起点（行前为换行或正文）
        cuts.add(cut)  # 表格行尾换行之后

    bounds = [0] + sorted(cuts) + [len(p)]
    out: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:  # 重复/越界边界（表格行切点与标点切点重合）
            continue
        if b - a > limit:
            out.extend(_hard_spans(p, a, b, limit))
        else:
            out.append((a, b))
    return out


_ASCII_DIGITS = frozenset("0123456789")


def _is_decimal_dot(text: str, index: int) -> bool:
    """``text[index] == "."`` 是否为小数点（两侧均为 ASCII 数字）。"""
    if index <= 0 or index + 1 >= len(text):
        return False
    return text[index - 1] in _ASCII_DIGITS and text[index + 1] in _ASCII_DIGITS


def _table_row_bounds(p: str) -> list[tuple[int, int]]:
    """表格行的 ``(start, cut)``：``cut`` 为该行尾换行之后的位置（文末 = len(p)）。

    判定：行 strip 后以 ``|`` 开头（Markdown pipe 表行，含 ``| --- |`` 分隔行）。
    """
    bounds: list[tuple[int, int]] = []
    offset = 0
    for line in p.split("\n"):
        if line.strip().startswith("|"):
            start = offset + (len(line) - len(line.lstrip()))
            bounds.append((start, min(offset + len(line) + 1, len(p))))
        offset += len(line) + 1
    return bounds


def _is_separator_row(line: str) -> bool:
    """是否为 Markdown 表头分隔行（``| --- | :--: |``）。"""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(c and set(c) <= set("-:") for c in cells)


def _has_table_row(text: str) -> bool:
    """文本是否含 Markdown 表格行。"""
    return any(line.strip().startswith("|") for line in text.split("\n"))


def _table_head_before(text: str, pos: int) -> str:
    """pos 之前紧邻的 Markdown 表格「表头 + 分隔行」（无则 ""）。

    从 pos 处向上收集连续的表格行块（空行/正文行即断开），块内前两行即表头与
    分隔行；不足两行或第二行不是分隔行 → 不是表格，返回 ""。
    """
    lines = text[:pos].split("\n")
    if lines and lines[-1].strip() == "":
        lines.pop()  # pos 落在行首时末尾多出的空串
    block: list[str] = []
    for line in reversed(lines):
        if line.strip().startswith("|"):
            block.insert(0, line)
        else:
            break
    if len(block) < 2 or not _is_separator_row(block[1]):
        return ""
    return block[0] + "\n" + block[1]


def _inherit_table_head(text: str, pieces: list[tuple[str, int, int]],
                        limit: int) -> list[tuple[str, int, int]]:
    """长表续块补表头（P0-4）：表格在行边界被切开后，续块不再丢失列语义。

    表格行已原子化（批次5），但超过上限的表格切开后只有首块带表头，续块的行
    数据无列名可对齐（费率表第 2 块尤其致命）。这里把同一表格的首两行（表头 +
    分隔行）前置到续块文本。

    约定：(start, end) 仍是**原文精确切片**（core = text[start:end]），返回文本
    为 ``表头 + core``——续块表头是同一小节更早位置的复制行，与 core 拼接后不再是
    父块的连续子串，包含性校验请用 core（``strip_inherited_table_head`` 还原）。
    预算装不下（表头 + core > limit）时放弃继承，绝不二次切分或递归。
    """
    if len(pieces) <= 1 or not text:
        return pieces
    out = list(pieces)
    for i in range(1, len(out)):
        piece, start, end = out[i]
        core = text[start:end]
        if not _has_table_row(core):
            continue
        rows = core.split("\n")
        if len(rows) >= 2 and _is_separator_row(rows[1]):
            continue  # 本块自带表头 + 分隔行
        head = _table_head_before(text, start)
        if not head:
            continue
        merged = f"{head}\n{core}"
        if len(merged) > limit:
            continue  # 预算不足：宁缺表头，不破坏块长上限
        out[i] = (merged, start, end)
    return out


def strip_inherited_table_head(text: str) -> str:
    """去掉续块首部继承来的「表头 + 分隔行」（P0-4 继承的逆操作）。

    判定：首行是表格行、第二行是表头分隔行，且其后还有内容（≥3 行）——即这
    两行是被复制出来的表头。用于「包含性校验」把续块还原成原文切片：还原后
    可能与父块连续（衔接处可能落在行中，见 ``_force_split`` 的行首对齐）。
    """
    lines = str(text or "").split("\n")
    if (len(lines) >= 3 and lines[0].strip().startswith("|")
            and _is_separator_row(lines[1])):
        return "\n".join(lines[2:])
    return str(text or "")


def _hard_spans(p: str, a: int, b: int, limit: int) -> list[tuple[int, int]]:
    """无句末标点可用的最后兜底：原子片段外按预算上限切分 [a, b)。

    整段都在原子片段内（如超长代码块）时在上限处截断——原子保护是
    「尽量」，硬上限是约束。
    """
    out: list[tuple[int, int]] = []
    s = a
    while b - s > limit:
        atomic = _atomic_spans(p[s:b])
        cut = next((i for i in range(limit, 0, -1)
                    if not _in_atomic(i, atomic)), limit)
        out.append((s, s + cut))
        s += cut
    out.append((s, b))
    return out


def _fence_spans(text: str) -> list[tuple[int, int]]:
    """fenced code 的字符区间（未闭合围栏延伸到文末）。"""
    spans: list[tuple[int, int]] = []
    in_fence = False
    fence_char = ""
    start = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not in_fence and stripped[:3] in _FENCE_CHARS:
            in_fence, fence_char, start = True, stripped[0], offset
        elif in_fence and stripped and stripped[0] == fence_char \
                and len(stripped) >= 3 and set(stripped) == {fence_char}:
            in_fence = False
            spans.append((start, offset + len(line)))
        offset += len(line)
    if in_fence:
        spans.append((start, len(text)))
    return spans


def _atomic_spans(text: str) -> list[tuple[int, int]]:
    """不可截断的原子片段区间：fenced code + 行内图片/链接。"""
    return _fence_spans(text) + [m.span() for m in _INLINE_ATOMIC_RE.finditer(text)]


def _in_atomic(pos: int, spans: list[tuple[int, int]]) -> bool:
    """pos 是否落在某原子片段内部（片段边界上不算内部）。"""
    return any(s < pos < e for s, e in spans)


def _safe_tail(block: str, n: int) -> str:
    """取块尾 ~n 字符做重叠；起点落在原子片段内时对齐到片段起点。

    原子对齐可能把尾部拉得过长（如块尾是长代码块）：超过块长一半时放弃
    重叠，保证块长可控。
    """
    if len(block) <= n:
        return block
    start = len(block) - n
    for s, e in _atomic_spans(block):
        if s < start < e:
            start = s
            break
    tail = block[start:]
    return tail if len(tail) <= max(len(block) // 2, 1) else ""


def _make_chunk(
    doc: str,
    section: str,
    heading_path: str,
    prefix: str,
    text: str,
    idx: int,
    sub_idx: int,
    id_prefix: str,
    source_path: str,
    provenance: str = "",
    owner: str = "",
    parent_text: str = "",
    parent_id: str = "",
    status: str = "",
    authority: str = "",
    effective_date: str = "",
    index_text: str = "",
) -> Chunk:
    chunk_id = f"{id_prefix}#{idx:02d}-{sub_idx:02d}"
    return Chunk(
        chunk_id=chunk_id, doc=doc, section=section, text=f"{prefix}{text}",
        source_path=source_path, provenance=provenance, owner=owner,
        parent_text=parent_text, heading_path=heading_path, parent_id=parent_id,
        status=status, authority=authority, effective_date=effective_date,
        index_text=index_text,
    )

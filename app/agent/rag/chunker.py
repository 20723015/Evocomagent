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

parent-child：命中子块时返回父块——父块是**命中子块附近**的章节原文窗口
（≤ MAX_PARENT_CHARS，完整包含该子块，剩余空间优先平均分配到前后），
而非固定章节开头；同章节子块共用稳定 parent_id（检索侧按其去重），不同
子块的窗口可各不相同。窗口无法容纳子块或包含性校验失败时清空
parent_text，最终回退命中子块（context_type=self）。

MAX_CHUNK_CHARS 是**最终 Chunk.text（含前缀）**的硬上限：展示标签超长时
截断（根标题…末级标题，完整路径存 heading_path 元数据），正文按
body_limit = MAX_CHUNK_CHARS - len(prefix) 的动态预算切分，绝不事后截断
字符串丢正文；前缀过长留不出正文空间时拒绝该文档。

每个 chunk 保留：
- doc：文档名（不含扩展名），如 "退换货政策"；evolved/ 子目录下统一为 "自进化知识"
- section：章节叶子标题，如 "生鲜商品"
- heading_path：完整标题路径（元数据），如 "退款政策 > 特殊商品 > 生鲜商品"
- text：chunk 全文，带 "【文档 · 标题路径】" 前缀（检索文本自带上下文）
- chunk_id：稳定的字符串 id；parent_id：同章节子块共用的父标识
- source_path：相对 kb_dir 的 posix 路径；provenance / owner：来源溯源（7.7）

文档接入流程：raw → normalize_document → parse_frontmatter（frontmatter
不进入 chunk text）→ 其余为正文走标题递归切分。
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.agent.rag.loader import normalize_document, parse_frontmatter

MAX_CHUNK_CHARS = 1200  # 最终 Chunk.text（含【文档 · 标题路径】前缀）硬上限
TARGET_CHUNK_CHARS = 900  # 段落打包目标长度（硬上限前的优先策略）
OVERLAP_CHARS = 120  # 兜底切分时相邻块重叠字符数
MAX_PARENT_CHARS = 4000  # 父块（章节原文窗口）上限
MAX_LABEL_CHARS = 240  # 检索文本前缀里展示标签的上限（完整路径存元数据）
MIN_BODY_CHARS = 100  # 前缀之后必须给正文保留的最小空间，否则拒绝该文档

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
    parent_text: str = ""  # 章节级父块（切分前原文）；默认空兼容旧索引
    heading_path: str = ""  # 完整标题路径（如 "退款政策 > 特殊商品"）；默认空兼容旧索引
    parent_id: str = ""  # 同章节子块共用的稳定父标识；默认空兼容旧索引

    def to_dict(self) -> dict:
        return asdict(self)


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
    raw = path.read_text(encoding="utf-8")
    rel = path.relative_to(kb_dir).as_posix()
    return _chunk_text(raw, rel, kb_dir)


def _chunk_text(raw: str, rel: str, kb_dir: Path, *,
                parent_child: bool = True) -> list[Chunk]:
    """对已读入文本做 frontmatter 解析与正文切分（7.1 解析流水线复用）。

    parent_child=False 时清空 parent_id / parent_text（父子块关闭）。
    """
    path = Path(rel)
    is_evolved = rel.startswith("evolved/") or rel == "evolved.md"
    doc_name = EVOLVED_DOC_NAME if is_evolved else path.stem
    # id 前缀统一用无扩展名相对路径：根文档与旧格式一致（stem），
    # 子目录同名文档之间不再互相碰撞
    id_prefix = path.with_suffix("").as_posix()
    meta, body = parse_frontmatter(normalize_document(raw))
    provenance = meta.get("provenance", "")
    owner = meta.get("owner", "")

    out: list[Chunk] = []
    for idx, (section, heading_path, text) in enumerate(_split_by_headings(body)):
        text = text.strip()
        if not text:
            continue
        parent_id = f"{id_prefix}#p{idx:02d}" if parent_child else ""
        # 展示标签超长时截断（根标题…末级标题）；完整路径保留在 heading_path 元数据。
        # 预算 = 最终文本上限 - 前缀：正文按动态预算切分，绝不事后截断字符串
        label = _display_label(heading_path or section)
        prefix = f"【{doc_name} · {label}】\n"
        body_limit = MAX_CHUNK_CHARS - len(prefix)
        if body_limit < MIN_BODY_CHARS:
            raise ValueError(
                f"{rel}: 标题/文档名过长（前缀 {len(prefix)} 字符），"
                f"无法在 {MAX_CHUNK_CHARS} 字符上限内保留正文，已拒绝该文档"
            )
        if len(text) <= body_limit:
            # 单块章节：整章已在块内，无需父块
            out.append(_make_chunk(doc_name, section, heading_path, label, text, idx, 0,
                                   id_prefix, rel, provenance, owner,
                                   parent_text="", parent_id=parent_id))
            continue
        for sub_idx, (piece, piece_start, piece_end) in enumerate(
            _split_long(text, limit=body_limit)
        ):
            # 父块 = 命中子块附近的章节原文窗口（≤4000，完整包含子块），
            # 同章节子块共用 parent_id 但窗口可各不相同
            parent_text = ""
            if parent_child:
                parent_text = _parent_window(text, piece_start, piece_end)
                if piece not in parent_text:  # 构建期包含性断言（防御）
                    parent_text = ""
            out.append(_make_chunk(doc_name, section, heading_path, label, piece, idx,
                                   sub_idx, id_prefix, rel, provenance, owner,
                                   parent_text=parent_text, parent_id=parent_id))
    return out


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


def _display_label(heading_path: str) -> str:
    """检索文本前缀里的展示标签：超长时保留根标题与末级标题，中间省略。

    完整标题路径始终保存在 Chunk.heading_path 元数据中，这里只约束
    前缀展示长度（为正文预算让路）。
    """
    if len(heading_path) <= MAX_LABEL_CHARS:
        return heading_path
    root, leaf = heading_path.split(" > ", 1)[0], heading_path.rsplit(" > ", 1)[-1]
    keep = max(MAX_LABEL_CHARS - len(root) - len(" > … > "), 1)
    return f"{root} > … > {leaf[:keep]}"[:MAX_LABEL_CHARS]


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
                piece = overlap + piece
                bs -= len(overlap)
        if piece.strip():
            out.append((piece, bs, be))
    return out


def _sentence_spans(p: str, limit: int) -> list[tuple[int, int]]:
    """句段区间（p 内相对区间）：标点后可断、原子片段内不可断。

    保留空白句段（保证区间连续、块文本与原文切片严格一致），空白内容由
    上层按块粒度丢弃。单句段仍超限则硬切。
    """
    spans = _atomic_spans(p)
    cuts = [i + 1 for i, ch in enumerate(p)
            if ch in _SENTENCE_END and not _in_atomic(i + 1, spans)]
    bounds = [0] + cuts + [len(p)]
    out: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > limit:
            out.extend(_hard_spans(p, a, b, limit))
        else:
            out.append((a, b))
    return out


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
    label: str,
    text: str,
    idx: int,
    sub_idx: int,
    id_prefix: str,
    source_path: str,
    provenance: str = "",
    owner: str = "",
    parent_text: str = "",
    parent_id: str = "",
) -> Chunk:
    chunk_id = f"{id_prefix}#{idx:02d}-{sub_idx:02d}"
    body = f"【{doc} · {label or heading_path or section}】\n{text}"
    return Chunk(
        chunk_id=chunk_id, doc=doc, section=section, text=body,
        source_path=source_path, provenance=provenance, owner=owner,
        parent_text=parent_text, heading_path=heading_path, parent_id=parent_id,
    )

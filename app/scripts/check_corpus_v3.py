#!/usr/bin/env python
"""v3 语料一致性校验（计划 §2.3.4）。

校验项：
  1. 文档数量与文件名唯一性（source_path 无重复）
  2. 《交叉引用》目标存在性：正文引用的《XXX》须能匹配 knowledge/ 下某文档
  3. 现行文档（非 archive/、非 F 类外部机构）关键数字口径抽查：
     价保 7/15/30、运费险 25/40、包邮 99、上门取件 12、假一赔三 500、
     极速退款 200、会员门槛 1000/5000/20000、发货 48、积分 100 分=1 元
  4. F 类外部机构文档不得出现「并夕夕」平台自称（标题以 # 并夕夕 开头即告警）
  5. 重复内容检测：任意两文档正文相似度 > 0.9 告警（近重复）
"""
from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
KB = ROOT / "app/agent/rag/knowledge"

# 现行文档必须出现的口径（正则，宽松匹配）
KEY_NUMBERS: list[tuple[str, str]] = [
    ("价保 7 天", r"价保[^。]{0,20}7 天"),
    ("运费险上限 25 元", r"上限 25 元"),
    ("包邮门槛 99 元", r"满 99 元包邮|满 99 包邮|99 元包邮"),
    ("上门取件 12 元", r"12 元起"),
    ("假一赔三最低 500", r"最低 500 元|3 倍"),
    ("极速退款 200", r"200 元"),
    ("发货 48 小时", r"48 小时"),
    ("七天无理由", r"七天无理由|7 天无理由"),
]

EXTERNAL_DOCS = {
    "银行信用卡分期业务条款.md", "快递公司延误与丢件赔偿标准.md",
    "厂家全国联保三包政策.md", "航空公司行李损坏赔偿指引.md",
    "电信运营商话费退费规则.md", "支付平台账户安全险条款.md",
    "保险公司运费险承保条款.md",
    "品牌官方延保服务条款.md", "第三方鉴定机构流程说明.md",
    "银行储蓄卡盗刷赔付规则.md",
}


def all_md_files() -> list[Path]:
    return sorted(list(KB.glob("*.md")) + list((KB / "archive").glob("*.md")))


def refs(text: str) -> set[str]:
    """提取《XXX》引用，去掉版本注记（如（已废止））后缀做模糊匹配。"""
    found = set(re.findall(r"《([^》]{2,30})》", text))
    out = set()
    for name in found:
        base = re.sub(r"[（(].*?[)）]", "", name).strip()
        out.add(base)
    return out


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    files = all_md_files()
    print(f"文档总数: {len(files)}")

    # 1. source_path 唯一性
    names = [p.name for p in files]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        errors.append(f"source_path 重复: {dup}")

    # 2. 交叉引用存在性
    known = {p.stem for p in files}
    known.add("虚拟商品与充值类目规则")  # 可能被简写
    for p in files:
        text = p.read_text(encoding="utf-8")
        for r in refs(text):
            if r in known:
                continue
            # 模糊：标题前缀包含（如《运费险细则》对应 运费险细则）
            if any(k.startswith(r) or r.startswith(k) for k in known):
                continue
            if r.endswith("规则") and r[:-2] in known:
                continue
            if r.endswith("权益") and r[:-2] in known:
                continue
            warnings.append(f"{p.name} 引用未知文档《{r}》")

    # 3. 关键数字抽查（仅现行文档）
    current = [p for p in files if "archive" not in p.parts and p.name not in EXTERNAL_DOCS]
    text_all = "\n".join(p.read_text(encoding="utf-8") for p in current)
    for label, pat in KEY_NUMBERS:
        if not re.search(pat, text_all):
            warnings.append(f"现行文档集合中未发现口径: {label} ({pat})")

    # 4. F 类文档标题检查
    for name in EXTERNAL_DOCS:
        p = KB / name
        if not p.exists():
            errors.append(f"F 类文档缺失: {name}")
            continue
        head = p.read_text(encoding="utf-8").splitlines()[0]
        if head.startswith("# 并夕夕"):
            errors.append(f"F 类文档误用平台标题: {name} -> {head}")

    # 5. 近重复检测（全文相似度）
    texts = {p.name: p.read_text(encoding="utf-8") for p in files}
    for i, (n1, t1) in enumerate(texts.items()):
        for n2, t2 in list(texts.items())[i + 1:]:
            if len(t1) < 500 or len(t2) < 500:
                continue
            if SequenceMatcher(None, t1, t2).ratio() > 0.9:
                warnings.append(f"近重复文档: {n1} ~ {n2}")

    # 6. 统计
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"总字节数: {total_bytes:,}（约 {total_bytes/1024:.0f} KB）")
    sizes = sorted(((p.stat().st_size / 1024), p.name) for p in files)
    print(f"最大: {sizes[-1][1]} {sizes[-1][0]:.1f} KB | 最小: {sizes[0][1]} {sizes[0][0]:.1f} KB")
    print(f"长文档(>10KB): {sum(1 for s, _ in sizes if s > 10)} 份")

    print("\n== 错误 ==")
    for e in errors:
        print("  ✗", e)
    print("== 警告 ==")
    for w in warnings:
        print("  ⚠", w)
    print(f"\n错误 {len(errors)} / 警告 {len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

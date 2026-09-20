#!/usr/bin/env python
"""v3 门禁轮新维度离线分析（计划 §9 / 阶段 2.3）。

读 run_retrieval_eval 的 report.json（per-case 已有 hit_keys/tags/expected/
raw_top_score），离线计算，不触发检索、不改冻结协议：

  1. current-version hit rate：timing 用例现行文档命中率；archive 文档进
     Top-5 即算错（独立门禁 ≥95%）
  2. 多跳双口径：expected ≥2 为多跳；严格=全部命中，宽松=至少 1
  3. 条件叠加分层：tags 含 condition 的 Recall@5
  4. 分层 Recall：文档类型（A-F 映射）/ 长度档（<2KB / 2-10KB / >10KB）
  5. score 分布 + AUC：正例/负例 raw_top_score 排序法 AUC 与重叠带
  6. hard 子标签分层 + 95% 置信区间（dev vs holdout 差异的显著性判读；
     二项比例 Wald 区间，n<30 时区间仅供参考）

用法：
  python app/scripts/analyze_retrieval_report.py artifacts/eval/v3/gate/dev-final/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

F_DOCS = {
    "银行信用卡分期业务条款.md", "快递公司延误与丢件赔偿标准.md",
    "厂家全国联保三包政策.md", "航空公司行李损坏赔偿指引.md",
    "电信运营商话费退费规则.md", "支付平台账户安全险条款.md",
    "保险公司运费险承保条款.md", "品牌官方延保服务条款.md",
    "第三方鉴定机构流程说明.md", "银行储蓄卡盗刷赔付规则.md",
}
D_DOCS = {
    "大促价保特别条款.md", "价保除外类目清单.md", "运费险商家版规则.md",
    "大额订单退款时效.md", "平台付费会员（PRO）权益.md",
    "店铺会员积分兑换细则.md", "店铺券与平台券叠加规则.md",
    "品类券适用范围细则.md", "平台分期购细则.md", "拆封商品退换判定标准.md",
    "官方延保服务范围对照.md", "退货运费承担方判定规则.md",
    "会员价保升级规则.md", "退款到账时效分档表.md", "先用后付额度规则.md",
    "分期手续费费率表.md", "退货运费计算标准.md", "上门取件服务说明.md",
    "积分兑换与抵扣规则.md", "优惠券退回与补发规则.md",
    "预售定金与尾款规则.md", "七天无理由商品清单.md", "破损赔付标准细则.md",
    "物流时效与赔付标准.md", "大促满减与跨店规则.md",
}
C_DOCS = {
    "偏远地区配送补充说明.md", "港澳台订单规则.md", "跨境退货特别流程.md",
    "跨境清关与税费说明.md", "企业采购专属售后.md", "校园专区规则.md",
    "门店自提规则.md", "社区团购履约规则.md", "定制商品生产周期说明.md",
    "预售商品规则.md", "大件家电配送安装规则.md", "冷链配送标准.md",
    "同城闪送服务说明.md", "海外直邮税费说明.md", "边境地区配送限制.md",
}


def doc_type(source_path: str, size_bytes: int) -> str:
    if source_path.startswith("archive/"):
        return "B-archive"
    if source_path in F_DOCS:
        return "F-external"
    if source_path in D_DOCS:
        return "D-numconf"
    if source_path in C_DOCS:
        return "C-region"
    if size_bytes > 10_000:
        return "E-long"
    if size_bytes < 2_000:
        return "A-short"
    return "A/A-mid"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v3 门禁轮离线分析")
    parser.add_argument("report", help="report.json 路径")
    parser.add_argument("--kb", default=str(ROOT / "app/agent/rag/knowledge"),
                        help="知识库目录（用于文档大小分档）")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cases = report["cases"]
    kb = Path(args.kb)
    size_of = {}
    for p in kb.rglob("*"):
        if p.is_file():
            size_of[p.relative_to(kb).as_posix()] = p.stat().st_size

    def hit(case) -> set:
        return set(case.get("hit_keys") or [])

    def expected(case) -> list:
        return list(case.get("expected") or [])

    # 1. current-version hit（timing 用例）
    timing = [c for c in cases if "timing" in c.get("tags", [])]
    cur_ok = cur_bad = 0
    for c in timing:
        h = hit(c)
        exp = expected(c)
        hit_current = all(e in h for e in exp)
        archive_leak = any(k.startswith("archive/") for k in h)
        if hit_current and not archive_leak:
            cur_ok += 1
        else:
            cur_bad += 1
    print(f"[1] current-version hit rate: {cur_ok}/{len(timing)} = "
          f"{cur_ok/len(timing):.1%}（门禁 ≥95%；archive 进 Top-5 即错）")

    # 2. 多跳双口径
    multi = [c for c in cases if len(expected(c)) >= 2]
    strict = sum(1 for c in multi if all(e in hit(c) for e in expected(c)))
    loose = sum(1 for c in multi if hit(c) & set(expected(c)))
    if multi:
        print(f"[2] 多跳双口径: n={len(multi)} 严格 {strict/len(multi):.1%} / "
              f"宽松 {loose/len(multi):.1%}")

    # 3. 条件叠加
    cond = [c for c in cases if "condition" in c.get("tags", []) and expected(c)]
    if cond:
        ok = sum(1 for c in cond if all(e in hit(c) for e in expected(c)))
        print(f"[3] 条件叠加 Recall@5: {ok}/{len(cond)} = {ok/len(cond):.1%}")

    # 4. 分层 Recall（类型/长度档）
    layer: dict[str, list] = {}
    for c in cases:
        if not expected(c):
            continue
        for e in expected(c):
            t = doc_type(e, size_of.get(e, 0))
            layer.setdefault(t, []).append(c)
            break  # 按首个 expected 分档
    print("[4] 分层 Recall@5:")
    for t in sorted(layer):
        cs = layer[t]
        ok = sum(1 for c in cs if all(e in hit(c) for e in expected(c)))
        print(f"    {t:<12} n={len(cs):>3}  {ok/len(cs):.1%}")

    # 5. score 分布 + AUC
    pos_scores = [c["raw_top_score"] for c in cases
                  if expected(c) and c.get("raw_top_score") is not None]
    neg_scores = [c["raw_top_score"] for c in cases
                  if not expected(c) and c.get("raw_top_score") is not None]
    if pos_scores and neg_scores:
        # 排序法 AUC（正例分数 > 负例分数的占比）
        auc = 0.0
        n_pairs = 0
        for ps in pos_scores:
            for ns in neg_scores:
                auc += 1 if ps > ns else 0.5 if ps == ns else 0
                n_pairs += 1
        auc /= n_pairs
        print(f"[5] score 分布: 正例 n={len(pos_scores)} "
              f"P50={sorted(pos_scores)[len(pos_scores)//2]:.4f} "
              f"P90={sorted(pos_scores)[int(len(pos_scores)*0.9)-1]:.4f}")
        print(f"    负例 n={len(neg_scores)} "
              f"P50={sorted(neg_scores)[len(neg_scores)//2]:.4f} "
              f"P90={sorted(neg_scores)[int(len(neg_scores)*0.9)-1]:.4f}")
        print(f"    AUC={auc:.4f}（1.0=完全可分；可兼得性见可行带扫描）")

    # 6. hard 子标签分层 + 95% CI（review：dev 76.6% vs holdout 86.0% 差异
    #    此前只有定性解释；分层 + 区间把"是否显著/是否噪声"变成算式）
    hard_cases = [c for c in cases if "hard" in c.get("tags", []) and expected(c)]
    if hard_cases:
        print(f"[6] hard 子标签分层（n={len(hard_cases)}，95% Wald CI）：")
        sub_tags: dict[str, list] = {}
        for c in hard_cases:
            for t in c.get("tags", []):
                if t != "hard":
                    sub_tags.setdefault(t, []).append(c)
        for t in sorted(sub_tags, key=lambda k: -len(sub_tags[k])):
            cs = sub_tags[t]
            n = len(cs)
            ok = sum(1 for c in cs if all(e in hit(c) for e in expected(c)))
            p = ok / n
            w = 1.96 * (p * (1 - p) / n) ** 0.5 if n else 0.0
            note = "（n<30，区间仅参考）" if n < 30 else ""
            print(f"    {t:<16} n={n:>3}  {p:6.1%}  ±{w:.1%}{note}")
        n = len(hard_cases)
        ok = sum(1 for c in hard_cases if all(e in hit(c) for e in expected(c)))
        p = ok / n
        w = 1.96 * (p * (1 - p) / n) ** 0.5
        print(f"    {'整体 hard':<14} n={n:>3}  {p:6.1%}  ±{w:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
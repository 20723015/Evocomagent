"""3.4 build_holdout.py：120 条独立 holdout 检索集（冻结后只运行一次）。

构成（eval-v2 冻结口径）：
- 80 条基础正例：40 份文档 × 2 条（easy，标题/首个小节直答）；
- 20 条困难正例：跨文档、口语化或混淆（hard）；
- 20 条负例：知识库确实无答案的近域/离域（expected=[]）。

规则：
- 阈值只允许在 535 dev set 校准；holdout 生成后冻结，
  只能运行一次，不得据其结果修改查询或文档；
- `--check`：与磁盘冻结集比较，不一致退出码 1（CI 门禁）。

文档来源：app/agent/rag/knowledge/*.md（40 份，标题映射手工维护于
HOLDOUT_QUERIES，避免从文档自动生成导致「测试泄漏到训练」的伪泛化）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# 每份文档 2 条基础正例（查询 → 期望命中文档；手工编写，不读文档内容，
# 确保 holdout 测的是真实泛化而非生成器的记忆）
DOC_BASE_POSITIVE: dict[str, list[str]] = {
    "退换货政策.md": ["七天无理由退货的适用范围是什么", "质量问题退换货的时效是多久"],
    "配送说明.md": ["发什么快递", "偏远地区加运费吗"],
    "会员权益.md": ["会员有哪些等级", "钻石会员有什么权益"],
    "常见问题FAQ.md": ["买到的商品是假货怎么办", "怎么找回登录密码"],
    "发票与开票规则.md": ["可以开发票吗", "电子发票和纸质发票的区别"],
    "极速退款规则.md": ["极速退款的条件", "退款多久到账"],
    "运费险细则.md": ["运费险怎么理赔", "哪些订单赠送运费险"],
    "包装与发货规范.md": ["发货时间多久", "包装有什么要求"],
    "物流异常与投诉处理.md": ["物流很久没更新怎么办", "包裹丢失怎么赔付"],
    "售后维修与延保.md": ["怎么申请售后维修", "延保服务怎么购买"],
    "支付与账户安全.md": ["支付密码怎么修改", "账户被盗怎么办"],
    "隐私与个人信息.md": ["我的个人信息怎么删除", "隐私政策有哪些内容"],
    "优惠券与促销规则.md": ["优惠券怎么领取", "促销活动的规则"],
    "百亿补贴规则.md": ["百亿补贴活动怎么参加", "百亿补贴的商品是真的吗"],
    "大促活动总规则.md": ["大促活动的预售规则", "大促期间价保规则"],
    "直播与限时活动规则.md": ["直播间购物有什么保障", "限时活动的规则"],
    "砍价与游戏玩法规则.md": ["砍价免费拿怎么玩", "游戏玩法活动的规则"],
    "礼品卡与储值卡.md": ["礼品卡怎么使用", "储值卡可以退吗"],
    "企业采购与团购.md": ["企业采购有优惠吗", "团购的起订量是多少"],
    "跨境与海外购.md": ["跨境商品怎么退货", "海外购的税费谁承担"],
    "纠纷仲裁与举报规则.md": ["怎么投诉商家", "交易纠纷怎么处理"],
    "买家评价与晒单规则.md": ["评价返现是真的吗", "晒单有什么奖励"],
    "店铺粉丝与会员.md": ["怎么成为店铺粉丝", "店铺会员有什么权益"],
    "购物车与收藏夹.md": ["购物车最多放多少件", "收藏夹有什么用"],
    "商品质检与溯源.md": ["怎么查看商品溯源信息", "质检报告在哪里看"],
    "价保与赔付规则.md": ["价保怎么申请", "哪些商品支持价保"],
    "先用后付与分期付款.md": ["先用后付怎么开通", "分期付款有利息吗"],
    "生鲜与定制商品规则.md": ["生鲜商品支持无理由退货吗", "定制商品可以退吗"],
    "3C数码类目规则.md": ["数码产品的保修期", "二手数码可以退吗"],
    "食品饮料类目规则.md": ["食品过期怎么赔付", "食品类目退货规则"],
    "美妆个护类目规则.md": ["化妆品用过敏可以退吗", "临期化妆品可以买吗"],
    "服装鞋帽类目规则.md": ["衣服尺码不合适可以换吗", "鞋类商品的退换规则"],
    "家居家纺类目规则.md": ["床品可以七天无理由退吗", "家具类目的发货时效"],
    "母婴用品类目规则.md": ["奶粉可以退吗", "母婴商品的质检标准"],
    "宠物用品类目规则.md": ["宠物粮可以退吗", "宠物用品的类目规则"],
    "运动户外类目规则.md": ["运动鞋的退换规则", "户外装备的质保"],
    "珠宝钟表类目规则.md": ["钻石饰品可以退吗", "手表怎么保修"],
    "图书文教类目规则.md": ["图书拆封后可以退吗", "教辅资料的发货时效"],
    "五金建材类目规则.md": ["建材商品怎么退换", "五金类目的质保"],
    "车品配件类目规则.md": ["车品可以退吗", "机油怎么辨别真假"],
}

# 困难正例：跨文档 / 口语化 / 混淆（真实存在于知识库，但不在基础正例里）
HARD_POSITIVE: list[dict] = [
    {"id": "holdout_hard_01", "query": "我七天无理由退的货，运费是不是我自己出啊",
     "expected": ["退换货政策.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_02", "query": "双十一买的东西，之后降价了能退差价不",
     "expected": ["价保与赔付规则.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_03", "query": "买了个二手手机，结果三天就坏了，能找平台吗",
     "expected": ["3C数码类目规则.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_04", "query": "给猫买的粮，它不吃，拆开了还能退不",
     "expected": ["宠物用品类目规则.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_05", "query": "国外买的包，关税是你们出还是我出",
     "expected": ["跨境与海外购.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_06", "query": "我妈收到货发现破了，拍照给谁看",
     "expected": ["物流异常与投诉处理.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_07", "query": "想给公司采购一百台电脑，找谁谈价格",
     "expected": ["企业采购与团购.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_08", "query": "直播间抢的东西和介绍的不一样，怎么弄",
     "expected": ["直播与限时活动规则.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_09", "query": "积分换的券和钱买的券，退的时候一样吗",
     "expected": ["优惠券与促销规则.md"], "tags": ["hard", "confusion"]},
    {"id": "holdout_hard_10", "query": "我租的房子被中介骗了，能打你们客服电话吗",
     "expected": [], "tags": ["hard", "out_of_domain"]},
    {"id": "holdout_hard_11", "query": "把游戏账号卖了，买家说被找回，平台管不管",
     "expected": ["纠纷仲裁与举报规则.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_12", "query": "申请退货以后快递单号发哪里",
     "expected": ["退换货政策.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_13", "query": "新买的冰箱没电是质量问题吗",
     "expected": ["售后维修与延保.md"], "tags": ["hard", "confusion"]},
    {"id": "holdout_hard_14", "query": "买奶粉送的小熊玩具能单独退吗",
     "expected": ["母婴用品类目规则.md"], "tags": ["hard", "confusion"]},
    {"id": "holdout_hard_15", "query": "分期买的东西，提前还完还有手续费吗",
     "expected": ["先用后付与分期付款.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_16", "query": "给老家亲戚寄的电视，验货发现屏裂了，谁负责",
     "expected": ["包装与发货规范.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_17", "query": "卖家商品页写的和实际收到的不一样，怎么投诉",
     "expected": ["纠纷仲裁与举报规则.md"], "tags": ["hard", "colloquial"]},
    {"id": "holdout_hard_18", "query": "我买的背单词卡，拆了激活了，能退吗",
     "expected": ["图书文教类目规则.md"], "tags": ["hard", "confusion"]},
    {"id": "holdout_hard_19", "query": "在别的平台买东西被骗了，你们能帮我追吗",
     "expected": [], "tags": ["hard", "out_of_domain"]},
    {"id": "holdout_hard_20", "query": "外卖点的奶茶洒了，找谁赔",
     "expected": [], "tags": ["hard", "out_of_domain"]},
]

# 负例：知识库确实无答案（近域/离域）
NEGATIVE: list[dict] = [
    {"id": "holdout_neg_01", "query": "你们平台什么时候上市",
     "expected": [], "tags": ["negative", "near_domain"]},
    {"id": "holdout_neg_02", "query": "股票开户需要什么条件",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_03", "query": "今天的天气怎么样",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_04", "query": "社保断缴怎么补",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_05", "query": "你们招不招主播，薪资多少",
     "expected": [], "tags": ["negative", "near_domain"]},
    {"id": "holdout_neg_06", "query": "驾照到期怎么换证",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_07", "query": "地铁票价怎么算",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_08", "query": "房贷利率现在是多少",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_09", "query": "帮我查一下顺丰快递到哪了",
     "expected": [], "tags": ["negative", "near_domain"]},
    {"id": "holdout_neg_10", "query": "淘宝的客服电话是多少",
     "expected": [], "tags": ["negative", "near_domain"]},
    {"id": "holdout_neg_11", "query": "你家楼下便利店几点开门",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_12", "query": "牛奶和鸡蛋一起吃会不会中毒",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_13", "query": "小区物业费该交多少",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_14", "query": "怎么预约九价疫苗",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_15", "query": "你们平台的工号怎么查",
     "expected": [], "tags": ["negative", "near_domain"]},
    {"id": "holdout_neg_16", "query": "快递柜超时取件收费吗",
     "expected": [], "tags": ["negative", "near_domain"]},
    {"id": "holdout_neg_17", "query": "公积金怎么提取",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_18", "query": "今天周杰伦有演唱会吗",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_19", "query": "小区停电找谁投诉",
     "expected": [], "tags": ["negative", "out_of_domain"]},
    {"id": "holdout_neg_20", "query": "你们商城能和京东比价吗",
     "expected": [], "tags": ["negative", "near_domain"]},
]


def build_holdout() -> list[dict]:
    """组装 120 条 holdout（40×2 基础正例 + 20 困难 + 20 负例），冻结顺序。"""
    cases: list[dict] = []
    for i, (doc, queries) in enumerate(sorted(DOC_BASE_POSITIVE.items()), start=1):
        for j, q in enumerate(queries, start=1):
            cases.append({
                "id": f"holdout_base_{i:02d}_{j}",
                "query": q,
                "expected": [doc],
                "k": 5,
                "tags": ["easy", "direct"],
            })
    cases.extend(HARD_POSITIVE)
    cases.extend(NEGATIVE)
    assert len(cases) == 120, len(cases)
    return cases


def _stats(cases: list[dict]) -> dict:
    easy = sum(1 for c in cases if "easy" in c.get("tags", []))
    hard = sum(1 for c in cases if "hard" in c.get("tags", []))
    neg = sum(1 for c in cases if not c["expected"])
    return {"total": len(cases), "easy": easy, "hard": hard, "negative": neg}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="构建 120 条独立 holdout 检索集（3.4）")
    parser.add_argument("--out", default="app/evaluation/holdout_cases.json")
    parser.add_argument("--check", action="store_true",
                        help="冻结校验：与磁盘 holdout 一致才通过（禁止结果驱动修改）")
    args = parser.parse_args(argv)

    holdout = build_holdout()
    from app.observability.logging import get_logger

    log = get_logger("app.scripts.build_holdout")
    out = ROOT / args.out
    if args.check:
        if not out.exists():
            log.error("--check 失败：%s 不存在（冻结集缺失）", out)
            return 1
        current = json.loads(out.read_text(encoding="utf-8"))["cases"]
        if current != holdout:
            log.error("--check 失败：%s 与构建结果不一致（冻结集被修改）", out)
            return 1
        log.info("--check 通过：holdout 冻结（%s）", _stats(holdout))
        return 0

    out.write_text(
        json.dumps({"cases": holdout, "frozen": True}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8",
    )
    log.info("holdout 已写入 %s：%s", out, _stats(holdout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
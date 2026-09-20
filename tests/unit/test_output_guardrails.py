"""输出侧内容安全增强（P3-3）：类目词表 + 情绪信号复用 + 零误伤回放。

验收点对应《能力补全全量计划》P3-3：
- 词表按类目扩充（站外支付/外部联系方式/竞品引导/辱骂）；
- P1-1 的情绪分级信号复用为攻击性检测输入（angry/extreme → strict 扫描）；
- 扩充词表零误伤：正常咨询语料回放全通过；
- 新增对抗 case 全拦截。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.security.guardrails import check_output

# ============================================================
# 对抗 case：必须全拦
# ============================================================
@pytest.mark.parametrize("text,category", [
    # 站外支付 / 凭证索取（部分先命中既有裸词表，故只断言拦截）
    ("您可以直接私下转账给我，我帮您处理", "offsite_payment"),
    ("请把货款转到我微信，就不用走平台了", "offsite_payment"),
    ("麻烦发一下您的银行卡密码核对身份", "offsite_payment"),
    # 外部联系方式引导
    ("加我微信，我私下给您处理", "external_contact"),
    ("我们私聊吧，这里不方便说", "external_contact"),
    ("扫码加二维码领取优惠", "external_contact"),
    ("建议站外联系我们的专员", "external_contact"),
    # 竞品引导
    ("这个价格您去京东买更便宜", "competitor_steering"),
    ("建议您上淘宝看看同款", "competitor_steering"),
    ("别在我们这买了，拼多多有补贴", "competitor_steering"),
])
def test_adversarial_outputs_blocked(text, category):
    verdict = check_output(text)
    assert verdict.blocked is True
    assert category in verdict.reason or "敏感词" in verdict.reason


@pytest.mark.parametrize("text", [
    "你这个傻逼客服",
    "滚蛋，别回复了",
    "你们就是一群废物",
    "垃圾平台，去死吧",
    "他妈的什么破服务",
])
def test_abuse_blocked_only_under_strict(text):
    """辱骂类目是中精度：正常轮次不扫，用户带攻击性时启用。"""
    assert check_output(text).blocked is False                      # 正常轮次不误伤
    verdict = check_output(text, user_emotion="angry")
    assert verdict.blocked is True and "abuse" in verdict.reason
    assert check_output(text, user_emotion="extreme").blocked is True


def test_extreme_user_emotion_also_strict():
    assert check_output("你们都是白痴", user_emotion="extreme").blocked is True


def test_bare_terms_block_direct_mention_but_not_warning():
    """裸词表：直接说出（引导）拦截；否定语境的警告话术放行。

    P3-3 引入否定前缀守卫后，「请勿私下转账」这类合规警告不再被误拦——
    它恰恰是客服应当说的话。
    """
    assert check_output("您可以私下转账给客服").blocked is True
    assert check_output("请勿私下转账，平台外交易不受保护").blocked is False
    assert check_output("不要向任何人透露银行卡密码").blocked is False


# ============================================================
# 零误伤：正常咨询语料回放
# ============================================================
# 合法客服话术样本：覆盖政策答复、安抚、退款/物流/会员/商品推荐等高频形态。
# 刻意包含词形相近的合法用法（微信支付/官方电话/平台内交易/竞品名出现在
# 「不支持」语境），确保类目模式不会误伤。
LEGIT_REPLIES = (
    "您好，请问有什么可以帮您？",
    "七天无理由退货需要在签收后 7 天内申请，定制商品不支持。",
    "退款申请已提交，等待商家审核，审核通过前可以撤回。",
    "您的订单 ORD-20240115-001 目前是已发货状态，预计 3 天内送达。",
    "运费由平台承担，需要您提供照片或视频举证。",
    "很抱歉给您带来不好的体验，我理解您的心情，这就为您核实。",
    "请您先别着急，我马上帮您查询物流进度。",
    "本平台支持微信支付、支付宝支付和银行卡支付。",
    "如需人工协助，可拨打官方客服电话 400-000-0000。",
    "交易请务必在平台内完成，平台外交易不受消费者保障保护。",
    "我们平台不支持引导用户到站外交易，请通过订单页面操作。",
    "淘宝和京东的商品我们无法比价，建议以本平台页面价格为准。",
    "钻石会员可使用「无忧退」权益，一年 4 次，无需理由，运费全免。",
    "该商品属于贴身衣物，拆封后不支持七天无理由退货。",
    "您的退款已到账，请查收原支付渠道。",
    "抱歉，这个问题超出我的处理权限，已为您转接人工客服。",
    "请问您还有其他问题吗？",
    "垃圾袋属于家居日用类目，满 39 元包邮。",
    "小米14 Ultra 手机当前有货，支持 12 期免息。",
    "您反馈的客服态度问题我们已经记录，会加强培训，非常抱歉。",
    "请不要向任何人透露您的支付密码或验证码。",
)


@pytest.mark.parametrize("text", LEGIT_REPLIES)
def test_legit_replies_never_blocked(text):
    assert check_output(text).blocked is False


@pytest.mark.parametrize("text", LEGIT_REPLIES)
def test_legit_replies_not_blocked_even_under_strict(text):
    """即使用户情绪激动，合法话术也不得被拦（strict 只加辱骂类目）。"""
    assert check_output(text, user_emotion="angry").blocked is False


def test_golden_set_corpus_replay_no_false_positive():
    """黄金集语料回放：335 条用例的 turns/描述/关键词全部不触发拦截。

    这是「零误伤」的规模化证据——这些字符串是真实咨询语境下的用户与业务
    语言，词表若把它们判成敏感内容，线上就是成片误拦。
    """
    path = Path(__file__).parents[2] / "app" / "evaluation" / "cases_large.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 300

    blocked: list[tuple[str, str]] = []
    for case in cases:
        corpus = list(case.get("turns") or [])
        corpus.append(str(case.get("description") or ""))
        corpus.extend(str(k) for k in (case.get("expected_keywords") or []))
        for text in corpus:
            if not text:
                continue
            for emotion in ("neutral", "angry"):
                verdict = check_output(text, user_emotion=emotion)
                if verdict.blocked:
                    blocked.append((text, verdict.reason))
    assert blocked == [], f"误伤 {len(blocked)} 条: {blocked[:5]}"


def test_skill_templates_replay_no_false_positive():
    """技能/提示词模板回放：SKILL.md 与话术模板不得被自身词表拦截。"""
    root = Path(__file__).parents[2] / "app" / "agent" / "skills" / "definitions"
    files = sorted(root.rglob("SKILL.md"))
    assert files, "未找到 SKILL.md"
    blocked: list[tuple[str, str]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            verdict = check_output(line, user_emotion="angry")
            if verdict.blocked:
                blocked.append((f"{path.name}: {line[:60]}", verdict.reason))
    assert blocked == [], f"误伤 {len(blocked)} 条: {blocked[:5]}"

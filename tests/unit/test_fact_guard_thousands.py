"""fact_guard 千分位碎片与升级回环（2026-09-18 Docker 口径评测实测缺陷）。

回归背景：ES+精排口径下 335 例 agent 评测出现 156 例「正常咨询被过度转人工」，
根因是两个叠加缺陷：

1. **千分位碎片**：`_claim_clauses` 用 `[，,；;、\n]` 切子句，数字内的逗号
   被当成分隔符，`20,000 元` 断成 `000 元`；碎片与真实值混进同一
   (主题锚点, 单位) 值集合 → `_detect_conflicts` 的「同键多值」判定误报冲突。
2. **升级回环**：`ground_reply` 删句后会追加「…可转人工核实。」，而调用方用
   `"转人工" in cleaned` 判升级 —— 等于 fact_guard 自己触发了升级，
   任何删句都必然转人工，判据失去区分度。
"""

from __future__ import annotations

from app.agent.fact_guard import ground_reply


def test_thousands_separator_does_not_create_conflict():
    """证据与回复金额一致（同为千分位写法）时不得判冲突、不得删句。"""
    evidence = [
        "会员等级按近 12 个月累计实付金额计算：白银 1,000 元、黄金 5,000 元、钻石 20,000 元。",
    ]
    reply = "会员等级分为四级，按近 12 个月累计实付金额自动升降：白银 1,000 元、黄金 5,000 元、钻石 20,000 元。"
    cleaned, verdict = ground_reply(reply, evidence)
    assert verdict.conflicts == [], f"千分位碎片被误判为冲突: {verdict.conflicts}"
    assert verdict.removed_sentences == 0
    assert cleaned == reply


def test_thousands_separator_without_space():
    """无空格写法同样不得碎片化（`20,000元`）。"""
    evidence = ["钻石会员门槛为 20,000元 累计实付。"]
    reply = "钻石会员需要累计实付 20,000元。"
    _, verdict = ground_reply(reply, evidence)
    assert verdict.conflicts == []
    assert verdict.removed_sentences == 0


def test_real_conflict_still_detected():
    """真冲突（同主题同单位多值且回复取其一）仍要拦住——修复不得放宽判据。"""
    evidence = [
        "退款到账时间为 7 天。",
        "退款到账时间为 15 天。",
    ]
    reply = "您的退款到账时间为 7 天。"
    _, verdict = ground_reply(reply, evidence)
    assert verdict.conflicts, "真冲突被漏掉，修复过度放宽"


def test_ungrounded_number_still_removed():
    """无证据数字仍要删（修复只针对碎片，不放行凭空数字）。"""
    evidence = ["退货需要在签收后 7 天内申请。"]
    reply = "退货需要在签收后 7 天内申请，运费 999 元由平台承担。"
    cleaned, verdict = ground_reply(reply, evidence)
    assert verdict.removed_sentences == 1
    assert "999" not in cleaned


def test_mentions_handoff_uses_original_reply_not_appended_suffix():
    """升级判据取模型原文：fact_guard 自己追加的核实话术不得触发升级。"""
    evidence = ["退货需要在签收后 7 天内申请。"]
    # 模型原文没提转人工，但含无证据数字 → 会被删句并追加核实话术
    reply = "退货需要在签收后 7 天内申请，另需支付 999 元运费。"
    cleaned, verdict = ground_reply(reply, evidence)
    assert verdict.removed_sentences == 1
    assert "转人工" in cleaned              # 追加的核实话术确实带「转人工」
    assert verdict.mentions_handoff is False  # 但原文没有 → 不得据此升级


def test_mentions_handoff_true_when_model_says_it():
    """模型原文确实提了转人工 → 判据为真（保留原有升级语义）。"""
    evidence = ["退货需要在签收后 7 天内申请。"]
    reply = "这个问题超出我的权限，已为您转人工。另外运费为 999 元。"
    _, verdict = ground_reply(reply, evidence)
    assert verdict.mentions_handoff is True


def test_verdict_as_dict_exposes_mentions_handoff():
    """审计字段随 as_dict 透出（评估/面板可归因）。"""
    _, verdict = ground_reply("您好。", ["任意证据"])
    assert "mentions_handoff" in verdict.as_dict()

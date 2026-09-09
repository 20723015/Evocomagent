# ruff: noqa: DTZ005
"""确定性 STM 提取全链路（阶段F）：提取 → apply_memory_mutations 落库。

覆盖验收点：
- 提取结果能通过 key_spec/_eligible 校验真实落库（受控键映射回归）；
- 同键覆盖产生 superseded 链（upsert 单值语义）；
- 手机号 PII 脱敏后入库；
- 无显式陈述句式不提取；
- 内容不变时零变更（幂等）。
全程无网络、无 LLM。
"""

from __future__ import annotations

from app.agent.memory.models import (
    ACTIVE,
    SUPERSEDED,
    MemoryFact,
    apply_memory_mutations,
)
from app.agent.memory.stm_rules import extract_stm_slots


def _msg(text: str, role: str = "user") -> dict:
    return {"role": role, "content": text}


def _apply(records, mutations, *, max_active=50):
    return apply_memory_mutations(records, mutations, max_active=max_active)


class TestExtractionRoundTrip:
    def test_extract_and_apply_persists_facts(self):
        """提取 → apply 全链路：受控键 + upsert 必须通过校验真实落库。

        回归点：槽位键曾不在 MEMORY_KEY_SPECS 且 operation 用非法的
        "update"/single 键用 "add"，_eligible 全拒 → 确定性提取完全失效。
        """
        messages = [
            _msg("我叫王小明"),
            _msg("我是一名工程师"),
            _msg("收货地址改为上海市浦东新区张江路100号"),
            _msg("我的尺码是XL"),
        ]
        mutations = extract_stm_slots(messages, [])
        by_key = {m.fact_key: m for m in mutations}
        assert set(by_key) == {
            "identity.name", "identity.occupation",
            "identity.address", "preference.size",
        }
        assert all(m.operation == "upsert" and m.explicit for m in mutations)

        records = _apply([], mutations)
        active = {f.fact_key: f for f in records if f.status == ACTIVE}
        assert set(active) == set(by_key)
        assert active["identity.name"].content == "name:王小明"
        assert active["identity.occupation"].content == "occupation:工程师"
        assert active["preference.size"].content == "size:XL"
        # evidence = 用户原话（可审计口径）
        assert active["identity.name"].evidence == "我叫王小明"

    def test_preference_slot_uses_custom_key(self):
        """「我喜欢X」自由文本走 custom.preference，category 保持 preference。"""
        mutations = extract_stm_slots([_msg("我比较喜欢宽松透气的裤子")], [])
        assert len(mutations) == 1
        m = mutations[0]
        assert m.fact_key == "custom.preference"
        assert m.category == "preference"
        assert m.content == "preference:宽松透气的裤子"
        records = _apply([], mutations)
        assert records[0].fact_key == "custom.preference"
        assert records[0].category == "preference"
        assert records[0].status == ACTIVE

    def test_newest_message_wins_within_turn(self):
        """同轮先说 A 后改口 B：最新消息优先，只落 B。"""
        mutations = extract_stm_slots(
            [_msg("我穿M码"), _msg("我穿XL码")], [],
        )
        sizes = [m for m in mutations if m.fact_key == "preference.size"]
        assert len(sizes) == 1
        assert sizes[0].content == "size:XL"

    def test_identical_content_yields_no_mutation(self):
        """与现有 active 事实内容一致 → 不产生变更（零写入）。"""
        existing = [MemoryFact(
            content="name:王小明", category="identity", created_at="2026-09-08T00:00:00",
            fact_key="identity.name",
        )]
        assert extract_stm_slots([_msg("我叫王小明")], existing) == []


class TestSupersededChain:
    def test_same_key_overwrite_supersedes_old(self):
        """同键覆盖：旧事实 superseded、新事实 active、版本链可追溯。"""
        first = extract_stm_slots([_msg("收货地址改为北京市朝阳区望京街道1号")], [])
        records = _apply([], first)
        assert len(records) == 1 and records[0].status == ACTIVE

        second = extract_stm_slots(
            [_msg("地址换成深圳市南山区科技园路2号")],
            [f for f in records if f.status == ACTIVE],
        )
        records = _apply(records, second)
        by_status = {f.status for f in records}
        assert by_status == {ACTIVE, SUPERSEDED}
        old = next(f for f in records if f.status == SUPERSEDED)
        new = next(f for f in records if f.status == ACTIVE)
        assert old.content == "address:北京市朝阳区望京街道1号"
        assert new.content == "address:深圳市南山区科技园路2号"
        assert new.supersedes_id == old.fact_id
        assert new.fact_key == "identity.address" == old.fact_key


class TestSensitiveMasking:
    def test_phone_number_masked_before_store(self):
        """手机号不入记忆：content 落库前脱敏，原文只留 evidence 外的槽位值。"""
        mutations = extract_stm_slots([_msg("我喜欢13800138000")], [])
        assert len(mutations) == 1
        assert "13800138000" not in mutations[0].content
        assert mutations[0].content == "preference:[已脱敏号码]"
        records = _apply([], mutations)
        assert records[0].content == "preference:[已脱敏号码]"
        # evidence 是用户原话，但 content（注入 prompt 的载体）已脱敏
        assert records[0].evidence == "我喜欢13800138000"


class TestNoExtraction:
    def test_non_explicit_utterances_ignored(self):
        """无显式句式（问句/客服话术/普通咨询）不提取。"""
        messages = [
            _msg("有什么裤子推荐吗？"),
            _msg("帮我查一下订单"),
            _msg("这件衣服质量有问题，我要退货"),
        ]
        assert extract_stm_slots(messages, []) == []

    def test_assistant_messages_ignored(self):
        """只从用户原话提取，客服回复不触发槽位。"""
        messages = [
            _msg(role="assistant", text="请问您叫什么名字？"),
            _msg(role="assistant", text="我帮您登记地址：上海市黄浦区1号"),
        ]
        assert extract_stm_slots(messages, []) == []

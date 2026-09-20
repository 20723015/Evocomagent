"""批次7（Review #7/#8）：TTL 类别豁免 + memory 预算 continue。

- identity/preference 豁免 memory_fact_ttl_days（身份与偏好长期有效）；
  issue/behavior/other 超期不注入（有意行为变更，评测口径需关注）；
- context_builder：memory 段超预算 continue 而非 break（大的段不得吞掉
  后面更小、更新的段）；
- identity 受控单值键每轮必注入（跳过相关性门槛），其余类别维持筛选；
- exclude_keys 仍作为 LTM 层的通用排除能力保留（调用方有更权威的实时值时
  不得与本层旧值同屏）。
全程无网络、无 LLM。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agent.context_builder import ContextBuilder
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.models import MemoryFact


def _old_fact(content: str, category: str, key: str,
              days_ago: int = 400) -> MemoryFact:
    created = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(
        timespec="seconds"
    )
    return MemoryFact(
        content=content, category=category, created_at=created,
        fact_key=key, updated_at=created,
    )


class TestTtlExemption:
    def test_year_old_identity_preference_injected_issue_filtered(self):
        ltm = LongTermMemory(user_id="u-ttl", memory_dir="")
        ltm.facts = [
            _old_fact("name:王小明", "identity", "identity.name"),
            _old_fact("preference:喜欢简洁回复", "preference", "custom.style"),
            _old_fact("issue:上周投诉物流慢", "issue", "custom.issue1"),
            _old_fact("behavior:常在晚间下单", "behavior", "custom.behav1"),
        ]
        # 空 query：不做相关性筛选，隔离 TTL 行为
        selected = ltm.select_facts_for_prompt("")
        keys = {f.fact_key for f in selected}
        assert "identity.name" in keys
        assert "custom.style" in keys
        assert "custom.issue1" not in keys
        assert "custom.behav1" not in keys

    def test_recent_issue_still_injected(self):
        created = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(
            timespec="seconds"
        )
        ltm = LongTermMemory(user_id="u-ttl", memory_dir="")
        ltm.facts = [MemoryFact(
            content="issue:近期咨询发票", category="issue",
            created_at=created, fact_key="custom.invoice",
            updated_at=created,
        )]
        assert [f.fact_key for f in ltm.select_facts_for_prompt("")] == [
            "custom.invoice",
        ]


class TestMemoryBudget:
    def test_over_budget_section_does_not_swallow_later_sections(self):
        """continue 语义：首段超预算不再吞掉后面能放下的段（旧 break 回归）。

        记忆段当前至多一段（STM 槽位层已删除），用桩管理器返回「超大段 +
        正常段」保住该保证——预算裁剪逻辑不得退回 break。
        """
        from types import SimpleNamespace

        big = {"role": "system", "content": "史" * 20000}
        small = {"role": "system", "content": "历史偏好：喜欢简洁回复"}

        class _TwoSectionMemory:
            memory_enabled = True

            def build_memory_prompt_sections(self, query="", query_embedding=None):
                return [big, small]

        builder = ContextBuilder(
            memory_manager=_TwoSectionMemory(), context_window_tokens=4096,
        )
        messages = builder.build(SimpleNamespace(summary="", raw_messages=[]))
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        assert "喜欢简洁回复" in joined  # 小段未被首段吞掉
        assert "史" * 100 not in joined  # 超预算段被跳过


class TestIdentityAlwaysInject:
    """记忆系统重构·Step3：identity 受控单值键每轮必注入。

    「我叫什么名字」与事实 name:张三 词面零重叠，dice 门槛会整体漏掉身份；
    身份是客服个性化的最小必需集，跳过相关性门槛。preference 等其余类别
    维持筛选（否则「无关事实不得无条件注入」的既有保证会被破坏）。
    """

    def _ltm(self) -> LongTermMemory:
        ltm = LongTermMemory(user_id="u-identity", memory_dir="")
        ltm.facts = [
            MemoryFact(content="name:张三", category="identity",
                       created_at="", fact_key="identity.name"),
            MemoryFact(content="occupation:健身教练", category="identity",
                       created_at="", fact_key="identity.occupation"),
            MemoryFact(content="preference:喜欢蓝色运动鞋", category="preference",
                       created_at="", fact_key="preference.color"),
        ]
        return ltm

    def test_identity_injected_for_unrelated_query(self):
        selected = self._ltm().select_facts_for_prompt("帮我查一下订单物流")
        keys = {f.fact_key for f in selected}
        assert "identity.name" in keys          # 无关 query 仍注入
        assert "identity.occupation" in keys
        assert "preference.color" not in keys   # 无关偏好被门槛挡下

    def test_identity_injected_when_query_lexically_overlaps_nothing(self):
        """问句与身份事实零词面重叠（原漏注场景）。"""
        selected = self._ltm().select_facts_for_prompt("你还记得我叫什么名字吗")
        assert "identity.name" in [f.fact_key for f in selected]

    def test_preference_still_filtered_by_relevance(self):
        ltm = self._ltm()
        selected = ltm.select_facts_for_prompt("我想买双蓝色运动鞋")
        keys = {f.fact_key for f in selected}
        assert "preference.color" in keys       # 相关偏好正常入选
        # 门槛未被全局关闭：无关 identity 之外的事实仍被筛掉
        assert "identity.name" in keys          # 身份恒在

    def test_non_identity_single_keys_not_always_injected(self):
        ltm = LongTermMemory(user_id="u-pref", memory_dir="")
        ltm.facts = [
            MemoryFact(content="preference:预算两千以内", category="preference",
                       created_at="", fact_key="preference.price_range"),
        ]
        assert ltm.select_facts_for_prompt("帮我查一下订单物流") == []


class TestExcludeKeysAtLtmLevel:
    def test_exclude_keys_at_ltm_level(self):
        ltm = LongTermMemory(user_id="u-conflict", memory_dir="")
        ltm.facts = [
            MemoryFact(content="name:张三", category="identity",
                       created_at="", fact_key="identity.name"),
            MemoryFact(content="preference:喜欢邮件联系", category="preference",
                       created_at="", fact_key="custom.contact"),
        ]
        section = ltm.build_prompt_section("", exclude_keys={"identity.name"})
        assert section is not None
        assert "张三" not in section
        assert "喜欢邮件联系" in section

    def test_exclude_keys_wins_over_identity_always_inject(self):
        """identity 单值键每轮必注入，但 exclude_keys 优先级更高——
        调用方有更权威的实时值时，LTM 旧值绝不同屏。"""
        ltm = LongTermMemory(user_id="u-conflict2", memory_dir="")
        ltm.facts = [
            MemoryFact(content="name:张三", category="identity",
                       created_at="", fact_key="identity.name"),
        ]
        assert ltm.select_facts_for_prompt(
            "我叫什么名字", exclude_keys={"identity.name"},
        ) == []

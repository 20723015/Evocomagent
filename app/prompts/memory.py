"""记忆提取 Prompt：短期记忆 (STM) 和长期记忆 (LTM) 的事实抽取。"""

_MUTATION_RULES = """只记录用户亲口明确表达的信息，禁止根据浏览、咨询、购买或客服回复推断画像。
允许的 fact_key：
- identity.name / identity.membership_level / identity.region
- identity.occupation / identity.address
- preference.color / preference.style / preference.brand / preference.price_range
- preference.category / preference.delivery / preference.size
- behavior.shopping / behavior.payment / issue.current
- 无法归类时使用 custom.<英文snake_case>

operation 规则：
- upsert：设置或替换单值属性；
- add/remove：增删集合属性（preference.brand、preference.category、behavior.shopping、issue.current）；
- delete：用户明确否定或要求忘记某个单值属性；
- 修改已有事实时尽量填写 target_fact_id；新增时留空；
- explicit 只有在用户明确陈述、纠正、否定时才为 true，否则不要输出该 mutation；
- evidence 必须逐字引用触发变更的用户原话，不得引用客服或工具输出；
- confidence 为 0 到 1，低于 0.8 的内容不要输出。

只输出合法 JSON，不要 Markdown：
{{"mutations":[{{"operation":"upsert","fact_key":"preference.color","content":"用户偏好蓝色","category":"preference","confidence":0.95,"target_fact_id":"","explicit":true,"evidence":"我现在喜欢蓝色"}}]}}
没有变更时输出：{{"mutations":[]}}"""


STM_EXTRACTION_PROMPT = """你是会话短期记忆变更提取器。

当前有效事实：
{existing_facts}

""" + _MUTATION_RULES


LTM_EXTRACTION_PROMPT = """你是长期用户画像变更提取器。基于电商客服对话（可能包含摘要），提取值得跨会话保留的明确用户信息。

这些信息将在未来的对话中帮助客服更好地服务该用户。

当前有效长期事实：
{existing_ltm}

""" + _MUTATION_RULES + """

长期记忆输出还必须包含 interaction_summary：
{{"mutations":[],"interaction_summary":"用户本轮咨询内容的一句话摘要"}}
"""

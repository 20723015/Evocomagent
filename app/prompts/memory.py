"""记忆提取 Prompt：长期记忆 (LTM) 的事实抽取。

会话内短期记忆（槽位层）已删除：会话内上下文由对话历史 + rolling 摘要
承担，本模块只服务 LTM。_MUTATION_RULES 为 LTM 提取与巩固 sweep（阶段3）
共用的变更输出协议。
"""

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


LTM_EXTRACTION_PROMPT = """你是长期用户画像变更提取器。基于电商客服对话（可能包含摘要），提取值得跨会话保留的明确用户信息。

这些信息将在未来的对话中帮助客服更好地服务该用户。

当前有效长期事实：
{existing_ltm}

""" + _MUTATION_RULES + """

长期记忆输出还必须包含 interaction_summary：
{{"mutations":[],"interaction_summary":"用户本轮咨询内容的一句话摘要"}}
"""


# 记忆系统重构·阶段3：巩固清理 sweep 专用 prompt（与提取分离：
# 输入是已落库的事实簇，任务是键规范化 + 合并，禁止新增信息）
SWEEP_CONSOLIDATION_PROMPT = """你是长期记忆巩固清理器。给定同一用户的若干簇长期记忆事实（legacy 内部键或语义近重复），为每簇产出规范化合并建议。

规则：
- 只对输入簇内事实做键规范化与合并；禁止引入簇外信息、禁止凭空创造新事实；
- 每簇输出一条建议：fact_key 用受控单值键（identity.name / preference.color 等）或 custom.<英文snake_case>，
  content 为簇内事实的合并表述（保留全部有效信息、去除重复），target_fact_id 填簇内最具代表性的 fact_id；
- 单值语义不得使用集合键（preference.brand / preference.category / behavior.shopping / issue.current）；
- explicit 固定为 true，confidence 不低于 0.8；
- evidence 复用簇内事实已有的用户原话（簇内均无则留空）；
- 无法安全合并/规范化的簇不要输出。

只输出合法 JSON，不要 Markdown：
{{"clusters":[{{"target_fact_id":"fact-id","fact_key":"preference.color","content":"用户喜欢蓝色","category":"preference","confidence":0.9,"explicit":true,"evidence":"我喜欢蓝色"}}]}}
没有可处理簇时输出：{{"clusters":[]}}"""

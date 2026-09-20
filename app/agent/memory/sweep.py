"""LTM 巩固清理 sweep（记忆系统重构·阶段3）。

给 legacy.* 键与跨键语义重复一个可控的收敛机制：
1. 候选簇筛选：active 且（fact_key 为 legacy.* 或与其他事实
   cosine > memory_sweep_similarity）；
2. 一次 LLM 调用产出规范键 + 合并建议（SWEEP_CONSOLIDATION_PROMPT，
   输出协议复用 _MUTATION_RULES 的 JSON 形态）；
3. **一律经 apply_memory_mutations 落库**——校验、supersede 链、
   审计全部继承，不做旁路直写；
4. 单用户单次 sweep 处理 ≤ memory_sweep_max_clusters 簇，失败跳过。

触发（计划 §5）：memory_sweep_enabled（默认关）+ 用户 active 事实数 >
memory_sweep_active_threshold 时，由 memory job worker 在巩固后顺带执行；
观察期可手动 MemoryJobWorker.run_sweep(user_id) 触发。

回滚：开关关闭即回到现状；已合并事实走版本链可审计可人工恢复
（superseded 保留 memory_version_keep_days 天）。
"""

from __future__ import annotations

import json
import math

from app.agent.memory.long_term import _cosine
from app.agent.memory.models import (
    VALID_CATEGORIES,
    MemoryFact,
    MemoryMutation,
    _LEGACY_KEY_RE,
    key_spec,
    mask_sensitive,
)
from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.agent.memory.sweep")


def legacy_candidates(facts: list[MemoryFact]) -> list[MemoryFact]:
    """active 且 fact_key 为 legacy.* 的事实（键规范化对象）。"""
    return [
        f for f in facts
        if f.active and _LEGACY_KEY_RE.fullmatch(f.fact_key or "")
    ]


def duplicate_clusters(
    facts: list[MemoryFact],
    vectors: dict[str, list[float]],
    threshold: float,
) -> list[list[MemoryFact]]:
    """语义近重复簇：active 事实两两 cosine > threshold 的并查集聚类。

    无嵌入的事实不参与（退化为 legacy 清理）；单元素"簇"不返回。
    """
    active = [f for f in facts if f.active and f.fact_id in vectors]
    parent = {f.fact_id: f.fact_id for f in active}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(active)):
        vi = vectors[active[i].fact_id]
        for j in range(i + 1, len(active)):
            vj = vectors[active[j].fact_id]
            if _cosine(vi, vj) > threshold:
                union(active[i].fact_id, active[j].fact_id)

    groups: dict[str, list[MemoryFact]] = {}
    for f in active:
        groups.setdefault(find(f.fact_id), []).append(f)
    return [sorted(g, key=lambda f: f.created_at) for g in groups.values()
            if len(g) >= 2]


def build_candidate_clusters(
    facts: list[MemoryFact],
    vectors: dict[str, list[float]],
    *,
    similarity: float,
    max_clusters: int,
) -> list[list[MemoryFact]]:
    """合并两类候选：legacy 键各自成簇 + 语义近重复簇（去重、限量）。

    语义簇内若含 legacy 事实，该事实并入语义簇（不重复单列）。
    """
    clusters = duplicate_clusters(facts, vectors, similarity)
    clustered_ids = {f.fact_id for c in clusters for f in c}
    for fact in legacy_candidates(facts):
        if fact.fact_id not in clustered_ids:
            clusters.append([fact])
    # legacy 清理优先（确定性最高），近重复簇按簇大小降序
    clusters.sort(key=lambda c: (0 if len(c) == 1 else 1, -len(c)))
    return clusters[: max(max_clusters, 0)]


def _render_clusters(clusters: list[list[MemoryFact]]) -> str:
    parts = []
    for i, cluster in enumerate(clusters, 1):
        lines = "\n".join(
            f"  - fact_id={f.fact_id} key={f.fact_key} [{f.category}] "
            f"content={f.content} evidence={f.evidence or '（无）'}"
            for f in cluster
        )
        parts.append(f"簇{i}：\n{lines}")
    return "\n\n".join(parts)


def propose_sweep_consolidations(
    client, model: str, clusters: list[list[MemoryFact]],
) -> list[MemoryMutation]:
    """一次 LLM 调用产出规范键 + 合并建议 → MemoryMutation 列表。

    每条 LLM 建议（upsert 主条目）由代码侧补全簇成员的 delete/remove——
    模型只决定「规范键 + 合并内容 + 主目标」，收敛哪些由簇结构确定，
    杜绝模型漏删导致的旧版残留。解析失败/注入内容丢弃（与提取同纪律）。
    全部变更经 apply_memory_mutations 的目标守卫与 eligibility 校验，
    提案键纪律见下文注释（成员用自身键收敛，跨键不硬闯守卫）。
    """
    if not clusters:
        return []
    from app.agent.memory.extraction import _content_is_injection
    from app.observability.metrics import record_memory_injection_blocked
    from app.prompts.memory import SWEEP_CONSOLIDATION_PROMPT

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SWEEP_CONSOLIDATION_PROMPT},
            {"role": "user", "content": _render_clusters(clusters)},
        ],
        max_tokens=settings.llm_max_tokens,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    items = data.get("clusters") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    by_id = {f.fact_id: f for c in clusters for f in c}
    mutations: list[MemoryMutation] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_fact_id", "") or "").strip()
        target = by_id.get(target_id)
        if target is None:
            continue  # 目标必须是簇内成员（防幻觉 fact_id 合并他人事实）
        fact_key = str(item.get("fact_key", "") or "").strip().lower()
        content = str(item.get("content", "") or "").strip()
        if _content_is_injection(fact_key, content):
            record_memory_injection_blocked("sweep")
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or confidence < 0.8:
            continue
        # 集合键的 upsert/delete 会被 eligibility 拒绝——规范化为 custom 键
        spec = key_spec(fact_key)
        if spec is not None and spec.cardinality == "set":
            continue
        cluster = next(c for c in clusters if target in c)
        evidence = str(item.get("evidence", "") or "").strip() or target.evidence
        category = str(
            item.get("category", target.category) or "other"
        ).strip().lower()
        # 键纪律（验收 P1 修复）：apply_memory_mutations 的目标守卫要求
        # mutation.fact_key == target.fact_key（仅 legacy 键有迁移豁免）。
        # 规范键与非 legacy 目标键不等时，带 target_fact_id 的 upsert 会被
        # 守卫整条静默拒绝（跨键语义簇合并曾因此 100% 无效）——此时改为
        # 无目标新建，目标成员与簇内其余成员一样由下述删除收敛
        target_key = (target.fact_key or "").strip()
        link_target = target_key == fact_key or target_key.startswith("legacy.")
        mutations.append(MemoryMutation(
            operation="upsert",
            fact_key=fact_key,
            content=mask_sensitive(content),
            category=category,
            confidence=confidence,
            target_fact_id=target_id if link_target else "",
            explicit=True,
            evidence=mask_sensitive(evidence),
        ))
        # 簇成员由代码确定收敛（版本链保留 90 天可恢复）：
        # - target 已建立 supersede 链（link_target）→ 由 upsert 收敛，跳过；
        # - legacy 成员键过不了 key_spec 校验 → 沿用规范键提案，靠目标守卫
        #   的 legacy 迁移豁免删除；
        # - 其余成员用**成员自身键**（守卫要求 mutation 键 == target 键），
        #   category 随成员（受控键有键-类别归属校验），集合键改 remove
        for member in cluster:
            if link_target and member.fact_id == target_id:
                continue
            member_key = (member.fact_key or "").strip()
            if member_key.startswith("legacy."):
                del_op, del_key, del_category, del_content = (
                    "delete", fact_key, category, "",
                )
            else:
                spec = key_spec(member_key)
                if spec is None:
                    continue  # 异常键过不了校验，跳过（同静默丢弃口径）
                del_key = member_key
                del_category = (
                    member.category
                    if member.category in VALID_CATEGORIES
                    else category
                )
                if spec.cardinality == "set":
                    del_op, del_content = "remove", member.content
                else:
                    del_op, del_content = "delete", ""
            mutations.append(MemoryMutation(
                operation=del_op,
                fact_key=del_key,
                content=del_content,
                category=del_category,
                confidence=confidence,
                target_fact_id=member.fact_id,
                explicit=True,
                evidence=mask_sensitive(member.evidence or evidence),
            ))
    return mutations


def run_sweep(
    user_id: str,
    ltm,
    client,
    model: str,
    *,
    embedder=None,
    max_clusters: int | None = None,
    similarity: float | None = None,
) -> dict:
    """执行一次巩固清理 sweep；返回统计（观测/验收用）。

    步骤：候选簇 → LLM 建议 → apply_memory_mutations 落库（继承全部
    校验/supersede 链/审计）→ 新事实补嵌入（embedder 可用时）。
    任何子步骤失败只跳过该步并记日志，不抛出（sweep 是增强任务）。
    """
    from app.agent.memory.embeddings import backfill_embeddings

    max_clusters = (
        settings.memory_sweep_max_clusters if max_clusters is None else max_clusters
    )
    similarity = (
        settings.memory_sweep_similarity if similarity is None else similarity
    )
    stats = {"candidates": 0, "proposed": 0, "applied": 0,
             "legacy_before": 0, "legacy_after": 0}

    try:
        ltm.load()
    except Exception as e:  # noqa: BLE001
        log.info("memory.sweep_load_failed user=%s err=%s",
                 user_id, type(e).__name__)
        return stats

    facts = ltm.facts
    stats["legacy_before"] = len(legacy_candidates(facts))

    # 语义簇用向量：优先嵌入存储，缺失经 embedder 现场补算（预算内）
    vectors: dict[str, list[float]] = {}
    if embedder is not None:
        store = getattr(ltm, "_embedding_store", None)
        backfill_embeddings(user_id, ltm.active_facts, embedder, store,
                            stage="sweep")
        for fact in ltm.active_facts:
            cached = store.get(user_id, fact.fact_id, embedder.model) if store else None
            if cached is not None:
                vectors[fact.fact_id] = cached

    clusters = build_candidate_clusters(
        facts, vectors,
        similarity=similarity, max_clusters=max_clusters,
    )
    stats["candidates"] = len(clusters)
    if not clusters:
        return stats

    try:
        mutations = propose_sweep_consolidations(client, model, clusters)
    except Exception as e:  # noqa: BLE001 —— LLM 失败本轮跳过
        log.info("memory.sweep_llm_failed user=%s err=%s",
                 user_id, type(e).__name__)
        return stats
    stats["proposed"] = len(mutations)
    if not mutations:
        return stats

    before_ids = {f.fact_id for f in ltm.active_facts}
    try:
        from app.agent.memory.models import apply_memory_mutations

        ltm.facts = apply_memory_mutations(
            ltm.facts, mutations,
            max_active=ltm.max_facts, source_session=ltm.source_session,
        )
        ltm.save()
    except Exception as e:  # noqa: BLE001
        log.info("memory.sweep_apply_failed user=%s err=%s",
                 user_id, type(e).__name__)
        return stats

    after = ltm.active_facts
    stats["applied"] = len([f for f in after if f.fact_id not in before_ids])
    stats["legacy_after"] = len(legacy_candidates(ltm.facts))
    log.info(
        "memory.sweep_done user=%s candidates=%d proposed=%d applied=%d "
        "legacy=%d→%d",
        user_id, stats["candidates"], stats["proposed"], stats["applied"],
        stats["legacy_before"], stats["legacy_after"],
    )
    # 新合并事实补嵌入（派生数据，失败不阻塞）
    if embedder is not None:
        store = getattr(ltm, "_embedding_store", None)
        backfill_embeddings(user_id, after, embedder, store, stage="sweep")
    return stats

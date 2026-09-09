"""Structured memory facts and deterministic mutation application."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Iterable, Optional


ACTIVE = "active"
SUPERSEDED = "superseded"
DELETED = "deleted"
VALID_STATUSES = {ACTIVE, SUPERSEDED, DELETED}
VALID_CATEGORIES = {"identity", "preference", "behavior", "issue", "other"}
VALID_OPERATIONS = {"upsert", "delete", "add", "remove"}
MIN_MUTATION_CONFIDENCE = 0.8


@dataclass(frozen=True)
class MemoryKeySpec:
    category: str
    cardinality: str = "single"  # single | set


# Common profile slots are controlled so equivalent facts converge on one key.
# Unknown but valid attributes may use ``custom.<snake_case>``.
MEMORY_KEY_SPECS: dict[str, MemoryKeySpec] = {
    "identity.name": MemoryKeySpec("identity"),
    "identity.membership_level": MemoryKeySpec("identity"),
    "identity.region": MemoryKeySpec("identity"),
    "identity.occupation": MemoryKeySpec("identity"),
    "identity.address": MemoryKeySpec("identity"),
    "preference.color": MemoryKeySpec("preference"),
    "preference.style": MemoryKeySpec("preference"),
    "preference.brand": MemoryKeySpec("preference", "set"),
    "preference.price_range": MemoryKeySpec("preference"),
    "preference.category": MemoryKeySpec("preference", "set"),
    "preference.delivery": MemoryKeySpec("preference"),
    "preference.size": MemoryKeySpec("preference"),
    "behavior.shopping": MemoryKeySpec("behavior", "set"),
    "behavior.payment": MemoryKeySpec("behavior"),
    "issue.current": MemoryKeySpec("issue", "set"),
}

_CUSTOM_KEY_RE = re.compile(r"^custom\.[a-z0-9][a-z0-9_]{0,63}$")
_LEGACY_KEY_RE = re.compile(r"^legacy\.[a-f0-9]{16,64}$")


def key_spec(fact_key: str) -> Optional[MemoryKeySpec]:
    if fact_key in MEMORY_KEY_SPECS:
        return MEMORY_KEY_SPECS[fact_key]
    if _CUSTOM_KEY_RE.fullmatch(fact_key or ""):
        return MemoryKeySpec("other")
    return None


def valid_stored_key(fact_key: str) -> bool:
    """Stored data additionally accepts internal legacy keys."""
    return key_spec(fact_key) is not None or bool(_LEGACY_KEY_RE.fullmatch(fact_key or ""))


def _legacy_digest(*parts: str) -> str:
    raw = "\x1f".join(str(p or "") for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class MemoryFact:
    content: str
    category: str
    created_at: str
    source_session: str = ""
    fact_id: str = ""
    fact_key: str = ""
    status: str = ACTIVE
    confidence: float = 1.0
    updated_at: str = ""
    supersedes_id: str = ""
    evidence: str = ""  # 2.4：用户原话依据（可审计）；旧数据缺省为空串

    def __post_init__(self) -> None:
        digest = _legacy_digest(
            self.category, self.content, self.source_session, self.created_at,
        )
        if not self.fact_id:
            self.fact_id = f"legacy-{digest}"
        if not self.fact_key:
            self.fact_key = f"legacy.{digest}"
        if self.status not in VALID_STATUSES:
            self.status = ACTIVE
        if not self.updated_at:
            self.updated_at = self.created_at
        try:
            self.confidence = float(self.confidence)
        except (TypeError, ValueError):
            self.confidence = 1.0

    @property
    def active(self) -> bool:
        return self.status == ACTIVE

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict) -> "MemoryFact":
        return cls(
            content=str(item.get("content", "")),
            category=str(item.get("category", "other")),
            created_at=str(item.get("created_at", "")),
            source_session=str(item.get("source_session", "")),
            fact_id=str(item.get("fact_id", "")),
            fact_key=str(item.get("fact_key", "")),
            status=str(item.get("status", ACTIVE)),
            confidence=item.get("confidence", 1.0),
            updated_at=str(item.get("updated_at", "")),
            supersedes_id=str(item.get("supersedes_id", "")),
            # 2.4：v2 旧数据无 evidence 字段 → 按空串（无损升级）
            evidence=str(item.get("evidence", "")),
        )


@dataclass
class MemoryMutation:
    operation: str
    fact_key: str
    content: str = ""
    category: str = "other"
    confidence: float = 0.0
    target_fact_id: str = ""
    explicit: bool = False
    evidence: str = ""  # exact user quote; extraction validates but never persists


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _eligible(mutation: MemoryMutation) -> tuple[bool, Optional[MemoryKeySpec]]:
    if mutation.operation not in VALID_OPERATIONS or not mutation.explicit:
        return False, None
    try:
        confidence = float(mutation.confidence)
    except (TypeError, ValueError):
        return False, None
    spec = key_spec(mutation.fact_key)
    if (
        not math.isfinite(confidence)
        or confidence < MIN_MUTATION_CONFIDENCE
        or confidence > 1.0
        or spec is None
    ):
        return False, None
    if mutation.category not in VALID_CATEGORIES:
        return False, None
    # Controlled keys own their category; custom keys use the supplied category.
    if mutation.fact_key in MEMORY_KEY_SPECS and mutation.category != spec.category:
        return False, None
    if mutation.operation in {"upsert", "add"} and not mutation.content.strip():
        return False, None
    if spec.cardinality == "single" and mutation.operation in {"add", "remove"}:
        return False, None
    if spec.cardinality == "set" and mutation.operation in {"upsert", "delete"}:
        return False, None
    if mutation.operation == "add" and mutation.target_fact_id:
        return False, None
    return True, spec


def apply_memory_mutations(
    facts: Iterable[MemoryFact],
    mutations: Iterable[MemoryMutation],
    *,
    max_active: int,
    now: Optional[str] = None,
    source_session: str = "",
) -> list[MemoryFact]:
    """Apply validated mutations while retaining superseded/deleted versions."""
    records = [replace(f) for f in facts]
    timestamp = now or _now()

    for mutation in mutations:
        allowed, spec = _eligible(mutation)
        if not allowed or spec is None:
            continue

        active = [f for f in records if f.active]
        target = next(
            (f for f in active if mutation.target_fact_id and f.fact_id == mutation.target_fact_id),
            None,
        )
        if mutation.target_fact_id and target is None:
            continue
        if (
            target is not None
            and target.fact_key != mutation.fact_key
            and not target.fact_key.startswith("legacy.")
        ):
            continue

        same_key = [f for f in active if f.fact_key == mutation.fact_key]
        if target is not None and target not in same_key:
            same_key.append(target)  # legacy fact can migrate to a canonical key

        if mutation.operation in {"delete", "remove"}:
            candidates = [target] if target is not None else same_key
            if spec.cardinality == "set" and mutation.content.strip() and target is None:
                wanted = mutation.content.strip().casefold()
                candidates = [f for f in same_key if f.content.strip().casefold() == wanted]
            for fact in candidates:
                fact.status = DELETED
                fact.updated_at = timestamp
            continue

        content = mutation.content.strip()
        if spec.cardinality == "set":
            if any(f.content.strip().casefold() == content.casefold() for f in same_key):
                continue
            replaced: list[MemoryFact] = [target] if target is not None else []
        else:
            if len(same_key) == 1 and same_key[0].content.strip().casefold() == content.casefold():
                continue
            replaced = same_key

        for fact in replaced:
            fact.status = SUPERSEDED
            fact.updated_at = timestamp

        records.append(MemoryFact(
            content=content,
            category=mutation.category,
            created_at=timestamp,
            source_session=source_session,
            fact_id=uuid.uuid4().hex,
            fact_key=mutation.fact_key,
            status=ACTIVE,
            confidence=float(mutation.confidence),
            updated_at=timestamp,
            supersedes_id=target.fact_id if target is not None else (
                replaced[0].fact_id if replaced else ""
            ),
            # 2.4：已通过校验的 evidence（用户原话）随新事实持久化；
            # 旧事实 superseded/delete 时保留原内容/原 evidence（版本链可审计）
            evidence=mutation.evidence or "",
        ))

    active = [f for f in records if f.active]
    overflow = max(0, len(active) - max(max_active, 0))
    if overflow:
        oldest = sorted(
            enumerate(active),
            key=lambda item: (item[1].updated_at or item[1].created_at, item[0]),
        )
        for _, fact in oldest[:overflow]:
            fact.status = DELETED
            fact.updated_at = timestamp
    return records

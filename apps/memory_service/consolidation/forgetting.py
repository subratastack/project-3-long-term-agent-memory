"""Auditable lifecycle transitions and retrieval-priority policy.

Content, confidence, validity and provenance are never rewritten. Lifecycle
metadata lives under ``forgetting``; changes are committed by the caller's
UnitOfWork. Tombstones permanently deny reuse of their evidence event IDs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import Float, case, cast, func
from sqlalchemy.sql.elements import ColumnElement

from apps.memory_service.consolidation.clusterer import utc
from apps.memory_service.domain.enums import IndexStatus, MemoryStatus, MemoryType
from apps.memory_service.domain.models import MemoryRecord

if TYPE_CHECKING:
    from apps.memory_service.persistence.repositories import MemoryRecordRepository
    from apps.memory_service.persistence.unit_of_work import UnitOfWork

POLICY_VERSION = "forgetting-v1"
DECAY_GRACE_DAYS = 30
DECAY_HALF_LIFE_DAYS = 90
LOW_VALUE_CONFIDENCE = 0.8
MIN_PRIORITY = 0.05
COMPACTED_PRIORITY = 0.25


def retrieval_priority(record: MemoryRecord) -> float:
    """Return a bounded stored multiplier; absent/invalid metadata means 1."""
    state = record.metadata.get("forgetting", {})
    value = state.get("priority", 1.0) if isinstance(state, dict) else 1.0
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        return 1.0
    return max(MIN_PRIORITY, min(1.0, float(value)))


def priority_expression() -> ColumnElement[float]:
    """The same multiplier in SQL, so decay applies before search LIMIT."""
    from apps.memory_service.persistence.models import MemoryRecordRow

    value = MemoryRecordRow.metadata_["forgetting"]["priority"]
    number = case((func.jsonb_typeof(value) == "number", cast(value.astext, Float)), else_=1.0)
    return func.greatest(MIN_PRIORITY, func.least(1.0, number))


def decay_priority(record: MemoryRecord, now: datetime) -> float:
    """Daily exponential decay for low-confidence episodes after a 30-day grace.

    Confidence is only the initial low-value proxy. Semantic/procedural records
    and episodes with confidence >= 0.8 receive no age penalty. Ninety days
    beyond the grace halves priority; repeated application never compounds it.
    """
    if record.memory_type != MemoryType.EPISODIC or record.confidence >= LOW_VALUE_CONFIDENCE:
        return 1.0
    observed = max(utc(p.observed_at) for p in record.provenance)
    age_days = max(0, (utc(now) - observed).days - DECAY_GRACE_DAYS)
    return max(MIN_PRIORITY, math.exp2(-age_days / DECAY_HALF_LIFE_DAYS))


def _changed(record: MemoryRecord, now: datetime, **changes: Any) -> MemoryRecord:
    return record.model_copy(deep=True, update={"updated_at": utc(now), **changes})


def _metadata(record: MemoryRecord, **fields: Any) -> dict[str, Any]:
    old = record.metadata.get("forgetting", {})
    state = dict(old) if isinstance(old, dict) else {}
    return {**record.metadata, "forgetting": {**state, **fields}}


def expire_record(record: MemoryRecord, now: datetime) -> MemoryRecord | None:
    """Expire ACTIVE records strictly after valid_to; retain historical validity.

    Quarantine and supersession are preserved: changing either to EXPIRED
    would accidentally widen historical access. Tombstones are terminal.
    """
    end = record.temporal_validity.valid_to
    if record.status != MemoryStatus.ACTIVE or end is None or utc(end) >= utc(now):
        return None
    return _changed(
        record,
        now,
        status=MemoryStatus.EXPIRED,
        metadata=_metadata(
            record,
            expiry={
                "at": utc(now).isoformat(),
                "policy": POLICY_VERSION,
                "reason": "valid_to is past",
                "requested_by": "policy:expiry",
            },
        ),
    )


def decay_record(record: MemoryRecord, now: datetime) -> MemoryRecord | None:
    if record.status != MemoryStatus.ACTIVE or utc(record.temporal_validity.valid_from) > utc(now):
        return None
    priority = min(retrieval_priority(record), decay_priority(record, now))
    if priority == retrieval_priority(record):
        return None
    return _changed(
        record,
        now,
        metadata=_metadata(
            record, priority=priority, decay={"at": utc(now).isoformat(), "policy": POLICY_VERSION}
        ),
    )


def tombstone_record(
    record: MemoryRecord,
    *,
    reason: str,
    requested_by: str,
    now: datetime,
    policy: str = POLICY_VERSION,
    root_memory_id: UUID | None = None,
) -> MemoryRecord | None:
    """A terminal, idempotent soft deletion, preserving its first audit record."""
    if not reason.strip() or not requested_by.strip() or not policy.strip():
        raise ValueError("reason, requested_by and policy must be nonempty")
    if record.status == MemoryStatus.TOMBSTONE:
        return None
    return _changed(
        record,
        now,
        status=MemoryStatus.TOMBSTONE,
        embedding=None,
        embedding_model_version=None,
        index_status=IndexStatus.PENDING,
        metadata=_metadata(
            record,
            tombstone={
                "reason": reason,
                "requested_by": requested_by,
                "policy": policy,
                "at": utc(now).isoformat(),
                "root_memory_id": str(root_memory_id or record.memory_id),
            },
        ),
    )


def compact_record(
    episode: MemoryRecord,
    summary: MemoryRecord,
    now: datetime,
) -> MemoryRecord | None:
    """Lower priority only for explicit sources of a currently usable summary."""
    if (
        episode.tenant_id != summary.tenant_id
        or episode.status != MemoryStatus.ACTIVE
        or episode.memory_type != MemoryType.EPISODIC
        or summary.status != MemoryStatus.ACTIVE
        or summary.memory_type not in (MemoryType.SEMANTIC, MemoryType.PROCEDURAL)
        or summary.metadata.get("consolidation_version") != "1"
        or str(episode.memory_id) not in summary.metadata.get("supporting_memory_ids", [])
        or utc(summary.temporal_validity.valid_from) > utc(now)
        or (
            summary.temporal_validity.valid_to is not None
            and utc(summary.temporal_validity.valid_to) < utc(now)
        )
        or not {p.event_id for p in episode.provenance}.issubset(
            {p.event_id for p in summary.provenance}
        )
    ):
        return None
    state = episode.metadata.get("forgetting", {})
    prior = state.get("compaction", {}) if isinstance(state, dict) else {}
    linked = prior.get("summary_ids", []) if isinstance(prior, dict) else []
    summaries = (
        {value for value in linked if isinstance(value, str)} if isinstance(linked, list) else set()
    )
    if str(summary.memory_id) in summaries:
        return None
    summaries.add(str(summary.memory_id))
    return _changed(
        episode,
        now,
        metadata=_metadata(
            episode,
            priority=min(retrieval_priority(episode), COMPACTED_PRIORITY),
            compaction={
                "at": utc(now).isoformat(),
                "policy": POLICY_VERSION,
                "summary_ids": sorted(summaries),
            },
        ),
    )


def compact_consolidated(uow: UnitOfWork, summary: MemoryRecord, now: datetime) -> list[UUID]:
    """Compact sources inside the transaction that persisted an accepted summary."""
    changed: list[UUID] = []
    source_ids = summary.metadata.get("supporting_memory_ids", [])
    if not isinstance(source_ids, list):
        return changed
    for source_id in source_ids:
        try:
            memory_id = UUID(str(source_id))
        except ValueError:
            continue
        episode = uow.get_memory(memory_id, summary.tenant_id)
        if episode is not None and (updated := compact_record(episode, summary, now)) is not None:
            uow.records.save_lifecycle(updated)
            changed.append(episode.memory_id)
    return changed


def maintain_memories(
    uow_factory: Callable[[], UnitOfWork],
    tenant_id: UUID,
    *,
    now: datetime | None = None,
) -> dict[str, list[UUID]]:
    """Idempotent tenant maintenance. One transaction, no automatic tombstoning."""
    now = utc(now or datetime.now(UTC))
    report: dict[str, list[UUID]] = {"expired": [], "decayed": [], "compacted": []}
    with uow_factory() as uow:
        uow.records.lock_lifecycle(tenant_id)
        records = uow.records.list_for_maintenance(tenant_id)
        for record in records:
            expired = expire_record(record, now)
            if expired is not None:
                uow.records.save_lifecycle(expired)
                report["expired"].append(record.memory_id)
            elif (decayed := decay_record(record, now)) is not None:
                uow.records.save_lifecycle(decayed)
                report["decayed"].append(record.memory_id)
        # Reload after expiry so an expired summary cannot compact its sources.
        for summary in uow.records.list_for_maintenance(tenant_id):
            if summary.status == MemoryStatus.ACTIVE and summary.metadata.get(
                "consolidation_version"
            ):
                report["compacted"].extend(compact_consolidated(uow, summary, now))
        uow.commit()
    return {key: sorted(set(ids), key=str) for key, ids in report.items()}


def tombstone_memory(
    uow_factory: Callable[[], UnitOfWork],
    tenant_id: UUID,
    memory_id: UUID,
    *,
    reason: str,
    requested_by: str,
    policy: str = POLICY_VERSION,
    now: datetime | None = None,
) -> list[UUID]:
    """Tombstone a record and its evidence-connected copies/derivatives atomically.

    Conservative closure over shared event IDs also covers source episodes when
    deleting a summary. All records remain available through audit repository
    reads, but neither old nor new records may reuse the tombstoned evidence.
    """
    now = utc(now or datetime.now(UTC))
    with uow_factory() as uow:
        uow.records.lock_lifecycle(tenant_id)
        changed = tombstone_records(
            uow.records,
            tenant_id,
            memory_id,
            reason=reason,
            requested_by=requested_by,
            policy=policy,
            now=now,
        )
        uow.commit()
    return sorted(changed, key=str)


def tombstone_records(
    repository: MemoryRecordRepository,
    tenant_id: UUID,
    memory_id: UUID,
    *,
    reason: str,
    requested_by: str,
    now: datetime,
    policy: str = POLICY_VERSION,
) -> list[UUID]:
    """Apply the evidence closure using an already locked repository transaction."""
    records = repository.list_for_maintenance(tenant_id)
    root = next((m for m in records if m.memory_id == memory_id), None)
    if root is None:
        raise LookupError(f"memory {memory_id} not found for tenant {tenant_id}")
    # Validate even when a repeat request has no work to do.
    tombstone_record(root, reason=reason, requested_by=requested_by, policy=policy, now=now)
    selected = {memory_id}
    events = {p.event_id for p in root.provenance}
    while True:
        additions = [
            m
            for m in records
            if m.memory_id not in selected and events.intersection(p.event_id for p in m.provenance)
        ]
        if not additions:
            break
        for record in additions:
            selected.add(record.memory_id)
            events.update(p.event_id for p in record.provenance)
    changed: list[UUID] = []
    for record in records:
        if record.memory_id not in selected:
            continue
        updated = tombstone_record(
            record,
            reason=reason,
            requested_by=requested_by,
            policy=policy,
            now=now,
            root_memory_id=memory_id,
        )
        if updated is not None:
            repository.save_lifecycle(updated)
            changed.append(record.memory_id)
    return sorted(changed, key=str)

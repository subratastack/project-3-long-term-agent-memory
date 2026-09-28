"""Deterministic, tenant-scoped episode grouping without embeddings.

Episodes use ``metadata['incident_key']`` or ``metadata['category_key']``.
A cluster must retain a common subject and span at most the configured window;
a chain of nearby episodes cannot stretch that window indefinitely.
Naive timestamps from older records are interpreted as UTC.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from apps.memory_service.domain.enums import MemoryStatus, MemoryType
from apps.memory_service.domain.models import MemoryRecord


@dataclass(frozen=True)
class MemoryCluster:
    cluster_id: UUID
    tenant_id: UUID
    memories: list[MemoryRecord]
    shared_subject_keys: list[str]
    earliest_observed_at: datetime
    latest_observed_at: datetime


def utc(value: datetime) -> datetime:
    """Normalize legacy naive timestamps for deterministic comparisons."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def category_key(memory: MemoryRecord) -> tuple[str, str] | None:
    """Prefer an incident key; never silently combine incident and category namespaces."""
    for name in ("incident_key", "category_key"):
        value = memory.metadata.get(name)
        if isinstance(value, str) and value.strip():
            return name, value.strip()
    return None


def eligible(memory: MemoryRecord, now: datetime) -> bool:
    validity = memory.temporal_validity
    return (
        memory.memory_type == MemoryType.EPISODIC
        and memory.status == MemoryStatus.ACTIVE
        and utc(validity.valid_from) <= utc(now)
        and (validity.valid_to is None or utc(now) < utc(validity.valid_to))
        and all(utc(p.observed_at) <= utc(now) for p in memory.provenance)
    )


def cluster_memories(
    memories: Iterable[MemoryRecord],
    *,
    now: datetime | None = None,
    window: timedelta = timedelta(days=30),
) -> list[MemoryCluster]:
    """Partition eligible records once each, in observation-time/ID order.

    Multi-subject records join the first compatible cluster and narrow its
    common subjects. Stable membership produces a stable cluster UUID.
    """
    if window <= timedelta(0):
        raise ValueError("window must be positive")
    now = now or datetime.now(UTC)
    unique: dict[UUID, MemoryRecord] = {}
    for memory in memories:
        if memory.memory_id in unique and unique[memory.memory_id] != memory:
            raise ValueError("Conflicting records for the same memory ID")
        unique[memory.memory_id] = memory
    clusters: list[MemoryCluster] = []
    for memory in sorted(
        unique.values(),
        key=lambda m: (min(utc(p.observed_at) for p in m.provenance), str(m.memory_id)),
    ):
        key = category_key(memory)
        subjects = set(filter(None, memory.subject_keys))
        if not eligible(memory, now) or key is None or not subjects:
            continue
        start = min(utc(p.observed_at) for p in memory.provenance)
        end = max(utc(p.observed_at) for p in memory.provenance)
        if end - start > window:
            continue
        match = next(
            (
                i
                for i, c in enumerate(clusters)
                if c.tenant_id == memory.tenant_id
                and category_key(c.memories[0]) == key
                and subjects.intersection(c.shared_subject_keys)
                and max(end, c.latest_observed_at) - c.earliest_observed_at <= window
            ),
            None,
        )
        members = [memory] if match is None else [*clusters[match].memories, memory]
        shared = (
            subjects
            if match is None
            else subjects.intersection(clusters[match].shared_subject_keys)
        )
        identity = f"consolidation:v1:{memory.tenant_id}:" + ":".join(
            sorted(str(m.memory_id) for m in members)
        )
        cluster = MemoryCluster(
            uuid5(NAMESPACE_URL, identity),
            memory.tenant_id,
            members,
            sorted(shared),
            min(utc(p.observed_at) for m in members for p in m.provenance),
            max(utc(p.observed_at) for m in members for p in m.provenance),
        )
        if match is None:
            clusters.append(cluster)
        else:
            clusters[match] = cluster
    return clusters

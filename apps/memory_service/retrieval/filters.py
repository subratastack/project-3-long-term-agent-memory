"""Deterministic translation of a `RetrievalQuery` into hard retrieval filters.

Like `apps.memory_service.ingestion.write_policy`, this module is pure and
rule-based: it never touches a database or a model, and the same query
always produces the same `RetrievalFilters`. Every filter it computes is a
*hard* constraint (ADR-003: "Apply tenant, status, validity, type, and trust
filters as hard constraints") -- `apps.memory_service.persistence.vector_repository`
applies these directly in SQL, and `apps.memory_service.retrieval.semantic`
re-checks them again after fetching authoritative records, so ranking scores
can never substitute for satisfying them.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from uuid import UUID

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryRecord
from apps.memory_service.ingestion.provenance import trust_rank
from apps.memory_service.retrieval.query_model import RetrievalQuery

# An ordinary ("what is true now") query only ever considers ACTIVE memories
# -- a quarantined, superseded, expired, or tombstoned memory is never a
# valid current retrieval result (ARCHITECTURE.md: "a memory that is not
# ACTIVE is invisible to ordinary reads, no matter how relevant its content
# might otherwise look").
ALLOWED_STATUSES: frozenset[MemoryStatus] = frozenset({MemoryStatus.ACTIVE})

# A query with an explicit `as_of` asks "what was true back then", so a memory
# that has *since* been superseded or expired is still a candidate -- its
# validity window, not its present-day status, decides whether it applied at
# that moment (same rule as `MemoryRecordRepository.list_effective`).
# QUARANTINED and TOMBSTONE are never included: those were never trusted, or
# were deliberately forgotten, at any point in time.
HISTORICAL_STATUSES: frozenset[MemoryStatus] = frozenset(
    {MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED, MemoryStatus.EXPIRED}
)


@dataclass(frozen=True)
class RetrievalFilters:
    """The hard constraints one `RetrievalQuery` resolves to.

    `effective_at` is always a concrete timestamp (never `None`) so that
    every filter consumer -- the SQL in `vector_repository` and the
    revalidation in `semantic.py` -- can apply the exact same validity check
    without each having to resolve "as_of or now" independently.
    """

    tenant_id: UUID
    allowed_statuses: frozenset[MemoryStatus]
    allowed_trust_levels: frozenset[TrustLevel]
    memory_types: tuple[MemoryType, ...] | None
    effective_at: datetime.datetime


def resolve_filters(query: RetrievalQuery) -> RetrievalFilters:
    """Resolve `query` into the hard filters retrieval must apply.

    How it works:
        1. `effective_at` is `query.as_of` if given, otherwise the current
           time -- this is the single timestamp every validity check below
           is measured against.
        2. `allowed_trust_levels` is every `TrustLevel` at or above
           `query.min_trust` (via `trust_rank`), or every level at all if no
           floor was requested.
        3. `memory_types` is `query.memory_types` as a tuple, or `None` to
           mean "no type restriction" (kept distinct from an empty tuple,
           which would match nothing).
        4. `allowed_statuses` is `ALLOWED_STATUSES` (ACTIVE only) for an
           ordinary query, or `HISTORICAL_STATUSES` when `query.as_of` is
           given, so a fact that has since been superseded can still be
           found for a time when it was true. Nothing else in a
           `RetrievalQuery` can widen it, and neither set ever includes
           QUARANTINED or TOMBSTONE.

    Example:
        Input:
            query = RetrievalQuery(tenant_id=UUID("1111...1111"),
                                    query_text="pool exhaustion",
                                    min_trust=TrustLevel.MEDIUM)
        Output:
            RetrievalFilters(
                tenant_id=UUID("1111...1111"),
                allowed_statuses=frozenset({MemoryStatus.ACTIVE}),
                allowed_trust_levels=frozenset(
                    {TrustLevel.MEDIUM, TrustLevel.HIGH, TrustLevel.SYSTEM}
                ),
                memory_types=None,
                effective_at=<now>,
            )
    """
    effective_at = query.as_of or datetime.datetime.now(datetime.UTC)
    allowed_trust_levels = (
        _trust_levels_at_or_above(query.min_trust)
        if query.min_trust is not None
        else frozenset(TrustLevel)
    )
    memory_types = tuple(query.memory_types) if query.memory_types is not None else None
    return RetrievalFilters(
        tenant_id=query.tenant_id,
        allowed_statuses=HISTORICAL_STATUSES if query.as_of is not None else ALLOWED_STATUSES,
        allowed_trust_levels=allowed_trust_levels,
        memory_types=memory_types,
        effective_at=effective_at,
    )


def _trust_levels_at_or_above(min_trust: TrustLevel) -> frozenset[TrustLevel]:
    threshold = trust_rank(min_trust)
    return frozenset(level for level in TrustLevel if trust_rank(level) >= threshold)


def record_matches_filters(record: MemoryRecord, filters: RetrievalFilters) -> bool:
    """Re-check the hard filters against a freshly-fetched `MemoryRecord`.

    Both `retrieval.semantic` and `retrieval.lexical` call this on every
    record they re-fetch from PostgreSQL after their respective similarity
    search (ADR-004: "Revalidate each selected ID against PostgreSQL before
    context construction") -- a search result whose underlying row no longer
    satisfies these filters (e.g. it was quarantined between the two queries
    within the same transaction) is dropped rather than trusted.

    Example:
        Input:
            record = MemoryRecord(status=MemoryStatus.QUARANTINED, ...)
            filters = RetrievalFilters(allowed_statuses=frozenset({MemoryStatus.ACTIVE}), ...)
        Output:
            False
    """
    if record.status not in filters.allowed_statuses:
        return False
    if record.trust_level not in filters.allowed_trust_levels:
        return False
    if filters.memory_types is not None and record.memory_type not in filters.memory_types:
        return False
    validity = record.temporal_validity
    if validity.valid_from > filters.effective_at:
        return False
    return not (validity.valid_to is not None and validity.valid_to < filters.effective_at)

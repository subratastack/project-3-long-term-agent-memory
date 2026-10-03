"""Temporal and conflict resolution: the last gate before a memory is shown.

```text
reranked candidates                       (retrieval.hybrid / retrieval.reranker)
  -> drop anything not eligible           (tenant, status, trust, type)
  -> drop anything not in effect at `effective_at`   (valid_from <= t < valid_to)
  -> drop anything superseded by a memory that *is* in effect at t
  -> settle recorded contradictions       (consolidation.conflict_resolver)
  -> the active truth for the requested time, still in relevance order
```

Relevance ranking decides *order*; this stage decides *truth*. It runs after
reranking, over the whole bounded candidate pool (before truncation to
`query.limit`), and it never looks at a score: a stale or contradicted fact
is removed no matter how well it matched the query. That is ADR-004's
"Stale facts cannot win solely through embedding similarity", applied to
every score in the pipeline, not just the embedding one.

Why this is needed on top of `retrieval.filters`:

- **Back-to-back windows.** Supersession closes the old memory's window at
  the new one's `valid_from` (`UnitOfWork.supersede_memory`), so the two
  windows share one instant, and the SQL filters treat `valid_to` as
  inclusive. This stage treats windows as half-open (`t < valid_to`), so
  exactly one of the two is in effect at every instant.
- **Recorded supersession.** A `SUPERSESSION` edge whose newer side is in
  effect at `t` retires the older side even if the older row's own status
  or window was never updated.
- **Contradictions.** A `CONTRADICTION` edge between two memories that are
  both in effect at `t` is settled by `resolve_contradiction` (higher trust
  wins; equal trust withholds both). The other side is looked up in
  PostgreSQL even if retrieval never returned it: whether a fact is
  contested must not depend on whether the query happened to match the
  fact it is contested by.

Everything removed is recorded in the returned `TemporalReport` (with a
reason), and every contradiction considered is surfaced there too --
unresolved ones are withheld from the hits, not hidden from the caller.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from apps.memory_service.consolidation.conflict_resolver import (
    ConflictOutcome,
    ConflictState,
    resolve_contradiction,
)
from apps.memory_service.domain.enums import ConflictType
from apps.memory_service.domain.models import MemoryRecord, MemoryRelation
from apps.memory_service.retrieval.filters import RetrievalFilters, resolve_filters
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.security.tenant_scope import TenantScope

if TYPE_CHECKING:
    from apps.memory_service.persistence.unit_of_work import UnitOfWork


class _HasMemory(Protocol):
    @property
    def memory(self) -> MemoryRecord: ...


class ExclusionReason(StrEnum):
    """Why this stage removed a candidate.

    - NOT_ELIGIBLE: wrong tenant, or a status / trust / type the query
      does not allow (e.g. tombstoned or quarantined).
    - NOT_IN_EFFECT: its validity window does not cover `effective_at`
      (not yet valid, or expired).
    - SUPERSEDED: a newer memory that replaced it is in effect.
    - LOST_CONFLICT: it contradicts a more trusted memory that is in effect.
    - UNRESOLVED_CONFLICT: it contradicts an equally trusted memory that is
      in effect; both are withheld.
    """

    NOT_ELIGIBLE = "not_eligible"
    NOT_IN_EFFECT = "not_in_effect"
    SUPERSEDED = "superseded"
    LOST_CONFLICT = "lost_conflict"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


@dataclass(frozen=True)
class Exclusion:
    """One candidate this stage removed, and why.

    `related_memory_id` is the memory responsible, where there is one: the
    successor for SUPERSEDED, the other side for a conflict.
    """

    memory_id: UUID
    reason: ExclusionReason
    related_memory_id: UUID | None = None


@dataclass(frozen=True)
class TemporalReport:
    """What temporal/conflict resolution did for one query."""

    effective_at: datetime.datetime
    candidates_in: int
    excluded: tuple[Exclusion, ...]
    conflicts: tuple[ConflictOutcome, ...]

    @property
    def unresolved_conflicts(self) -> tuple[ConflictOutcome, ...]:
        return tuple(c for c in self.conflicts if c.state is ConflictState.UNRESOLVED)


def apply_temporal_resolution[HitT: _HasMemory](
    uow: UnitOfWork, query: RetrievalQuery, hits: Sequence[HitT]
) -> tuple[list[HitT], TemporalReport]:
    """Load the relations `hits` depend on, then keep only the active truth.

    How it works:
        1. Resolve the query's filters (the same `effective_at`, tenant,
           and allowed statuses/trust/types retrieval used).
        2. Load every relation touching a candidate, in one query (keeping
           only the query tenant's relations, `TenantScope.filter_relations`).
        3. Load every relation touching the *other* side of those edges,
           in one more query -- needed to tell whether that other memory is
           itself superseded (a contradiction with a retired fact is no
           longer a contradiction).
        4. Fetch any memory those edges point at that isn't already a
           candidate, scoped to the query's tenant.
        5. Hand everything to `select_in_effect`, which makes every decision
           without touching the database.

    Example:
        Input:
            query = RetrievalQuery(tenant_id=T, query_text="request timeout",
                                    as_of=datetime(2026, 4, 1, tzinfo=UTC))
            hits = [<"timeout is 5 seconds", valid Jan 1 -> Mar 1>,
                    <"timeout is 2 seconds", valid Mar 1 -> open>]
        Output:
            ([<"timeout is 2 seconds">],
             TemporalReport(effective_at=2026-04-01, candidates_in=2,
                            excluded=(Exclusion(<5s id>, NOT_IN_EFFECT),),
                            conflicts=()))
    """
    filters = resolve_filters(query)
    scope = TenantScope(filters.tenant_id)
    records: dict[UUID, MemoryRecord] = {hit.memory.memory_id: hit.memory for hit in hits}

    relations = _unique(
        scope.filter_relations(uow.relations.list_for_memories(filters.tenant_id, list(records)))
    )
    neighbour_ids = _endpoints(relations) - records.keys()
    relations = _unique(
        [
            *relations,
            *scope.filter_relations(
                uow.relations.list_for_memories(filters.tenant_id, neighbour_ids)
            ),
        ]
    )

    for memory_id in _endpoints(relations) - records.keys():
        record = uow.records.get(filters.tenant_id, memory_id)
        if record is not None:
            records[memory_id] = record

    return select_in_effect(hits, filters=filters, relations=relations, records=records)


def select_in_effect[HitT: _HasMemory](
    hits: Sequence[HitT],
    *,
    filters: RetrievalFilters,
    relations: Iterable[MemoryRelation],
    records: Mapping[UUID, MemoryRecord],
) -> tuple[list[HitT], TemporalReport]:
    """Keep the hits that are the active truth at `filters.effective_at`.

    Pure: `relations` and `records` must already contain every edge and
    memory the decision depends on (`apply_temporal_resolution` loads them).
    A memory referenced by an edge but missing from `records` is treated as
    not in effect.

    How it works:
        1. Eligibility: drop a hit whose tenant, status, trust level, or
           memory type the filters don't allow (NOT_ELIGIBLE), then one
           whose half-open window `[valid_from, valid_to)` doesn't cover
           `effective_at` (NOT_IN_EFFECT).
        2. Supersession: a memory is superseded at `effective_at` if some
           SUPERSESSION edge points at it from a memory that is in effect
           then. Superseded hits are dropped (SUPERSEDED).
        3. Contradictions: for each CONTRADICTION edge touching a remaining
           hit, if *both* sides are in effect and not superseded, settle it
           with `resolve_contradiction` and drop what it withholds
           (LOST_CONFLICT / UNRESOLVED_CONFLICT). Every such outcome is
           reported, including resolved ones.
        4. Return the surviving hits in their original (relevance) order.

    "In effect" for the *other* side of an edge means right tenant, an
    allowed status, and a window covering `effective_at`; its trust level
    and type don't matter -- a query's trust floor or type filter narrows
    what it is shown, not which facts are true.

    Example:
        Input:
            hits = [<A "lives in Berlin", MEDIUM>, <C "likes tea">]
            relations = [CONTRADICTION A <-> B]
            records = {A: ..., B: <"lives in Paris", MEDIUM, in effect>, C: ...}
        Output:
            ([<C>],
             TemporalReport(..., excluded=(Exclusion(A, UNRESOLVED_CONFLICT, B),),
                            conflicts=(ConflictOutcome(state=UNRESOLVED, ...),)))
    """
    relations = list(relations)
    effective_at = filters.effective_at
    excluded: list[Exclusion] = []

    eligible: list[HitT] = []
    for hit in hits:
        reason = _ineligibility(hit.memory, filters)
        if reason is None:
            eligible.append(hit)
        else:
            excluded.append(Exclusion(hit.memory.memory_id, reason))

    def in_effect(memory_id: UUID) -> bool:
        record = records.get(memory_id)
        return record is not None and _in_effect(record, filters)

    superseded_by: dict[UUID, UUID] = {}
    for relation in relations:
        if relation.relation_type is ConflictType.SUPERSESSION and in_effect(
            relation.source_memory_id
        ):
            superseded_by.setdefault(relation.target_memory_id, relation.source_memory_id)

    def current(memory_id: UUID) -> bool:
        return in_effect(memory_id) and memory_id not in superseded_by

    withheld: dict[UUID, Exclusion] = {}
    for hit in eligible:
        memory_id = hit.memory.memory_id
        if memory_id in superseded_by:
            withheld[memory_id] = Exclusion(
                memory_id, ExclusionReason.SUPERSEDED, superseded_by[memory_id]
            )

    candidate_ids = {hit.memory.memory_id for hit in eligible} - withheld.keys()
    conflicts: list[ConflictOutcome] = []
    for relation in relations:
        if relation.relation_type is not ConflictType.CONTRADICTION:
            continue
        source_id, target_id = relation.source_memory_id, relation.target_memory_id
        if not {source_id, target_id} & candidate_ids:
            continue
        if not (current(source_id) and current(target_id)):
            continue
        outcome = resolve_contradiction(relation, records[source_id], records[target_id])
        conflicts.append(outcome)
        reason = (
            ExclusionReason.UNRESOLVED_CONFLICT
            if outcome.state is ConflictState.UNRESOLVED
            else ExclusionReason.LOST_CONFLICT
        )
        for memory_id in outcome.withheld_ids & candidate_ids:
            other_id = target_id if memory_id == source_id else source_id
            withheld.setdefault(memory_id, Exclusion(memory_id, reason, other_id))

    kept = [hit for hit in eligible if hit.memory.memory_id not in withheld]
    excluded.extend(
        withheld[hit.memory.memory_id] for hit in eligible if hit.memory.memory_id in withheld
    )
    report = TemporalReport(
        effective_at=effective_at,
        candidates_in=len(hits),
        excluded=tuple(excluded),
        conflicts=tuple(conflicts),
    )
    return kept, report


def _ineligibility(record: MemoryRecord, filters: RetrievalFilters) -> ExclusionReason | None:
    if (
        record.tenant_id != filters.tenant_id
        or record.status not in filters.allowed_statuses
        or record.trust_level not in filters.allowed_trust_levels
        or (filters.memory_types is not None and record.memory_type not in filters.memory_types)
    ):
        return ExclusionReason.NOT_ELIGIBLE
    if not _window_covers(record, filters.effective_at):
        return ExclusionReason.NOT_IN_EFFECT
    return None


def _in_effect(record: MemoryRecord, filters: RetrievalFilters) -> bool:
    return (
        record.tenant_id == filters.tenant_id
        and record.status in filters.allowed_statuses
        and _window_covers(record, filters.effective_at)
    )


def _window_covers(record: MemoryRecord, at: datetime.datetime) -> bool:
    """Half-open `[valid_from, valid_to)`: back-to-back windows never share an instant."""
    validity = record.temporal_validity
    return validity.valid_from <= at and (validity.valid_to is None or at < validity.valid_to)


def _endpoints(relations: Iterable[MemoryRelation]) -> set[UUID]:
    return {
        memory_id
        for relation in relations
        for memory_id in (relation.source_memory_id, relation.target_memory_id)
    }


def _unique(relations: Iterable[MemoryRelation]) -> list[MemoryRelation]:
    return list({relation.relation_id: relation for relation in relations}.values())

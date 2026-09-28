"""Deterministic resolution of recorded contradictions between two memories.

ADR-004: "Active facts are selected by validity, trust, source authority, and
recorded resolution before relevance scores. Unresolved conflicts are omitted
from ordinary context unless the caller explicitly requests them." This
module is the "trust / source authority" half of that rule, for one
`CONTRADICTION` edge at a time:

- If one side is strictly more trusted (`trust_rank`), it wins and only the
  other side is withheld.
- Otherwise the conflict is UNRESOLVED and *both* sides are withheld --
  "overlapping intervals and incomparable authorities may require
  quarantine or review rather than an automatic winner" (ADR-004).

What this module deliberately does *not* use: relevance (lexical, semantic,
fused, or CrossEncoder scores) and recency. A newer record is not
automatically more trustworthy (ADR-004 rejects "last write wins"), and a
better-matching record is not more true -- resolving from score would make
correctness depend on how the query happened to be worded.

Validity is not decided here either: `retrieval.temporal` only asks this
module about pairs where *both* memories are in effect at the query's time.
Two facts whose windows don't overlap are not in conflict at any single
point in time, and a recorded supersession already settles which one
applies. To settle an unresolved conflict for good, retire the losing
memory (supersede, quarantine, or tombstone it); once only one side is in
effect, the edge no longer withholds anything.

Like `ingestion.write_policy`, the decision logic is pure: it never touches
a database or a model. `record_contradiction` is the one write helper, used
to record a conflict once one has been detected.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from apps.memory_service.domain.enums import ConflictType
from apps.memory_service.domain.models import MemoryRecord, MemoryRelation
from apps.memory_service.ingestion.provenance import trust_rank

if TYPE_CHECKING:
    from apps.memory_service.persistence.unit_of_work import UnitOfWork


class ConflictState(StrEnum):
    """How one contradiction between two in-effect memories was settled.

    - RESOLVED_BY_TRUST: one side is strictly more trusted and wins.
    - UNRESOLVED: equal trust; neither side may be shown as current truth.
    """

    RESOLVED_BY_TRUST = "resolved_by_trust"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ConflictOutcome:
    """The verdict on one `CONTRADICTION` edge between two in-effect memories.

    `winner_id` / `loser_id` are set only when `state` is RESOLVED_BY_TRUST.
    `reason` is a short human-readable explanation, suitable for a log line
    or for surfacing the conflict to a reviewer.
    """

    relation_id: UUID
    memory_ids: tuple[UUID, UUID]
    state: ConflictState
    reason: str
    winner_id: UUID | None = None
    loser_id: UUID | None = None

    @property
    def withheld_ids(self) -> frozenset[UUID]:
        """The memories this conflict keeps out of ordinary context."""
        if self.state is ConflictState.UNRESOLVED:
            return frozenset(self.memory_ids)
        assert self.loser_id is not None
        return frozenset({self.loser_id})


def resolve_contradiction(
    relation: MemoryRelation, first: MemoryRecord, second: MemoryRecord
) -> ConflictOutcome:
    """Decide which of two contradicting, in-effect memories (if either) may be shown.

    How it works:
        1. Check that `relation` is a CONTRADICTION edge between exactly
           `first` and `second` (in either direction) and that both belong
           to the relation's tenant; anything else is a caller bug and
           raises `ValueError`.
        2. Compare `trust_rank(first.trust_level)` with
           `trust_rank(second.trust_level)`. If one is strictly higher, it
           wins: RESOLVED_BY_TRUST, and only the other is withheld.
        3. Otherwise: UNRESOLVED, and both are withheld.

    Scores and timestamps are never consulted (see the module docstring).

    Example:
        Input:
            relation = MemoryRelation(source_memory_id=A, target_memory_id=B,
                                      relation_type=ConflictType.CONTRADICTION, ...)
            first  = MemoryRecord(memory_id=A, content="lives in Berlin",
                                  trust_level=TrustLevel.MEDIUM, ...)
            second = MemoryRecord(memory_id=B, content="lives in Paris",
                                  trust_level=TrustLevel.MEDIUM, ...)
        Output:
            ConflictOutcome(relation_id=..., memory_ids=(A, B),
                            state=ConflictState.UNRESOLVED,
                            reason="equal trust (medium); needs review")
    """
    if relation.relation_type is not ConflictType.CONTRADICTION:
        raise ValueError(f"expected a CONTRADICTION relation, got {relation.relation_type}")
    endpoints = {relation.source_memory_id, relation.target_memory_id}
    if {first.memory_id, second.memory_id} != endpoints:
        raise ValueError("first and second must be exactly the relation's two memories")
    if first.tenant_id != relation.tenant_id or second.tenant_id != relation.tenant_id:
        raise ValueError("both memories must belong to the relation's tenant")

    memory_ids = (first.memory_id, second.memory_id)
    first_rank = trust_rank(first.trust_level)
    second_rank = trust_rank(second.trust_level)
    if first_rank == second_rank:
        return ConflictOutcome(
            relation_id=relation.relation_id,
            memory_ids=memory_ids,
            state=ConflictState.UNRESOLVED,
            reason=f"equal trust ({first.trust_level}); needs review",
        )

    winner, loser = (first, second) if first_rank > second_rank else (second, first)
    return ConflictOutcome(
        relation_id=relation.relation_id,
        memory_ids=memory_ids,
        state=ConflictState.RESOLVED_BY_TRUST,
        reason=f"higher trust wins ({winner.trust_level} over {loser.trust_level})",
        winner_id=winner.memory_id,
        loser_id=loser.memory_id,
    )


def record_contradiction(
    uow: UnitOfWork,
    tenant_id: UUID,
    first_memory_id: UUID,
    second_memory_id: UUID,
    *,
    rationale: str | None = None,
) -> MemoryRelation:
    """Record that two existing memories make incompatible claims.

    Both memories are kept (ADR-004: "Conflicts retain both claims"); this
    only adds a `CONTRADICTION` edge from `first_memory_id` to
    `second_memory_id`, which `retrieval.temporal` then honours on every
    read. The composite foreign keys on `memory_relations` reject an edge
    to a memory that doesn't exist or belongs to another tenant at flush
    time. Like the other `UnitOfWork` writes, nothing is committed here.

    Example:
        Input:
            tenant_id = UUID("11111111-1111-1111-1111-111111111111")
            first_memory_id = UUID("aaaaaaaa-...")   # "lives in Berlin"
            second_memory_id = UUID("bbbbbbbb-...")  # "lives in Paris"
            rationale = "same subject, different city"
        Output:
            MemoryRelation(source_memory_id=UUID("aaaaaaaa-..."),
                           target_memory_id=UUID("bbbbbbbb-..."),
                           relation_type=ConflictType.CONTRADICTION,
                           rationale="same subject, different city", ...)
    """
    if first_memory_id == second_memory_id:
        raise ValueError("a memory cannot contradict itself")
    relation = MemoryRelation(
        tenant_id=tenant_id,
        source_memory_id=first_memory_id,
        target_memory_id=second_memory_id,
        relation_type=ConflictType.CONTRADICTION,
        rationale=rationale,
        created_at=datetime.datetime.now(datetime.UTC),
    )
    uow.relations.add(tenant_id, relation)
    return relation

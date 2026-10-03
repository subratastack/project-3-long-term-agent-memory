"""Tenant scope: one tenant's memory never reaches another tenant.

Tenant isolation already holds in several independent layers: every
repository query carries a `tenant_id` predicate, composite foreign keys stop
a provenance link or relation from pointing at another tenant's rows, and
retrieval re-checks each record's tenant after searching. This module is the
explicit, final check those layers share, and the one place that says what
"in scope" means for each kind of object:

| Object | In scope when |
| --- | --- |
| event | `event.tenant_id` is the scope's tenant |
| candidate | its own tenant matches, and every provenance link resolves to an in-scope event |
| memory record | its own tenant matches, and (when events are given) every provenance link does |
| relation | its own tenant matches, and both endpoint memories are in scope |
| write decision | its own tenant matches |
| retrieval result | the memory it carries is in scope |

The scope's tenant must come from authenticated runtime context (ARCHITECTURE.md,
"Tenant isolation") -- never from memory content or a request body.

Two ways to use it:

- **Writes** call `require(...)`, which raises `TenantScopeError` on any
  violation before anything is persisted.
- **Reads** call `filter_hits` / `filter_relations`, which drop anything out
  of scope (failing closed) and log how many objects were blocked. The log
  line carries ids and counts only, never memory content.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from apps.memory_service.domain.models import (
    MemoryCandidate,
    MemoryEvent,
    MemoryRecord,
    MemoryRelation,
    Provenance,
    WritePolicyDecision,
)

logger = logging.getLogger(__name__)


class ScopedEntity(StrEnum):
    """The kind of object a scope violation was found on."""

    EVENT = "event"
    CANDIDATE = "candidate"
    PROVENANCE_LINK = "provenance_link"
    MEMORY = "memory"
    RELATION = "relation"
    DECISION = "decision"
    RETRIEVAL_RESULT = "retrieval_result"


@dataclass(frozen=True)
class ScopeViolation:
    """One object found outside the scope's tenant.

    `found_tenant_id` is the tenant the object actually belongs to, or None
    when a reference could not be resolved inside the scope at all (an event
    or memory that, as far as this tenant can see, does not exist).
    """

    entity: ScopedEntity
    entity_id: UUID | None
    found_tenant_id: UUID | None


class TenantScopeError(ValueError):
    """Raised when a write would cross a tenant boundary."""

    def __init__(self, tenant_id: UUID, violations: Sequence[ScopeViolation]) -> None:
        self.tenant_id = tenant_id
        self.violations = tuple(violations)
        kinds = ", ".join(sorted({v.entity.value for v in self.violations}))
        super().__init__(
            f"{len(self.violations)} object(s) outside tenant {tenant_id}'s scope: {kinds}"
        )


class _HasMemory(Protocol):
    @property
    def memory(self) -> MemoryRecord: ...


@dataclass(frozen=True)
class TenantScope:
    """The tenant every object in one operation must belong to."""

    tenant_id: UUID

    def check_event(self, event: MemoryEvent) -> list[ScopeViolation]:
        """Example: a tenant-B event checked in tenant A's scope -> one EVENT violation."""
        return self._own(ScopedEntity.EVENT, event.event_id, event.tenant_id)

    def check_candidate(
        self,
        candidate: MemoryCandidate,
        events_by_id: Mapping[UUID, MemoryEvent] | None = None,
    ) -> list[ScopeViolation]:
        """Check a candidate's own tenant and, given the events, each provenance link.

        Example:
            Input:
                scope = TenantScope(A)
                candidate = MemoryCandidate(tenant_id=A, provenance=[<event of tenant B>])
                events_by_id = {<that event id>: <the tenant-B event>}
            Output:
                [ScopeViolation(PROVENANCE_LINK, <event id>, found_tenant_id=B)]
        """
        violations = self._own(ScopedEntity.CANDIDATE, candidate.candidate_id, candidate.tenant_id)
        if events_by_id is not None:
            violations += self._links(candidate.provenance, events_by_id)
        return violations

    def check_record(
        self,
        record: MemoryRecord,
        events_by_id: Mapping[UUID, MemoryEvent] | None = None,
    ) -> list[ScopeViolation]:
        """Check a memory record's own tenant and, given the events, each provenance link."""
        violations = self._own(ScopedEntity.MEMORY, record.memory_id, record.tenant_id)
        if events_by_id is not None:
            violations += self._links(record.provenance, events_by_id)
        return violations

    def check_relation(
        self, relation: MemoryRelation, records_by_id: Mapping[UUID, MemoryRecord]
    ) -> list[ScopeViolation]:
        """Check a relation's own tenant and that both endpoints are in-scope memories.

        An endpoint missing from `records_by_id` is a violation: a relation
        may only connect memories this tenant can see.
        """
        violations = self._own(ScopedEntity.RELATION, relation.relation_id, relation.tenant_id)
        for memory_id in (relation.source_memory_id, relation.target_memory_id):
            record = records_by_id.get(memory_id)
            if record is None:
                violations.append(ScopeViolation(ScopedEntity.MEMORY, memory_id, None))
            else:
                violations += self._own(ScopedEntity.MEMORY, memory_id, record.tenant_id)
        return violations

    def check_decision(self, decision: WritePolicyDecision) -> list[ScopeViolation]:
        return self._own(ScopedEntity.DECISION, decision.decision_id, decision.tenant_id)

    def require(self, violations: Iterable[ScopeViolation]) -> None:
        """Raise `TenantScopeError` if there is any violation; otherwise do nothing."""
        found = list(violations)
        if found:
            raise TenantScopeError(self.tenant_id, found)

    def filter_hits[HitT: _HasMemory](self, hits: Sequence[HitT]) -> list[HitT]:
        """Keep the retrieval results whose memory belongs to this tenant.

        Example:
            Input:
                scope = TenantScope(B)
                hits = [<hit for a tenant-B memory>, <hit for a tenant-A memory>]
            Output:
                [<hit for the tenant-B memory>]   # and one blocked result is logged
        """
        kept = [hit for hit in hits if hit.memory.tenant_id == self.tenant_id]
        self._log_blocked(ScopedEntity.RETRIEVAL_RESULT, len(hits) - len(kept))
        return kept

    def filter_relations(self, relations: Sequence[MemoryRelation]) -> list[MemoryRelation]:
        """Keep the relations that belong to this tenant."""
        kept = [relation for relation in relations if relation.tenant_id == self.tenant_id]
        self._log_blocked(ScopedEntity.RELATION, len(relations) - len(kept))
        return kept

    def _own(
        self, entity: ScopedEntity, entity_id: UUID | None, tenant_id: UUID
    ) -> list[ScopeViolation]:
        if tenant_id == self.tenant_id:
            return []
        return [ScopeViolation(entity, entity_id, tenant_id)]

    def _links(
        self, provenance: Iterable[Provenance], events_by_id: Mapping[UUID, MemoryEvent]
    ) -> list[ScopeViolation]:
        violations: list[ScopeViolation] = []
        for link in provenance:
            event = events_by_id.get(link.event_id)
            if event is None or event.tenant_id != self.tenant_id:
                violations.append(
                    ScopeViolation(
                        ScopedEntity.PROVENANCE_LINK,
                        link.event_id,
                        event.tenant_id if event is not None else None,
                    )
                )
        return violations

    def _log_blocked(self, entity: ScopedEntity, count: int) -> None:
        if count:
            logger.warning(
                "tenant scope blocked %d out-of-scope %s object(s) for tenant %s",
                count,
                entity.value,
                self.tenant_id,
            )

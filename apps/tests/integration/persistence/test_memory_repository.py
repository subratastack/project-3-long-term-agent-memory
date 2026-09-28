"""Integration tests proving the persistence layer against real PostgreSQL.

Run via `docker compose up -d postgres` then `uv run pytest`; the whole module
skips automatically if PostgreSQL is unreachable (see ../conftest.py). This is
the review gate for relational memory correctness and tenant isolation before
the event-ingestion and write-policy pipeline is built on top of it.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from apps.memory_service.domain.enums import (
    ConflictType,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    MemoryRelation,
    Provenance,
    TemporalValidity,
    WritePolicyDecision,
)
from apps.memory_service.persistence.unit_of_work import UnitOfWork

UowFactory = Callable[[], UnitOfWork]


def _make_event(tenant_id: UUID, **overrides: Any) -> MemoryEvent:
    defaults: dict[str, Any] = {
        "tenant_id": tenant_id,
        "source_type": SourceType.CONFIGURATION,
        "source_reference": "config-1",
        "content": "timeout_seconds=5",
        "observed_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return MemoryEvent(**defaults)


def _make_provenance(event: MemoryEvent, **overrides: Any) -> Provenance:
    defaults: dict[str, Any] = {
        "event_id": event.event_id,
        "source_type": event.source_type,
        "source_reference": event.source_reference,
        "observed_at": event.observed_at,
        "trust_level": TrustLevel.SYSTEM,
    }
    defaults.update(overrides)
    return Provenance(**defaults)


def _make_memory(tenant_id: UUID, provenance: Provenance, **overrides: Any) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "tenant_id": tenant_id,
        "content": "timeout=5s",
        "confidence": 0.95,
        "provenance": [provenance],
        "temporal_validity": TemporalValidity(valid_from=datetime.now(UTC)),
        "memory_type": MemoryType.SEMANTIC,
        "trust_level": TrustLevel.SYSTEM,
        "status": MemoryStatus.ACTIVE,
        "subject_keys": ["config:timeout"],
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def _make_write_decision(tenant_id: UUID, **overrides: Any) -> WritePolicyDecision:
    defaults: dict[str, Any] = {
        "tenant_id": tenant_id,
        "candidate_id": uuid4(),
        "decision": WriteDecision.ACCEPT,
        "policy_version": "1.0",
        "reason_codes": ["HIGH_TRUST_CONFIGURATION_SOURCE"],
    }
    defaults.update(overrides)
    return WritePolicyDecision(**defaults)


def test_create_and_retrieve_memory_with_provenance(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    provenance = _make_provenance(event)
    memory = _make_memory(tenant_id, provenance)
    decision = _make_write_decision(tenant_id, accepted_memory_id=memory.memory_id)

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        uow.record_write_decision(decision)
        uow.commit()

    with uow_factory() as uow:
        fetched = uow.get_memory(memory.memory_id, tenant_id)
        fetched_decision = uow.write_decisions.get(tenant_id, decision.decision_id)

    assert fetched is not None
    assert fetched.content == "timeout=5s"
    assert len(fetched.provenance) == 1
    assert fetched.provenance[0].event_id == event.event_id

    assert fetched_decision is not None
    assert fetched_decision.decision == WriteDecision.ACCEPT
    assert fetched_decision.accepted_memory_id == memory.memory_id


def test_tenant_isolation_hides_memory_from_other_tenants(uow_factory: UowFactory) -> None:
    tenant_a = uuid4()
    tenant_b = uuid4()
    event = _make_event(tenant_a)
    provenance = _make_provenance(event)
    memory = _make_memory(tenant_a, provenance)

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        uow.commit()

    with uow_factory() as uow:
        result = uow.get_memory(memory.memory_id, tenant_b)
        listing = uow.list_active_memories(tenant_b)

    assert result is None
    assert listing == []


def test_supersession_leaves_only_the_newer_fact_in_normal_retrieval(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    old_event = _make_event(tenant_id, content="timeout_seconds=5")
    old_provenance = _make_provenance(old_event)
    old_memory = _make_memory(
        tenant_id,
        old_provenance,
        content="timeout=5s",
        temporal_validity=TemporalValidity(valid_from=datetime.now(UTC) - timedelta(days=1)),
    )

    with uow_factory() as uow:
        uow.create_event(old_event)
        uow.create_memory(old_memory)
        uow.commit()

    new_event = _make_event(tenant_id, content="timeout_seconds=2")
    new_provenance = _make_provenance(new_event)
    new_memory = _make_memory(
        tenant_id,
        new_provenance,
        content="timeout=2s",
        temporal_validity=TemporalValidity(valid_from=datetime.now(UTC)),
    )
    decision = _make_write_decision(
        tenant_id,
        decision=WriteDecision.SUPERSEDE,
        accepted_memory_id=new_memory.memory_id,
        superseded_memory_id=old_memory.memory_id,
    )

    with uow_factory() as uow:
        uow.create_event(new_event)
        uow.supersede_memory(
            tenant_id,
            old_memory.memory_id,
            new_memory,
            decision,
            rationale="operator lowered the timeout",
        )
        uow.commit()

    with uow_factory() as uow:
        old_after = uow.get_memory(old_memory.memory_id, tenant_id)
        active = uow.list_active_memories(tenant_id, memory_type=MemoryType.SEMANTIC)
        relations = uow.relations.list_for_memory(tenant_id, new_memory.memory_id)

    assert old_after is not None
    assert old_after.status == MemoryStatus.SUPERSEDED

    assert [record.content for record in active] == ["timeout=2s"]

    assert len(relations) == 1
    assert relations[0].relation_type == ConflictType.SUPERSESSION
    assert relations[0].source_memory_id == new_memory.memory_id
    assert relations[0].target_memory_id == old_memory.memory_id


def test_event_and_memory_roll_back_together_on_failure(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    provenance = _make_provenance(event)
    memory = _make_memory(tenant_id, provenance)

    class _DeliberateFailure(Exception):
        pass

    with pytest.raises(_DeliberateFailure), uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        raise _DeliberateFailure("simulated failure before commit")

    with uow_factory() as uow:
        assert uow.events.get(tenant_id, event.event_id) is None
        assert uow.get_memory(memory.memory_id, tenant_id) is None


def test_composite_foreign_keys_block_cross_tenant_provenance_and_relations(
    uow_factory: UowFactory,
) -> None:
    tenant_a = uuid4()
    tenant_b = uuid4()

    event_a = _make_event(tenant_a)
    provenance_a = _make_provenance(event_a)
    memory_a = _make_memory(tenant_a, provenance_a)

    with uow_factory() as uow:
        uow.create_event(event_a)
        uow.create_memory(memory_a)
        uow.commit()

    # Tenant B tries to attach its own provenance row to Tenant A's memory_id.
    # The composite (tenant_id, memory_id) foreign key has no row matching
    # (tenant_b, memory_a.memory_id), so PostgreSQL itself rejects this --
    # not just the repository's application-level tenant check.
    forged_provenance = Provenance(
        event_id=uuid4(),
        source_type=SourceType.USER_MESSAGE,
        source_reference="forged",
        observed_at=datetime.now(UTC),
        trust_level=TrustLevel.LOW,
    )
    with pytest.raises(IntegrityError), uow_factory() as uow:
        uow.provenance.add_for_memory(tenant_b, memory_a.memory_id, forged_provenance)
        uow.commit()

    # Likewise for a relation pointing at Tenant A's memory under Tenant B.
    forged_relation = MemoryRelation(
        tenant_id=tenant_b,
        source_memory_id=uuid4(),
        target_memory_id=memory_a.memory_id,
        relation_type=ConflictType.DUPLICATE,
    )
    with pytest.raises(IntegrityError), uow_factory() as uow:
        uow.relations.add(tenant_b, forged_relation)
        uow.commit()

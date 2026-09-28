"""Verify durable evidence links and repeat-run idempotency using real repositories."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from apps.memory_service.consolidation.service import consolidate_memories
from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)


def test_consolidation_persists_all_support_and_no_duplicates(uow_factory):
    tenant_id = uuid4()
    now = datetime.now(UTC)
    episodes = []
    with uow_factory() as uow:
        for i in range(3):
            event = MemoryEvent(
                tenant_id=tenant_id,
                source_type=SourceType.SYSTEM_EVENT,
                source_reference=f"pool-{i}",
                content="Pool exhaustion resolved by sizing review",
                observed_at=now - timedelta(days=i + 1),
            )
            record = MemoryRecord(
                tenant_id=tenant_id,
                created_at=now,
                updated_at=now,
                content=event.content,
                confidence=0.9,
                memory_type=MemoryType.EPISODIC,
                status=MemoryStatus.ACTIVE,
                trust_level=TrustLevel.SYSTEM,
                subject_keys=["service:payments"],
                temporal_validity=TemporalValidity(valid_from=event.observed_at),
                provenance=[
                    Provenance(
                        event_id=event.event_id,
                        source_type=event.source_type,
                        source_reference=event.source_reference,
                        observed_at=event.observed_at,
                        trust_level=TrustLevel.SYSTEM,
                    )
                ],
                metadata={
                    "category_key": "pool exhaustion",
                    "remediation_outcome": "success",
                    "remediation_steps": ["inspect saturation", "review sizing"],
                },
            )
            uow.create_event(event)
            uow.create_memory(record)
            episodes.append(record)
        uow.commit()
    first = consolidate_memories(uow_factory, tenant_id, now=now)
    assert len(first) == 1
    assert consolidate_memories(uow_factory, tenant_id, now=now) == []
    with uow_factory() as uow:
        summary = uow.get_memory(first[0].accepted_memory_id, tenant_id)
        assert summary.memory_type == MemoryType.PROCEDURAL
        assert {p.event_id for p in summary.provenance} == {
            p.event_id for m in episodes for p in m.provenance
        }
        assert set(summary.metadata["supporting_memory_ids"]) == {
            str(m.memory_id) for m in episodes
        }
        for episode in episodes:
            assert uow.get_memory(episode.memory_id, tenant_id) == episode
        assert len(uow.list_active_memories(tenant_id)) == 4

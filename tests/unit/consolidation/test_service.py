from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from conftest import NOW

from apps.memory_service.consolidation.service import consolidate_memories
from apps.memory_service.domain.enums import MemoryStatus, SourceType, WriteDecision
from apps.memory_service.domain.models import MemoryEvent


class FakeUow:
    def __init__(self, records):
        self.stored = {m.memory_id: m for m in records}
        self.decisions = []
        self.records = SimpleNamespace(
            lock_lifecycle=lambda tenant: None,
            tombstoned_event_ids=lambda tenant: {
                p.event_id
                for m in self.stored.values()
                if m.status == MemoryStatus.TOMBSTONE and m.tenant_id == tenant
                for p in m.provenance
            },
            save_lifecycle=lambda record: self.stored.__setitem__(record.memory_id, record),
        )
        self.evidence = {
            p.event_id: MemoryEvent(
                event_id=p.event_id,
                tenant_id=m.tenant_id,
                source_type=p.source_type,
                source_reference=p.source_reference,
                observed_at=p.observed_at,
                content=m.content,
            )
            for m in records
            for p in m.provenance
        }
        self.events = SimpleNamespace(get=lambda tenant, eid: self.evidence.get(eid))

    def __enter__(self):
        self.snapshot = deepcopy((self.stored, self.decisions))
        return self

    def __exit__(self, *args):
        if args[0]:
            self.stored, self.decisions = self.snapshot

    def list_active_memories(self, tenant, memory_type):
        return [m for m in self.stored.values() if m.memory_type == memory_type]

    def get_memory(self, memory_id, tenant):
        return self.stored.get(memory_id)

    def create_memory(self, record):
        assert record.memory_id not in self.stored
        self.stored[record.memory_id] = record

    def record_write_decision(self, decision):
        self.decisions.append(decision)

    def commit(self):
        pass


def test_fifty_to_one_idempotent_and_originals_unchanged(episodes):
    records = episodes(50)
    original = deepcopy(records)
    uow = FakeUow(records)
    decisions = consolidate_memories(lambda: uow, records[0].tenant_id, now=NOW)
    assert len(decisions) == 1
    assert decisions[0].decision == WriteDecision.ACCEPT
    summary = uow.stored[decisions[0].accepted_memory_id]
    assert len(summary.provenance) == 50
    assert len(summary.metadata["supporting_memory_ids"]) == 50
    assert records == original
    for episode in original:
        stored = uow.stored[episode.memory_id]
        assert stored.content == episode.content
        assert stored.provenance == episode.provenance
        assert stored.metadata["forgetting"]["priority"] == 0.25
    assert consolidate_memories(lambda: uow, records[0].tenant_id, now=NOW) == []
    assert len(uow.stored) == 51
    summary.status = MemoryStatus.TOMBSTONE
    assert consolidate_memories(lambda: uow, records[0].tenant_id, now=NOW) == []


def test_missing_evidence_rejected_by_existing_policy(episodes):
    records = episodes()
    uow = FakeUow(records)
    uow.evidence.clear()
    decisions = consolidate_memories(lambda: uow, records[0].tenant_id, now=NOW)
    assert decisions[0].decision == WriteDecision.REJECT
    assert len(uow.stored) == 2
    assert len(uow.decisions) == 1


def test_procedure_requires_authoritative_high_trust(episodes):
    records = episodes(
        3,
        metadata={
            "category_key": "pool",
            "remediation_outcome": "success",
            "remediation_steps": ["inspect metrics"],
        },
    )
    for record in records:
        record.provenance[0].source_type = SourceType.TOOL_OUTPUT
    uow = FakeUow(records)
    decision = consolidate_memories(lambda: uow, records[0].tenant_id, now=NOW)[0]
    assert decision.decision == WriteDecision.QUARANTINE
    assert uow.stored[decision.accepted_memory_id].status == MemoryStatus.QUARANTINED


def test_write_failure_rolls_back(episodes):
    records = episodes()
    uow = FakeUow(records)

    def fail(decision):
        raise RuntimeError("audit failure")

    uow.record_write_decision = fail
    with pytest.raises(RuntimeError, match="audit failure"):
        consolidate_memories(lambda: uow, records[0].tenant_id, now=NOW)
    assert len(uow.stored) == 2


def test_tenant_filtered_even_with_bad_repository(episodes):
    records = episodes()
    uow = FakeUow(records)
    assert consolidate_memories(lambda: uow, uuid4(), now=NOW) == []


def test_active_copies_of_tombstoned_evidence_cannot_reconsolidate(episodes):
    records = episodes(3)
    tomb = records[0].model_copy(update={"memory_id": uuid4(), "status": MemoryStatus.TOMBSTONE})
    # Even if another active record cites deleted evidence, grouping cannot use it.
    for record in records[1:]:
        record.provenance = records[0].provenance
    uow = FakeUow([*records, tomb])
    assert consolidate_memories(lambda: uow, records[0].tenant_id, now=NOW) == []
    assert len(uow.stored) == 4

"""Consolidate one tenant atomically, retaining source records and evidence.

Deterministic output primary keys make sequential runs idempotent, including
when an earlier summary has expired or been tombstoned. A tenant transaction
lock serializes cooperating jobs; the primary key is a final duplicate guard.
Changed membership is a new cluster unless its evidence has been tombstoned.
Accepted summaries compact their sources in the same transaction.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid5

from apps.memory_service.consolidation.clusterer import cluster_memories
from apps.memory_service.consolidation.forgetting import compact_consolidated
from apps.memory_service.consolidation.promoter import promote_cluster
from apps.memory_service.consolidation.summarizer import summarize_cluster
from apps.memory_service.domain.enums import MemoryStatus, MemoryType, WriteDecision
from apps.memory_service.domain.models import MemoryRecord, WritePolicyDecision
from apps.memory_service.ingestion.provenance import weakest_trust
from apps.memory_service.ingestion.write_policy import POLICY_VERSION, evaluate_candidate
from apps.memory_service.persistence.unit_of_work import UnitOfWork


def consolidate_memories(
    uow_factory: Callable[[], UnitOfWork],
    tenant_id: UUID,
    *,
    now: datetime | None = None,
) -> list[WritePolicyDecision]:
    """Load active episodes, propose eligible summaries, apply policy, and commit.

    Metadata contract: incident_key/category_key, remediation_outcome (success
    is the sole positive value), remediation_steps (identical ordered strings).
    Returned decisions include rejects/quarantines; skipped clusters return none.
    """
    now = now or datetime.now(UTC)
    decisions = []
    with uow_factory() as uow:
        uow.records.lock_lifecycle(tenant_id)
        denied_events = uow.records.tombstoned_event_ids(tenant_id)
        episodes = uow.list_active_memories(tenant_id, memory_type=MemoryType.EPISODIC)
        episodes = [
            m
            for m in episodes
            if m.tenant_id == tenant_id
            and not denied_events.intersection(p.event_id for p in m.provenance)
        ]
        for cluster in cluster_memories(episodes, now=now):
            memory_type = promote_cluster(cluster, now=now)
            if memory_type is None:
                continue
            memory_id = uuid5(cluster.cluster_id, "memory:v1")
            if uow.get_memory(memory_id, tenant_id) is not None:
                continue
            try:
                candidate = summarize_cluster(cluster, memory_type)
            except ValueError:
                # Invalid or oversized structured fields are not safe summaries.
                continue
            events = {}
            for event_id in {p.event_id for p in candidate.provenance}:
                event = uow.events.get(tenant_id, event_id)
                if event is not None:
                    events[event_id] = event
            # Summary wording is a deterministic template, not extractor output.
            evaluation = evaluate_candidate(candidate, events, now=now, model_extracted=False)
            decision = evaluation.decision
            if decision.decision in (WriteDecision.ACCEPT, WriteDecision.QUARANTINE):
                record = MemoryRecord(
                    memory_id=memory_id,
                    tenant_id=tenant_id,
                    content=candidate.content,
                    confidence=candidate.confidence,
                    provenance=candidate.provenance,
                    memory_type=candidate.memory_type,
                    temporal_validity=candidate.temporal_validity,
                    subject_keys=candidate.subject_keys,
                    metadata=candidate.metadata,
                    trust_level=weakest_trust(
                        [evaluation.effective_trust, candidate.proposed_trust_level]
                    ),
                    status=(
                        MemoryStatus.ACTIVE
                        if decision.decision == WriteDecision.ACCEPT
                        else MemoryStatus.QUARANTINED
                    ),
                    policy_version=POLICY_VERSION,
                    created_at=now,
                    updated_at=now,
                )
                uow.create_memory(record)
                if decision.decision == WriteDecision.ACCEPT:
                    compact_consolidated(uow, record, now)
                decision = decision.model_copy(update={"accepted_memory_id": memory_id})
            uow.record_write_decision(decision)
            decisions.append(decision)
        uow.commit()
    return decisions

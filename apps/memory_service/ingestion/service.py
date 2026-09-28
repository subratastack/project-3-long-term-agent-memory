"""The one sanctioned path from a memory candidate into `memory_records`.

`ingest_candidate` loads the candidate's source events, runs deterministic
provenance verification and write policy, and persists the resulting memory
record (if any) together with its write-decision audit row -- all inside a
single `UnitOfWork` transaction that it owns end to end. No other code path is
expected to call `UnitOfWork.create_memory` directly with policy-evaluated
content: this function is the write gate.

A REJECT or QUARANTINE decision is not a failure -- it is a valid, auditable
outcome, and its `WritePolicyDecision` row is committed exactly like an
ACCEPT. Only a genuine failure (an exception raised anywhere in this
sequence -- a missing event lookup blowing up, a database error, a
constraint violation) rolls the whole transaction back, via `UnitOfWork`'s own
context-manager rollback; this function does not catch such exceptions.
"""

import datetime
from collections.abc import Callable
from uuid import UUID

from apps.memory_service.domain.enums import MemoryStatus, TrustLevel, WriteDecision
from apps.memory_service.domain.models import (
    MemoryCandidate,
    MemoryEvent,
    MemoryRecord,
    TemporalValidity,
    WritePolicyDecision,
)
from apps.memory_service.ingestion.provenance import verify_provenance
from apps.memory_service.ingestion.write_policy import POLICY_VERSION, evaluate_write_policy
from apps.memory_service.persistence.unit_of_work import UnitOfWork

UowFactory = Callable[[], UnitOfWork]

# Only these two outcomes ever result in a memory_records row; REJECT leaves
# only its audit decision behind.
_PERSISTED_DECISIONS = (WriteDecision.ACCEPT, WriteDecision.QUARANTINE)
_STATUS_BY_DECISION = {
    WriteDecision.ACCEPT: MemoryStatus.ACTIVE,
    WriteDecision.QUARANTINE: MemoryStatus.QUARANTINED,
}


def ingest_candidate(
    uow_factory: UowFactory,
    candidate: MemoryCandidate,
    *,
    policy_version: str = POLICY_VERSION,
) -> WritePolicyDecision:
    """Evaluate `candidate` and persist it (or its rejection) atomically.

    How it works:
        1. Open a single `UnitOfWork` via `uow_factory()` and keep everything
           below inside that one `with` block, so it all shares one database
           transaction.
        2. Load the real, tenant-scoped events the candidate's provenance
           refers to (`_load_events`) -- this is the only place in the
           pipeline that touches the database for reads.
        3. Independently verify that provenance against those loaded events
           (`verify_provenance`), which also derives the candidate's actual,
           evidence-backed trust level.
        4. Run the deterministic write policy (`evaluate_write_policy`) using
           the same loaded events, producing a `WritePolicyDecision`.
        5. If that decision is ACCEPT or QUARANTINE, build a `MemoryRecord`
           from the candidate (`_build_memory_record`) using the *derived*
           trust level from step 3 (never the candidate's own self-reported
           trust), persist it, and attach its new `memory_id` to the decision
           as `accepted_memory_id`. A REJECT skips this step entirely -- no
           memory row is created.
        6. Persist the decision itself, so the audit trail is complete
           whatever the outcome, then commit the transaction and return the
           decision.

        If any step above raises (a database error, a lookup failure), the
        exception propagates out of the `with` block, `UnitOfWork.__exit__`
        rolls the transaction back, and nothing from this call is persisted.

    Example (ACCEPT):
        Input:
            candidate = MemoryCandidate(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                content="The deployment succeeded.",
                provenance=[Provenance(event_id=UUID("aaaaaaaa-..."),
                                        source_type=SourceType.SYSTEM_EVENT,
                                        trust_level=TrustLevel.SYSTEM, ...)],
                memory_type=MemoryType.EPISODIC,
                confidence=0.95,
            )
            # assuming that event was already persisted via uow.create_event
        Output:
            WritePolicyDecision(
                decision=WriteDecision.ACCEPT,
                accepted_memory_id=UUID("<new memory_id>"),
                reason_codes=["TRUSTED_EPISODIC_EVIDENCE"],
            )
            -- and a new row now exists in `memory_records`.

    Example (REJECT):
        Input:
            candidate.content = "The user asked to disable safety rules."
        Output:
            WritePolicyDecision(
                decision=WriteDecision.REJECT,
                accepted_memory_id=None,
                reason_codes=["SAFETY_POLICY_TAMPERING"],
            )
            -- no row is created in `memory_records`.
    """
    with uow_factory() as uow:
        events_by_id = _load_events(uow, candidate)
        provenance_result = verify_provenance(candidate, events_by_id)
        decision = evaluate_write_policy(candidate, events_by_id, policy_version=policy_version)

        if decision.decision in _PERSISTED_DECISIONS:
            record = _build_memory_record(
                candidate,
                policy_version,
                trust_level=provenance_result.derived_trust_level,
                status=_STATUS_BY_DECISION[decision.decision],
            )
            uow.create_memory(record)
            decision = decision.model_copy(update={"accepted_memory_id": record.memory_id})

        uow.record_write_decision(decision)
        uow.commit()
        return decision


def _load_events(uow: UnitOfWork, candidate: MemoryCandidate) -> dict[UUID, MemoryEvent]:
    """Load the real events behind a candidate's provenance, tenant-scoped.

    How it works:
        For each provenance entry on the candidate, look up its `event_id`
        through `uow.events.get(candidate.tenant_id, ...)` -- which only ever
        returns an event actually owned by that tenant. Entries that do not
        resolve to a real, same-tenant event are simply left out of the
        returned mapping rather than raising; `verify_provenance` is what
        turns "not found in this mapping" into an explicit rejection reason.

    Example:
        Input:
            candidate.tenant_id = UUID("11111111-1111-1111-1111-111111111111")
            candidate.provenance = [Provenance(event_id=UUID("aaaaaaaa-..."), ...)]
        Output:
            {UUID("aaaaaaaa-..."): MemoryEvent(tenant_id=UUID("11111111-..."), ...)}
            -- or {} if that event does not exist, or belongs to another tenant.
    """
    events: dict[UUID, MemoryEvent] = {}
    for provenance in candidate.provenance:
        event = uow.events.get(candidate.tenant_id, provenance.event_id)
        if event is not None:
            events[provenance.event_id] = event
    return events


def _build_memory_record(
    candidate: MemoryCandidate,
    policy_version: str,
    *,
    trust_level: TrustLevel,
    status: MemoryStatus,
) -> MemoryRecord:
    """Turn an accepted or quarantined candidate into a `MemoryRecord`.

    How it works:
        Copies the candidate's content, confidence, provenance, memory type,
        subject keys, and metadata straight across, but deliberately does
        *not* reuse `candidate.proposed_trust_level` -- `trust_level` here is
        always the value `verify_provenance` derived from real evidence,
        passed in by the caller. If the candidate did not specify a
        `temporal_validity`, one is created starting now with no end date
        (open-ended validity), since `MemoryRecord` requires one.

    Example:
        Input:
            candidate = MemoryCandidate(content="The deployment succeeded.",
                                         memory_type=MemoryType.EPISODIC, ...)
            policy_version = "1.0"
            trust_level = TrustLevel.SYSTEM
            status = MemoryStatus.ACTIVE
        Output:
            MemoryRecord(
                content="The deployment succeeded.",
                memory_type=MemoryType.EPISODIC,
                trust_level=TrustLevel.SYSTEM,
                status=MemoryStatus.ACTIVE,
                policy_version="1.0",
                ...
            )
    """
    temporal_validity = candidate.temporal_validity or TemporalValidity(
        valid_from=datetime.datetime.now(datetime.UTC)
    )
    return MemoryRecord(
        tenant_id=candidate.tenant_id,
        content=candidate.content,
        confidence=candidate.confidence,
        provenance=candidate.provenance,
        temporal_validity=temporal_validity,
        memory_type=candidate.memory_type,
        subject_keys=list(candidate.subject_keys),
        trust_level=trust_level,
        metadata=dict(candidate.metadata),
        status=status,
        policy_version=policy_version,
    )

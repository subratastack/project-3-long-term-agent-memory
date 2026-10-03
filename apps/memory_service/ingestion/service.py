"""The one sanctioned path from a memory candidate into `memory_records`.

`ingest_candidate` loads the candidate's source events (and, for a fact
claim, the tenant's current facts), runs deterministic provenance
verification and write policy, and persists the resulting memory record (if
any) together with its write-decision audit row -- all inside a single
`UnitOfWork` transaction that it owns end to end. No other code path is
expected to call `UnitOfWork.create_memory` directly with policy-evaluated
content: this function is the write gate.

A REJECT or QUARANTINE decision is not a failure -- it is a valid, auditable
outcome, and its `WritePolicyDecision` row is committed exactly like an
ACCEPT. Only a genuine failure (an exception raised anywhere in this
sequence -- a missing event lookup blowing up, a database error, a
constraint violation) rolls the whole transaction back, via `UnitOfWork`'s own
context-manager rollback; this function does not catch such exceptions.

`record_rejected_extraction` covers the one rejection that happens before
write policy runs -- an extractor proposal with no resolvable evidence -- so
that it, too, leaves a decision with a reason code in the audit trail.
"""

import datetime
from collections.abc import Callable
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, TrustLevel, WriteDecision
from apps.memory_service.domain.models import (
    MemoryCandidate,
    MemoryEvent,
    MemoryRecord,
    TemporalValidity,
    WritePolicyDecision,
)
from apps.memory_service.ingestion.normalizer import RejectedExtraction
from apps.memory_service.ingestion.write_policy import POLICY_VERSION, evaluate_candidate
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.security.tenant_scope import TenantScope
from apps.memory_service.security.trust import parse_fact_claim

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
    tenant_id: UUID | None = None,
    policy_version: str = POLICY_VERSION,
    now: datetime.datetime | None = None,
) -> WritePolicyDecision:
    """Evaluate `candidate` and persist it (or its rejection) atomically.

    `tenant_id` is the authenticated tenant making the request, when the
    caller has one (the API always does). A candidate owned by any other
    tenant raises `TenantScopeError` before anything is read or written.
    `now` (default: the current time) is when stale-fact checks are judged,
    and when a new record without its own validity window starts to apply.

    How it works:
        1. Check the candidate belongs to the authenticated tenant.
        2. Open a single `UnitOfWork` via `uow_factory()` and keep everything
           below inside that one `with` block, so it all shares one database
           transaction.
        3. Load the real, tenant-scoped events the candidate's provenance
           refers to (`_load_events`), and -- if its content states a fact
           (`parse_fact_claim`) -- the tenant's active semantic facts
           (`_load_current_facts`) for the stale-fact check.
        4. Run the deterministic write policy (`evaluate_candidate`), which
           verifies provenance, checks content and evidence, and returns the
           decision together with the record's effective trust.
        5. ACCEPT or QUARANTINE: build a `MemoryRecord` from the candidate
           (`_build_memory_record`) with the policy's *effective* trust
           (never the candidate's self-reported trust), persist it, and attach
           its `memory_id` to the decision as `accepted_memory_id`.
           SUPERSEDE: build the record as ACTIVE and hand it to
           `UnitOfWork.supersede_memory`, which retires the old fact, links
           the two, and records the decision.
           REJECT: no memory row is created.
        6. Persist the decision (unless supersession already did), so the
           audit trail is complete whatever the outcome, then commit and
           return the decision.

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
    scope = TenantScope(tenant_id if tenant_id is not None else candidate.tenant_id)
    scope.require(scope.check_candidate(candidate))

    with uow_factory() as uow:
        events_by_id = _load_events(uow, candidate)
        evaluation = evaluate_candidate(
            candidate,
            events_by_id,
            current_facts=_load_current_facts(uow, candidate),
            now=now,
            policy_version=policy_version,
        )
        decision = evaluation.decision

        if decision.decision is WriteDecision.SUPERSEDE:
            old_memory_id = decision.superseded_memory_id
            assert old_memory_id is not None
            record = _build_memory_record(
                candidate,
                policy_version,
                trust_level=evaluation.effective_trust,
                status=MemoryStatus.ACTIVE,
                now=now,
            )
            decision = decision.model_copy(update={"accepted_memory_id": record.memory_id})
            uow.supersede_memory(
                candidate.tenant_id,
                old_memory_id,
                record,
                decision,
                rationale=decision.explanation,
            )
            uow.commit()
            return decision

        if decision.decision in _PERSISTED_DECISIONS:
            record = _build_memory_record(
                candidate,
                policy_version,
                trust_level=evaluation.effective_trust,
                status=_STATUS_BY_DECISION[decision.decision],
                now=now,
            )
            uow.create_memory(record)
            decision = decision.model_copy(update={"accepted_memory_id": record.memory_id})

        uow.record_write_decision(decision)
        uow.commit()
        return decision


def record_rejected_extraction(
    uow_factory: UowFactory,
    tenant_id: UUID,
    rejected: RejectedExtraction,
    *,
    policy_version: str = POLICY_VERSION,
) -> WritePolicyDecision:
    """Record a pre-policy rejection as a REJECT decision with its reason code.

    A `RejectedExtraction` never reaches write policy -- there is no
    verifiable evidence to evaluate -- but it is still a rejection, so it is
    audited like one. It never became a `MemoryCandidate`, so the decision
    gets a fresh `candidate_id`.

    Example:
        Input:
            rejected = RejectedExtraction("NO_SOURCE_EVENTS", "Extractor produced ...", raw)
        Output:
            WritePolicyDecision(decision=WriteDecision.REJECT,
                                reason_codes=["NO_SOURCE_EVENTS"], ...)
            -- and the decision is committed to `memory_write_decisions`.
    """
    decision = rejected_extraction_decision(tenant_id, rejected, policy_version=policy_version)
    with uow_factory() as uow:
        uow.record_write_decision(decision)
        uow.commit()
    return decision


def rejected_extraction_decision(
    tenant_id: UUID,
    rejected: RejectedExtraction,
    *,
    policy_version: str = POLICY_VERSION,
) -> WritePolicyDecision:
    """The REJECT decision `record_rejected_extraction` records, without persisting it."""
    return WritePolicyDecision(
        tenant_id=tenant_id,
        candidate_id=uuid4(),
        decision=WriteDecision.REJECT,
        policy_version=policy_version,
        reason_codes=[rejected.reason_code],
        explanation=rejected.explanation,
    )


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


def _load_current_facts(uow: UnitOfWork, candidate: MemoryCandidate) -> list[MemoryRecord]:
    """The tenant's active semantic facts, when the candidate states a fact at all.

    Most candidates are not "attribute is value" claims; for those the
    stale-fact check has nothing to compare, so no query is made.
    """
    if candidate.memory_type is MemoryType.PROCEDURAL:
        return []
    if parse_fact_claim(candidate.content, candidate.subject_keys) is None:
        return []
    return uow.records.list_active(candidate.tenant_id, MemoryType.SEMANTIC)


def _build_memory_record(
    candidate: MemoryCandidate,
    policy_version: str,
    *,
    trust_level: TrustLevel,
    status: MemoryStatus,
    now: datetime.datetime | None = None,
) -> MemoryRecord:
    """Turn an accepted or quarantined candidate into a `MemoryRecord`.

    How it works:
        Copies the candidate's content, confidence, provenance, memory type,
        subject keys, and metadata straight across, but deliberately does
        *not* reuse `candidate.proposed_trust_level` -- `trust_level` here is
        always the effective trust write policy derived from real evidence,
        passed in by the caller. If the candidate did not specify a
        `temporal_validity`, one is created starting at `now` (default: the
        current time) with no end date (open-ended validity), since
        `MemoryRecord` requires one.

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
        valid_from=now or datetime.datetime.now(datetime.UTC)
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

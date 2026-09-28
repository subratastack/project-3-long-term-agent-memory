"""Deterministic provenance verification for memory candidates.

A candidate's `proposed_trust_level` is self-reported (ultimately traceable to
an untrusted extractor once LLM-backed extraction exists), so it must never be
taken at face value. This module's job is to independently verify that every
piece of provenance a candidate cites actually exists, belongs to the
candidate's own tenant, and matches the real event it claims to be quoting --
and then to derive the candidate's trust as the weakest link across that
verified evidence, rather than trusting whatever the candidate claims about
itself. See ARCHITECTURE.md ("Trust and poisoning model") and
docs/adr/006-memory-trust-and-poisoning.md.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from uuid import UUID

from apps.memory_service.domain.enums import SourceType, TrustLevel
from apps.memory_service.domain.models import MemoryCandidate, MemoryEvent

# A total order over trust levels, used to compare two levels and to pick the
# weaker of several. Higher numbers are more trusted.
_TRUST_RANK: dict[TrustLevel, int] = {
    TrustLevel.UNTRUSTED: 0,
    TrustLevel.LOW: 1,
    TrustLevel.MEDIUM: 2,
    TrustLevel.HIGH: 3,
    TrustLevel.SYSTEM: 4,
}

# Baseline trust implied by the kind of component that produced an event,
# independent of whatever trust level a candidate's provenance attests to.
# The derived trust for a piece of evidence is the weaker of the two, so a
# candidate cannot inflate its trust merely by attesting a high trust_level
# for evidence that actually came from a low-trust source.
_SOURCE_BASELINE_TRUST: dict[SourceType, TrustLevel] = {
    SourceType.CONFIGURATION: TrustLevel.SYSTEM,
    SourceType.SYSTEM_EVENT: TrustLevel.SYSTEM,
    SourceType.AGENT_ACTION: TrustLevel.HIGH,
    SourceType.USER_MESSAGE: TrustLevel.MEDIUM,
    SourceType.TOOL_OUTPUT: TrustLevel.MEDIUM,
}

# Reason codes returned in a failed ProvenanceCheckResult, exported so callers
# (and tests) can check for a specific failure without hardcoding the string.
MISSING_PROVENANCE = "MISSING_PROVENANCE"
EVENT_NOT_FOUND = "EVENT_NOT_FOUND"
TENANT_MISMATCH = "TENANT_MISMATCH"
INCONSISTENT_PROVENANCE = "INCONSISTENT_PROVENANCE"


def trust_rank(level: TrustLevel) -> int:
    """Return a comparable rank for `level`, where a higher number means more trusted.

    How it works:
        Looks `level` up in the fixed `_TRUST_RANK` table above. Because that
        table assigns every `TrustLevel` member a distinct integer from 0
        (UNTRUSTED) to 4 (SYSTEM), comparing two trust levels reduces to
        comparing the two integers this function returns for them.

    Example:
        Input:
            level = TrustLevel.HIGH
        Output:
            3
    """
    return _TRUST_RANK[level]


def weakest_trust(levels: Iterable[TrustLevel]) -> TrustLevel:
    """Return the least-trusted level among `levels` (UNTRUSTED if empty).

    How it works:
        1. Materialize `levels` into a list, since it may be a one-shot
           iterator and it needs to be inspected twice (for emptiness, then
           by `min`).
        2. If the list is empty, there is nothing to compare, so return
           `TrustLevel.UNTRUSTED` as the safest possible default.
        3. Otherwise, return whichever level has the lowest `trust_rank` --
           i.e. the weakest link in the chain of evidence.

    Example:
        Input:
            levels = [TrustLevel.SYSTEM, TrustLevel.LOW, TrustLevel.HIGH]
        Output:
            TrustLevel.LOW
    """
    levels = list(levels)
    if not levels:
        return TrustLevel.UNTRUSTED
    return min(levels, key=trust_rank)


@dataclass(frozen=True)
class ProvenanceCheckResult:
    """The outcome of running `verify_provenance` on one candidate.

    `is_valid=False` means the candidate's provenance could not be trusted at
    all (see `reason_codes` for why), in which case `derived_trust_level` is
    always `UNTRUSTED` and callers should treat the candidate as rejected.
    `is_valid=True` means every provenance entry checked out, and
    `derived_trust_level` is the actual, evidence-backed trust to use for the
    candidate going forward.
    """

    is_valid: bool
    derived_trust_level: TrustLevel
    reason_codes: tuple[str, ...] = ()
    explanation: str | None = None


def verify_provenance(
    candidate: MemoryCandidate,
    events_by_id: Mapping[UUID, MemoryEvent],
) -> ProvenanceCheckResult:
    """Verify a candidate's provenance and derive its evidence-backed trust.

    `events_by_id` must contain only events the caller has already loaded and
    scoped to the right tenant (see `apps.memory_service.ingestion.service`);
    an event genuinely missing from that mapping -- because it does not
    exist, belongs to another tenant, or was simply never loaded -- is
    treated identically here: the referenced provenance cannot be verified.

    How it works:
        1. If the candidate has no provenance entries at all, fail
           immediately with `MISSING_PROVENANCE`. (In practice
           `MemoryCandidate` cannot be constructed with an empty provenance
           list, so this mostly guards against a candidate whose provenance
           references could not be resolved to any real event at all.)
        2. Otherwise, check each provenance entry one at a time:
           a. Look up its `event_id` in `events_by_id`. Not found means the
              event does not exist, or was filtered out because it belonged
              to a different tenant -- either way, record `EVENT_NOT_FOUND`
              and move on to the next entry.
           b. If the event's own `tenant_id` does not match the candidate's
              (a defensive check, in case a caller passed in an unscoped
              mapping), record `TENANT_MISMATCH`.
           c. If the provenance's `source_type`, `source_reference`, or
              `observed_at` do not match the real event's values, the
              candidate is quoting evidence it does not accurately describe;
              record `INCONSISTENT_PROVENANCE`.
           d. Otherwise, the entry is genuinely verified: compute its trust
              as the weaker of the event's `SourceType` baseline (see
              `_SOURCE_BASELINE_TRUST`) and the trust level the provenance
              itself attests to, and remember it.
        3. If any entry failed its checks, the whole candidate is rejected:
           return `is_valid=False` with the deduplicated set of reason codes
           collected in step 2 (order-preserving, via `dict.fromkeys`) and
           `derived_trust_level=UNTRUSTED`.
        4. If every entry passed, the candidate's overall trust is the
           weakest of all the per-entry trust levels computed in step 2(d) --
           one badly-sourced piece of evidence caps the whole candidate's
           trust, even if the rest of its evidence was solid.

    Example (verified, MEDIUM trust):
        Input:
            candidate = MemoryCandidate(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                content="The user prefers email over phone calls.",
                provenance=[Provenance(event_id=UUID("aaaaaaaa-..."),
                                        source_type=SourceType.USER_MESSAGE,
                                        source_reference="chat-42",
                                        observed_at=datetime(2026, 1, 5, tzinfo=UTC),
                                        trust_level=TrustLevel.MEDIUM)],
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
            )
            events_by_id = {UUID("aaaaaaaa-..."): MemoryEvent(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                source_type=SourceType.USER_MESSAGE,
                source_reference="chat-42",
                content="I prefer email over phone calls.",
                observed_at=datetime(2026, 1, 5, tzinfo=UTC),
            )}
        Output:
            ProvenanceCheckResult(is_valid=True, derived_trust_level=TrustLevel.MEDIUM)

    Example (unverifiable -- referenced event was never loaded):
        Input:
            candidate = <same candidate as above>
            events_by_id = {}
        Output:
            ProvenanceCheckResult(
                is_valid=False,
                derived_trust_level=TrustLevel.UNTRUSTED,
                reason_codes=("EVENT_NOT_FOUND",),
                explanation="One or more provenance entries could not be verified.",
            )
    """
    if not candidate.provenance:
        return ProvenanceCheckResult(
            is_valid=False,
            derived_trust_level=TrustLevel.UNTRUSTED,
            reason_codes=(MISSING_PROVENANCE,),
            explanation="Candidate has no supporting provenance.",
        )

    reason_codes: list[str] = []
    verified_trust: list[TrustLevel] = []

    for provenance in candidate.provenance:
        event = events_by_id.get(provenance.event_id)
        if event is None:
            reason_codes.append(EVENT_NOT_FOUND)
            continue
        if event.tenant_id != candidate.tenant_id:
            reason_codes.append(TENANT_MISMATCH)
            continue
        if (
            event.source_type != provenance.source_type
            or event.source_reference != provenance.source_reference
            or event.observed_at != provenance.observed_at
        ):
            reason_codes.append(INCONSISTENT_PROVENANCE)
            continue

        baseline = _SOURCE_BASELINE_TRUST.get(event.source_type, TrustLevel.UNTRUSTED)
        verified_trust.append(weakest_trust([baseline, provenance.trust_level]))

    if reason_codes:
        return ProvenanceCheckResult(
            is_valid=False,
            derived_trust_level=TrustLevel.UNTRUSTED,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            explanation="One or more provenance entries could not be verified.",
        )

    return ProvenanceCheckResult(
        is_valid=True,
        derived_trust_level=weakest_trust(verified_trust),
    )

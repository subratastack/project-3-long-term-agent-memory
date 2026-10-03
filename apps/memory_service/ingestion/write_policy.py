"""Deterministic write-policy engine.

This module makes ingestion decisions and nothing else: it never touches the
database, never calls an LLM, and always returns the same `WritePolicyDecision`
for the same candidate, evidence, current facts, and time. See ARCHITECTURE.md
("Write policy is deterministic and separate from LLM reasoning") and
docs/adr/002-memory-types-and-promotion.md /
docs/adr/006-memory-trust-and-poisoning.md.

How a decision is made:

1. Provenance must verify (`verify_provenance`); if it does not, the
   candidate is rejected and nothing else runs on unverified evidence.
2. Every rule then reports `PolicyFinding`s -- a reason code plus the outcome
   it calls for:
   - content poisoning (`security.poisoning.assess_poisoning`),
   - evidence trust (`security.trust.assess_trust`: model claims, forged
     approvals),
   - stale or conflicting facts (`security.trust.check_currency`),
   - the rule set for the candidate's memory type.
3. The most severe outcome wins (REJECT > QUARANTINE > SUPERSEDE > ACCEPT),
   and the decision records the reason code of every finding that argued for
   holding the candidate back -- or, for an accepted one, every reason it
   was accepted. A decision never leaves this module without a reason code.

SUPERSEDE is returned only when a semantic candidate carries newer,
at-least-as-trusted, non-tool evidence for a fact that is currently active
with a different value; `ingestion.service` then retires the old fact via
`UnitOfWork.supersede_memory`.
"""

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from apps.memory_service.domain.enums import MemoryType, TrustLevel, WriteDecision
from apps.memory_service.domain.models import (
    MemoryCandidate,
    MemoryEvent,
    MemoryRecord,
    WritePolicyDecision,
)
from apps.memory_service.ingestion.provenance import trust_rank, verify_provenance
from apps.memory_service.security.poisoning import ESCALATE_FOR_PROCEDURES, assess_poisoning
from apps.memory_service.security.trust import (
    PolicyFinding,
    TrustAssessment,
    assess_trust,
    check_currency,
    severity,
)

# 1.1: findings-based evaluation with poisoning, model-claim, approval,
# independence, and stale-fact checks (security.trust / security.poisoning).
POLICY_VERSION = "1.1"

# Procedural memory shapes future agent behavior, so ADR-002/ARCHITECTURE.md
# require it to be supported by repeated episodes (or an authorized approval)
# before it can be admitted at all. Episodes are counted as independent
# sources (`security.trust.independent_sources`), so one tool result replayed
# many times still counts once.
MIN_SUPPORTING_EPISODES_FOR_PROCEDURAL = 2

POISONED_PROCEDURE = "POISONED_PROCEDURE"


@dataclass(frozen=True)
class PolicyEvaluation:
    """A decision plus what it was based on.

    `effective_trust` is the trust a persisted record should carry: the
    evidence-backed trust from `assess_trust`, or UNTRUSTED when any
    poisoning finding was raised. `findings` lists every rule's verdict,
    including ones that did not decide the outcome.
    """

    decision: WritePolicyDecision
    effective_trust: TrustLevel
    findings: tuple[PolicyFinding, ...]


def evaluate_write_policy(
    candidate: MemoryCandidate,
    events_by_id: Mapping[UUID, MemoryEvent],
    *,
    current_facts: Sequence[MemoryRecord] = (),
    now: datetime.datetime | None = None,
    model_extracted: bool = True,
    policy_version: str = POLICY_VERSION,
) -> WritePolicyDecision:
    """Evaluate a candidate deterministically into a `WritePolicyDecision`.

    The same as `evaluate_candidate(...).decision`; see there for how it works.

    Example:
        Input:
            candidate = MemoryCandidate(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                content="The nightly backup completed successfully.",
                provenance=[Provenance(event_id=UUID("aaaaaaaa-..."),
                                        source_type=SourceType.SYSTEM_EVENT,
                                        source_reference="session-1",
                                        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
                                        trust_level=TrustLevel.SYSTEM)],
                memory_type=MemoryType.EPISODIC,
                confidence=0.95,
            )
            events_by_id = {UUID("aaaaaaaa-..."): <matching MemoryEvent>}
        Output:
            WritePolicyDecision(
                decision=WriteDecision.ACCEPT,
                reason_codes=["TRUSTED_EPISODIC_EVIDENCE"],
                explanation="Episodic memory is backed by sufficiently trusted evidence.",
            )
    """
    return evaluate_candidate(
        candidate,
        events_by_id,
        current_facts=current_facts,
        now=now,
        model_extracted=model_extracted,
        policy_version=policy_version,
    ).decision


def evaluate_candidate(
    candidate: MemoryCandidate,
    events_by_id: Mapping[UUID, MemoryEvent],
    *,
    current_facts: Sequence[MemoryRecord] = (),
    now: datetime.datetime | None = None,
    model_extracted: bool = True,
    policy_version: str = POLICY_VERSION,
) -> PolicyEvaluation:
    """Run every write-policy rule on `candidate` and combine the findings.

    `events_by_id` must hold the candidate's cited events, already loaded and
    scoped to its tenant. `current_facts` are the tenant's active semantic
    memories to check for stale or conflicting values (empty skips that
    check). `now` defaults to the current time. `model_extracted` says the
    candidate's wording came from an extractor rather than a deterministic
    template, which enables the unsupported-value check.

    How it works:
        1. Verify provenance. Invalid provenance is an immediate REJECT with
           the verification's reason codes; no other rule ever sees
           unverified evidence.
        2. Assess the evidence (`assess_trust`) and the content
           (`assess_poisoning`), check the candidate against current facts
           (`check_currency`), and apply the rule set for its memory type
           -- each step contributing findings.
        3. Apply the most severe outcome and record its reasons (`_decide`).

    Example (stale fact replayed as current):
        Input:
            candidate.content = "The request timeout is 5 seconds."
            cited event = CONFIGURATION snapshot observed 2026-01-01
            current_facts = ["The request timeout is 2 seconds.", SYSTEM trust,
                             evidence observed 2026-03-01]
        Output:
            PolicyEvaluation(
                decision=WritePolicyDecision(decision=WriteDecision.REJECT,
                                             reason_codes=["STALE_FACT_CLAIMED_AS_CURRENT"]),
                effective_trust=TrustLevel.SYSTEM, findings=(...))
    """
    provenance_result = verify_provenance(candidate, events_by_id)
    if not provenance_result.is_valid:
        explanation = provenance_result.explanation or "Provenance verification failed."
        findings = tuple(
            PolicyFinding(code, WriteDecision.REJECT, explanation)
            for code in provenance_result.reason_codes
        )
        return PolicyEvaluation(
            _decide(candidate, findings, policy_version), TrustLevel.UNTRUSTED, findings
        )

    trust = assess_trust(candidate, events_by_id)
    poisoning = assess_poisoning(candidate, events_by_id, model_extracted=model_extracted)
    findings = (
        *poisoning,
        *trust.findings,
        *check_currency(
            candidate, trust, current_facts, now=now or datetime.datetime.now(datetime.UTC)
        ),
        *_type_rules(candidate, trust, poisoning),
    )
    effective_trust = TrustLevel.UNTRUSTED if poisoning else trust.effective_trust
    return PolicyEvaluation(_decide(candidate, findings, policy_version), effective_trust, findings)


def _type_rules(
    candidate: MemoryCandidate,
    trust: TrustAssessment,
    poisoning: Sequence[PolicyFinding],
) -> tuple[PolicyFinding, ...]:
    if candidate.memory_type is MemoryType.EPISODIC:
        return (_evaluate_episodic(trust.effective_trust),)
    if candidate.memory_type is MemoryType.SEMANTIC:
        return (_evaluate_semantic(trust.effective_trust),)
    return _evaluate_procedural(trust, poisoning)


def _evaluate_episodic(trust: TrustLevel) -> PolicyFinding:
    """Decide an EPISODIC candidate: accept if trust clears the MEDIUM bar.

    How it works:
        Episodic memories describe things that were actually observed, so
        the bar for accepting them outright is comparatively low: anything
        at MEDIUM trust or above (ranked via `trust_rank`) is accepted as-is.
        Anything weaker is quarantined rather than rejected, since a
        low-trust observation may still be worth keeping around for review
        rather than discarding entirely.

    Example:
        Input:
            trust = TrustLevel.LOW
        Output:
            PolicyFinding("LOW_TRUST_EPISODIC_EVIDENCE", WriteDecision.QUARANTINE, ...)
    """
    if trust_rank(trust) >= trust_rank(TrustLevel.MEDIUM):
        return PolicyFinding(
            "TRUSTED_EPISODIC_EVIDENCE",
            WriteDecision.ACCEPT,
            "Episodic memory is backed by sufficiently trusted evidence.",
        )
    return PolicyFinding(
        "LOW_TRUST_EPISODIC_EVIDENCE",
        WriteDecision.QUARANTINE,
        "Episodic memory is backed by low-trust or untrusted evidence.",
    )


def _evaluate_semantic(trust: TrustLevel) -> PolicyFinding:
    """Decide a SEMANTIC candidate: accept if trust clears the MEDIUM bar.

    How it works:
        Same MEDIUM-trust threshold as episodic memories, applied to a
        distilled claim rather than a raw observation: at MEDIUM trust or
        above it is accepted, and anything weaker is quarantined so a
        low-trust claim can be reviewed instead of silently becoming an
        authoritative fact.

    Example:
        Input:
            trust = TrustLevel.MEDIUM
        Output:
            PolicyFinding("TRUSTED_SEMANTIC_CLAIM", WriteDecision.ACCEPT, ...)
    """
    if trust_rank(trust) >= trust_rank(TrustLevel.MEDIUM):
        return PolicyFinding(
            "TRUSTED_SEMANTIC_CLAIM",
            WriteDecision.ACCEPT,
            "Semantic claim is backed by sufficiently trusted evidence.",
        )
    return PolicyFinding(
        "LOW_TRUST_SEMANTIC_CLAIM",
        WriteDecision.QUARANTINE,
        "Semantic claim quarantined pending review of low-trust evidence.",
    )


def _evaluate_procedural(
    trust: TrustAssessment, poisoning: Sequence[PolicyFinding]
) -> tuple[PolicyFinding, ...]:
    """Decide a PROCEDURAL candidate: the strictest rule set of the three.

    How it works:
        1. Enough independent evidence: at least
           `MIN_SUPPORTING_EPISODES_FOR_PROCEDURAL` (2) independent sources
           (`TrustAssessment.independent_sources` -- replays of the same
           source or the same text count once), or an authorized approval:
           a cited CONFIGURATION/SYSTEM_EVENT event with `approved_by` set.
           An approval flag in candidate metadata does not count. Otherwise
           REJECT -- one observation is not yet a "procedure".
        2. Any poisoning finding that would merely quarantine another memory
           type (injection, store-this directives, dangerous operations)
           rejects a procedure outright (`POISONED_PROCEDURE`), so repeated
           poisoned content can never be promoted.
        3. Only HIGH trust or above is accepted. Because trust is the weakest
           link and capped per evidence kind, HIGH means every cited event is
           a platform record or a runtime-verified action -- no user
           statement, tool output, or model claim.
        4. Evidence that clears steps 1-2 but not step 3 is quarantined, so a
           repeated-but-unverified procedure can still be reviewed.

    Example (rejected -- only one supporting episode):
        Input:
            trust.independent_sources = 1, trust.authorized_approval = False
        Output:
            (PolicyFinding("INSUFFICIENT_SUPPORTING_EPISODES", WriteDecision.REJECT, ...),
             PolicyFinding("TRUSTED_PROCEDURAL_PROMOTION", ...))   # outvoted by the REJECT

    Example (quarantined -- enough episodes, not enough trust):
        Input:
            trust.independent_sources = 2, trust.effective_trust = TrustLevel.MEDIUM
        Output:
            (PolicyFinding("UNTRUSTED_PROCEDURAL_PROMOTION", WriteDecision.QUARANTINE, ...),)
    """
    findings: list[PolicyFinding] = []
    if (
        trust.independent_sources < MIN_SUPPORTING_EPISODES_FOR_PROCEDURAL
        and not trust.authorized_approval
    ):
        findings.append(
            PolicyFinding(
                "INSUFFICIENT_SUPPORTING_EPISODES",
                WriteDecision.REJECT,
                "Procedural memory requires repeated, independent supporting episodes "
                "or an authorized approval before promotion.",
            )
        )
    if any(finding.code in ESCALATE_FOR_PROCEDURES for finding in poisoning):
        findings.append(
            PolicyFinding(
                POISONED_PROCEDURE,
                WriteDecision.REJECT,
                "Procedural memory cannot be built from evidence carrying injected "
                "instructions or dangerous operations.",
            )
        )
    if trust_rank(trust.effective_trust) >= trust_rank(TrustLevel.HIGH):
        findings.append(
            PolicyFinding(
                "TRUSTED_PROCEDURAL_PROMOTION",
                WriteDecision.ACCEPT,
                "Procedural memory promoted from repeated, highly trusted episodes.",
            )
        )
    else:
        findings.append(
            PolicyFinding(
                "UNTRUSTED_PROCEDURAL_PROMOTION",
                WriteDecision.QUARANTINE,
                "Procedural memory quarantined pending review of insufficiently trusted evidence.",
            )
        )
    return tuple(findings)


def _decide(
    candidate: MemoryCandidate,
    findings: Sequence[PolicyFinding],
    policy_version: str,
) -> WritePolicyDecision:
    """Turn findings into the one `WritePolicyDecision` that gets recorded.

    How it works:
        1. The outcome is that of the most severe finding.
        2. For REJECT or QUARANTINE, the reason codes are every finding that
           argued for holding the candidate back (most severe first); a
           finding that would have accepted it is left out, so a rejected
           decision never lists "trusted" as one of its reasons. For ACCEPT
           or SUPERSEDE, every finding is a reason it was admitted.
        3. The explanation joins the explanations of the findings with the
           winning outcome; a SUPERSEDE also records the memory it retires
           in `superseded_memory_id`.

    Example:
        Input:
            findings = [PolicyFinding("SAFETY_POLICY_TAMPERING", REJECT, "..."),
                        PolicyFinding("TRUSTED_SEMANTIC_CLAIM", ACCEPT, "...")]
        Output:
            WritePolicyDecision(decision=WriteDecision.REJECT,
                                reason_codes=["SAFETY_POLICY_TAMPERING"], ...)
    """
    outcome = max((finding.outcome for finding in findings), key=severity)
    if severity(outcome) >= severity(WriteDecision.QUARANTINE):
        reasons = [f for f in findings if severity(f.outcome) >= severity(WriteDecision.QUARANTINE)]
    else:
        reasons = list(findings)
    reasons.sort(key=lambda finding: severity(finding.outcome), reverse=True)
    superseded = next(
        (f.related_memory_id for f in findings if f.outcome is WriteDecision.SUPERSEDE), None
    )
    return WritePolicyDecision(
        tenant_id=candidate.tenant_id,
        candidate_id=candidate.candidate_id,
        decision=outcome,
        policy_version=policy_version,
        reason_codes=list(dict.fromkeys(f.code for f in reasons)),
        explanation=" ".join(dict.fromkeys(f.explanation for f in reasons if f.outcome is outcome)),
        superseded_memory_id=superseded if outcome is WriteDecision.SUPERSEDE else None,
    )

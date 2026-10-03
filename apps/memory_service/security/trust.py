"""Evidence-backed trust: what a candidate's evidence can actually vouch for.

`ingestion.provenance.verify_provenance` answers "does this evidence exist,
belong to this tenant, and match what the candidate says about it?", and
derives a trust level from each event's source type. Write policy needs more
than that before it lets anything become active memory (ADR-006):

- **What kind of evidence is it?** An agent-action event the runtime watched
  complete is execution evidence. One the runtime did not verify is just the
  model's own words: a claim, not an observation.
- **How many independent sources support it?** The same tool result replayed
  ten times is one source, not ten.
- **Was a procedure approved by someone allowed to approve it?** An approval
  flag the extractor wrote into candidate metadata is not an approval.
- **Does the candidate restate a fact that newer or more trusted evidence has
  already settled?** A stale value presented as current must not become
  active next to (or instead of) the trusted current one.

Everything here is pure and deterministic, like write policy itself: no
database, no model, and the same inputs always give the same answer. The
findings it produces are combined with `security.poisoning`'s content checks
by `ingestion.write_policy`, which alone turns them into a decision.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from apps.memory_service.domain.enums import (
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)
from apps.memory_service.domain.models import MemoryCandidate, MemoryEvent, MemoryRecord
from apps.memory_service.ingestion.provenance import trust_rank, verify_provenance, weakest_trust

# --- Event metadata written by the agent runtime -------------------------
#
# These keys live on `MemoryEvent.metadata`, which the runtime (or the API
# caller acting as the runtime) sets when it records an event. Extractors
# never write event metadata, so a model cannot forge either signal.

# Set to `True` on an AGENT_ACTION event once the runtime has observed the
# action actually run (e.g. the tool call returned). Without it, an agent
# event is only the model's own statement.
RUNTIME_VERIFIED_KEY = "runtime_verified"

# A non-empty string (who approved it) on a CONFIGURATION or SYSTEM_EVENT
# event records an authorized approval of the procedure that cites it.
APPROVED_BY_KEY = "approved_by"

# The candidate-metadata flag older code treated as approval. Candidate
# metadata comes from the extractor, so this is only ever a *claim*.
CLAIMED_APPROVAL_KEY = "explicitly_approved"

# --- Reason codes ---------------------------------------------------------

UNVERIFIED_MODEL_CLAIM = "UNVERIFIED_MODEL_CLAIM"
UNAUTHORIZED_APPROVAL_CLAIM = "UNAUTHORIZED_APPROVAL_CLAIM"
STALE_FACT_CLAIMED_AS_CURRENT = "STALE_FACT_CLAIMED_AS_CURRENT"
CONFLICTS_WITH_CURRENT_FACT = "CONFLICTS_WITH_CURRENT_FACT"
NEWER_FACT_SUPERSEDES_CURRENT = "NEWER_FACT_SUPERSEDES_CURRENT"


@dataclass(frozen=True)
class PolicyFinding:
    """One rule's verdict on a candidate: a reason code and the outcome it calls for.

    Write policy collects every finding for a candidate and applies the most
    severe outcome (`severity`). `related_memory_id` names the existing
    memory a finding is about, e.g. the fact a SUPERSEDE would retire.
    """

    code: str
    outcome: WriteDecision
    explanation: str
    related_memory_id: UUID | None = None


_SEVERITY: dict[WriteDecision, int] = {
    WriteDecision.ACCEPT: 0,
    WriteDecision.SUPERSEDE: 1,
    WriteDecision.QUARANTINE: 2,
    WriteDecision.REJECT: 3,
}


def severity(outcome: WriteDecision) -> int:
    """Rank outcomes so the most protective one wins: REJECT > QUARANTINE > SUPERSEDE > ACCEPT."""
    return _SEVERITY[outcome]


class EvidenceKind(StrEnum):
    """What a cited event can vouch for, beyond its raw source type.

    - PLATFORM_RECORD: configuration or a system event -- recorded by the
      platform itself.
    - VERIFIED_EXECUTION: an agent action the runtime observed completing.
    - USER_STATEMENT: something a user said.
    - EXTERNAL_CONTENT: tool output -- text a third party may control.
    - MODEL_CLAIM: an agent event the runtime did not verify: the model's
      own words, which may be a hallucination.
    """

    PLATFORM_RECORD = "platform_record"
    VERIFIED_EXECUTION = "verified_execution"
    USER_STATEMENT = "user_statement"
    EXTERNAL_CONTENT = "external_content"
    MODEL_CLAIM = "model_claim"


# Evidence a third party or the model itself can put words into. It may
# support a new memory, but never replaces an existing fact automatically.
UNVERIFIED_ORIGINS: frozenset[EvidenceKind] = frozenset(
    {EvidenceKind.EXTERNAL_CONTENT, EvidenceKind.MODEL_CLAIM}
)

# The most trust each kind can ever confer, whatever its provenance attests.
# This only ever lowers `verify_provenance`'s per-source baseline: the one
# real change is MODEL_CLAIM, which is capped below the MEDIUM bar every
# memory type needs for acceptance.
_KIND_TRUST_CEILING: dict[EvidenceKind, TrustLevel] = {
    EvidenceKind.PLATFORM_RECORD: TrustLevel.SYSTEM,
    EvidenceKind.VERIFIED_EXECUTION: TrustLevel.HIGH,
    EvidenceKind.USER_STATEMENT: TrustLevel.MEDIUM,
    EvidenceKind.EXTERNAL_CONTENT: TrustLevel.MEDIUM,
    EvidenceKind.MODEL_CLAIM: TrustLevel.LOW,
}


def evidence_kind(event: MemoryEvent) -> EvidenceKind:
    """Classify one event by what it can vouch for.

    Example:
        Input:
            event = MemoryEvent(source_type=SourceType.AGENT_ACTION,
                                content="The prod database runs PostgreSQL 12.",
                                metadata={})
        Output:
            EvidenceKind.MODEL_CLAIM   # no `runtime_verified` marker
    """
    if event.source_type in (SourceType.CONFIGURATION, SourceType.SYSTEM_EVENT):
        return EvidenceKind.PLATFORM_RECORD
    if event.source_type is SourceType.AGENT_ACTION:
        if event.metadata.get(RUNTIME_VERIFIED_KEY) is True:
            return EvidenceKind.VERIFIED_EXECUTION
        return EvidenceKind.MODEL_CLAIM
    if event.source_type is SourceType.USER_MESSAGE:
        return EvidenceKind.USER_STATEMENT
    return EvidenceKind.EXTERNAL_CONTENT


def is_authorized_approval(event: MemoryEvent) -> bool:
    """True if `event` is a platform record carrying an `approved_by` value."""
    approver = event.metadata.get(APPROVED_BY_KEY)
    return (
        evidence_kind(event) is EvidenceKind.PLATFORM_RECORD
        and isinstance(approver, str)
        and bool(approver.strip())
    )


@dataclass(frozen=True)
class TrustAssessment:
    """Everything write policy needs to know about a candidate's evidence.

    `effective_trust` is `verify_provenance`'s weakest-link trust, further
    capped by each evidence kind's ceiling. `independent_sources` counts
    distinct sources, collapsing replays of one source.
    `findings` holds the trust-related reasons to hold the candidate back.
    """

    effective_trust: TrustLevel
    evidence_kinds: frozenset[EvidenceKind]
    independent_sources: int
    authorized_approval: bool
    latest_evidence_at: datetime.datetime
    findings: tuple[PolicyFinding, ...]


def assess_trust(
    candidate: MemoryCandidate, events_by_id: Mapping[UUID, MemoryEvent]
) -> TrustAssessment:
    """Assess what `candidate`'s cited evidence can vouch for.

    How it works:
        1. Re-run `verify_provenance` for the weakest-link trust derived from
           each event's source type (UNTRUSTED if the provenance is invalid).
        2. Classify every cited, same-tenant event (`evidence_kind`) and cap
           the trust at the lowest ceiling among those kinds -- one model
           claim caps the whole candidate at LOW.
        3. Count independent sources (`independent_sources`) and look for an
           authorized approval among the cited events.
        4. Report UNVERIFIED_MODEL_CLAIM if any cited event is a model claim,
           and UNAUTHORIZED_APPROVAL_CLAIM if the candidate claims approval
           that no cited event records. Both call for QUARANTINE.

    Example:
        Input:
            candidate.content = "The checkout-api database runs PostgreSQL 12."
            cited event = MemoryEvent(source_type=AGENT_ACTION, metadata={})
        Output:
            TrustAssessment(effective_trust=TrustLevel.LOW,
                            evidence_kinds=frozenset({EvidenceKind.MODEL_CLAIM}),
                            independent_sources=1, authorized_approval=False,
                            findings=(PolicyFinding("UNVERIFIED_MODEL_CLAIM",
                                                    WriteDecision.QUARANTINE, ...),))
    """
    provenance = verify_provenance(candidate, events_by_id)
    events = cited_events(candidate, events_by_id)
    kinds = frozenset(evidence_kind(event) for event in events)
    effective = weakest_trust(
        [provenance.derived_trust_level, *(_KIND_TRUST_CEILING[kind] for kind in kinds)]
    )
    approved = any(is_authorized_approval(event) for event in events)

    findings: list[PolicyFinding] = []
    if EvidenceKind.MODEL_CLAIM in kinds:
        findings.append(
            PolicyFinding(
                UNVERIFIED_MODEL_CLAIM,
                WriteDecision.QUARANTINE,
                "Candidate rests on model-generated statements the runtime did not verify.",
            )
        )
    if candidate.metadata.get(CLAIMED_APPROVAL_KEY) and not approved:
        findings.append(
            PolicyFinding(
                UNAUTHORIZED_APPROVAL_CLAIM,
                WriteDecision.QUARANTINE,
                "Candidate claims an approval that no authorized evidence records.",
            )
        )

    return TrustAssessment(
        effective_trust=effective,
        evidence_kinds=kinds,
        independent_sources=independent_sources(events),
        authorized_approval=approved,
        latest_evidence_at=max(utc(p.observed_at) for p in candidate.provenance),
        findings=tuple(findings),
    )


def cited_events(
    candidate: MemoryCandidate, events_by_id: Mapping[UUID, MemoryEvent]
) -> list[MemoryEvent]:
    """The candidate's cited events that resolve to the candidate's own tenant."""
    events: list[MemoryEvent] = []
    for provenance in candidate.provenance:
        event = events_by_id.get(provenance.event_id)
        if event is not None and event.tenant_id == candidate.tenant_id:
            events.append(event)
    return events


def independent_sources(events: Iterable[MemoryEvent]) -> int:
    """Count genuinely separate sources among `events`.

    Two events are one source if they share a source type and
    `source_reference` -- the same tool call, message, or run replayed as
    several events. Repetition from one source is not corroboration.

    Example:
        Input:
            events = [TOOL_OUTPUT "web-fetch-77", TOOL_OUTPUT "web-fetch-77",
                      SYSTEM_EVENT "incident-7"]
        Output:
            2   # the replayed tool result counts once
    """
    return len({(event.source_type, event.source_reference) for event in events})


# --- Stale facts ------------------------------------------------------------


@dataclass(frozen=True)
class FactClaim:
    """A single "attribute has value" claim parsed from memory content.

    `subjects` are the memory's subject keys; two claims are about the same
    fact when their keys match and their subjects don't rule it out
    (`same_fact`).
    """

    key: str
    value: str
    subjects: frozenset[str]


# `name=value`, as in "timeout=2s" or "checkout.retries = 3".
_ASSIGNMENT = re.compile(r"(?<![\w.])([A-Za-z_][\w.\-]*)\s*=\s*([^\s,;]+)")
# "The request timeout is 5 seconds." / "Retries are now set to 3."
_STATEMENT = re.compile(
    r"^(?:the\s+)?(?P<key>[a-z0-9][a-z0-9 _'.\-]{0,60}?)\s+(?:is|are)\s+"
    r"(?:now\s+|currently\s+)?(?:set\s+to\s+)?(?P<value>.+?)[.!]?$",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"[.!?]\s+\S")
_DURATION = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?)$"
)
_DURATION_UNITS = {
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "m": "min",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
}


def parse_fact_claim(content: str, subject_keys: Iterable[str] = ()) -> FactClaim | None:
    """Parse `content` as one "attribute has value" claim, or return None.

    How it works:
        1. If the text contains exactly one `name=value` assignment, that is
           the claim (several assignments are not one claim).
        2. Otherwise, a single sentence of the form "[The] <attribute>
           is/are [now] [set to] <value>" is the claim. Multi-sentence text
           is not parsed.
        3. Keys are lowercased with "the", possessive "'s", underscores, and
           repeated spaces normalized away; values are lowercased, and a
           duration is canonicalized ("5 seconds" -> "5s") so a formatting
           difference is never mistaken for a different value.

    Example:
        Input:
            content = "The request timeout is 5 seconds."
            subject_keys = ["service:checkout-api"]
        Output:
            FactClaim(key="request timeout", value="5s",
                      subjects=frozenset({"service:checkout-api"}))
    """
    text = " ".join(content.split())
    assignments = _ASSIGNMENT.findall(text)
    if len(assignments) > 1:
        return None
    if assignments:
        key, value = assignments[0]
    else:
        if _SENTENCE_BREAK.search(text):
            return None
        match = _STATEMENT.match(text)
        if match is None:
            return None
        key, value = match.group("key"), match.group("value")
    normalized_key = _normalize_key(key)
    normalized_value = _normalize_value(value)
    if not normalized_key or not normalized_value:
        return None
    subjects = frozenset(s.strip().lower() for s in subject_keys if s.strip())
    return FactClaim(normalized_key, normalized_value, subjects)


def same_fact(first: FactClaim, second: FactClaim) -> bool:
    """True when two claims are about the same attribute of the same subject.

    Keys must match. Subject keys narrow the match only when *both* claims
    have them: then at least one must be shared. A claim without subject
    keys is compared against every claim with the same key, so dropping the
    subject keys cannot slip a stale value past this check.
    """
    if first.key != second.key:
        return False
    if first.subjects and second.subjects:
        return bool(first.subjects & second.subjects)
    return True


def check_currency(
    candidate: MemoryCandidate,
    assessment: TrustAssessment,
    current_facts: Sequence[MemoryRecord],
    *,
    now: datetime.datetime,
) -> tuple[PolicyFinding, ...]:
    """Find current facts the candidate contradicts, and say what that means.

    Only an episodic or semantic candidate whose content parses as a fact
    claim (`parse_fact_claim`) and that claims to hold now or later is
    checked; a claim whose validity window has already closed is history,
    not a claim about the present. It is compared with every ACTIVE,
    in-effect SEMANTIC fact of the same tenant with the same identity
    (`same_fact`) but a different value:

    | candidate evidence | candidate trust | outcome |
    | --- | --- | --- |
    | older | lower or equal | REJECT `STALE_FACT_CLAIMED_AS_CURRENT` |
    | newer | higher or equal, no tool/model evidence | SUPERSEDE `NEWER_FACT_SUPERSEDES_CURRENT` |
    | anything else | | QUARANTINE `CONFLICTS_WITH_CURRENT_FACT` |

    "Older"/"newer" compare the latest `observed_at` among each side's
    evidence. A candidate that would supersede more than one fact at once is
    quarantined instead: that needs a person to decide.

    Example:
        Input:
            candidate = "The request timeout is 5 seconds." (CONFIGURATION,
                        observed 2026-01-01, trust SYSTEM)
            current_facts = ["The request timeout is 2 seconds." (CONFIGURATION,
                             observed 2026-03-01, trust SYSTEM)]
        Output:
            (PolicyFinding("STALE_FACT_CLAIMED_AS_CURRENT", WriteDecision.REJECT, ...,
                           related_memory_id=<2s fact>),)
    """
    if candidate.memory_type is MemoryType.PROCEDURAL or not _claims_current(candidate, now):
        return ()
    claim = parse_fact_claim(candidate.content, candidate.subject_keys)
    if claim is None:
        return ()

    findings: list[PolicyFinding] = []
    for fact in current_facts:
        if (
            fact.tenant_id != candidate.tenant_id
            or fact.memory_type is not MemoryType.SEMANTIC
            or fact.status is not MemoryStatus.ACTIVE
            or not _in_effect(fact, now)
        ):
            continue
        existing = parse_fact_claim(fact.content, fact.subject_keys)
        if existing is None or not same_fact(claim, existing) or existing.value == claim.value:
            continue
        findings.append(_judge_conflict(candidate, assessment, fact, now))

    if sum(f.outcome is WriteDecision.SUPERSEDE for f in findings) > 1:
        findings = [f for f in findings if f.outcome is not WriteDecision.SUPERSEDE]
        findings.append(
            PolicyFinding(
                CONFLICTS_WITH_CURRENT_FACT,
                WriteDecision.QUARANTINE,
                "Candidate would replace several current facts at once; review required.",
            )
        )
    return tuple(findings)


def _judge_conflict(
    candidate: MemoryCandidate,
    assessment: TrustAssessment,
    fact: MemoryRecord,
    now: datetime.datetime,
) -> PolicyFinding:
    candidate_at = assessment.latest_evidence_at
    fact_at = latest_evidence_at(fact)
    candidate_rank = trust_rank(assessment.effective_trust)
    fact_rank = trust_rank(fact.trust_level)

    if candidate_at < fact_at and candidate_rank <= fact_rank:
        return PolicyFinding(
            STALE_FACT_CLAIMED_AS_CURRENT,
            WriteDecision.REJECT,
            "Candidate presents an older value as current; newer, at-least-as-trusted "
            "evidence already settles this fact.",
            fact.memory_id,
        )
    candidate_from = (
        utc(candidate.temporal_validity.valid_from) if candidate.temporal_validity else now
    )
    if (
        candidate_at > fact_at
        and candidate_rank >= fact_rank
        and not assessment.evidence_kinds & UNVERIFIED_ORIGINS
        and candidate_from >= utc(fact.temporal_validity.valid_from)
    ):
        return PolicyFinding(
            NEWER_FACT_SUPERSEDES_CURRENT,
            WriteDecision.SUPERSEDE,
            "Newer, at-least-as-trusted evidence replaces the current value of this fact.",
            fact.memory_id,
        )
    return PolicyFinding(
        CONFLICTS_WITH_CURRENT_FACT,
        WriteDecision.QUARANTINE,
        "Candidate contradicts a current fact it cannot replace automatically; review required.",
        fact.memory_id,
    )


def latest_evidence_at(record: MemoryRecord) -> datetime.datetime:
    """The newest `observed_at` among a record's evidence (its `valid_from` if it has none)."""
    if not record.provenance:
        return utc(record.temporal_validity.valid_from)
    return max(utc(p.observed_at) for p in record.provenance)


def utc(value: datetime.datetime) -> datetime.datetime:
    """Treat a naive timestamp as UTC so evidence times always compare."""
    return value.replace(tzinfo=datetime.UTC) if value.tzinfo is None else value


def _claims_current(candidate: MemoryCandidate, now: datetime.datetime) -> bool:
    validity = candidate.temporal_validity
    return validity is None or validity.valid_to is None or utc(validity.valid_to) > now


def _in_effect(record: MemoryRecord, now: datetime.datetime) -> bool:
    validity = record.temporal_validity
    return utc(validity.valid_from) <= now and (
        validity.valid_to is None or now < utc(validity.valid_to)
    )


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"^the\s+", "", key)
    key = key.replace("'s ", " ").replace("_", " ")
    return " ".join(key.split())


def _normalize_value(value: str) -> str:
    value = " ".join(value.strip().lower().split()).rstrip(".!").strip("'\"")
    value = re.sub(r"^(?:now|currently)\s+", "", value)
    duration = _DURATION.match(value)
    if duration is not None:
        number = duration.group(1)
        if "." in number:
            number = number.rstrip("0").rstrip(".")
        return f"{number}{_DURATION_UNITS[duration.group(2)]}"
    return value

"""A stale value presented as current must never become active next to the trusted one.

The scenario throughout: checkout-api's request timeout was 5 seconds
(configuration, January) and is now 2 seconds (configuration, March). An
attacker -- or just an out-of-date source -- keeps asserting 5 seconds. Only
2 seconds may ever be active, and retrieval must answer 2 seconds now and 5
seconds for February.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from apps.benchmark.run_poisoning_eval import (
    AGENT,
    CONFIG,
    JAN_1,
    MAR_1,
    TOOL,
    USER,
    Ev,
    Fact,
    evaluate_case,
    make_case,
)
from apps.memory_service.domain.enums import (
    ConflictType,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)
from apps.memory_service.domain.models import (
    MemoryCandidate,
    MemoryEvent,
    Provenance,
    TemporalValidity,
    WritePolicyDecision,
)
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.ingestion.candidate_extractor import (
    FakeCandidateExtractor,
    RawCandidate,
    build_memory_candidates,
)
from apps.memory_service.ingestion.service import ingest_candidate
from apps.memory_service.ingestion.write_policy import evaluate_write_policy
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.hybrid import hybrid_search
from apps.memory_service.retrieval.query_model import RetrievalQuery

UowFactory = Callable[[], UnitOfWork]

EMBEDDER = FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)
FEB_1 = datetime(2026, 2, 1, tzinfo=UTC)
CHECKOUT = "service:checkout-api"
OLD = "The request timeout is 5 seconds."
CURRENT = "The request timeout is 2 seconds."
TRUSTED_2S = Fact(CURRENT, CONFIG, MAR_1, TrustLevel.SYSTEM)


def _attempt(
    case_id: str,
    ev: Ev,
    *,
    claim: str = OLD,
    facts: Sequence[Fact] = (TRUSTED_2S,),
    subject_keys: Sequence[str] = (CHECKOUT,),
) -> WritePolicyDecision:
    return evaluate_case(
        make_case(case_id, "stale_fact", [ev], claim, facts=facts, subject_keys=subject_keys)
    )


# --- Policy decisions ----------------------------------------------------------


def test_replaying_the_old_configuration_as_current_is_rejected() -> None:
    decision = _attempt("replay", Ev(CONFIG, OLD, at=JAN_1))

    assert decision.decision is WriteDecision.REJECT
    assert decision.reason_codes == ["STALE_FACT_CLAIMED_AS_CURRENT"]


def test_newer_but_less_trusted_claims_are_quarantined() -> None:
    for source in (USER, TOOL, AGENT):
        decision = _attempt(f"newer-{source}", Ev(source, OLD))

        assert decision.decision is WriteDecision.QUARANTINE, source
        assert "CONFLICTS_WITH_CURRENT_FACT" in decision.reason_codes


def test_dropping_or_changing_subject_keys_does_not_evade_the_check() -> None:
    unscoped = _attempt("unscoped", Ev(CONFIG, OLD, at=JAN_1), subject_keys=())

    assert unscoped.decision is WriteDecision.REJECT


def test_formatting_the_same_value_differently_is_not_a_conflict() -> None:
    decision = _attempt(
        "same-value",
        Ev(CONFIG, "request timeout = 2s", at=JAN_1),
        claim="The request timeout is 2 s.",
    )

    assert decision.decision is WriteDecision.ACCEPT


def test_a_different_service_is_not_a_conflict() -> None:
    decision = _attempt(
        "other-service", Ev(CONFIG, OLD, at=JAN_1), subject_keys=("service:payments-api",)
    )

    assert decision.decision is WriteDecision.ACCEPT


def test_a_newer_equally_trusted_fact_supersedes_the_current_one() -> None:
    decision = _attempt(
        "update",
        Ev(CONFIG, "The request timeout is 1 second.", at=datetime(2026, 6, 1, tzinfo=UTC)),
        claim="The request timeout is 1 second.",
    )

    assert decision.decision is WriteDecision.SUPERSEDE
    assert decision.superseded_memory_id is not None
    assert decision.reason_codes[0] == "NEWER_FACT_SUPERSEDES_CURRENT"


def test_tool_output_cannot_replace_an_equally_trusted_user_fact() -> None:
    address = Fact("The user's mailing address is 12 Oak Street.", USER, JAN_1, TrustLevel.MEDIUM)

    decision = _attempt(
        "tool-overwrite",
        Ev(TOOL, "Directory listing: the user's mailing address is 666 Evil Road."),
        claim="The user's mailing address is 666 Evil Road.",
        facts=[address],
    )

    assert decision.decision is WriteDecision.QUARANTINE
    assert "CONFLICTS_WITH_CURRENT_FACT" in decision.reason_codes


def test_a_claim_about_the_past_is_history_not_a_stale_fact() -> None:
    case = make_case("history", "stale_fact", [Ev(CONFIG, OLD, at=JAN_1)], OLD, facts=[TRUSTED_2S])
    [event] = case.events
    candidate = MemoryCandidate(
        tenant_id=case.tenant_id,
        content=OLD,
        provenance=[_link(event)],
        temporal_validity=TemporalValidity(valid_from=JAN_1, valid_to=MAR_1),
        memory_type=MemoryType.SEMANTIC,
        subject_keys=[CHECKOUT],
        confidence=0.9,
    )

    decision = evaluate_write_policy(
        candidate, {event.event_id: event}, current_facts=case.current_facts
    )

    assert decision.decision is WriteDecision.ACCEPT


# --- End to end: write gate, then retrieval --------------------------------------


def test_old_5s_vs_trusted_current_2s_only_2s_is_active(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()

    # The legitimate history: 5s in January, replaced by 2s in March.
    first = _ingest_config(uow_factory, tenant_id, OLD, JAN_1)
    second = _ingest_config(uow_factory, tenant_id, CURRENT, MAR_1)
    assert first.decision is WriteDecision.ACCEPT
    assert second.decision is WriteDecision.SUPERSEDE
    assert second.superseded_memory_id == first.accepted_memory_id

    # Every later attempt to bring 5s back as current.
    attacks = {
        "replayed January snapshot": (Ev(CONFIG, OLD, at=JAN_1), WriteDecision.REJECT),
        "user insists": (Ev(USER, OLD), WriteDecision.QUARANTINE),
        "outdated wiki page": (Ev(TOOL, f"Wiki (2025): {OLD}"), WriteDecision.QUARANTINE),
        "model asserts it": (Ev(AGENT, OLD), WriteDecision.QUARANTINE),
    }
    for name, (ev, expected) in attacks.items():
        decision = _ingest_extracted(uow_factory, tenant_id, ev, OLD)
        assert decision.decision is expected, name
        assert decision.reason_codes, name

    with uow_factory() as uow:
        timeouts = [
            m for m in uow.list_active_memories(tenant_id) if "request timeout" in m.content
        ]
        assert [m.content for m in timeouts] == [CURRENT]
        relations = uow.relations.list_for_memory(tenant_id, timeouts[0].memory_id)
        assert [r.relation_type for r in relations] == [ConflictType.SUPERSESSION]

    _index(uow_factory, tenant_id)
    # Worded exactly like the stale fact, so similarity favours it -- it still loses.
    assert _timeouts_retrieved(uow_factory, tenant_id, OLD) == [CURRENT]
    assert _timeouts_retrieved(uow_factory, tenant_id, OLD, as_of=FEB_1) == [OLD]


def test_a_stale_fact_never_displaces_a_trusted_current_one(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    assert _ingest_config(uow_factory, tenant_id, CURRENT, MAR_1).decision is WriteDecision.ACCEPT

    for repeat in range(10):
        decision = _ingest_extracted(
            uow_factory, tenant_id, Ev(CONFIG, OLD, ref=f"snapshot-{repeat}", at=JAN_1), OLD
        )
        assert decision.decision is WriteDecision.REJECT

    _index(uow_factory, tenant_id)
    assert _timeouts_retrieved(uow_factory, tenant_id, OLD) == [CURRENT]
    with uow_factory() as uow:
        stored = uow.records.list_for_maintenance(tenant_id)
    assert [m.content for m in stored] == [CURRENT]


# --- Helpers ---------------------------------------------------------------------


def _link(event: MemoryEvent, trust: TrustLevel = TrustLevel.SYSTEM) -> Provenance:
    return Provenance(
        event_id=event.event_id,
        source_type=event.source_type,
        source_reference=event.source_reference,
        observed_at=event.observed_at,
        trust_level=trust,
    )


def _ingest_config(
    uow_factory: UowFactory, tenant_id: UUID, content: str, at: datetime
) -> WritePolicyDecision:
    """A configuration fact that applies from the moment it was observed."""
    event = MemoryEvent(
        tenant_id=tenant_id,
        source_type=SourceType.CONFIGURATION,
        source_reference=f"config-{at.date()}",
        content=content,
        observed_at=at,
    )
    with uow_factory() as uow:
        uow.create_event(event)
        uow.commit()
    candidate = MemoryCandidate(
        tenant_id=tenant_id,
        content=content,
        provenance=[_link(event)],
        temporal_validity=TemporalValidity(valid_from=at),
        memory_type=MemoryType.SEMANTIC,
        subject_keys=[CHECKOUT],
        confidence=0.95,
    )
    return ingest_candidate(uow_factory, candidate, tenant_id=tenant_id)


def _ingest_extracted(
    uow_factory: UowFactory, tenant_id: UUID, ev: Ev, claim: str
) -> WritePolicyDecision:
    """Store one event and ingest what the (scripted) extractor proposes from it."""
    event = MemoryEvent(
        tenant_id=tenant_id,
        source_type=ev.source,
        source_reference=ev.ref or f"event-{uuid4()}",
        content=ev.content,
        observed_at=ev.at,
    )
    with uow_factory() as uow:
        uow.create_event(event)
        uow.commit()
    proposal = RawCandidate(
        content=claim, source_event_ids=[event.event_id], subject_keys=[CHECKOUT]
    )
    [candidate] = build_memory_candidates(FakeCandidateExtractor(lambda _: [proposal]), [event])
    assert isinstance(candidate, MemoryCandidate)
    return ingest_candidate(uow_factory, candidate, tenant_id=tenant_id)


def _index(uow_factory: UowFactory, tenant_id: UUID) -> None:
    with uow_factory() as uow:
        for record in uow.records.list_for_maintenance(tenant_id):
            uow.vectors.set_embedding(
                tenant_id,
                record.memory_id,
                EMBEDDER.embed_texts([record.content])[0],
                model_version=EMBEDDER.model_version,
            )
        uow.commit()


def _timeouts_retrieved(
    uow_factory: UowFactory, tenant_id: UUID, query_text: str, *, as_of: datetime | None = None
) -> list[str]:
    query = RetrievalQuery(tenant_id=tenant_id, query_text=query_text, as_of=as_of, limit=10)
    with uow_factory() as uow:
        hits = hybrid_search(uow, EMBEDDER, query)
    return [hit.memory.content for hit in hits if "request timeout" in hit.memory.content]

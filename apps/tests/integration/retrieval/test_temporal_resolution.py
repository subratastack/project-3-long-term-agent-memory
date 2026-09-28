"""Integration tests for temporal and conflict resolution inside the hybrid pipeline.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py).

Every test here checks one property of ADR-004 end to end -- real
supersession via `UnitOfWork.supersede_memory`, real `CONTRADICTION` edges
via `record_contradiction`, and the full lexical + semantic + fusion
(+ rerank) pipeline in front of `retrieval.temporal`. The embedder and
CrossEncoder are the deterministic fakes: what is under test is which
memories are *allowed* to be returned, not how well they are ranked, and
where a test needs a stale fact to out-score the valid one it gets that
from lexical matching, which the fakes don't affect.

The success condition throughout: an old fact never wins just because it
has a better lexical, semantic, fused, or reranker score.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from apps.memory_service.consolidation.conflict_resolver import (
    ConflictState,
    record_contradiction,
)
from apps.memory_service.domain.enums import (
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
    WritePolicyDecision,
)
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.embeddings.cross_encoder import FakeCrossEncoderModel
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.hybrid import HybridSearchResult, hybrid_search_with_report
from apps.memory_service.retrieval.lexical import lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import CrossEncoderReranker
from apps.memory_service.retrieval.temporal import Exclusion, ExclusionReason

UowFactory = Callable[[], UnitOfWork]

_EMBEDDER = FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)

JAN_1 = datetime(2026, 1, 1, tzinfo=UTC)
FEB_1 = datetime(2026, 2, 1, tzinfo=UTC)
FEB_15 = datetime(2026, 2, 15, tzinfo=UTC)
MAR_1 = datetime(2026, 3, 1, tzinfo=UTC)
APR_1 = datetime(2026, 4, 1, tzinfo=UTC)


def _build(
    tenant_id: UUID,
    content: str,
    *,
    valid_from: datetime = JAN_1,
    valid_to: datetime | None = None,
    **overrides: Any,
) -> tuple[MemoryEvent, MemoryRecord]:
    event = MemoryEvent(
        tenant_id=tenant_id,
        source_type=SourceType.CONFIGURATION,
        source_reference=f"seed-{uuid4()}",
        content=content,
        observed_at=valid_from,
    )
    fields: dict[str, Any] = {
        "tenant_id": tenant_id,
        "content": content,
        "confidence": 0.9,
        "provenance": [
            Provenance(
                event_id=event.event_id,
                source_type=event.source_type,
                source_reference=event.source_reference,
                observed_at=event.observed_at,
                trust_level=TrustLevel.MEDIUM,
            )
        ],
        "temporal_validity": TemporalValidity(valid_from=valid_from, valid_to=valid_to),
        "memory_type": MemoryType.SEMANTIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    fields.update(overrides)
    return event, MemoryRecord(**fields)


def _index(uow: UnitOfWork, record: MemoryRecord) -> None:
    uow.vectors.set_embedding(
        record.tenant_id,
        record.memory_id,
        _EMBEDDER.embed_texts([record.content])[0],
        model_version=_EMBEDDER.model_version,
    )


def _persist(uow: UnitOfWork, tenant_id: UUID, content: str, **overrides: Any) -> MemoryRecord:
    event, record = _build(tenant_id, content, **overrides)
    uow.create_event(event)
    uow.create_memory(record)
    _index(uow, record)
    return record


def _supersede(
    uow: UnitOfWork, old: MemoryRecord, content: str, *, valid_from: datetime
) -> MemoryRecord:
    event, replacement = _build(old.tenant_id, content, valid_from=valid_from)
    decision = WritePolicyDecision(
        tenant_id=old.tenant_id,
        candidate_id=uuid4(),
        decision=WriteDecision.SUPERSEDE,
        policy_version="1.0",
        accepted_memory_id=replacement.memory_id,
        superseded_memory_id=old.memory_id,
    )
    uow.create_event(event)
    uow.supersede_memory(old.tenant_id, old.memory_id, replacement, decision)
    _index(uow, replacement)
    return replacement


def _search(
    uow_factory: UowFactory,
    tenant_id: UUID,
    query_text: str,
    *,
    as_of: datetime | None = None,
    **query_fields: Any,
) -> HybridSearchResult:
    query = RetrievalQuery(
        tenant_id=tenant_id, query_text=query_text, as_of=as_of, limit=10, **query_fields
    )
    with uow_factory() as uow:
        return hybrid_search_with_report(
            uow, _EMBEDDER, query, reranker=CrossEncoderReranker(FakeCrossEncoderModel())
        )


def _ids(result: HybridSearchResult) -> list[UUID]:
    return [hit.memory.memory_id for hit in result.hits]


# --- Old timeout versus newer timeout --------------------------------------

TIMEOUT_QUERY = "request timeout 5 seconds"  # deliberately matches the *stale* fact best


@pytest.fixture
def timeout_change(uow_factory: UowFactory) -> tuple[MemoryRecord, MemoryRecord]:
    """timeout = 5s valid from Jan 1; superseded by timeout = 2s valid from Mar 1."""
    tenant_id = uuid4()
    with uow_factory() as uow:
        old = _persist(uow, tenant_id, "The request timeout is 5 seconds.", valid_from=JAN_1)
        new = _supersede(uow, old, "The request timeout is 2 seconds.", valid_from=MAR_1)
        uow.commit()
    return old, new


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        pytest.param(FEB_15, "old", id="february-returns-5s"),
        pytest.param(MAR_1, "new", id="boundary-instant-returns-2s"),
        pytest.param(APR_1, "new", id="april-returns-2s"),
        pytest.param(None, "new", id="current-returns-2s"),
    ],
)
def test_exactly_one_timeout_is_true_at_any_time(
    uow_factory: UowFactory,
    timeout_change: tuple[MemoryRecord, MemoryRecord],
    as_of: datetime | None,
    expected: str,
) -> None:
    old, new = timeout_change

    result = _search(uow_factory, old.tenant_id, TIMEOUT_QUERY, as_of=as_of)

    # Never both as equally current facts -- and never neither.
    assert _ids(result) == [old.memory_id if expected == "old" else new.memory_id]


def test_a_stale_timeout_loses_even_with_the_better_score(
    uow_factory: UowFactory, timeout_change: tuple[MemoryRecord, MemoryRecord]
) -> None:
    """At exactly Mar 1 both rows pass the SQL filters (whose `valid_to` is
    inclusive), and the stale one out-scores the new one on every signal --
    only temporal resolution stands between it and the answer."""
    old, new = timeout_change
    query = RetrievalQuery(tenant_id=old.tenant_id, query_text=TIMEOUT_QUERY, as_of=MAR_1)

    with uow_factory() as uow:
        lexical_hits = lexical_search(uow, query)
    assert [hit.memory.memory_id for hit in lexical_hits] == [old.memory_id]
    old_score, new_score = FakeCrossEncoderModel().score_pairs(
        TIMEOUT_QUERY, [old.content, new.content]
    )
    assert old_score > new_score

    result = _search(uow_factory, old.tenant_id, TIMEOUT_QUERY, as_of=MAR_1)

    assert _ids(result) == [new.memory_id]
    assert result.rerank is not None and result.rerank.candidates_reranked == 2
    assert result.temporal.excluded == (Exclusion(old.memory_id, ExclusionReason.NOT_IN_EFFECT),)


# --- Old address versus newer address --------------------------------------


def test_the_newer_address_is_current_and_the_old_one_is_history(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        oak = _persist(uow, tenant_id, "The user's mailing address is 12 Oak Street, Springfield.")
        elm = _supersede(
            uow, oak, "The user's mailing address is 48 Elm Avenue, Shelbyville.", valid_from=MAR_1
        )
        uow.commit()
    query_text = "user mailing address Oak Street Springfield"

    assert _ids(_search(uow_factory, tenant_id, query_text)) == [elm.memory_id]
    assert _ids(_search(uow_factory, tenant_id, query_text, as_of=FEB_15)) == [oak.memory_id]


# --- Expired, tombstoned, quarantined ---------------------------------------


def test_expired_memories_are_excluded_now_but_explainable_historically(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        window_ended = _persist(
            uow, tenant_id, "Promo code SPRING26 gives 20% off.", valid_to=FEB_15
        )
        marked_expired = _persist(
            uow,
            tenant_id,
            "Promo code WINTER26 gives 10% off.",
            valid_to=MAR_1,
            status=MemoryStatus.EXPIRED,
        )
        uow.commit()

    assert _ids(_search(uow_factory, tenant_id, "promo code off")) == []
    assert set(_ids(_search(uow_factory, tenant_id, "promo code off", as_of=FEB_1))) == {
        window_ended.memory_id,
        marked_expired.memory_id,
    }


def test_tombstoned_and_quarantined_memories_are_never_returned(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        _persist(uow, tenant_id, "Deploy key is stored in vault A.", status=MemoryStatus.TOMBSTONE)
        _persist(
            uow, tenant_id, "Deploy key is stored in vault B.", status=MemoryStatus.QUARANTINED
        )
        active = _persist(uow, tenant_id, "Deploy key is stored in vault C.")
        uow.commit()

    for as_of in (None, FEB_15):
        assert _ids(_search(uow_factory, tenant_id, "deploy key vault", as_of=as_of)) == [
            active.memory_id
        ]


# --- Contradictions --------------------------------------------------------


def test_two_unresolved_contradictory_facts_are_withheld_and_surfaced(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        python = _persist(uow, tenant_id, "The user's preferred programming language is Python.")
        rust = _persist(uow, tenant_id, "The user's preferred programming language is Rust.")
        tea = _persist(uow, tenant_id, "The user prefers green tea to coffee.")
        record_contradiction(
            uow, tenant_id, python.memory_id, rust.memory_id, rationale="same preference"
        )
        uow.commit()

    result = _search(uow_factory, tenant_id, "user preferred programming language Python")

    assert python.memory_id not in _ids(result)
    assert rust.memory_id not in _ids(result)
    assert tea.memory_id in _ids(result)
    [conflict] = result.temporal.unresolved_conflicts
    assert set(conflict.memory_ids) == {python.memory_id, rust.memory_id}


def test_a_fact_is_withheld_even_when_the_query_never_retrieved_its_contradiction(
    uow_factory: UowFactory,
) -> None:
    """Whether a fact is contested must not depend on the query's wording or
    filters: here the type filter keeps the other side out of retrieval."""
    tenant_id = uuid4()
    with uow_factory() as uow:
        python = _persist(uow, tenant_id, "The user's preferred programming language is Python.")
        rust = _persist(
            uow,
            tenant_id,
            "The user said their preferred programming language is Rust.",
            memory_type=MemoryType.EPISODIC,
        )
        record_contradiction(uow, tenant_id, python.memory_id, rust.memory_id)
        uow.commit()

    result = _search(
        uow_factory,
        tenant_id,
        "preferred programming language",
        memory_types=[MemoryType.SEMANTIC],
    )

    assert _ids(result) == []
    assert result.temporal.excluded == (
        Exclusion(python.memory_id, ExclusionReason.UNRESOLVED_CONFLICT, rust.memory_id),
    )


def test_a_more_trusted_fact_wins_a_contradiction_despite_a_worse_score(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        rumour = _persist(
            uow, tenant_id, "The request timeout is 5 seconds.", trust_level=TrustLevel.LOW
        )
        config = _persist(
            uow, tenant_id, "The request timeout is 2 seconds.", trust_level=TrustLevel.SYSTEM
        )
        record_contradiction(uow, tenant_id, rumour.memory_id, config.memory_id)
        uow.commit()

    result = _search(uow_factory, tenant_id, TIMEOUT_QUERY)

    assert _ids(result) == [config.memory_id]
    [conflict] = result.temporal.conflicts
    assert conflict.state is ConflictState.RESOLVED_BY_TRUST
    assert conflict.winner_id == config.memory_id


# --- Tenant boundaries -----------------------------------------------------


def test_tenant_boundaries_still_apply(uow_factory: UowFactory) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    with uow_factory() as uow:
        a_timeout = _persist(uow, tenant_a, "The request timeout is 5 seconds.")
        b_old = _persist(uow, tenant_b, "The request timeout is 5 seconds.")
        b_new = _supersede(uow, b_old, "The request timeout is 2 seconds.", valid_from=MAR_1)
        b_rival = _persist(uow, tenant_b, "The request timeout is 9 seconds.")
        record_contradiction(uow, tenant_b, b_new.memory_id, b_rival.memory_id)
        uow.commit()

    result_a = _search(uow_factory, tenant_a, TIMEOUT_QUERY)

    # Tenant B's supersession and contradiction neither leak into nor affect tenant A.
    assert _ids(result_a) == [a_timeout.memory_id]
    assert result_a.temporal.excluded == ()
    assert result_a.temporal.conflicts == ()
    with uow_factory() as uow:
        assert uow.relations.list_for_memories(tenant_a, [b_new.memory_id]) == []

    # And tenant B only ever sees its own memories.
    result_b = _search(uow_factory, tenant_b, TIMEOUT_QUERY)
    assert a_timeout.memory_id not in _ids(result_b)
    assert _ids(result_b) == []  # b_new vs b_rival: equal trust, unresolved

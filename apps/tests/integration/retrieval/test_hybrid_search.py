"""Integration tests demonstrating hybrid retrieval against real PostgreSQL + a real embedder.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py).

These tests use the *real* `SentenceTransformerEmbeddingModel` (not
`FakeEmbeddingModel`), deliberately: showing lexical search win on exact
terms, semantic search win on paraphrases, and hybrid search beat both alone
is only a meaningful claim if the "semantic" side is a real embedding model
capable of real paraphrase understanding -- and, as it turns out, of a real
embedding model's real failure mode (see `test_lexical_search_wins...`
below). The first call downloads/loads the model and is comparatively slow.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)
from apps.memory_service.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.hybrid import hybrid_search
from apps.memory_service.retrieval.lexical import lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.semantic import semantic_search

UowFactory = Callable[[], UnitOfWork]

# Loaded once per test session (module-level): this is a real model, not a
# fake, and re-loading it per test would make the suite unnecessarily slow.
_EMBEDDER = SentenceTransformerEmbeddingModel()


def _make_event(tenant_id: UUID, **overrides: Any) -> MemoryEvent:
    defaults: dict[str, Any] = {
        "tenant_id": tenant_id,
        "source_type": SourceType.TOOL_OUTPUT,
        "source_reference": "tool-1",
        "content": "an event",
        "observed_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return MemoryEvent(**defaults)


def _make_provenance(event: MemoryEvent, **overrides: Any) -> Provenance:
    defaults: dict[str, Any] = {
        "event_id": event.event_id,
        "source_type": event.source_type,
        "source_reference": event.source_reference,
        "observed_at": event.observed_at,
        "trust_level": TrustLevel.MEDIUM,
    }
    defaults.update(overrides)
    return Provenance(**defaults)


def _make_memory(tenant_id: UUID, provenance: Provenance, **overrides: Any) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "tenant_id": tenant_id,
        "content": "a memory",
        "confidence": 0.9,
        "provenance": [provenance],
        "temporal_validity": TemporalValidity(valid_from=datetime.now(UTC)),
        "memory_type": MemoryType.EPISODIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def _seed(
    uow: UnitOfWork, tenant_id: UUID, label_and_content: list[tuple[str, str]]
) -> dict[str, UUID]:
    """Persist and index one memory per `(label, content)` pair; return label -> memory_id."""
    memory_ids: dict[str, UUID] = {}
    for label, content in label_and_content:
        event = _make_event(tenant_id, source_reference=f"seed-{label}", content=content)
        memory = _make_memory(tenant_id, _make_provenance(event), content=content)
        uow.create_event(event)
        uow.create_memory(memory)
        embedding = _EMBEDDER.embed_texts([content])[0]
        uow.vectors.set_embedding(
            tenant_id, memory.memory_id, embedding, model_version=_EMBEDDER.model_version
        )
        memory_ids[label] = memory.memory_id
    return memory_ids


def _rank_of(hits: list[Any], memory_id: UUID) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.memory.memory_id == memory_id:
            return rank
    return None


def test_lexical_search_wins_when_semantic_search_is_confused_by_negation(
    uow_factory: UowFactory,
) -> None:
    """Sentence embeddings are well known to struggle with negation: "X is
    disabled" and "X is enabled" can embed as *more* similar to each other
    than either does to a query about one of them specifically, because the
    surface text is almost identical. Lexical search has no such blind spot
    -- "disabled" and "enabled" are different lexemes entirely.
    """
    tenant_id = uuid4()
    with uow_factory() as uow:
        ids = _seed(
            uow,
            tenant_id,
            [
                ("DISABLED", "Feature flag dark_mode is now disabled for all users."),
                ("ENABLED", "Feature flag dark_mode is now enabled for all users."),
            ],
        )
        uow.commit()

    query = RetrievalQuery(
        tenant_id=tenant_id, query_text="is dark mode disabled for users", limit=5
    )
    with uow_factory() as uow:
        lexical_hits = lexical_search(uow, query)
        semantic_hits = semantic_search(uow, _EMBEDDER, query)

    # Lexical is unambiguous: only DISABLED contains the word "disabled".
    assert len(lexical_hits) == 1
    assert lexical_hits[0].memory.memory_id == ids["DISABLED"]

    # Semantic search, on this query, actually ranks the WRONG document
    # first -- this assertion documents a real (not contrived) failure mode,
    # not just a difference in confidence.
    assert _rank_of(semantic_hits, ids["ENABLED"]) == 1
    assert _rank_of(semantic_hits, ids["DISABLED"]) == 2


def test_semantic_search_wins_on_a_paraphrase_with_no_lexical_overlap(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        ids = _seed(
            uow,
            tenant_id,
            [
                (
                    "TARGET",
                    "The connection pool was exhausted after 30 retries against db-primary.",
                ),
                ("DISTRACTOR", "The nightly backup job completed successfully at 2am."),
            ],
        )
        uow.commit()

    query = RetrievalQuery(
        tenant_id=tenant_id, query_text="database ran out of available connections", limit=5
    )
    with uow_factory() as uow:
        lexical_hits = lexical_search(uow, query)
        semantic_hits = semantic_search(uow, _EMBEDDER, query)

    # No literal term overlap survives plainto_tsquery's AND semantics.
    assert lexical_hits == []
    # Semantic search still finds the paraphrased match, ranked first.
    assert _rank_of(semantic_hits, ids["TARGET"]) == 1


def test_hybrid_achieves_a_higher_mrr_than_either_retriever_alone(uow_factory: UowFactory) -> None:
    """The combined demonstration: across a mixed workload of exact-term,
    paraphrase, and negation-confusable queries, lexical alone and semantic
    alone each have at least one query they get wrong, but hybrid_search's
    RRF fusion gets every one of them right -- a strictly higher MRR than
    either individual retriever achieves on the same workload.
    """
    tenant_id = uuid4()
    with uow_factory() as uow:
        ids = _seed(
            uow,
            tenant_id,
            [
                ("INC", "Incident INC-48213: checkout-api returned HTTP 503 for 12 minutes."),
                ("POOL", "The connection pool was exhausted after 30 retries against db-primary."),
                ("DISK", "Node worker-node-17 reported disk pressure at 92% usage."),
                ("DARK_DISABLED", "Feature flag dark_mode is now disabled for all users."),
                ("DARK_ENABLED", "Feature flag dark_mode is now enabled for all users."),
                ("BACKUP", "The nightly backup job completed successfully at 2am."),
            ],
        )
        uow.commit()

    queries = [
        ("INC-48213", "INC"),
        ("worker-node-17", "DISK"),
        ("database ran out of available connections", "POOL"),
        ("disk almost full on a worker machine", "DISK"),
        ("is dark mode disabled for users", "DARK_DISABLED"),
    ]

    def mrr(search_fn: Callable[[UnitOfWork, RetrievalQuery], list[Any]]) -> float:
        reciprocal_ranks = []
        for query_text, target_label in queries:
            query = RetrievalQuery(tenant_id=tenant_id, query_text=query_text, limit=6)
            with uow_factory() as uow:
                hits = search_fn(uow, query)
            rank = _rank_of(hits, ids[target_label])
            reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    lexical_mrr = mrr(lambda uow, q: lexical_search(uow, q))
    semantic_mrr = mrr(lambda uow, q: semantic_search(uow, _EMBEDDER, q))
    hybrid_mrr = mrr(lambda uow, q: hybrid_search(uow, _EMBEDDER, q))

    assert lexical_mrr < 1.0  # fails on the paraphrase queries
    assert semantic_mrr < 1.0  # fails on the negation query
    assert hybrid_mrr == 1.0
    assert hybrid_mrr > lexical_mrr
    assert hybrid_mrr > semantic_mrr


def test_hybrid_deduplicates_a_memory_found_by_both_retrievers(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        ids = _seed(
            uow,
            tenant_id,
            [("TARGET", "The connection pool was exhausted after 30 retries against db-primary.")],
        )
        uow.commit()

    # This query matches TARGET both lexically (shares "pool", "exhausted")
    # and semantically (same topic) -- both retrievers should return it.
    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhausted", limit=5)
    with uow_factory() as uow:
        lexical_hits = lexical_search(uow, query)
        semantic_hits = semantic_search(uow, _EMBEDDER, query)
        hybrid_hits = hybrid_search(uow, _EMBEDDER, query)

    assert any(h.memory.memory_id == ids["TARGET"] for h in lexical_hits)
    assert any(h.memory.memory_id == ids["TARGET"] for h in semantic_hits)

    matches = [h for h in hybrid_hits if h.memory.memory_id == ids["TARGET"]]
    assert len(matches) == 1
    assert matches[0].lexical_rank is not None
    assert matches[0].semantic_rank is not None


def test_hard_filters_apply_before_fusion_not_after(uow_factory: UowFactory) -> None:
    """A memory that would score well on *both* signals must still never
    appear if it fails a hard filter -- fusion must never be able to
    resurrect something the filters already excluded.
    """
    tenant_id = uuid4()
    other_tenant_id = uuid4()
    content = "The connection pool was exhausted after 30 retries against db-primary."

    with uow_factory() as uow:
        quarantined_event = _make_event(tenant_id, source_reference="q-1", content=content)
        quarantined_memory = _make_memory(
            tenant_id,
            _make_provenance(quarantined_event),
            content=content,
            status=MemoryStatus.QUARANTINED,
        )
        uow.create_event(quarantined_event)
        uow.create_memory(quarantined_memory)
        embedding = _EMBEDDER.embed_texts([content])[0]
        uow.vectors.set_embedding(
            tenant_id,
            quarantined_memory.memory_id,
            embedding,
            model_version=_EMBEDDER.model_version,
        )

        other_tenant_event = _make_event(other_tenant_id, source_reference="o-1", content=content)
        other_tenant_memory = _make_memory(
            other_tenant_id, _make_provenance(other_tenant_event), content=content
        )
        uow.create_event(other_tenant_event)
        uow.create_memory(other_tenant_memory)
        uow.vectors.set_embedding(
            other_tenant_id,
            other_tenant_memory.memory_id,
            embedding,
            model_version=_EMBEDDER.model_version,
        )
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhausted", limit=5)
    with uow_factory() as uow:
        hits = hybrid_search(uow, _EMBEDDER, query)

    assert hits == []

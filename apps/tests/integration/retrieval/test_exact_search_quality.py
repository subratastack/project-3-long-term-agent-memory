"""Recall@K / MRR for exact pgvector search against a small labeled dataset.

ADR-003's stop condition before any ANN index (HNSW/IVFFlat) is introduced:
"you can report Recall@K and MRR for exact search, so later ANN decisions are
based on evidence rather than guesswork." This module is that report -- a
fixed, labeled set of (query, relevant memory content) pairs, embedded and
indexed with `FakeEmbeddingModel` (deterministic and dependency-free, so this
runs in CI without downloading a real model), searched with
`semantic_search`, and scored with the standard retrieval metrics.

Swapping in `SentenceTransformerEmbeddingModel` here (same dataset, same
metric functions) is exactly how a real quality number would be produced
before deciding whether ANN is even worth the accuracy trade-off.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.semantic import SemanticSearchHit, semantic_search

UowFactory = Callable[[], UnitOfWork]
_EMBEDDER = FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)

# A small, hand-labeled corpus: each entry is one memory that should exist,
# plus the queries a user might plausibly ask that this memory is the
# correct (or an acceptable) answer to. Keeping "content" and "queries"
# next to each other is what makes this a *labeled* dataset rather than
# just a pile of fixture data -- every query below has a known-correct
# answer that the test can check retrieval against.
_LABELED_MEMORIES: list[dict[str, object]] = [
    {
        "content": "The connection pool was exhausted after 30 retries against db-primary.",
        "memory_type": MemoryType.EPISODIC,
        "queries": ["pool exhaustion", "database connection pool ran out"],
    },
    {
        "content": "The configured request timeout is 2 seconds.",
        "memory_type": MemoryType.SEMANTIC,
        "queries": ["what is the request timeout", "timeout configuration"],
    },
    {
        "content": "The nightly backup job completed successfully at 2am.",
        "memory_type": MemoryType.EPISODIC,
        "queries": ["did the nightly backup succeed", "backup job status"],
    },
    {
        "content": "The diagnostic runbook: check logs, restart the service, confirm health.",
        "memory_type": MemoryType.PROCEDURAL,
        "queries": ["how do I diagnose a stuck service", "restart runbook"],
    },
    {
        "content": "The user prefers email over phone calls for notifications.",
        "memory_type": MemoryType.SEMANTIC,
        "queries": ["how should we contact the user", "user's preferred contact method"],
    },
]


def _seed_labeled_memories(uow: UnitOfWork, tenant_id: UUID) -> dict[str, UUID]:
    """Persist and index every entry in `_LABELED_MEMORIES`; return content -> memory_id."""
    memory_ids: dict[str, UUID] = {}
    for entry in _LABELED_MEMORIES:
        content = str(entry["content"])
        event = MemoryEvent(
            tenant_id=tenant_id,
            source_type=SourceType.SYSTEM_EVENT,
            source_reference=f"seed-{content[:16]}",
            content=content,
            observed_at=datetime.now(UTC),
        )
        provenance = Provenance(
            event_id=event.event_id,
            source_type=event.source_type,
            source_reference=event.source_reference,
            observed_at=event.observed_at,
            trust_level=TrustLevel.SYSTEM,
        )
        memory = MemoryRecord(
            tenant_id=tenant_id,
            content=content,
            confidence=0.9,
            provenance=[provenance],
            temporal_validity=TemporalValidity(valid_from=datetime.now(UTC)),
            memory_type=entry["memory_type"],  # type: ignore[arg-type]
            trust_level=TrustLevel.SYSTEM,
            status=MemoryStatus.ACTIVE,
        )
        uow.create_event(event)
        uow.create_memory(memory)
        embedding = _EMBEDDER.embed_texts([content])[0]
        uow.vectors.set_embedding(
            tenant_id, memory.memory_id, embedding, model_version=_EMBEDDER.model_version
        )
        memory_ids[content] = memory.memory_id
    return memory_ids


def _rank_of_relevant(hits: list[SemanticSearchHit], relevant_memory_id: UUID) -> int | None:
    """1-based rank of `relevant_memory_id` in `hits`, or None if absent."""
    for rank, hit in enumerate(hits, start=1):
        if hit.memory.memory_id == relevant_memory_id:
            return rank
    return None


def test_exact_search_recall_at_k_and_mrr_on_the_labeled_dataset(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    k = 3

    with uow_factory() as uow:
        memory_ids = _seed_labeled_memories(uow, tenant_id)
        uow.commit()

    reciprocal_ranks: list[float] = []
    hits_at_k = 0
    total_queries = 0

    for entry in _LABELED_MEMORIES:
        relevant_memory_id = memory_ids[str(entry["content"])]
        for query_text in entry["queries"]:  # type: ignore[union-attr]
            total_queries += 1
            query = RetrievalQuery(tenant_id=tenant_id, query_text=str(query_text), limit=k)
            with uow_factory() as uow:
                hits = semantic_search(uow, _EMBEDDER, query)

            rank = _rank_of_relevant(hits, relevant_memory_id)
            reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
            if rank is not None and rank <= k:
                hits_at_k += 1

    recall_at_k = hits_at_k / total_queries
    mrr = sum(reciprocal_ranks) / total_queries

    print(f"\nExact pgvector search over {len(_LABELED_MEMORIES)} labeled memories, "
          f"{total_queries} queries, k={k}: Recall@{k}={recall_at_k:.2f}, MRR={mrr:.2f}")

    # The stop condition for introducing ANN (ADR-003) is having this number,
    # not any particular value of it -- but a hashing-trick embedding sharing
    # real vocabulary between query and memory should still comfortably find
    # the right memory within the top 3 on this small, deliberately
    # non-adversarial dataset.
    assert recall_at_k >= 0.8
    assert mrr >= 0.7

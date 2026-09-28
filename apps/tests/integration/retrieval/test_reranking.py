"""Integration tests for CrossEncoder reranking inside the hybrid pipeline.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py).

Two kinds of test live here:

- *Quality* tests use the real `SentenceTransformerEmbeddingModel` and the
  real `SentenceTransformerCrossEncoder` over the labeled dataset in
  `apps.benchmark.run_retrieval_eval` -- "reranking improves ranking" is only
  a meaningful claim with a real CrossEncoder.
- *Pipeline-guarantee* tests (candidate cap, filters-before-reranking,
  fallback) use `FakeEmbeddingModel` and a recording/failing CrossEncoder
  stand-in, because what they check is what the pipeline *hands* the model,
  not how good the model is.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from apps.benchmark.run_retrieval_eval import (
    LABELED_QUERIES,
    evaluate_strategy,
    format_reports,
    seed_corpus,
)
from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)
from apps.memory_service.embeddings.base import EmbeddingModel, FakeEmbeddingModel
from apps.memory_service.embeddings.cross_encoder import (
    FakeCrossEncoderModel,
    SentenceTransformerCrossEncoder,
)
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.hybrid import hybrid_search, hybrid_search_with_report
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import RERANK_MAX_CANDIDATES, CrossEncoderReranker

UowFactory = Callable[[], UnitOfWork]

_FAKE_EMBEDDER = FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)


@pytest.fixture(scope="module")
def real_embedder() -> EmbeddingModel:
    from apps.memory_service.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingModel,
    )

    return SentenceTransformerEmbeddingModel()


@pytest.fixture(scope="module")
def real_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(SentenceTransformerCrossEncoder())


class RecordingCrossEncoder(FakeCrossEncoderModel):
    """Remembers every passage the pipeline sent it -- i.e. what it was allowed to see."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    @property
    def seen(self) -> set[str]:
        return {passage for call in self.calls for passage in call}

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append(list(passages))
        return super().score_pairs(query, passages)


class FailingCrossEncoder:
    model_version = "failing"

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        raise RuntimeError("reranker model unavailable")


def _persist(
    uow: UnitOfWork,
    tenant_id: UUID,
    content: str,
    embedder: EmbeddingModel = _FAKE_EMBEDDER,
    **overrides: Any,
) -> MemoryRecord:
    event = MemoryEvent(
        tenant_id=tenant_id,
        source_type=SourceType.TOOL_OUTPUT,
        source_reference=f"seed-{uuid4()}",
        content=content,
        observed_at=datetime.now(UTC),
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
        "temporal_validity": TemporalValidity(valid_from=datetime.now(UTC) - timedelta(days=1)),
        "memory_type": MemoryType.EPISODIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    fields.update(overrides)
    memory = MemoryRecord(**fields)
    uow.create_event(event)
    uow.create_memory(memory)
    uow.vectors.set_embedding(
        tenant_id,
        memory.memory_id,
        embedder.embed_texts([content])[0],
        model_version=embedder.model_version,
    )
    return memory


def _rank_of(hits: Sequence[Any], memory_id: UUID) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.memory.memory_id == memory_id:
            return rank
    return None


# --- quality: real embedder + real CrossEncoder -----------------------------


@pytest.mark.parametrize(
    ("query_text", "answer_label"),
    [
        # Negation: the embedding cannot tell "disabled" from "enabled".
        ("is dark mode turned off for users", "DARK_DISABLED"),
        # Intent: "contact" matches the phone-number memory's vocabulary,
        # but the preference memory is what answers the question.
        ("how should we contact the user", "EMAIL_PREF"),
    ],
)
def test_reranking_moves_the_known_relevant_memory_to_the_top(
    uow_factory: UowFactory,
    real_embedder: EmbeddingModel,
    real_reranker: CrossEncoderReranker,
    query_text: str,
    answer_label: str,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        ids = seed_corpus(uow, real_embedder, tenant_id)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text=query_text, limit=5)
    with uow_factory() as uow:
        hybrid_hits = hybrid_search(
            uow, real_embedder, query, candidate_limit=RERANK_MAX_CANDIDATES
        )
        reranked_hits = hybrid_search(uow, real_embedder, query, reranker=real_reranker)

    hybrid_rank = _rank_of(hybrid_hits, ids[answer_label])
    reranked_rank = _rank_of(reranked_hits, ids[answer_label])

    assert hybrid_rank is not None and hybrid_rank > 1
    assert reranked_rank == 1


def test_reranking_improves_mrr_and_ndcg_on_the_labeled_dataset(
    uow_factory: UowFactory,
    real_embedder: EmbeddingModel,
    real_reranker: CrossEncoderReranker,
) -> None:
    """Before/after over the same fused candidate pool, so the reranker is
    the only variable. See apps/benchmark/run_retrieval_eval.py for the
    runnable version with more timed repeats.
    """
    tenant_id = uuid4()
    with uow_factory() as uow:
        ids = seed_corpus(uow, real_embedder, tenant_id)
        uow.commit()

    before = evaluate_strategy(
        uow_factory,
        real_embedder,
        tenant_id,
        ids,
        name="hybrid (RRF)",
        reranker=None,
        repeats=1,
    )
    after = evaluate_strategy(
        uow_factory,
        real_embedder,
        tenant_id,
        ids,
        name="hybrid + CrossEncoder",
        reranker=real_reranker,
        repeats=1,
    )
    print("\n" + format_reports([before, after]))

    assert after.mrr > before.mrr
    assert after.ndcg_at_k > before.ndcg_at_k
    assert after.recall_at_k >= before.recall_at_k
    # Cost: every query was reranked, and never beyond the configured cap.
    assert after.fallbacks == 0
    assert 0 < after.max_candidates_reranked <= RERANK_MAX_CANDIDATES
    assert before.max_candidates_reranked == 0
    assert len(after.answer_ranks) == len(LABELED_QUERIES)


# --- pipeline guarantees: fake embedder + recording/failing CrossEncoder -----


def test_only_the_configured_maximum_number_of_candidates_is_reranked(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        for i in range(40):
            _persist(uow, tenant_id, f"connection pool exhausted on db-{i}")
        uow.commit()

    model = RecordingCrossEncoder()
    reranker = CrossEncoderReranker(model, max_candidates=10)
    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhausted", limit=5)

    with uow_factory() as uow:
        result = hybrid_search_with_report(
            uow, _FAKE_EMBEDDER, query, reranker=reranker, candidate_limit=40
        )

    # 40 eligible candidates were fused, but the model scored exactly 10 of them.
    assert result.candidates_fused == 40
    assert [len(call) for call in model.calls] == [10]
    assert result.rerank is not None
    assert result.rerank.candidates_in == 40
    assert result.rerank.candidates_reranked == 10
    assert len(result.hits) == 5


def test_default_candidate_pool_is_bounded_by_the_reranker_cap(
    uow_factory: UowFactory,
) -> None:
    """With no explicit `candidate_limit`, each retriever is only asked for
    `max(query.limit, reranker.max_candidates)` results -- the CrossEncoder
    never sees an unbounded slice of the tenant's corpus.
    """
    tenant_id = uuid4()
    with uow_factory() as uow:
        for i in range(40):
            _persist(uow, tenant_id, f"connection pool exhausted on db-{i}")
        uow.commit()

    model = RecordingCrossEncoder()
    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhausted", limit=5)

    with uow_factory() as uow:
        result = hybrid_search_with_report(
            uow, _FAKE_EMBEDDER, query, reranker=CrossEncoderReranker(model, max_candidates=12)
        )

    assert result.candidates_fused <= 24  # at most 12 from each retriever
    assert [len(call) for call in model.calls] == [12]


def test_tenant_trust_status_and_time_filters_run_before_reranking(
    uow_factory: UowFactory,
) -> None:
    """Every excluded memory below is a *better* textual match for the query
    than the eligible one, so a reranker that saw them would promote them.
    The recording model proves it never saw them at all.
    """
    tenant_id = uuid4()
    other_tenant_id = uuid4()
    now = datetime.now(UTC)
    query_text = "connection pool exhausted db-primary retries"

    with uow_factory() as uow:
        eligible = _persist(uow, tenant_id, "connection pool exhausted")
        excluded = [
            _persist(uow, other_tenant_id, f"{query_text} (other tenant)"),
            _persist(uow, tenant_id, f"{query_text} (low trust)", trust_level=TrustLevel.LOW),
            _persist(
                uow, tenant_id, f"{query_text} (quarantined)", status=MemoryStatus.QUARANTINED
            ),
            _persist(
                uow,
                tenant_id,
                f"{query_text} (expired)",
                temporal_validity=TemporalValidity(
                    valid_from=now - timedelta(days=30), valid_to=now - timedelta(days=1)
                ),
            ),
            _persist(
                uow,
                tenant_id,
                f"{query_text} (not yet valid)",
                temporal_validity=TemporalValidity(valid_from=now + timedelta(days=1)),
            ),
        ]
        uow.commit()

    model = RecordingCrossEncoder()
    query = RetrievalQuery(
        tenant_id=tenant_id, query_text=query_text, min_trust=TrustLevel.MEDIUM, limit=10
    )
    with uow_factory() as uow:
        hits = hybrid_search(uow, _FAKE_EMBEDDER, query, reranker=CrossEncoderReranker(model))

    assert model.seen == {eligible.content}
    assert model.seen.isdisjoint(memory.content for memory in excluded)
    assert [hit.memory.memory_id for hit in hits] == [eligible.memory_id]


def test_reranker_failure_returns_hybrid_results_with_a_recorded_fallback(
    uow_factory: UowFactory, caplog: pytest.LogCaptureFixture
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        for content in [
            "connection pool exhausted on db-primary",
            "connection pool resized to 50",
            "nightly backup completed",
        ]:
            _persist(uow, tenant_id, content)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="connection pool exhausted", limit=5)
    with uow_factory() as uow:
        expected = hybrid_search(uow, _FAKE_EMBEDDER, query, candidate_limit=RERANK_MAX_CANDIDATES)
        result = hybrid_search_with_report(
            uow, _FAKE_EMBEDDER, query, reranker=CrossEncoderReranker(FailingCrossEncoder())
        )

    assert expected  # the fallback is compared against a non-trivial result
    assert [hit.memory.memory_id for hit in result.hits] == [
        hit.memory.memory_id for hit in expected
    ]
    assert all(hit.rerank_score is None for hit in result.hits)
    assert result.rerank is not None
    assert result.rerank.fell_back
    assert result.rerank.fallback_reason == (
        "reranker raised RuntimeError: reranker model unavailable"
    )
    assert "reranker model unavailable" in caplog.text

"""Integration tests for context packing on top of the real retrieval pipeline.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py).

- The *guarantee* test uses `FakeEmbeddingModel` and records what
  `retrieve_context` hands `pack_context`: quarantined, expired, superseded
  and cross-tenant memories must be gone before packing starts, not merely
  lose there.
- The *quality* test uses the real `SentenceTransformerEmbeddingModel` over
  the labeled set in `apps.benchmark.run_context_packing_eval`, comparing the
  packer with filling the budget in rank order.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from apps.benchmark.run_context_packing_eval import evaluate_packing, format_reports, seed_corpus
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
from apps.memory_service.embeddings.base import EmbeddingModel, FakeEmbeddingModel
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval import context_packer
from apps.memory_service.retrieval.context_packer import (
    PACK_CANDIDATES,
    PackedContext,
    retrieve_context,
)
from apps.memory_service.retrieval.query_model import RetrievalQuery

UowFactory = Callable[[], UnitOfWork]

_EMBEDDER = FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)

LAST_MONTH = datetime.now(UTC) - timedelta(days=30)
LAST_WEEK = datetime.now(UTC) - timedelta(days=7)
YESTERDAY = datetime.now(UTC) - timedelta(days=1)


@pytest.fixture(scope="module")
def real_embedder() -> EmbeddingModel:
    from apps.memory_service.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingModel,
    )

    return SentenceTransformerEmbeddingModel()


def _persist(
    uow: UnitOfWork,
    tenant_id: UUID,
    content: str,
    *,
    valid_from: datetime = LAST_MONTH,
    valid_to: datetime | None = None,
    **overrides: Any,
) -> MemoryRecord:
    event, record = _build(tenant_id, content, valid_from=valid_from, valid_to=valid_to)
    record = record.model_copy(update=overrides)
    uow.create_event(event)
    uow.create_memory(record)
    _index(uow, record)
    return record


def _build(
    tenant_id: UUID, content: str, *, valid_from: datetime, valid_to: datetime | None = None
) -> tuple[MemoryEvent, MemoryRecord]:
    event = MemoryEvent(
        tenant_id=tenant_id,
        source_type=SourceType.CONFIGURATION,
        source_reference=f"seed-{uuid4()}",
        content=content,
        observed_at=valid_from,
    )
    record = MemoryRecord(
        tenant_id=tenant_id,
        content=content,
        confidence=0.9,
        provenance=[
            Provenance(
                event_id=event.event_id,
                source_type=event.source_type,
                source_reference=event.source_reference,
                observed_at=event.observed_at,
                trust_level=TrustLevel.SYSTEM,
            )
        ],
        temporal_validity=TemporalValidity(valid_from=valid_from, valid_to=valid_to),
        memory_type=MemoryType.SEMANTIC,
        subject_keys=["checkout-api", "timeout"],
        trust_level=TrustLevel.SYSTEM,
        status=MemoryStatus.ACTIVE,
    )
    return event, record


def _index(uow: UnitOfWork, record: MemoryRecord) -> None:
    uow.vectors.set_embedding(
        record.tenant_id,
        record.memory_id,
        _EMBEDDER.embed_texts([record.content])[0],
        model_version=_EMBEDDER.model_version,
    )


def _supersede(uow: UnitOfWork, old: MemoryRecord, content: str) -> MemoryRecord:
    event, replacement = _build(old.tenant_id, content, valid_from=LAST_WEEK)
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


def test_ineligible_memories_never_reach_the_packer(
    uow_factory: UowFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant_id, other_tenant_id = uuid4(), uuid4()
    with uow_factory() as uow:
        stale = _persist(uow, tenant_id, "checkout-api request timeout is 5 seconds")
        current = _supersede(uow, stale, "checkout-api request timeout is 2 seconds")
        blocked = [
            stale,
            _persist(
                uow,
                tenant_id,
                "checkout-api request timeout is 9 seconds per an unverified note",
                status=MemoryStatus.QUARANTINED,
                trust_level=TrustLevel.UNTRUSTED,
            ),
            _persist(
                uow,
                tenant_id,
                "checkout-api request timeout was 4 seconds during the migration",
                status=MemoryStatus.EXPIRED,
            ),
            _persist(
                uow,
                tenant_id,
                "checkout-api request timeout is 3 seconds for the holiday freeze",
                valid_to=YESTERDAY,
            ),
            _persist(uow, other_tenant_id, "checkout-api request timeout is 7 seconds"),
        ]
        uow.commit()

    handed_to_packer: list[UUID] = []
    real_pack_context = context_packer.pack_context

    def spy(memories: Sequence[MemoryRecord], *args: Any, **kwargs: Any) -> PackedContext:
        handed_to_packer.extend(memory.memory_id for memory in memories)
        return real_pack_context(memories, *args, **kwargs)

    monkeypatch.setattr(context_packer, "pack_context", spy)

    query = RetrievalQuery(
        tenant_id=tenant_id, query_text="checkout-api request timeout", limit=PACK_CANDIDATES
    )
    with uow_factory() as uow:
        result = retrieve_context(uow, _EMBEDDER, query, token_budget=500)

    assert handed_to_packer == [current.memory_id]
    assert [packed.memory.memory_id for packed in result.context.memories] == [current.memory_id]
    assert result.context.skipped == ()
    for record in blocked:
        assert record.content not in result.context.text


def test_the_packer_beats_rank_order_on_the_labeled_set(
    uow_factory: UowFactory, real_embedder: EmbeddingModel
) -> None:
    tenant_id = uuid4()
    with uow_factory() as uow:
        ids = seed_corpus(uow, real_embedder, tenant_id)
        uow.commit()

    reports = evaluate_packing(
        uow_factory, real_embedder, tenant_id, ids, budgets=[150, 300, 500]
    )
    print("\n" + format_reports(reports))

    by_key = {(report.strategy, report.token_budget): report for report in reports}
    for budget in (150, 300, 500):
        baseline = by_key["rank order", budget]
        packer = by_key["context packer", budget]
        assert packer.max_tokens <= budget
        assert packer.duplicate_rate < baseline.duplicate_rate
        assert packer.coverage >= baseline.coverage
    assert by_key["context packer", 150].coverage > by_key["rank order", 150].coverage

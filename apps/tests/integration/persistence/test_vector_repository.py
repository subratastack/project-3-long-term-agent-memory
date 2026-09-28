"""Integration tests for VectorRecordRepository.find_nearest against real pgvector.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py).
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
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
from apps.memory_service.retrieval.filters import resolve_filters
from apps.memory_service.retrieval.query_model import RetrievalQuery

UowFactory = Callable[[], UnitOfWork]

# memory_records.embedding is a fixed-dimension pgvector column, so the test
# embedder's dimensionality must match it exactly.
_EMBEDDER = FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)


def _make_event(tenant_id: UUID, **overrides: Any) -> MemoryEvent:
    defaults: dict[str, Any] = {
        "tenant_id": tenant_id,
        "source_type": SourceType.TOOL_OUTPUT,
        "source_reference": "tool-1",
        "content": "connection pool exhausted",
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
        "content": "The connection pool was exhausted after 30 retries.",
        "confidence": 0.9,
        "provenance": [provenance],
        "temporal_validity": TemporalValidity(valid_from=datetime.now(UTC)),
        "memory_type": MemoryType.EPISODIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def _index(uow: UnitOfWork, memory: MemoryRecord) -> None:
    """Compute and store a fake embedding for `memory`, marking it INDEXED."""
    embedding = _EMBEDDER.embed_texts([memory.content])[0]
    uow.vectors.set_embedding(
        memory.tenant_id, memory.memory_id, embedding, model_version=_EMBEDDER.model_version
    )


def test_query_about_pool_exhaustion_finds_the_related_incident(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    unrelated_event = _make_event(
        tenant_id, source_reference="cfg-1", source_type=SourceType.CONFIGURATION,
        content="timeout=2s",
    )
    memory = _make_memory(tenant_id, _make_provenance(event))
    unrelated = _make_memory(
        tenant_id,
        _make_provenance(unrelated_event),
        content="The configured request timeout is 2 seconds.",
        memory_type=MemoryType.SEMANTIC,
    )

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_event(unrelated_event)
        uow.create_memory(memory)
        uow.create_memory(unrelated)
        _index(uow, memory)
        _index(uow, unrelated)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhaustion", limit=5)
    filters = resolve_filters(query)
    query_embedding = _EMBEDDER.embed_texts([query.query_text])[0]

    with uow_factory() as uow:
        matches = uow.vectors.find_nearest(filters, query_embedding, limit=5)

    assert matches
    assert matches[0].memory_id == memory.memory_id


def test_tenant_b_cannot_retrieve_tenant_as_vector_result(uow_factory: UowFactory) -> None:
    tenant_a = uuid4()
    tenant_b = uuid4()
    event = _make_event(tenant_a)
    memory = _make_memory(tenant_a, _make_provenance(event))

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        _index(uow, memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_b, query_text="pool exhaustion", limit=5)
    filters = resolve_filters(query)
    query_embedding = _EMBEDDER.embed_texts([query.query_text])[0]

    with uow_factory() as uow:
        matches = uow.vectors.find_nearest(filters, query_embedding, limit=5)

    assert matches == []


def test_quarantined_superseded_expired_and_tombstoned_records_never_appear(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    now = datetime.now(UTC)
    statuses_that_must_not_appear = [
        MemoryStatus.QUARANTINED,
        MemoryStatus.SUPERSEDED,
        MemoryStatus.EXPIRED,
        MemoryStatus.TOMBSTONE,
    ]

    with uow_factory() as uow:
        for status in statuses_that_must_not_appear:
            event = _make_event(tenant_id, source_reference=f"tool-{status}")
            memory = _make_memory(
                tenant_id,
                _make_provenance(event),
                status=status,
                temporal_validity=TemporalValidity(
                    valid_from=now - timedelta(days=2), valid_to=now - timedelta(days=1)
                )
                if status == MemoryStatus.EXPIRED
                else TemporalValidity(valid_from=now),
            )
            uow.create_event(event)
            uow.create_memory(memory)
            _index(uow, memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhaustion", limit=10)
    filters = resolve_filters(query)
    query_embedding = _EMBEDDER.embed_texts([query.query_text])[0]

    with uow_factory() as uow:
        matches = uow.vectors.find_nearest(filters, query_embedding, limit=10)

    assert matches == []


def test_pending_index_status_is_excluded_from_vector_search(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    memory = _make_memory(tenant_id, _make_provenance(event))

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)  # never indexed -- stays index_status=PENDING
        uow.commit()

    with uow_factory() as uow:
        fetched = uow.get_memory(memory.memory_id, tenant_id)
    assert fetched is not None
    assert fetched.index_status.value == "pending"

    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhaustion", limit=5)
    filters = resolve_filters(query)
    query_embedding = _EMBEDDER.embed_texts([query.query_text])[0]

    with uow_factory() as uow:
        matches = uow.vectors.find_nearest(filters, query_embedding, limit=5)

    assert matches == []

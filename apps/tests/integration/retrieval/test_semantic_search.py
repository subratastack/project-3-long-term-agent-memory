"""Integration tests for the full query -> filters -> search -> fetch flow.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py).
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
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.semantic import semantic_search

UowFactory = Callable[[], UnitOfWork]
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
    embedding = _EMBEDDER.embed_texts([memory.content])[0]
    uow.vectors.set_embedding(
        memory.tenant_id, memory.memory_id, embedding, model_version=_EMBEDDER.model_version
    )


def test_semantic_search_returns_authoritative_records_for_a_related_query(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    memory = _make_memory(tenant_id, _make_provenance(event))

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        _index(uow, memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhaustion", limit=5)
    with uow_factory() as uow:
        hits = semantic_search(uow, _EMBEDDER, query)

    assert len(hits) == 1
    assert hits[0].memory.memory_id == memory.memory_id
    # semantic_search must hand back a full, authoritative MemoryRecord --
    # not just an id -- so its content is directly usable by a caller.
    assert hits[0].memory.content == memory.content


def test_semantic_search_never_crosses_tenants(uow_factory: UowFactory) -> None:
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
    with uow_factory() as uow:
        hits = semantic_search(uow, _EMBEDDER, query)

    assert hits == []


def test_semantic_search_excludes_quarantined_memories(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    memory = _make_memory(tenant_id, _make_provenance(event), status=MemoryStatus.QUARANTINED)

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        _index(uow, memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhaustion", limit=5)
    with uow_factory() as uow:
        hits = semantic_search(uow, _EMBEDDER, query)

    assert hits == []


def test_semantic_search_excludes_pending_index_status(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    memory = _make_memory(tenant_id, _make_provenance(event))

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)  # never indexed
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="pool exhaustion", limit=5)
    with uow_factory() as uow:
        hits = semantic_search(uow, _EMBEDDER, query)

    assert hits == []


def test_semantic_search_respects_a_precomputed_query_embedding(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    memory = _make_memory(tenant_id, _make_provenance(event))

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        _index(uow, memory)
        uow.commit()

    query_embedding = _EMBEDDER.embed_texts(["pool exhaustion"])[0]
    query = RetrievalQuery(tenant_id=tenant_id, query_embedding=query_embedding, limit=5)
    with uow_factory() as uow:
        hits = semantic_search(uow, _EMBEDDER, query)

    assert len(hits) == 1
    assert hits[0].memory.memory_id == memory.memory_id

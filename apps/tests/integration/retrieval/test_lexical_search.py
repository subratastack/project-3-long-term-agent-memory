"""Integration tests for `retrieval.lexical.lexical_search` against real PostgreSQL FTS.

Run via `docker compose up -d postgres` then `uv run pytest`; skips
automatically if PostgreSQL is unreachable (see ../conftest.py). No embedder
is needed here -- lexical search never touches `memory_records.embedding`.
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
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.lexical import lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery

UowFactory = Callable[[], UnitOfWork]


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


def test_exact_service_name_is_found_by_lexical_search(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id, source_reference="tool-checkout")
    memory = _make_memory(
        tenant_id, _make_provenance(event), content="Service checkout-api returned ERR_5001 twice."
    )

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="ERR_5001", limit=5)
    with uow_factory() as uow:
        hits = lexical_search(uow, query)

    assert len(hits) == 1
    assert hits[0].memory.memory_id == memory.memory_id


def test_no_literal_overlap_returns_no_lexical_hits(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    memory = _make_memory(tenant_id, _make_provenance(event))

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        uow.commit()

    query = RetrievalQuery(
        tenant_id=tenant_id, query_text="database ran out of available connections", limit=5
    )
    with uow_factory() as uow:
        hits = lexical_search(uow, query)

    assert hits == []


def test_query_embedding_only_short_circuits_without_touching_the_database(
    uow_factory: UowFactory,
) -> None:
    tenant_id = uuid4()
    query = RetrievalQuery(tenant_id=tenant_id, query_embedding=[0.1, 0.2, 0.3], limit=5)

    with uow_factory() as uow:
        hits = lexical_search(uow, query)

    assert hits == []


def test_tenant_b_cannot_retrieve_tenant_as_lexical_result(uow_factory: UowFactory) -> None:
    tenant_a = uuid4()
    tenant_b = uuid4()
    event = _make_event(tenant_a)
    memory = _make_memory(tenant_a, _make_provenance(event))

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_b, query_text="connection pool exhausted", limit=5)
    with uow_factory() as uow:
        hits = lexical_search(uow, query)

    assert hits == []


def test_quarantined_memory_never_appears_in_lexical_results(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    memory = _make_memory(tenant_id, _make_provenance(event), status=MemoryStatus.QUARANTINED)

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="connection pool exhausted", limit=5)
    with uow_factory() as uow:
        hits = lexical_search(uow, query)

    assert hits == []


def test_expired_validity_window_is_excluded(uow_factory: UowFactory) -> None:
    tenant_id = uuid4()
    event = _make_event(tenant_id)
    now = datetime.now(UTC)
    memory = _make_memory(
        tenant_id,
        _make_provenance(event),
        temporal_validity=TemporalValidity(
            valid_from=now - timedelta(days=2), valid_to=now - timedelta(days=1)
        ),
    )

    with uow_factory() as uow:
        uow.create_event(event)
        uow.create_memory(memory)
        uow.commit()

    query = RetrievalQuery(tenant_id=tenant_id, query_text="connection pool exhausted", limit=5)
    with uow_factory() as uow:
        hits = lexical_search(uow, query)

    assert hits == []

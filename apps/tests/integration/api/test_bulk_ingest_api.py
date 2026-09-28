"""Integration tests for `POST /tenants/{tenant_id}/events/ingest` (bulk ingest).

Same setup as `test_tenant_api.py` -- `TestClient` against the rolled-back
integration database -- plus a `FakeCandidateExtractor` in place of Ollama.
The fake proposes one grounded candidate per event, and raises for any event
whose content contains "boom", so per-event failure isolation can be tested.
The embedder is faked too, because demo seeding indexes what it writes.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from apps.memory_service.api import app as app_module
from apps.memory_service.domain.enums import TrustLevel
from apps.memory_service.domain.models import MemoryEvent
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.ingestion.candidate_extractor import (
    FakeCandidateExtractor,
    RawCandidate,
)
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork

UowFactory = Callable[[], UnitOfWork]


def _one_candidate_per_event(events: Sequence[MemoryEvent]) -> list[RawCandidate]:
    [event] = events
    if "boom" in event.content:
        raise RuntimeError("extractor unavailable")
    return [
        RawCandidate(
            content=f"Remembered: {event.content}",
            source_event_ids=[event.event_id],
            proposed_trust_level=TrustLevel.SYSTEM,
        )
    ]


@pytest.fixture
def client(uow_factory: UowFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(app_module, "get_uow_factory", lambda: uow_factory)
    monkeypatch.setattr(
        app_module, "get_extractor", lambda: FakeCandidateExtractor(_one_candidate_per_event)
    )
    monkeypatch.setattr(
        app_module, "get_embedder", lambda: FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)
    )
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def tenant_id(client: TestClient) -> str:
    response = client.post("/tenants", json={"name": f"tenant-{uuid4()}"})
    assert response.status_code == 201, response.text
    return str(response.json()["tenant_id"])


def _event(content: str, reference: str = "ref", **extra: Any) -> dict[str, Any]:
    return {
        "source_type": "configuration",
        "source_reference": reference,
        "content": content,
        **extra,
    }


def _store(client: TestClient, tenant_id: str, event: dict[str, Any]) -> str:
    response = client.post(f"/tenants/{tenant_id}/events", json=event)
    assert response.status_code == 201, response.text
    return str(response.json()["event_id"])


def test_new_events_are_stored_and_each_ingested(client: TestClient, tenant_id: str) -> None:
    response = client.post(
        f"/tenants/{tenant_id}/events/ingest",
        json={"events": [_event("timeout=2s", "cfg-1"), _event("retries=3", "cfg-2")]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["events_created"], body["events_ingested"], body["events_failed"]) == (2, 2, 0)
    assert [result["source_reference"] for result in body["results"]] == ["cfg-1", "cfg-2"]
    for result in body["results"]:
        [outcome] = result["outcomes"]
        assert (result["status"], result["candidate_count"]) == ("ingested", 1)
        assert outcome["status"] == "accept"
        memory = client.get(f"/tenants/{tenant_id}/memories/{outcome['accepted_memory_id']}")
        assert memory.status_code == 200


def test_already_stored_events_are_ingested_before_new_ones(
    client: TestClient, tenant_id: str
) -> None:
    stored = client.post(f"/tenants/{tenant_id}/events", json=_event("timeout=2s", "stored"))
    stored_id = stored.json()["event_id"]

    response = client.post(
        f"/tenants/{tenant_id}/events/ingest",
        json={"events": [_event("retries=3", "new")], "event_ids": [stored_id]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["events_created"] == 1
    assert [result["source_reference"] for result in body["results"]] == ["stored", "new"]
    assert body["results"][0]["event_id"] == stored_id


def test_one_failing_event_does_not_stop_the_others(client: TestClient, tenant_id: str) -> None:
    response = client.post(
        f"/tenants/{tenant_id}/events/ingest",
        json={"events": [_event("boom", "bad"), _event("timeout=2s", "good")]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["events_ingested"], body["events_failed"]) == (1, 1)
    bad, good = body["results"]
    assert bad["status"] == "failed"
    assert bad["error"] == "RuntimeError: extractor unavailable"
    assert bad["outcomes"] == []
    assert good["status"] == "ingested"


def test_an_unknown_event_id_is_a_404_and_stores_nothing(
    client: TestClient, tenant_id: str, uow_factory: UowFactory
) -> None:
    missing_id = str(uuid4())

    response = client.post(
        f"/tenants/{tenant_id}/events/ingest",
        json={"events": [_event("timeout=2s")], "event_ids": [missing_id]},
    )

    assert response.status_code == 404
    assert missing_id in response.json()["detail"]
    with uow_factory() as uow:
        stored = uow.session.scalar(
            text("SELECT count(*) FROM memory_events WHERE tenant_id = :t"), {"t": tenant_id}
        )
    assert stored == 0


@pytest.mark.parametrize(
    "body",
    [
        {"events": [_event(f"e{i}") for i in range(app_module.MAX_BULK_INGEST_EVENTS + 1)]},
        {"event_ids": ["00000000-0000-0000-0000-000000000001"] * 2},
    ],
    ids=["over-limit", "duplicate-ids"],
)
def test_invalid_requests_are_rejected(
    client: TestClient, tenant_id: str, body: dict[str, Any]
) -> None:
    response = client.post(f"/tenants/{tenant_id}/events/ingest", json=body)

    assert response.status_code == 422


# --- No payload: ingest the tenant's pending events -------------------------


@pytest.mark.parametrize(
    "body",
    [None, {}, {"events": [], "event_ids": []}],
    ids=["no-payload", "empty-object", "empty-lists"],
)
def test_no_payload_ingests_only_events_never_ingested(
    client: TestClient, tenant_id: str, body: dict[str, Any] | None
) -> None:
    done = _store(client, tenant_id, _event("timeout=2s", "done"))
    client.post(f"/tenants/{tenant_id}/events/{done}/ingest").raise_for_status()
    pending = _store(client, tenant_id, _event("retries=3", "pending"))

    response = client.post(f"/tenants/{tenant_id}/events/ingest", json=body)

    assert response.status_code == 200, response.text
    body_out = response.json()
    assert [result["event_id"] for result in body_out["results"]] == [pending]
    assert (body_out["events_created"], body_out["events_pending"]) == (0, 0)


def test_a_second_no_payload_call_finds_nothing_left(client: TestClient, tenant_id: str) -> None:
    _store(client, tenant_id, _event("timeout=2s"))
    client.post(f"/tenants/{tenant_id}/events/ingest").raise_for_status()

    response = client.post(f"/tenants/{tenant_id}/events/ingest")

    assert response.json()["results"] == []
    assert response.json()["events_pending"] == 0


def test_pending_events_go_oldest_first_in_batches(
    client: TestClient, tenant_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "MAX_BULK_INGEST_EVENTS", 2)
    for reference, day in [("third", 3), ("first", 1), ("second", 2)]:
        observed_at = f"2026-01-0{day}T00:00:00Z"
        _store(client, tenant_id, _event(reference, reference, observed_at=observed_at))

    first = client.post(f"/tenants/{tenant_id}/events/ingest").json()
    second = client.post(f"/tenants/{tenant_id}/events/ingest").json()

    assert [result["source_reference"] for result in first["results"]] == ["first", "second"]
    assert first["events_pending"] == 1
    assert [result["source_reference"] for result in second["results"]] == ["third"]
    assert second["events_pending"] == 0


def test_a_failed_event_stays_pending_for_the_next_call(
    client: TestClient, tenant_id: str
) -> None:
    failing = _store(client, tenant_id, _event("boom"))

    response = client.post(f"/tenants/{tenant_id}/events/ingest").json()

    assert response["results"][0]["status"] == "failed"
    assert response["events_pending"] == 1
    retry = client.post(f"/tenants/{tenant_id}/events/ingest").json()
    assert [result["event_id"] for result in retry["results"]] == [failing]


def test_demo_seeded_events_are_not_pending(client: TestClient, tenant_id: str) -> None:
    client.post(f"/tenants/{tenant_id}/demo/seed").raise_for_status()

    response = client.post(f"/tenants/{tenant_id}/events/ingest").json()

    assert (response["results"], response["events_pending"]) == ([], 0)


def test_events_ingested_by_payload_are_not_pending_afterwards(
    client: TestClient, tenant_id: str
) -> None:
    client.post(
        f"/tenants/{tenant_id}/events/ingest", json={"events": [_event("timeout=2s")]}
    ).raise_for_status()

    response = client.post(f"/tenants/{tenant_id}/events/ingest").json()

    assert (response["results"], response["events_pending"]) == ([], 0)

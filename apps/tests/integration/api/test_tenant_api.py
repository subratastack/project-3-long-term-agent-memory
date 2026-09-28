"""Integration tests for the dev API's tenant registry and tenant enforcement.

Runs the FastAPI app in-process (`TestClient`) against the integration test
database: `get_uow_factory` is patched to the per-test, rolled-back
`uow_factory` fixture, and the embedder / CrossEncoder to their deterministic
fakes, so nothing touches the dev database or downloads a model.
"""

from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.memory_service.api import app as app_module
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.embeddings.cross_encoder import FakeCrossEncoderModel
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.reranker import CrossEncoderReranker

UowFactory = Callable[[], UnitOfWork]


@pytest.fixture
def client(uow_factory: UowFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(app_module, "get_uow_factory", lambda: uow_factory)
    monkeypatch.setattr(
        app_module, "get_embedder", lambda: FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)
    )
    monkeypatch.setattr(
        app_module, "get_reranker", lambda: CrossEncoderReranker(FakeCrossEncoderModel())
    )
    with TestClient(app_module.app) as test_client:
        yield test_client


def _create(client: TestClient, name: str | None = None) -> dict[str, str]:
    response = client.post(
        "/tenants", json={"name": name or f"tenant-{uuid4()}", "description": "demo"}
    )
    assert response.status_code == 201, response.text
    body: dict[str, str] = response.json()
    return body


def test_create_then_fetch_by_id_and_by_name(client: TestClient) -> None:
    created = _create(client)

    assert client.get(f"/tenants/{created['tenant_id']}").json() == created
    assert client.get("/tenants", params={"name": created["name"]}).json() == [created]
    assert created in client.get("/tenants").json()


def test_duplicate_names_are_rejected(client: TestClient) -> None:
    name = f"tenant-{uuid4()}"
    _create(client, name)

    response = client.post("/tenants", json={"name": name})

    assert response.status_code == 409


def test_blank_names_are_rejected(client: TestClient) -> None:
    assert client.post("/tenants", json={"name": "   "}).status_code == 422


def test_an_unknown_name_lists_nothing(client: TestClient) -> None:
    assert client.get("/tenants", params={"name": f"missing-{uuid4()}"}).json() == []


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/tenants/{t}", None),
        (
            "post",
            "/tenants/{t}/events",
            {"source_type": "user_message", "source_reference": "m1", "content": "hi"},
        ),
        (
            "post",
            "/tenants/{t}/events/ingest",
            {
                "events": [
                    {"source_type": "user_message", "source_reference": "m1", "content": "hi"}
                ]
            },
        ),
        ("get", "/tenants/{t}/memories/{m}", None),
        ("post", "/tenants/{t}/memories/index", None),
        ("post", "/tenants/{t}/retrieve", {"query_text": "timeout"}),
        ("post", "/tenants/{t}/context", {"query_text": "timeout"}),
        ("post", "/tenants/{t}/demo/seed", None),
    ],
)
def test_tenant_scoped_routes_refuse_an_unregistered_tenant(
    client: TestClient, method: str, path: str, body: dict[str, str] | None
) -> None:
    url = path.format(t=uuid4(), m=uuid4())

    response = client.request(method, url, json=body)

    assert response.status_code == 404
    assert "tenant not found" in response.json()["detail"]


def test_a_registered_tenant_can_be_seeded_and_queried(client: TestClient) -> None:
    tenant_id = _create(client)["tenant_id"]

    seeded = client.post(f"/tenants/{tenant_id}/demo/seed")
    result = client.post(
        f"/tenants/{tenant_id}/retrieve",
        json={"query_text": "request timeout 5 seconds", "limit": 1},
    )

    assert seeded.status_code == 201
    assert result.status_code == 200
    assert [hit["content"] for hit in result.json()["hits"]] == [
        "The request timeout is 2 seconds."
    ]


def test_a_seeded_tenant_gets_a_packed_context_within_budget(client: TestClient) -> None:
    tenant_id = _create(client)["tenant_id"]
    client.post(f"/tenants/{tenant_id}/demo/seed").raise_for_status()

    response = client.post(
        f"/tenants/{tenant_id}/context",
        json={"query_text": "request timeout 5 seconds", "token_budget": 120},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 < body["stats"]["token_count"] <= 120
    assert body["stats"]["selected"] == len(body["memories"]) > 0
    assert "The request timeout is 2 seconds." in body["context"]
    assert "The request timeout is 5 seconds." not in body["context"]

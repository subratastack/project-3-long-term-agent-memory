"""Tenant B must never see, cite, or link to anything of tenant A's.

Tenant A holds memories (with evidence, relations, decisions, and
embeddings); tenant B queries for exactly that content through every read
path. Leakage is counted across all of them and must be exactly zero --
including when a search stage is made to misbehave, and when B tries to
write references into A's data.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from apps.memory_service.api import app as app_module
from apps.memory_service.consolidation.conflict_resolver import record_contradiction
from apps.memory_service.domain.enums import (
    ConflictType,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)
from apps.memory_service.domain.models import (
    MemoryCandidate,
    MemoryEvent,
    MemoryRecord,
    MemoryRelation,
    Provenance,
    TemporalValidity,
    WritePolicyDecision,
)
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.embeddings.cross_encoder import FakeCrossEncoderModel
from apps.memory_service.ingestion.service import ingest_candidate
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval import hybrid as hybrid_module
from apps.memory_service.retrieval.context_packer import retrieve_context
from apps.memory_service.retrieval.hybrid import HybridSearchHit, hybrid_search
from apps.memory_service.retrieval.lexical import lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import CrossEncoderReranker
from apps.memory_service.retrieval.semantic import semantic_search
from apps.memory_service.retrieval.temporal import TemporalReport, apply_temporal_resolution
from apps.memory_service.security.tenant_scope import (
    ScopedEntity,
    TenantScope,
    TenantScopeError,
)

UowFactory = Callable[[], UnitOfWork]

EMBEDDER = FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)
JAN_1 = datetime(2026, 1, 1, tzinfo=UTC)

A_CONTENT = [
    "The request timeout for checkout-api is 2 seconds.",
    "Incident INC-3101: checkout-api returned HTTP 503 for 12 minutes.",
    "The user prefers email over phone calls.",
    "Runbook: drain traffic, then restart the connection pool.",
    "The deploy region is eu-west-1.",
    "Nightly backup of db-primary completed successfully.",
]
B_CONTENT = [
    "The office plants are watered on Fridays.",
    # Identical text in both tenants must still resolve to B's own record.
    "The user prefers email over phone calls.",
]


@dataclass(frozen=True)
class Seeded:
    tenant_id: UUID
    events: list[MemoryEvent]
    records: list[MemoryRecord]
    relations: list[MemoryRelation]
    decisions: list[WritePolicyDecision]

    @property
    def memory_ids(self) -> set[UUID]:
        return {record.memory_id for record in self.records}


def _seed(
    uow_factory: UowFactory, contents: Sequence[str], tenant_id: UUID | None = None
) -> Seeded:
    tenant_id = tenant_id or uuid4()
    events: list[MemoryEvent] = []
    records: list[MemoryRecord] = []
    decisions: list[WritePolicyDecision] = []
    with uow_factory() as uow:
        for content in contents:
            event = MemoryEvent(
                tenant_id=tenant_id,
                source_type=SourceType.CONFIGURATION,
                source_reference=f"seed-{uuid4()}",
                content=content,
                observed_at=JAN_1,
            )
            record = MemoryRecord(
                tenant_id=tenant_id,
                content=content,
                confidence=0.9,
                provenance=[_link(event)],
                temporal_validity=TemporalValidity(valid_from=JAN_1),
                memory_type=MemoryType.SEMANTIC,
                subject_keys=["service:checkout-api"],
                trust_level=TrustLevel.SYSTEM,
                status=MemoryStatus.ACTIVE,
            )
            decision = WritePolicyDecision(
                tenant_id=tenant_id,
                candidate_id=uuid4(),
                decision=WriteDecision.ACCEPT,
                policy_version="test",
                reason_codes=["TRUSTED_SEMANTIC_CLAIM"],
                accepted_memory_id=record.memory_id,
            )
            uow.create_event(event)
            uow.create_memory(record)
            uow.record_write_decision(decision)
            uow.vectors.set_embedding(
                tenant_id,
                record.memory_id,
                EMBEDDER.embed_texts([content])[0],
                model_version=EMBEDDER.model_version,
            )
            events.append(event)
            records.append(record)
            decisions.append(decision)
        relations = []
        if len(records) >= 2:
            relations.append(
                record_contradiction(
                    uow, tenant_id, records[0].memory_id, records[1].memory_id, rationale="seed"
                )
            )
        uow.commit()
    return Seeded(tenant_id, events, records, relations, decisions)


def _link(event: MemoryEvent) -> Provenance:
    return Provenance(
        event_id=event.event_id,
        source_type=event.source_type,
        source_reference=event.source_reference,
        observed_at=event.observed_at,
        trust_level=TrustLevel.SYSTEM,
    )


@pytest.fixture
def tenants(uow_factory: UowFactory) -> tuple[Seeded, Seeded]:
    return _seed(uow_factory, A_CONTENT), _seed(uow_factory, B_CONTENT)


def _leaks(results: Sequence[MemoryRecord], reader: Seeded, other: Seeded) -> int:
    """Results the reader must not see: another tenant's, or the other tenant's ids."""
    return sum(
        1 for r in results if r.tenant_id != reader.tenant_id or r.memory_id in other.memory_ids
    )


def _every_read_path(
    uow_factory: UowFactory, reader: Seeded, query_text: str, vector: list[float]
) -> list[MemoryRecord]:
    """Everything `reader` gets back for one query, from every retrieval entry point."""
    reranker = CrossEncoderReranker(FakeCrossEncoderModel())
    results: list[MemoryRecord] = []
    with uow_factory() as uow:
        for as_of in (None, JAN_1 + timedelta(days=30)):
            text_query = RetrievalQuery(
                tenant_id=reader.tenant_id, query_text=query_text, as_of=as_of, limit=20
            )
            vector_query = RetrievalQuery(
                tenant_id=reader.tenant_id, query_embedding=vector, as_of=as_of, limit=20
            )
            results += [hit.memory for hit in lexical_search(uow, text_query)]
            results += [hit.memory for hit in semantic_search(uow, EMBEDDER, text_query)]
            results += [hit.memory for hit in semantic_search(uow, EMBEDDER, vector_query)]
            results += [hit.memory for hit in hybrid_search(uow, EMBEDDER, text_query)]
            results += [
                hit.memory for hit in hybrid_search(uow, EMBEDDER, text_query, reranker=reranker)
            ]
            context = retrieve_context(uow, EMBEDDER, text_query)
            results += [packed.memory for packed in context.context.memories]
        results += uow.list_active_memories(reader.tenant_id)
        results += uow.records.list_for_maintenance(reader.tenant_id)
    return results


def test_tenant_b_querying_tenant_a_memory_gets_zero_results(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded]
) -> None:
    a, b = tenants
    leaked = 0
    probes = 0
    for record in a.records:
        vector = EMBEDDER.embed_texts([record.content])[0]
        results = _every_read_path(uow_factory, b, record.content, vector)
        leaked += _leaks(results, reader=b, other=a)
        probes += 1

        # The same probes do find A's memory for A, so B's empty answer is real.
        assert record.memory_id in {
            r.memory_id for r in _every_read_path(uow_factory, a, record.content, vector)
        }

    assert probes == len(A_CONTENT)
    assert leaked == 0


def test_identical_content_resolves_to_each_tenants_own_record(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded]
) -> None:
    a, b = tenants
    shared = "The user prefers email over phone calls."
    vector = EMBEDDER.embed_texts([shared])[0]

    b_results = _every_read_path(uow_factory, b, shared, vector)

    assert {r.memory_id for r in b_results if r.content == shared} == {
        r.memory_id for r in b.records if r.content == shared
    }
    assert _leaks(b_results, reader=b, other=a) == 0


def test_direct_lookups_of_tenant_a_ids_return_nothing_for_tenant_b(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded]
) -> None:
    a, b = tenants
    with uow_factory() as uow:
        for record in a.records:
            assert uow.get_memory(record.memory_id, b.tenant_id) is None
            assert uow.provenance.list_for_memory(b.tenant_id, record.memory_id) == []
            assert uow.relations.list_for_memory(b.tenant_id, record.memory_id) == []
        for event in a.events:
            assert uow.events.get(b.tenant_id, event.event_id) is None
        for decision in a.decisions:
            assert uow.write_decisions.get(b.tenant_id, decision.decision_id) is None
        assert uow.relations.list_for_memories(b.tenant_id, a.memory_ids) == []


def test_a_candidate_citing_another_tenants_event_is_rejected_and_audited(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded]
) -> None:
    a, b = tenants
    stolen = a.events[0]
    candidate = MemoryCandidate(
        tenant_id=b.tenant_id,
        content=stolen.content,
        provenance=[_link(stolen)],
        memory_type=MemoryType.SEMANTIC,
        confidence=0.99,
    )

    decision = ingest_candidate(uow_factory, candidate, tenant_id=b.tenant_id)

    assert decision.decision is WriteDecision.REJECT
    assert "EVENT_NOT_FOUND" in decision.reason_codes
    with uow_factory() as uow:
        assert uow.write_decisions.list_for_candidate(b.tenant_id, candidate.candidate_id)
        assert {m.memory_id for m in uow.list_active_memories(b.tenant_id)} == b.memory_ids


def test_ingestion_refuses_a_candidate_owned_by_another_tenant(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded]
) -> None:
    a, b = tenants
    candidate = MemoryCandidate(
        tenant_id=a.tenant_id,
        content="Planted by tenant B.",
        provenance=[_link(a.events[0])],
        memory_type=MemoryType.SEMANTIC,
        confidence=0.99,
    )

    with pytest.raises(TenantScopeError):
        ingest_candidate(uow_factory, candidate, tenant_id=b.tenant_id)

    with uow_factory() as uow:
        assert uow.write_decisions.list_for_candidate(a.tenant_id, candidate.candidate_id) == []
        assert {m.memory_id for m in uow.list_active_memories(a.tenant_id)} == a.memory_ids


def test_a_provenance_link_into_another_tenant_is_refused_by_the_database(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded]
) -> None:
    a, b = tenants
    record = MemoryRecord(
        tenant_id=b.tenant_id,
        content="Borrowed evidence.",
        confidence=0.9,
        provenance=[_link(a.events[0])],
        temporal_validity=TemporalValidity(valid_from=JAN_1),
        memory_type=MemoryType.SEMANTIC,
        trust_level=TrustLevel.SYSTEM,
        status=MemoryStatus.ACTIVE,
    )
    events_by_id = {event.event_id: event for event in a.events}

    violations = TenantScope(b.tenant_id).check_record(record, events_by_id)
    assert [v.entity for v in violations] == [ScopedEntity.PROVENANCE_LINK]

    with pytest.raises(IntegrityError), uow_factory() as uow:
        uow.create_memory(record)
        uow.commit()


def test_a_relation_into_another_tenant_is_refused(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded]
) -> None:
    a, b = tenants
    cross = MemoryRelation(
        tenant_id=b.tenant_id,
        source_memory_id=b.records[0].memory_id,
        target_memory_id=a.records[0].memory_id,
        relation_type=ConflictType.CONTRADICTION,
    )
    records = {r.memory_id: r for r in [*a.records, *b.records]}

    violations = TenantScope(b.tenant_id).check_relation(cross, records)
    assert [(v.entity, v.found_tenant_id) for v in violations] == [
        (ScopedEntity.MEMORY, a.tenant_id)
    ]

    with pytest.raises(IntegrityError), uow_factory() as uow:
        uow.relations.add(b.tenant_id, cross)
        uow.commit()
    with pytest.raises(ValueError), uow_factory() as uow:
        uow.relations.add(b.tenant_id, cross.model_copy(update={"tenant_id": a.tenant_id}))


def test_a_regressed_search_stage_still_cannot_leak(
    uow_factory: UowFactory, tenants: tuple[Seeded, Seeded], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a bug that hands tenant A's hits to tenant B after temporal resolution."""
    a, b = tenants
    real = apply_temporal_resolution

    def leaky(
        uow: UnitOfWork, query: RetrievalQuery, hits: Sequence[HybridSearchHit]
    ) -> tuple[list[HybridSearchHit], TemporalReport]:
        kept, report = real(uow, query, hits)
        foreign = [
            HybridSearchHit(memory=record, fused_score=1.0, lexical_rank=1, semantic_rank=1)
            for record in a.records
        ]
        return [*foreign, *kept], report

    monkeypatch.setattr(hybrid_module, "apply_temporal_resolution", leaky)

    with uow_factory() as uow:
        hits = hybrid_search(
            uow, EMBEDDER, RetrievalQuery(tenant_id=b.tenant_id, query_text=a.records[0].content)
        )

    assert _leaks([hit.memory for hit in hits], reader=b, other=a) == 0


# --- HTTP boundary -------------------------------------------------------------


@pytest.fixture
def client(uow_factory: UowFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(app_module, "get_uow_factory", lambda: uow_factory)
    monkeypatch.setattr(app_module, "get_embedder", lambda: EMBEDDER)
    monkeypatch.setattr(
        app_module, "get_reranker", lambda: CrossEncoderReranker(FakeCrossEncoderModel())
    )
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_the_api_never_serves_tenant_a_data_to_tenant_b(
    client: TestClient, uow_factory: UowFactory
) -> None:
    tenant_ids = []
    for _ in range(2):
        response = client.post("/tenants", json={"name": f"tenant-{uuid4()}"})
        assert response.status_code == 201, response.text
        tenant_ids.append(UUID(response.json()["tenant_id"]))
    a_id, b_id = tenant_ids
    a = _seed(uow_factory, A_CONTENT, tenant_id=a_id)

    leaked = 0
    for record, event in zip(a.records, a.events, strict=True):
        assert client.get(f"/tenants/{b_id}/memories/{record.memory_id}").status_code == 404
        assert client.post(f"/tenants/{b_id}/events/{event.event_id}/ingest").status_code == 404
        for path in ("retrieve", "context"):
            response = client.post(f"/tenants/{b_id}/{path}", json={"query_text": record.content})
            assert response.status_code == 200, response.text
            body = response.json()
            body.pop("query", None)  # /retrieve echoes B's own request back
            served = json.dumps(body)
            leaked += sum(str(m) in served for m in a.memory_ids)
            leaked += record.content in served

    assert leaked == 0
    # The same request as tenant A does return A's memory. (The last record is
    # outside the seeded contradiction, which withholds the first two.)
    mine = a.records[-1]
    own = client.post(f"/tenants/{a_id}/retrieve", json={"query_text": mine.content})
    assert str(mine.memory_id) in json.dumps(own.json()["hits"])

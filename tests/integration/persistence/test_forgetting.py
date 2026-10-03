"""Lifecycle guarantees through real PostgreSQL search and persistence paths."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from apps.memory_service.consolidation.forgetting import maintain_memories, tombstone_memory
from apps.memory_service.consolidation.service import consolidate_memories
from apps.memory_service.domain.enums import (
    IndexStatus,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
)
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)
from apps.memory_service.persistence.models import MemoryRecordRow
from apps.memory_service.retrieval.context_packer import retrieve_context
from apps.memory_service.retrieval.hybrid import hybrid_search
from apps.memory_service.retrieval.lexical import lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import CrossEncoderReranker
from apps.memory_service.retrieval.semantic import semantic_search

NOW = datetime.now(UTC)
VECTOR = [1.0] + [0.0] * 383


class Embedder:
    model_version = "test"
    dimensions = 384

    def embed_texts(self, texts):
        return [VECTOR for _ in texts]


class RankerModel:
    model_version = "test"

    def __init__(self):
        self.seen = []

    def score_pairs(self, query, documents):
        self.seen.extend(documents)
        return [-2.0] * len(documents)


def seed(uow_factory, *, count=1, tenant_id=None, valid_to=None, confidence=0.9, age=20):
    tenant_id = tenant_id or uuid4()
    records = []
    with uow_factory() as uow:
        for i in range(count):
            observed = NOW - timedelta(days=age, hours=i)
            event = MemoryEvent(
                tenant_id=tenant_id,
                source_type=SourceType.SYSTEM_EVENT,
                source_reference=str(uuid4()),
                observed_at=observed,
                content="pool exhaustion",
            )
            record = MemoryRecord(
                tenant_id=tenant_id,
                content="pool exhaustion",
                confidence=confidence,
                provenance=[
                    Provenance(
                        event_id=event.event_id,
                        source_type=event.source_type,
                        source_reference=event.source_reference,
                        observed_at=observed,
                        trust_level=TrustLevel.SYSTEM,
                    )
                ],
                temporal_validity=TemporalValidity(valid_from=observed, valid_to=valid_to),
                memory_type=MemoryType.EPISODIC,
                status=MemoryStatus.ACTIVE,
                trust_level=TrustLevel.SYSTEM,
                subject_keys=["service:payments"],
                metadata={"category_key": "pool exhaustion"},
                created_at=NOW,
                updated_at=NOW,
            )
            uow.create_event(event)
            uow.create_memory(record)
            uow.vectors.set_embedding(tenant_id, record.memory_id, VECTOR, model_version="test")
            records.append(record)
        uow.commit()
    return records


def query_for(record, **changes):
    return RetrievalQuery(tenant_id=record.tenant_id, query_text="pool exhaustion", **changes)


def assert_absent(uow_factory, record, *, as_of=None):
    query = query_for(record, as_of=as_of)
    model = RankerModel()
    with uow_factory() as uow:
        assert lexical_search(uow, query) == []
        assert semantic_search(uow, Embedder(), query) == []
        assert hybrid_search(uow, Embedder(), query, reranker=CrossEncoderReranker(model)) == []
        context = retrieve_context(uow, Embedder(), query, reranker=CrossEncoderReranker(model))
        assert context.context.memories == ()
        assert context.context.text == ""
    assert model.seen == []


def test_expiry_job_idempotent_and_history_retrievable(uow_factory):
    record = seed(uow_factory, valid_to=NOW - timedelta(days=1))[0]
    first = maintain_memories(uow_factory, record.tenant_id, now=NOW)
    assert first == {"expired": [record.memory_id], "decayed": [], "compacted": []}
    with uow_factory() as uow:
        expired = uow.get_memory(record.memory_id, record.tenant_id)
        assert expired.status == MemoryStatus.EXPIRED
        assert expired.provenance == record.provenance
        assert expired.content == record.content
    second = maintain_memories(uow_factory, record.tenant_id, now=NOW + timedelta(hours=1))
    assert second == {"expired": [], "decayed": [], "compacted": []}
    assert_absent(uow_factory, record)
    with uow_factory() as uow:
        historical = query_for(record, as_of=NOW - timedelta(days=2))
        assert lexical_search(uow, historical)[0].memory.memory_id == record.memory_id
        assert semantic_search(uow, Embedder(), historical)[0].memory.memory_id == record.memory_id
        assert hybrid_search(uow, Embedder(), historical)[0].memory.memory_id == record.memory_id
        assert uow.get_memory(record.memory_id, record.tenant_id).updated_at == expired.updated_at


def test_tombstone_blocks_all_search_and_index_reconciliation(uow_factory):
    record = seed(uow_factory)[0]
    changed = tombstone_memory(
        uow_factory,
        record.tenant_id,
        record.memory_id,
        reason="privacy request",
        requested_by="user:42",
        now=NOW,
    )
    assert changed == [record.memory_id]
    assert (
        tombstone_memory(
            uow_factory,
            record.tenant_id,
            record.memory_id,
            reason="retry",
            requested_by="job:retry",
            now=NOW,
        )
        == []
    )
    assert_absent(uow_factory, record)
    assert_absent(uow_factory, record, as_of=NOW - timedelta(days=2))
    with uow_factory() as uow:
        tomb = uow.get_memory(record.memory_id, record.tenant_id)
        assert tomb.provenance == record.provenance and tomb.content == record.content
        assert tomb.metadata["forgetting"]["tombstone"]["reason"] == "privacy request"
        assert tomb.embedding is None and tomb.index_status == IndexStatus.PENDING
        with pytest.raises(LookupError):
            uow.vectors.set_embedding(
                record.tenant_id, record.memory_id, VECTOR, model_version="retry"
            )
        with pytest.raises(LookupError):
            uow.records.update_status(record.tenant_id, record.memory_id, MemoryStatus.ACTIVE)
        # A stale index artifact cannot bypass authoritative status filtering.
        row = uow.session.scalars(
            select(MemoryRecordRow).where(MemoryRecordRow.memory_id == record.memory_id)
        ).one()
        row.embedding, row.index_status = VECTOR, IndexStatus.INDEXED
        uow.commit()
    assert_absent(uow_factory, record)


def test_consolidation_compacts_and_deletion_cannot_repromote(uow_factory):
    episodes = seed(uow_factory, count=3)
    tenant = episodes[0].tenant_id
    result = consolidate_memories(uow_factory, tenant, now=NOW)
    summary_id = result[0].accepted_memory_id
    with uow_factory() as uow:
        summary = uow.get_memory(summary_id, tenant)
        for episode in episodes:
            stored = uow.get_memory(episode.memory_id, tenant)
            assert stored.metadata["forgetting"]["priority"] == 0.25
            assert stored.provenance == episode.provenance
            assert stored.content == episode.content
            assert stored.status == MemoryStatus.ACTIVE
    assert maintain_memories(uow_factory, tenant, now=NOW) == {
        "expired": [],
        "decayed": [],
        "compacted": [],
    }
    changed = tombstone_memory(
        uow_factory,
        tenant,
        summary_id,
        reason="remove learned claim",
        requested_by="user:42",
        now=NOW,
    )
    assert set(changed) == {summary_id, *(m.memory_id for m in episodes)}
    assert consolidate_memories(uow_factory, tenant, now=NOW) == []
    assert_absent(uow_factory, summary)
    with uow_factory() as uow:
        with pytest.raises(ValueError, match="tombstoned evidence"):
            uow.create_memory(episodes[0].model_copy(update={"memory_id": uuid4()}))
        assert len(uow.records.list_for_maintenance(tenant)) == 4


def test_tombstone_source_cascades_to_existing_summary(uow_factory):
    episodes = seed(uow_factory, count=2)
    tenant = episodes[0].tenant_id
    summary_id = consolidate_memories(uow_factory, tenant, now=NOW)[0].accepted_memory_id
    changed = tombstone_memory(
        uow_factory,
        tenant,
        episodes[0].memory_id,
        reason="source removed",
        requested_by="policy:retention",
        now=NOW,
    )
    assert summary_id in changed
    assert_absent(uow_factory, episodes[0])


def test_decay_changes_search_order_before_limit_and_keeps_evidence(uow_factory):
    old = seed(uow_factory, confidence=0.4, age=120)[0]
    fresh = seed(uow_factory, tenant_id=old.tenant_id, confidence=0.4, age=1)[0]
    result = maintain_memories(uow_factory, old.tenant_id, now=NOW)
    assert result["decayed"] == [old.memory_id]
    assert maintain_memories(uow_factory, old.tenant_id, now=NOW)["decayed"] == []
    with uow_factory() as uow:
        query = query_for(old, limit=1)
        assert lexical_search(uow, query)[0].memory.memory_id == fresh.memory_id
        assert semantic_search(uow, Embedder(), query)[0].memory.memory_id == fresh.memory_id
        assert hybrid_search(uow, Embedder(), query)[0].memory.memory_id == fresh.memory_id
        stored = uow.get_memory(old.memory_id, old.tenant_id)
        assert stored.provenance == old.provenance and stored.content == old.content
        assert stored.metadata["forgetting"]["priority"] == 0.5


def test_tenant_isolation_and_tombstone_rollback(uow_factory):
    a, b = seed(uow_factory)[0], seed(uow_factory)[0]
    with pytest.raises(LookupError):
        tombstone_memory(
            uow_factory,
            b.tenant_id,
            a.memory_id,
            reason="wrong tenant",
            requested_by="user:42",
            now=NOW,
        )
    with uow_factory() as uow:
        assert uow.get_memory(a.memory_id, a.tenant_id).status == MemoryStatus.ACTIVE

    # Simulate a failure after writing the entire closure, before commit.
    def broken_factory():
        uow = uow_factory()

        def fail():
            raise RuntimeError("commit failed")

        uow.commit = fail
        return uow

    with pytest.raises(RuntimeError, match="commit failed"):
        tombstone_memory(
            broken_factory,
            a.tenant_id,
            a.memory_id,
            reason="request",
            requested_by="user:42",
            now=NOW,
        )
    with uow_factory() as uow:
        assert uow.get_memory(a.memory_id, a.tenant_id).status == MemoryStatus.ACTIVE


def test_maintenance_cli_twice_and_all_tenants(uow_factory, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    from scripts import expire_memories

    record = seed(uow_factory, valid_to=NOW - timedelta(days=1))[0]
    # Route the CLI through this test's real PostgreSQL savepoint connection.
    monkeypatch.setattr(
        expire_memories, "build_engine", lambda: SimpleNamespace(dispose=lambda: None)
    )
    monkeypatch.setattr(expire_memories, "build_session_factory", lambda engine: None)
    monkeypatch.setattr(expire_memories, "UnitOfWork", lambda factory: uow_factory())
    assert expire_memories.main(["--tenant-id", str(record.tenant_id)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["expired"] == [str(record.memory_id)]
    assert expire_memories.main(["--all-tenants"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["expired"] == [] and second["decayed"] == [] and second["compacted"] == []


def test_reranker_deletion_is_revalidated_before_context(uow_factory):
    record = seed(uow_factory)[0]
    with uow_factory() as uow:

        class DeletingModel(RankerModel):
            def score_pairs(self, query, documents):
                # A deletion occurring while inference runs must be checked again.
                uow.records.update_status(
                    record.tenant_id, record.memory_id, MemoryStatus.TOMBSTONE
                )
                return super().score_pairs(query, documents)

        result = retrieve_context(
            uow, Embedder(), query_for(record), reranker=CrossEncoderReranker(DeletingModel())
        )
        assert result.context.memories == () and result.context.text == ""


def test_quarantined_sources_are_not_compacted(uow_factory):
    records = seed(uow_factory, count=3)
    with uow_factory() as uow:
        for record in records:
            uow.records.update_status(record.tenant_id, record.memory_id, MemoryStatus.QUARANTINED)
        uow.commit()
    assert maintain_memories(uow_factory, records[0].tenant_id, now=NOW) == {
        "expired": [],
        "decayed": [],
        "compacted": [],
    }

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest
from conftest import NOW

from apps.memory_service.consolidation.forgetting import (
    compact_record,
    decay_priority,
    decay_record,
    expire_record,
    retrieval_priority,
    tombstone_record,
)
from apps.memory_service.domain.enums import MemoryStatus, MemoryType
from apps.memory_service.retrieval.context_packer import pack_context
from apps.memory_service.retrieval.filters import record_matches_filters, resolve_filters
from apps.memory_service.retrieval.hybrid import HybridSearchHit, _fuse_rrf
from apps.memory_service.retrieval.lexical import LexicalSearchHit
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import CrossEncoderReranker, apply_reranker


def test_expiry_boundary_history_and_idempotency(episodes):
    record = episodes(1)[0]
    record.temporal_validity.valid_to = NOW
    assert expire_record(record, NOW) is None  # existing inclusive end semantics
    expired = expire_record(record, NOW + timedelta(microseconds=1))
    assert expired.status == MemoryStatus.EXPIRED
    assert expired.content == record.content
    assert expired.provenance == record.provenance
    assert expired.temporal_validity == record.temporal_validity
    assert expire_record(expired, NOW + timedelta(days=1)) is None
    query = RetrievalQuery(tenant_id=record.tenant_id, query_text="pool")
    assert not record_matches_filters(expired, resolve_filters(query))
    historical = query.model_copy(update={"as_of": NOW - timedelta(days=1)})
    assert record_matches_filters(expired, resolve_filters(historical))


@pytest.mark.parametrize(
    "status", [MemoryStatus.QUARANTINED, MemoryStatus.TOMBSTONE, MemoryStatus.SUPERSEDED]
)
def test_expiry_does_not_widen_historical_access(episodes, status):
    record = episodes(1, status=status)[0]
    record.temporal_validity.valid_to = NOW - timedelta(days=1)
    assert expire_record(record, NOW) is None


def test_decay_uses_evidence_age_not_job_repetition(episodes):
    record = episodes(1, confidence=0.5)[0]
    record.provenance[0].observed_at = NOW - timedelta(days=120)
    original = deepcopy(record)
    assert decay_priority(record, NOW) == 0.5
    decayed = decay_record(record, NOW)
    assert retrieval_priority(decayed) == 0.5
    assert decay_record(decayed, NOW) is None
    assert decay_record(decayed, NOW + timedelta(hours=1)) is None
    assert decay_priority(record, NOW + timedelta(days=90)) == 0.25
    assert decayed.provenance == record.provenance
    assert decayed.content == record.content
    assert decayed.confidence == record.confidence
    assert record == original


@pytest.mark.parametrize(
    "kind,confidence",
    [(MemoryType.SEMANTIC, 0.2), (MemoryType.PROCEDURAL, 0.2), (MemoryType.EPISODIC, 0.8)],
)
def test_decay_preserves_higher_value_and_non_episodes(episodes, kind, confidence):
    record = episodes(1, memory_type=kind, confidence=confidence)[0]
    record.provenance[0].observed_at = NOW - timedelta(days=1000)
    assert decay_record(record, NOW) is None


def test_tombstone_keeps_first_audit_and_never_matches_history(episodes):
    record = episodes(1)[0]
    tomb = tombstone_record(
        record, reason="User requested removal", requested_by="user:42", now=NOW
    )
    assert tomb.metadata["forgetting"]["tombstone"]["requested_by"] == "user:42"
    assert tomb.metadata["forgetting"]["tombstone"]["at"] == NOW.isoformat()
    assert tomb.provenance == record.provenance and tomb.content == record.content
    assert tomb.embedding is None
    assert tombstone_record(tomb, reason="again", requested_by="policy:test", now=NOW) is None
    query = RetrievalQuery(tenant_id=record.tenant_id, query_text="pool", as_of=NOW)
    filters = resolve_filters(query)
    assert not record_matches_filters(tomb, filters)
    assert not record_matches_filters(
        tomb, replace(filters, allowed_statuses=frozenset(MemoryStatus))
    )
    packed = pack_context([tomb], filters)
    assert not packed.memories and not packed.text
    with pytest.raises(ValueError):
        tombstone_record(record, reason="", requested_by="user:42", now=NOW)


def test_compaction_requires_accepted_same_tenant_summary_and_all_evidence(episodes):
    episode = episodes(1)[0]
    summary = episode.model_copy(
        deep=True,
        update={
            "memory_id": uuid4(),
            "memory_type": MemoryType.SEMANTIC,
            "metadata": {
                "consolidation_version": "1",
                "supporting_memory_ids": [str(episode.memory_id)],
            },
        },
    )
    compacted = compact_record(episode, summary, NOW)
    assert retrieval_priority(compacted) == 0.25
    assert compacted.status == MemoryStatus.ACTIVE
    assert compacted.content == episode.content and compacted.provenance == episode.provenance
    assert compact_record(compacted, summary, NOW) is None
    assert compact_record(episode, summary.model_copy(update={"tenant_id": uuid4()}), NOW) is None
    assert (
        compact_record(
            episode, summary.model_copy(update={"status": MemoryStatus.QUARANTINED}), NOW
        )
        is None
    )
    summary.provenance[0].event_id = uuid4()
    assert compact_record(episode, summary, NOW) is None


class ConstantModel:
    model_version = "test"

    def __init__(self):
        self.seen = []

    def score_pairs(self, query, documents):
        self.seen.extend(documents)
        return [-2.0] * len(documents)  # negative logits must also be demoted


def test_decay_affects_fusion_reranking_and_packing(episodes):
    old, fresh = episodes(2)
    old.metadata["forgetting"] = {"priority": 0.1}
    old.content = "old pool exhaustion"
    fresh.content = "fresh pool exhaustion"
    hits = _fuse_rrf([LexicalSearchHit(old, 1), LexicalSearchHit(fresh, 1)], [], rrf_k=60)
    assert hits[0].fused_score < hits[1].fused_score
    model = ConstantModel()
    reranked, _ = apply_reranker(CrossEncoderReranker(model), "pool", hits)
    assert reranked[0].memory.memory_id == fresh.memory_id
    filters = resolve_filters(RetrievalQuery(tenant_id=old.tenant_id, query_text="pool", as_of=NOW))
    packed = pack_context([old, fresh], filters)
    assert [p.memory.memory_id for p in packed.memories] == [fresh.memory_id]


@pytest.mark.parametrize("fails", [False, True])
def test_reranker_never_sees_tombstones_including_fallback(episodes, fails):
    record = episodes(1, status=MemoryStatus.TOMBSTONE)[0]
    hit = HybridSearchHit(record, 1.0, 1, 1)
    model = ConstantModel()
    reranker = CrossEncoderReranker(model)
    assert reranker.rerank("pool", [hit]) == []
    output, _ = apply_reranker(reranker, None if fails else "pool", [hit])
    assert output == [] and model.seen == []


def test_invalid_priority_metadata_cannot_boost_a_record(episodes):
    record = episodes(1)[0]
    for value in [True, "0.1", float("nan"), float("inf"), None, {}]:
        record.metadata["forgetting"] = {"priority": value}
        assert retrieval_priority(record) == 1.0
    record.metadata["forgetting"] = {"priority": 10}
    assert retrieval_priority(record) == 1.0
    record.metadata["forgetting"] = {"priority": -1}
    assert retrieval_priority(record) == 0.05

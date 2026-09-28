from datetime import timedelta
from uuid import uuid4

import pytest
from conftest import NOW

from apps.memory_service.consolidation.clusterer import cluster_memories
from apps.memory_service.domain.enums import MemoryStatus


def test_related_and_deterministic(episodes):
    records = episodes(50)
    clusters = cluster_memories(records, now=NOW)
    assert len(clusters) == 1
    assert len(clusters[0].memories) == 50
    assert cluster_memories(reversed(records), now=NOW) == clusters
    assert cluster_memories([*records, records[0]], now=NOW) == clusters


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", uuid4()),
        ("subject_keys", ["service:other"]),
        ("metadata", {"category_key": "different"}),
    ],
)
def test_separate_groups(episodes, field, value):
    records = episodes()
    records[1] = records[1].model_copy(update={field: value})
    assert len(cluster_memories(records, now=NOW)) == 2


@pytest.mark.parametrize("status", [s for s in MemoryStatus if s != MemoryStatus.ACTIVE])
def test_inactive_excluded(episodes, status):
    assert not cluster_memories(episodes(status=status), now=NOW)


def test_expired_and_future_excluded(episodes):
    records = episodes()
    records[0].temporal_validity.valid_to = NOW
    records[1].temporal_validity.valid_from = NOW + timedelta(days=1)
    assert not cluster_memories(records, now=NOW)


def test_window_does_not_chain(episodes):
    records = episodes(3)
    for i, record in enumerate(records):
        record.provenance[0].observed_at = NOW - timedelta(days=60 - 20 * i)
    assert [len(c.memories) for c in cluster_memories(records, now=NOW)] == [2, 1]


def test_subject_overlap_does_not_bridge(episodes):
    records = episodes(3)
    for record, subjects in zip(records, [["a"], ["a", "b"], ["b"]], strict=True):
        record.subject_keys = subjects
    assert [len(c.memories) for c in cluster_memories(records, now=NOW)] == [2, 1]


def test_missing_keys_skipped(episodes):
    assert not cluster_memories(episodes(metadata={}), now=NOW)
    assert not cluster_memories(episodes(subject_keys=[]), now=NOW)

import pytest
from conftest import NOW

from apps.memory_service.consolidation.clusterer import cluster_memories
from apps.memory_service.consolidation.summarizer import MAX_SUMMARY_CHARS, summarize_cluster
from apps.memory_service.domain.enums import MemoryType


def test_bounded_summary_keeps_all_evidence(episodes):
    records = episodes(50)
    candidate = summarize_cluster(cluster_memories(records, now=NOW)[0])
    assert candidate.memory_type == MemoryType.SEMANTIC
    assert len(candidate.content) <= MAX_SUMMARY_CHARS
    assert candidate.provenance == [p for m in records for p in m.provenance]
    assert set(candidate.metadata["supporting_memory_ids"]) == {str(m.memory_id) for m in records}
    assert len(candidate.metadata["episode_event_ids"]) == 50
    assert "peak load" not in candidate.content


def test_procedure_preserves_steps(episodes):
    records = episodes(
        3,
        metadata={
            "category_key": "pool exhaustion",
            "remediation_outcome": "success",
            "remediation_steps": ["inspect saturation metrics", "evaluate pool sizing"],
        },
    )
    candidate = summarize_cluster(cluster_memories(records, now=NOW)[0], MemoryType.PROCEDURAL)
    assert "inspect saturation metrics; then evaluate pool sizing" in candidate.content
    assert len(candidate.provenance) == 3


def test_oversized_procedure_not_truncated(episodes):
    records = episodes(3, metadata={"category_key": "pool", "remediation_steps": ["x" * 1001]})
    with pytest.raises(ValueError, match="bound"):
        summarize_cluster(cluster_memories(records, now=NOW)[0], MemoryType.PROCEDURAL)


def test_cross_tenant_forgery_rejected(episodes):
    cluster = cluster_memories(episodes(), now=NOW)[0]
    from uuid import uuid4

    cluster.memories[1].tenant_id = uuid4()
    with pytest.raises(ValueError, match="tenant"):
        summarize_cluster(cluster)

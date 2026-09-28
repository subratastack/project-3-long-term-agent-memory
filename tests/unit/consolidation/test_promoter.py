from dataclasses import replace

import pytest
from conftest import NOW

from apps.memory_service.consolidation.clusterer import cluster_memories
from apps.memory_service.consolidation.promoter import promote_cluster
from apps.memory_service.domain.enums import MemoryType, TrustLevel


@pytest.mark.parametrize("count,expected", [(1, None), (2, MemoryType.SEMANTIC)])
def test_threshold(episodes, count, expected):
    assert promote_cluster(cluster_memories(episodes(count), now=NOW)[0], now=NOW) == expected


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("success", MemoryType.PROCEDURAL),
        ("failed", MemoryType.SEMANTIC),
        ("unknown", MemoryType.SEMANTIC),
    ],
)
def test_remediation_outcomes(episodes, outcome, expected):
    records = episodes(
        3,
        metadata={
            "category_key": "pool",
            "remediation_outcome": "success",
            "remediation_steps": ["inspect saturation", "evaluate pool size"],
        },
    )
    records[2].metadata = {**records[2].metadata, "remediation_outcome": outcome}
    assert promote_cluster(cluster_memories(records, now=NOW)[0], now=NOW) == expected


def test_untrusted_rejected(episodes):
    records = episodes(trust_level=TrustLevel.UNTRUSTED)
    assert promote_cluster(cluster_memories(records, now=NOW)[0], now=NOW) is None


def test_repeated_event_not_independent(episodes):
    records = episodes(3)
    for record in records[1:]:
        record.provenance = records[0].provenance
    assert promote_cluster(cluster_memories(records, now=NOW)[0], now=NOW) is None


def test_forged_tenant_rejected(episodes):
    cluster = cluster_memories(episodes(), now=NOW)[0]
    from uuid import uuid4

    assert promote_cluster(replace(cluster, tenant_id=uuid4()), now=NOW) is None

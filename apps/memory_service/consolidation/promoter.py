"""Conservative evidence thresholds independent of summary wording."""

from datetime import datetime
from uuid import UUID

from apps.memory_service.consolidation.clusterer import MemoryCluster, cluster_memories
from apps.memory_service.consolidation.summarizer import remediation_steps
from apps.memory_service.domain.enums import MemoryType, TrustLevel
from apps.memory_service.ingestion.provenance import trust_rank

MIN_EPISODES_FOR_SEMANTIC = 2
MIN_SUCCESSFUL_EPISODES_FOR_PROCEDURAL = 3


def promote_cluster(cluster: MemoryCluster, *, now: datetime | None = None) -> MemoryType | None:
    """Select at most one candidate type, or reject ineligible/untrusted evidence.

    Every episode must have a distinct evidence set. Shared events do not count
    as independent repetitions. Unknown, failed, or mixed remediation outcomes
    permit only an observation; only unanimous ``remediation_outcome='success'``
    and identical structured steps permit a procedure.
    """
    rebuilt = cluster_memories(cluster.memories, now=now)
    if len(rebuilt) != 1 or rebuilt[0] != cluster:
        return None
    if any(
        trust_rank(level) < trust_rank(TrustLevel.MEDIUM)
        for m in cluster.memories
        for level in [m.trust_level, *(p.trust_level for p in m.provenance)]
    ):
        return None
    seen: set[UUID] = set()
    independent = 0
    for memory in cluster.memories:
        events = {p.event_id for p in memory.provenance}
        if events.isdisjoint(seen):
            independent += 1
        seen.update(events)
    if independent < MIN_EPISODES_FOR_SEMANTIC:
        return None
    if (
        independent >= MIN_SUCCESSFUL_EPISODES_FOR_PROCEDURAL
        and all(m.metadata.get("remediation_outcome") == "success" for m in cluster.memories)
        and remediation_steps(cluster) is not None
    ):
        return MemoryType.PROCEDURAL
    return MemoryType.SEMANTIC

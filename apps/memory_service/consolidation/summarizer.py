"""Bounded structured summaries: proposals only, never authoritative records.

Procedures require identical ``remediation_steps`` (a list of nonempty strings)
across the cluster. Wording never invents environmental conditions or steps.
Evidence and episode links are not truncated when summary text is bounded.
"""

from uuid import uuid5

from apps.memory_service.consolidation.clusterer import MemoryCluster, category_key, utc
from apps.memory_service.domain.enums import MemoryType
from apps.memory_service.domain.models import MemoryCandidate, TemporalValidity
from apps.memory_service.ingestion.provenance import weakest_trust

MAX_SUMMARY_CHARS = 1000
MAX_KEY_CHARS = 160


def remediation_steps(cluster: MemoryCluster) -> list[str] | None:
    steps = cluster.memories[0].metadata.get("remediation_steps")
    if (
        not isinstance(steps, list)
        or not steps
        or any(not isinstance(s, str) or not s.strip() for s in steps)
    ):
        return None
    if any(m.metadata.get("remediation_steps") != steps for m in cluster.memories):
        return None
    return [s.strip() for s in steps]


def summarize_cluster(
    cluster: MemoryCluster,
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> MemoryCandidate:
    """Build a candidate preserving every episode's complete provenance.

    Oversized structured fields fail closed instead of truncating a procedure.
    Promotion eligibility and authoritative evidence verification are separate.
    """
    if not cluster.memories or any(m.tenant_id != cluster.tenant_id for m in cluster.memories):
        raise ValueError("A summary requires nonempty, same-tenant evidence")
    key = category_key(cluster.memories[0])
    if key is None or any(category_key(m) != key for m in cluster.memories):
        raise ValueError("A summary requires one shared incident/category key")
    if not cluster.shared_subject_keys or any(
        not set(cluster.shared_subject_keys).issubset(m.subject_keys) for m in cluster.memories
    ):
        raise ValueError("A summary requires a common subject")
    subject = cluster.shared_subject_keys[0]
    if max(len(subject), len(key[1])) > MAX_KEY_CHARS:
        raise ValueError("Structured summary keys exceed the bound")
    content = f"{subject} has recurring {key[1]} ({len(cluster.memories)} episodes)."
    if memory_type == MemoryType.PROCEDURAL:
        steps = remediation_steps(cluster)
        if steps is None:
            raise ValueError("A procedure requires shared structured remediation steps")
        content = f"For {subject} / {key[1]}: " + "; then ".join(steps) + "."
    elif memory_type != MemoryType.SEMANTIC:
        raise ValueError("Consolidation only proposes semantic or procedural candidates")
    if len(content) > MAX_SUMMARY_CHARS:
        raise ValueError("Summary exceeds the content bound")
    provenance = [p.model_copy(deep=True) for m in cluster.memories for p in m.provenance]
    expires = [
        m.temporal_validity.valid_to
        for m in cluster.memories
        if m.temporal_validity.valid_to is not None
    ]

    return MemoryCandidate(
        candidate_id=uuid5(cluster.cluster_id, f"candidate:v1:{memory_type}"),
        tenant_id=cluster.tenant_id,
        content=content,
        provenance=provenance,
        memory_type=memory_type,
        subject_keys=list(cluster.shared_subject_keys),
        confidence=min(m.confidence for m in cluster.memories),
        proposed_trust_level=weakest_trust(
            [*(m.trust_level for m in cluster.memories), *(p.trust_level for p in provenance)]
        ),
        temporal_validity=TemporalValidity(
            valid_from=max(
                cluster.latest_observed_at,
                *(utc(m.temporal_validity.valid_from) for m in cluster.memories),
            ),
            valid_to=min(map(utc, expires)) if expires else None,
        ),
        metadata={
            "consolidation_version": "1",
            "cluster_id": str(cluster.cluster_id),
            "supporting_memory_ids": sorted(str(m.memory_id) for m in cluster.memories),
            "episode_event_ids": {
                str(m.memory_id): [str(p.event_id) for p in m.provenance] for m in cluster.memories
            },
            key[0]: key[1],
        },
    )

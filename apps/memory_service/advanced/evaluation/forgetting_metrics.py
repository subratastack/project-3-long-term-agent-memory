"""Lifecycle absence, poison admission, and tenant leakage use external fixture labels."""

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    BenchmarkObservation,
    in_effect,
)


def forgetting_correctness(case: BenchmarkCase, observation: BenchmarkObservation) -> bool | None:
    if case.category != "forgetting":
        return None
    returned = {hit.memory_id for hit in observation.retrieved} | set(
        observation.presented_memory_ids
    )
    return (
        observation.error is None
        and not returned.intersection(case.expected_absence_ids)
        and all(
            hit.status != "tombstone"
            and not (hit.status == "expired" and not in_effect(hit, case.query_time))
            for hit in observation.retrieved
        )
    )


def safety_counts(case: BenchmarkCase, observation: BenchmarkObservation) -> dict[str, int]:
    poisoned = {event.memory_id for event in case.writes if event.poisoned}
    returned = {hit.memory_id for hit in observation.retrieved}
    owners = {event.memory_id: event.tenant_id for event in case.writes}
    foreign = {
        hit.memory_id
        for hit in observation.retrieved
        if hit.tenant_id != case.tenant_id or owners.get(hit.memory_id) != case.tenant_id
    }
    accepted = set(observation.accepted_memory_ids) | set(observation.active_memory_ids)
    return {
        "poison_attempts": len(poisoned),
        "poison_accepted": len(poisoned & accepted),
        "poison_returned": len(poisoned & returned),
        "cross_tenant_returned": len(foreign),
        "returned_memories": len(returned),
        "tenant_leak_query": int(bool(foreign)),
        "unknown_returned": len(returned - owners.keys()),
    }

"""Compare observed facts with external labels at the case's requested time."""

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    BenchmarkObservation,
    in_effect,
)


def temporal_correctness(case: BenchmarkCase, observation: BenchmarkObservation) -> bool | None:
    if case.category not in ("temporal", "update"):
        return None
    hits = {hit.memory_id: hit for hit in observation.retrieved}
    shown = set(observation.presented_memory_ids)
    correct_facts = all(
        fact.memory_id in shown
        and fact.memory_id in hits
        and hits[fact.memory_id].content == fact.content
        and in_effect(hits[fact.memory_id], case.query_time)
        for fact in case.expected_active_facts
    )
    return (
        observation.error is None
        and correct_facts
        and set(case.expected_memory_ids) <= shown
        and not set(case.expected_absence_ids) & hits.keys()
        and all(in_effect(hit, case.query_time) for hit in observation.retrieved)
        and all(hit.status not in ("tombstone", "quarantined") for hit in observation.retrieved)
    )


def update_correctness(case: BenchmarkCase, observation: BenchmarkObservation) -> bool | None:
    return temporal_correctness(case, observation) if case.category == "update" else None

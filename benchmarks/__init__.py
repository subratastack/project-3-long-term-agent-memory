"""Small original benchmark-style subsets; no public benchmark redistribution."""

from pathlib import Path

from apps.memory_service.advanced.evaluation.benchmark_adapter import BenchmarkCase

DEFAULT_FIXTURES = Path(__file__).parent / "fixtures" / "memory_cases.jsonl"


def load_cases(path: Path = DEFAULT_FIXTURES) -> tuple[BenchmarkCase, ...]:
    from benchmarks.longmemeval_subset import load_subset as longmemeval
    from benchmarks.membench_subset import load_subset as membench
    from benchmarks.memoryagentbench_subset import load_subset as memoryagentbench

    cases = sorted(
        (*longmemeval(path), *memoryagentbench(path), *membench(path)), key=lambda c: c.case_id
    )
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("benchmark corpus must have unique case IDs and at least one case")
    return tuple(cases)

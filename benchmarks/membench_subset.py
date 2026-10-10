"""Translate original MemBench-like effectiveness and capacity corpora."""

from pathlib import Path
from typing import Any

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    assemble_case,
    source_examples,
)
from benchmarks import DEFAULT_FIXTURES

SOURCE = "membench-like"


def adapt_example(example: dict[str, Any]) -> BenchmarkCase:
    sessions = [
        {"session_id": s["batch_id"], "at": s["time"], "events": s["records"]}
        for s in example["corpus"]
    ]
    return assemble_case(source=SOURCE, example=example, sessions=sessions, query=example["prompt"])


def load_subset(path: Path = DEFAULT_FIXTURES) -> tuple[BenchmarkCase, ...]:
    return tuple(adapt_example(example) for example in source_examples(path, SOURCE))

"""Translate original LongMemEval-like session facts, questions, and updates."""

from pathlib import Path
from typing import Any

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    assemble_case,
    source_examples,
)
from benchmarks import DEFAULT_FIXTURES

SOURCE = "longmemeval-like"


def adapt_example(example: dict[str, Any]) -> BenchmarkCase:
    sessions = [
        {"session_id": s["session_id"], "at": s["date"], "events": s["facts"]}
        for s in example["history"]
    ]
    return assemble_case(
        source=SOURCE, example=example, sessions=sessions, query=example["question"]
    )


def load_subset(path: Path = DEFAULT_FIXTURES) -> tuple[BenchmarkCase, ...]:
    return tuple(adapt_example(example) for example in source_examples(path, SOURCE))

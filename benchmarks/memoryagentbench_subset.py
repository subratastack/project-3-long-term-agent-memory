"""Translate original MemoryAgentBench-like learning episodes and forgetting requests."""

from pathlib import Path
from typing import Any

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    assemble_case,
    source_examples,
)
from benchmarks import DEFAULT_FIXTURES

SOURCE = "memoryagentbench-like"


def adapt_example(example: dict[str, Any]) -> BenchmarkCase:
    sessions = [
        {"session_id": s["episode_id"], "at": s["timestamp"], "events": s["memories"]}
        for s in example["episodes"]
    ]
    return assemble_case(source=SOURCE, example=example, sessions=sessions, query=example["task"])


def load_subset(path: Path = DEFAULT_FIXTURES) -> tuple[BenchmarkCase, ...]:
    return tuple(adapt_example(example) for example in source_examples(path, SOURCE))

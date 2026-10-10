"""Standard IR scoring through ir_measures, and paired full-run comparisons via ranx."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    BenchmarkObservation,
)


def qrels_and_run(
    cases: Sequence[BenchmarkCase], observations: Sequence[BenchmarkObservation]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """Exclude no-support cases from IR; score their abstention separately."""
    qrels = {
        c.case_id: dict.fromkeys(c.expected_memory_ids, 1) for c in cases if c.expected_memory_ids
    }
    run = {
        o.case_id: {hit.memory_id: hit.score for hit in o.retrieved}
        for o in observations
        if o.case_id in qrels
    }
    return qrels, run


def evaluate_ir(
    cases: Sequence[BenchmarkCase], observations: Sequence[BenchmarkObservation], *, k: int
) -> dict[str, Any]:
    import ir_measures

    if k < 1:
        raise ValueError("k must be positive")
    qrels, run = qrels_and_run(cases, observations)
    names = {
        ir_measures.R @ k: "recall_at_k",
        ir_measures.RR: "mrr",
        ir_measures.nDCG @ k: "ndcg_at_k",
    }
    if not qrels:
        return {"scored_queries": 0, **dict.fromkeys(names.values()), "per_case": {}}
    aggregate = ir_measures.calc_aggregate(list(names), qrels, run)
    per_case: dict[str, dict[str, float]] = {}
    for metric in ir_measures.iter_calc(list(names), qrels, run):
        per_case.setdefault(metric.query_id, {})[names[metric.measure]] = float(metric.value)
    return {
        "scored_queries": len(qrels),
        **{name: float(aggregate[measure]) for measure, name in names.items()},
        "per_case": per_case,
    }


def compare_runs(
    cases: Sequence[BenchmarkCase], runs: Mapping[str, Sequence[BenchmarkObservation]], *, k: int
) -> dict[str, Any]:
    from ranx import Qrels, Run, compare  # type: ignore[import-untyped]

    qrels, _ = qrels_and_run(cases, ())
    if not qrels:
        return {"scored_queries": 0, "results": {}, "table": "No relevant-memory queries."}
    compared_runs = []
    for name, observations in runs.items():
        _, scored = qrels_and_run(cases, observations)
        scored = {query_id: hits for query_id, hits in scored.items() if hits}
        # ranx 0.3's dictionary constructor cannot infer document types from
        # an entirely empty run. Its empty constructor and make_comparable
        # preserve zero-result queries without inventing a retrieved document.
        compared_runs.append(Run(scored, name=name) if scored else Run(name=name))
    report = compare(
        Qrels(qrels),
        compared_runs,
        metrics=[f"recall@{k}", "mrr", f"ndcg@{k}"],
        make_comparable=True,
        stat_test="fisher",
        random_seed=42,
        n_permutations=1000,
        threads=1,
    )
    # Keep ranx's serializable export, including paired tests and win/tie/loss counts.
    return {"scored_queries": len(qrels), "report": report.to_dict(), "table": str(report)}

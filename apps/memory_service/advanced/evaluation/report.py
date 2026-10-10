"""The single evaluator for all benchmark adapters and retrieval strategies."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    BenchmarkObservation,
    in_effect,
)
from apps.memory_service.advanced.evaluation.forgetting_metrics import (
    forgetting_correctness,
    safety_counts,
)
from apps.memory_service.advanced.evaluation.ir_metrics import compare_runs, evaluate_ir
from apps.memory_service.advanced.evaluation.resource_metrics import summarize_resources
from apps.memory_service.advanced.evaluation.temporal_metrics import (
    temporal_correctness,
    update_correctness,
)


def score_case(
    case: BenchmarkCase, observation: BenchmarkObservation, *, token_budget: int
) -> dict[str, Any]:
    if observation.case_id != case.case_id:
        raise ValueError("observation does not match the benchmark case")
    safety = safety_counts(case, observation)
    hits = {hit.memory_id: hit for hit in observation.retrieved}
    shown = set(observation.presented_memory_ids)
    absent_returned = sorted(set(case.expected_absence_ids) & hits.keys())
    missing = sorted(set(case.expected_memory_ids) - shown)
    wrong_facts = [
        fact.memory_id
        for fact in case.expected_active_facts
        if fact.memory_id not in hits or hits[fact.memory_id].content != fact.content
    ]
    invalid_time = sorted(
        hit.memory_id for hit in hits.values() if not in_effect(hit, case.query_time)
    )
    invalid_status = sorted(
        hit.memory_id
        for hit in hits.values()
        if hit.status in ("tombstone", "quarantined")
        or (hit.status == "expired" and not in_effect(hit, case.query_time))
    )
    no_support_correct = (
        observation.error is None and not hits and not shown and not observation.context.strip()
        if not case.expected_memory_ids
        else None
    )
    budget_breach = observation.context_tokens > token_budget
    passed = (
        observation.error is None
        and not missing
        and not wrong_facts
        and not absent_returned
        and not invalid_time
        and not invalid_status
        and not budget_breach
        and not safety["poison_accepted"]
        and not safety["poison_returned"]
        and not safety["cross_tenant_returned"]
        and not safety["unknown_returned"]
        and no_support_correct is not False
    )
    return {
        "case_id": case.case_id,
        "source": case.source,
        "category": case.category,
        "scenario": case.scenario,
        "passed": passed,
        "temporal_correct": temporal_correctness(case, observation),
        "update_correct": update_correctness(case, observation),
        "forgetting_correct": forgetting_correctness(case, observation),
        "abstention_correct": no_support_correct if case.scenario == "abstention" else None,
        "no_support_correct": no_support_correct,
        "missing_presented_ids": missing,
        "wrong_fact_ids": wrong_facts,
        "absence_violations": absent_returned,
        "invalid_time_ids": invalid_time,
        "invalid_status_ids": invalid_status,
        "budget_breach": budget_breach,
        "safety": safety,
    }


def _accuracy(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    applicable = [row[field] for row in rows if row[field] is not None]
    return {"cases": len(applicable), "accuracy": mean(applicable) if applicable else None}


def build_report(
    cases: Sequence[BenchmarkCase],
    runs: Mapping[str, Sequence[BenchmarkObservation]],
    *,
    k: int,
    token_budget: int,
    storage: Mapping[str, dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require complete paired runs; quality uses first reads, safety/resources use every read."""
    if k < 1 or token_budget < 1:
        raise ValueError("k and token_budget must be positive")
    ids = {case.case_id for case in cases}
    if not ids or len(ids) != len(cases) or not runs:
        raise ValueError("cases must be nonempty and unique, with at least one strategy")
    case_of = {case.case_id: case for case in cases}
    first_runs: dict[str, list[BenchmarkObservation]] = {}
    strategies: dict[str, dict[str, Any]] = {}
    attempt_counts: dict[str, dict[str, int]] = {}
    for strategy, observations in runs.items():
        if {o.case_id for o in observations} != ids or any(
            o.strategy != strategy for o in observations
        ):
            raise ValueError(f"{strategy}: run must cover exactly the cases and match the strategy")
        first = {o.case_id: o for o in reversed(observations)}
        first_runs[strategy] = [first[case.case_id] for case in cases]
        attempt_counts[strategy] = {
            case_id: sum(o.case_id == case_id for o in observations) for case_id in ids
        }
        rows = [score_case(case_of[o.case_id], o, token_budget=token_budget) for o in observations]
        first_rows = [
            score_case(case, first[case.case_id], token_budget=token_budget) for case in cases
        ]
        safety = {key: sum(row["safety"][key] for row in rows) for key in rows[0]["safety"]}
        # Admission is measured once per source candidate, not once per repeated read.
        poisoned_by_case = {
            case.case_id: {event.memory_id for event in case.writes if event.poisoned}
            for case in cases
        }
        admitted_by_case: dict[str, set[str]] = {case_id: set() for case_id in ids}
        for observation in observations:
            admitted_by_case[observation.case_id].update(observation.accepted_memory_ids)
            admitted_by_case[observation.case_id].update(observation.active_memory_ids)
        admission = {
            "poison_attempts": sum(len(poisoned) for poisoned in poisoned_by_case.values()),
            "poison_accepted": sum(
                len(poisoned & admitted_by_case[case_id])
                for case_id, poisoned in poisoned_by_case.items()
            ),
        }
        safety.update(admission)
        safety["poison_acceptance_rate"] = (
            admission["poison_accepted"] / admission["poison_attempts"]
            if admission["poison_attempts"]
            else None
        )
        safety["tenant_leakage_rate"] = (
            safety["cross_tenant_returned"] / safety["returned_memories"]
            if safety["returned_memories"]
            else 0.0
        )
        safety["tenant_leak_query_rate"] = safety["tenant_leak_query"] / len(observations)
        strategies[strategy] = {
            "queries": len(cases),
            "attempts": len(observations),
            "ir": evaluate_ir(cases, first_runs[strategy], k=k),
            "temporal": _accuracy(first_rows, "temporal_correct"),
            "update": _accuracy(first_rows, "update_correct"),
            "forgetting": _accuracy(first_rows, "forgetting_correct"),
            "abstention": _accuracy(first_rows, "abstention_correct"),
            "task_accuracy": mean(row["passed"] for row in first_rows),
            "safety": safety,
            "resources": summarize_resources(observations),
            "storage": storage[strategy],
            "by_scenario": {
                scenario: {
                    "cases": len(group),
                    "task_accuracy": mean(row["passed"] for row in group),
                }
                for scenario in sorted({case.scenario for case in cases})
                if (group := [row for row in first_rows if row["scenario"] == scenario])
            },
            "safety_passed": all(
                not row["safety"]["cross_tenant_returned"]
                and not row["safety"]["poison_accepted"]
                and not row["safety"]["poison_returned"]
                and not row["safety"]["unknown_returned"]
                and not row["budget_breach"]
                and not row["absence_violations"]
                and not row["invalid_status_ids"]
                and observation.error is None
                for row, observation in zip(rows, observations, strict=True)
            ),
            "cases": first_rows,
            "observations": [o.model_dump(mode="json") for o in observations],
        }
    counts = [count for run_counts in attempt_counts.values() for count in run_counts.values()]
    if len(set(counts)) != 1:
        raise ValueError("all cases and strategies must have the same number of attempts")
    return {
        "schema_version": "advanced-memory-eval-v1",
        "k": k,
        "token_budget": token_budget,
        "case_count": len(cases),
        "repeats": counts[0],
        "scoring_scope": (
            "Labeled retrieval and memory-state contracts; no generated answer grading."
        ),
        "quality_sampling": (
            "First read per case; every repeat counts toward safety and resources."
        ),
        "metadata": dict(metadata or {}),
        "strategies": strategies,
        "ranx_comparison": compare_runs(cases, first_runs, k=k),
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    def value(number: float | None) -> str:
        return "n/a" if number is None else f"{number:.3f}"

    lines = [
        "# Advanced memory evaluation",
        "",
        f"{report['case_count']} original synthetic cases; "
        f"{report['repeats']} timed reads per case; "
        f"K={report['k']}; context budget={report['token_budget']} estimated tokens.",
        "",
        report["scoring_scope"],
        "",
        report["quality_sampling"],
        "",
        "Recall@K measures how much expected memory appears in the first K results. "
        "MRR rewards the first relevant result; nDCG@K rewards ordering against an ideal run. "
        "All three use binary relevance labels and are higher-is-better.",
        "",
        "Temporal and update accuracy require correct facts in the final context at query time "
        "and no stale retrieval. Forgetting accuracy checks absence before packing. Abstention "
        "accuracy covers the explicit unsupported-question cases. Task accuracy also checks "
        "required context, safety, execution errors, and the token budget.",
        "",
        "Poison acceptance is admitted poisoned candidates divided by attempted poison writes. "
        "Tenant leakage is foreign results divided by all retrieved results, with zero when "
        "nothing is retrieved. Their denominators and query leakage counts are in JSON.",
        "",
        "| Strategy | Recall@K | MRR | nDCG@K | Temporal | Update | Forgetting | "
        "Abstention | Task | Poison acceptance | Tenant leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["strategies"].items():
        ir, safety = summary["ir"], summary["safety"]
        metrics = [
            ir["recall_at_k"],
            ir["mrr"],
            ir["ndcg_at_k"],
            *[
                summary[key]["accuracy"]
                for key in ("temporal", "update", "forgetting", "abstention")
            ],
            summary["task_accuracy"],
            safety["poison_acceptance_rate"],
            safety["tenant_leakage_rate"],
        ]
        lines.append(f"| {name} | " + " | ".join(value(metric) for metric in metrics) + " |")
    lines += [
        "",
        "P50/P95 latency in milliseconds. Retrieval excludes measured reranking time; "
        "total includes all stages.",
        "",
        "| Strategy | Retrieval P50/P95 | Rerank P50/P95 | Packing P50/P95 | Total P50/P95 | "
        "Mean context tokens | Table bytes | Index bytes | Errors | Safety passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, summary in report["strategies"].items():
        resources, footprint = summary["resources"], summary["storage"]
        latencies = [
            f"{value(resources['latency_ms'][stage]['p50'])}/{value(resources['latency_ms'][stage]['p95'])}"
            for stage in ("retrieval", "reranking", "packing", "total")
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(latencies)
            + f" | {value(resources['mean_context_tokens'])} | {footprint['table_bytes']} | "
            f"{footprint['index_bytes']} | {resources['errors']} | {summary['safety_passed']} |"
        )
    lines += [
        "",
        "Memory-enabled strategies share one seeded PostgreSQL corpus and its measured storage. "
        "No-memory uses zero retained-memory storage. Physical sizes include evidence and audit "
        "tables; bytes per source write and per-table sizes are in JSON.",
        "",
        "IR averages exclude cases without relevant IDs. Abstention requires empty retrieval "
        "and context without an execution error. Safety checks retrieval before packing and "
        "admission before querying; target tenant leakage and poison acceptance are zero.",
        "",
        "## Scenario outcomes",
        "",
        "| Strategy | Scenario | Cases | Task accuracy |",
        "|---|---|---:|---:|",
    ]
    for name, summary in report["strategies"].items():
        for scenario, outcomes in summary["by_scenario"].items():
            lines.append(
                f"| {name} | {scenario} | {outcomes['cases']} | "
                f"{value(outcomes['task_accuracy'])} |"
            )
    lines += [
        "",
        "## Reproduction metadata",
        "",
        "```json",
        json.dumps(report["metadata"], indent=2, sort_keys=True),
        "```",
        "",
        "## Paired ranx comparison",
        "",
        "```text",
        report["ranx_comparison"]["table"],
        "```",
        "",
        "These small fixtures and deterministic models are a regression check. They do not "
        "establish public benchmark scores or production model quality. See the JSON for "
        "per-case violations, raw scores, stage samples, errors, and paired comparison details.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output: Path) -> tuple[Path, Path]:
    json_path, markdown_path = output.with_suffix(".json"), output.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(format_markdown(report))
    return json_path, markdown_path

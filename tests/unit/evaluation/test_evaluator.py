"""Adversarial scoring checks and standard-library metric reference runs."""

import json
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from importlib.util import find_spec
from uuid import uuid4

import pytest
from pydantic import ValidationError

from apps.benchmark import run_advanced_memory_eval as runner
from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    BenchmarkObservation,
    LatencySample,
    RetrievedMemory,
)
from apps.memory_service.advanced.evaluation.ir_metrics import compare_runs, evaluate_ir
from apps.memory_service.advanced.evaluation.report import build_report, score_case, write_report
from apps.memory_service.advanced.evaluation.resource_metrics import summarize_resources
from apps.memory_service.embeddings.base import FakeEmbeddingModel
from apps.memory_service.embeddings.cross_encoder import FakeCrossEncoderModel
from apps.memory_service.retrieval.reranker import CrossEncoderReranker
from benchmarks import DEFAULT_FIXTURES, load_cases
from benchmarks.longmemeval_subset import adapt_example as longmemeval
from benchmarks.membench_subset import adapt_example as membench
from benchmarks.memoryagentbench_subset import adapt_example as memoryagentbench

CASES = {case.case_id: case for case in load_cases()}
requires_metrics = pytest.mark.skipif(
    find_spec("ir_measures") is None or find_spec("ranx") is None,
    reason="install the benchmark extra for IR and full-run comparisons",
)


def observation(case, ids=None, *, strategy="test", error=None, accepted=()):
    writes = {event.memory_id: event for event in case.writes}
    ids = case.expected_memory_ids if ids is None else ids
    hits = tuple(
        RetrievedMemory(
            memory_id=memory_id,
            tenant_id=writes[memory_id].tenant_id,
            content=writes[memory_id].content,
            score=float(len(ids) - rank),
            valid_from=writes[memory_id].observed_at,
            valid_to=writes[memory_id].valid_to,
        )
        for rank, memory_id in enumerate(ids)
    )
    context = "\n".join(hit.content for hit in hits)
    return BenchmarkObservation(
        case_id=case.case_id,
        strategy=strategy,
        retrieved=hits,
        presented_memory_ids=tuple(ids),
        accepted_memory_ids=accepted,
        context=context,
        context_tokens=len(context.split()),
        error=error,
        latency=LatencySample(retrieval_ms=2, reranking_ms=1, packing_ms=1, total_ms=4),
    )


@requires_metrics
def test_perfect_full_run_and_ranx_agree():
    cases = tuple(CASES.values())
    perfect = [observation(case) for case in cases]
    scores = evaluate_ir(cases, perfect, k=5)
    assert scores["recall_at_k"] == scores["mrr"] == scores["ndcg_at_k"] == 1.0
    assert scores["scored_queries"] == 30
    assert len(scores["per_case"]) == 30
    empty = [observation(case, (), strategy="empty") for case in cases]
    compared = compare_runs(cases, {"test": perfect, "empty": empty}, k=5)["report"]
    assert compared["test"]["scores"] == {"recall@5": 1.0, "mrr": 1.0, "ndcg@5": 1.0}
    assert compared["empty"]["scores"] == {"recall@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0}
    assert all(
        score_case(case, obs, token_budget=160)["passed"]
        for case, obs in zip(cases, perfect, strict=True)
    )


@requires_metrics
def test_ir_cutoff_rank_and_empty_query_denominator():
    case = CASES["static-1"]
    wrong_first = observation(case, ("noise-0", "office"))
    scores = evaluate_ir((case,), (wrong_first,), k=1)
    assert scores["recall_at_k"] == scores["ndcg_at_k"] == 0
    assert scores["mrr"] == 0.5
    empty = observation(CASES["static-2"], ())
    scores = evaluate_ir((case, CASES["static-2"]), (observation(case), empty), k=5)
    assert scores["recall_at_k"] == scores["mrr"] == scores["ndcg_at_k"] == 0.5


@requires_metrics
def test_stale_fact_fails_even_when_the_correct_fact_is_also_retrieved():
    case = CASES["temporal-1"]
    scored = score_case(case, observation(case, ("new", "old")), token_budget=160)
    assert not scored["temporal_correct"]
    assert not scored["update_correct"]
    assert not scored["passed"]
    assert scored["absence_violations"] == ["old"]
    assert evaluate_ir((case,), (observation(case, ("new", "old")),), k=5)["recall_at_k"] == 1


def test_content_and_half_open_query_time_are_checked():
    case = CASES["temporal-3"]
    obs = observation(case)
    bad = obs.retrieved[0].model_copy(update={"content": "The timeout is 5 seconds."})
    assert not score_case(case, obs.model_copy(update={"retrieved": (bad,)}), token_budget=160)[
        "temporal_correct"
    ]
    expired = obs.retrieved[0].model_copy(update={"valid_to": case.query_time})
    scored = score_case(case, obs.model_copy(update={"retrieved": (expired,)}), token_budget=160)
    assert not scored["temporal_correct"]
    assert scored["invalid_time_ids"] == ["new"]


@pytest.mark.parametrize("case_id", ["forgetting-1", "forgetting-2", "forgetting-3"])
def test_tombstoned_and_expired_memories_must_stay_absent(case_id):
    case = CASES[case_id]
    obs = observation(case, ("forgotten",))
    hit = obs.retrieved[0].model_copy(update={"status": "tombstone"})
    # Still fails if packing removes the record before constructing context.
    obs = obs.model_copy(
        update={"retrieved": (hit,), "presented_memory_ids": (), "context": "", "context_tokens": 0}
    )
    scored = score_case(case, obs, token_budget=160)
    assert not scored["forgetting_correct"]
    assert not scored["passed"]


@requires_metrics
def test_tenant_b_retrieving_tenant_a_fails_run_and_safety_gate():
    case = CASES["tenant-1"]
    obs = observation(case, ("tenant-b", "tenant-a"))
    report = build_report((case,), {"test": (obs,)}, k=5, token_budget=160, storage={"test": {}})
    strategy = report["strategies"]["test"]
    assert strategy["ir"]["recall_at_k"] == 1
    assert not strategy["safety_passed"]
    assert strategy["safety"]["tenant_leakage_rate"] == 0.5
    assert strategy["safety"]["tenant_leak_query_rate"] == 1
    # Lying about the observation tenant does not defeat the source-owner check.
    forged = obs.retrieved[1].model_copy(update={"tenant_id": "B"})
    assert not score_case(
        case, obs.model_copy(update={"retrieved": (obs.retrieved[0], forged)}), token_budget=160
    )["passed"]


def test_poison_acceptance_detected_without_retrieval():
    case = CASES["poison-1"]
    scored = score_case(case, observation(case, accepted=("safe", "poisoned")), token_budget=160)
    assert scored["safety"]["poison_accepted"] == 1
    assert scored["safety"]["poison_returned"] == 0
    assert not scored["passed"]


@requires_metrics
def test_abstention_and_safe_errors_keep_tokens_and_latency():
    case = CASES["abstention-1"]
    safe = observation(case)
    assert score_case(case, safe, token_budget=160)["abstention_correct"]
    assert evaluate_ir((case,), (safe,), k=5)["recall_at_k"] is None
    failed = observation(case, error="Backend timed out")
    assert not score_case(case, failed, token_budget=160)["passed"]
    assert not score_case(case, failed, token_budget=160)["abstention_correct"]
    resources = summarize_resources((safe, failed))
    assert resources["samples"] == 2
    assert resources["mean_context_tokens"] == 0
    assert resources["latency_ms"]["total"] == {"p50": 4.0, "p95": 4.0}
    assert resources["errors"] == 1


def test_runner_records_exception_and_does_not_credit_abstention(monkeypatch):
    case = CASES["abstention-1"]

    def failing_search(*args):
        raise RuntimeError("forced retrieval failure")

    monkeypatch.setattr(runner, "semantic_search", failing_search)
    obs = runner.read_case(
        lambda: nullcontext(object()),
        FakeEmbeddingModel(),
        runner.PreparedCase(case, {"A": uuid4()}),
        "exact_semantic",
        k=5,
        candidates=30,
        token_budget=160,
        reranker=CrossEncoderReranker(FakeCrossEncoderModel()),
    )
    assert obs.error == "RuntimeError: forced retrieval failure"
    assert obs.context_tokens == 0
    assert obs.latency.retrieval_ms > 0
    assert obs.latency.total_ms >= obs.latency.retrieval_ms
    assert not score_case(case, obs, token_budget=160)["passed"]


@requires_metrics
def test_repeated_reads_do_not_multiply_poison_admission_but_do_check_safety():
    case = CASES["poison-1"]
    first = observation(case)
    second = observation(case, ("safe", "poisoned"), accepted=("poisoned",))
    report = build_report(
        (case,), {"test": (first, second)}, k=5, token_budget=160, storage={"test": {}}
    )
    summary = report["strategies"]["test"]
    assert summary["safety"]["poison_attempts"] == 1
    assert summary["safety"]["poison_acceptance_rate"] == 1
    assert summary["resources"]["samples"] == 2
    assert not summary["safety_passed"]


def test_contract_rejects_ambiguous_and_incomplete_runs():
    case = CASES["static-1"]
    payload = case.model_dump()
    payload["expected_absence_ids"] = case.expected_memory_ids
    with pytest.raises(ValidationError, match="disjoint"):
        BenchmarkCase.model_validate(payload)
    payload = case.model_dump()
    payload["query_time"] = datetime(2026, 3, 1)  # Time zone is required.
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate(payload)
    with pytest.raises(ValueError, match="cover exactly"):
        build_report((case,), {"test": ()}, k=5, token_budget=160, storage={"test": {}})
    obs = observation(case)
    payload = obs.model_dump()
    payload["retrieved"] = (obs.retrieved[0], obs.retrieved[0])
    with pytest.raises(ValidationError, match="unique"):
        BenchmarkObservation.model_validate(payload)


def test_fixture_coverage_and_all_adapters_share_the_same_contract():
    assert Counter(case.scenario for case in CASES.values()) == dict.fromkeys(
        (
            "static",
            "preference",
            "temporal",
            "workflow",
            "forgetting",
            "poison",
            "tenant",
            "abstention",
        ),
        5,
    )
    row = json.loads(DEFAULT_FIXTURES.read_text().splitlines()[0])["example"]
    sessions = row["history"]
    translated = dict(id=row["id"], evaluation=row["evaluation"])
    agent = memoryagentbench(
        {
            **translated,
            "task": row["question"],
            "episodes": [
                dict(episode_id=s["session_id"], timestamp=s["date"], memories=s["facts"])
                for s in sessions
            ],
        }
    )
    mem = membench(
        {
            **translated,
            "prompt": row["question"],
            "corpus": [
                dict(batch_id=s["session_id"], time=s["date"], records=s["facts"]) for s in sessions
            ],
        }
    )
    reference = longmemeval(row).model_dump(exclude={"source"})
    assert agent.model_dump(exclude={"source"}) == mem.model_dump(exclude={"source"}) == reference


@requires_metrics
def test_report_writes_json_and_markdown(tmp_path):
    case = CASES["static-1"]
    storage = {"table_bytes": 10, "index_bytes": 5}
    report = build_report(
        (case,), {"test": (observation(case),)}, k=5, token_budget=160, storage={"test": storage}
    )
    json_path, md_path = write_report(report, tmp_path / "eval")
    assert json.loads(json_path.read_text())["schema_version"] == "advanced-memory-eval-v1"
    assert "Recall@K" in md_path.read_text()
    assert "Table bytes" in md_path.read_text()

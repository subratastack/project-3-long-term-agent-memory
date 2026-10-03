"""Multi-session state, temporal truth, poison handling and forgetting in PostgreSQL."""

from dataclasses import asdict

import pytest

from apps.benchmark.run_longitudinal_eval import format_report, run_longitudinal


def test_longitudinal_memory_contracts_and_baselines(uow_factory):
    report = run_longitudinal(uow_factory)
    summaries = {s.backend: s for s in report.summaries}
    custom = summaries["custom_memory_os"]
    native = summaries["langgraph_store"]
    empty = summaries["no_memory"]
    assert report.sessions == 7
    assert len(report.reads) == 51
    assert len(report.writes) == 27
    assert custom.recall_tasks == custom.recalled_tasks == 13
    assert custom.passed_tasks == custom.queries == 17
    assert custom.preference_tasks == custom.preference_recalls == 3
    assert custom.temporal_probes == 6
    assert custom.temporal_violations == 0
    assert custom.poison_exposure_queries == custom.tenant_leak_queries == 0
    assert custom.budget_breaches == custom.forgotten_memory_exposures == 0
    assert custom.forgetting_probes == 2
    assert custom.audited_writes == custom.write_attempts == 9
    assert custom.audit_coverage == 1.0
    assert custom.final_record_states == {
        "active": 4,
        "quarantined": 1,
        "superseded": 1,
        "tombstone": 1,
    }
    # Native primitives still find useful information. Missing governance is measured separately.
    assert native.recalled_tasks == 13
    assert native.preference_recalls == 3
    assert native.temporal_violations > 0
    assert native.poison_exposure_queries == native.poison_opportunities == 8
    assert native.forgotten_memory_exposures == native.tenant_leak_queries == 0
    assert native.audit_coverage is None
    assert empty.recalled_tasks == 0
    assert empty.passed_tasks == 4
    assert empty.poison_exposure_queries == 0
    assert "Labeled retrieval/state contracts" in format_report(report)
    assert asdict(report)["scenario_version"] == "longitudinal-v1"
    writes = {
        w.observation.label: w.observation
        for w in report.writes
        if w.observation.backend == custom.backend
    }
    assert writes["stale_timeout"].outcome == "reject"
    assert writes["stale_timeout"].reason_codes == ("STALE_FACT_CLAIMED_AS_CURRENT",)
    assert writes["safety_override"].outcome == "reject"
    early = [r for r in report.reads if r.session == "1_bootstrap"]
    assert all("timeout_new" not in r.observation.retrieved for r in early)


@pytest.mark.parametrize("budget", [1, 80, 500])
def test_longitudinal_budget_pressure_is_measured_without_changing_policy(uow_factory, budget):
    report = run_longitudinal(uow_factory, token_budget=budget)
    summaries = {s.backend: s for s in report.summaries}
    custom = summaries["custom_memory_os"]
    assert custom.budget_breaches == custom.poison_exposure_queries == 0
    assert custom.max_tokens <= budget
    assert custom.audited_writes == 9
    if budget == 1:
        assert custom.recalled_tasks == 0
        assert summaries["langgraph_store"].budget_breaches > 0
    else:
        assert custom.recalled_tasks > 0


def test_fresh_longitudinal_runs_preserve_recall_safety_and_isolation(uow_factory):
    first = run_longitudinal(uow_factory)
    second = run_longitudinal(uow_factory)
    # UUID tie-breaking may change a redundant episode and its token cost.
    for left, right in zip(first.summaries, second.summaries, strict=True):
        a, b = asdict(left), asdict(right)
        for field in ("mean_tokens", "max_tokens"):
            a.pop(field)
            b.pop(field)
        assert a == b
    assert [(r.recalled, r.task_passed) for r in first.reads] == [
        (r.recalled, r.task_passed) for r in second.reads
    ]

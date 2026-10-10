"""Run real governed writes, historical reads, deletion, and private-schema storage."""

import pytest
from sqlalchemy import text

from apps.benchmark.run_advanced_memory_eval import evaluation_database, run_evaluation
from benchmarks import load_cases


def test_full_corpus_through_production_service_and_isolated_storage(engine):
    pytest.importorskip("ir_measures")
    pytest.importorskip("ranx")
    with evaluation_database(engine.url.render_as_string(hide_password=False)) as (factory, schema):
        report = run_evaluation(factory, schema=schema, cases=load_cases(), repeats=1)
        assert report["case_count"] == 40
        assert len(report["strategies"]) == 4
        for name, summary in report["strategies"].items():
            assert summary["safety"]["tenant_leakage_rate"] == 0
            assert summary["resources"]["errors"] == 0
            assert summary["resources"]["max_context_tokens"] <= 160
            assert summary["resources"]["samples"] == 40
            if name != "memory_disabled":
                assert summary["temporal"]["accuracy"] == 1
                assert summary["forgetting"]["accuracy"] == 1
                # The evaluator must expose policy regressions, rather than assume
                # the production detector recognizes every fixture attack.
                admitted_poison = sum(
                    decision["memory_id"] == "poisoned"
                    and decision["decision"] in ("accept", "supersede")
                    for case_id, decisions in report["write_decisions"].items()
                    if case_id.startswith("poison-")
                    for decision in decisions
                )
                assert summary["safety"]["poison_accepted"] == admitted_poison
                assert summary["safety"]["poison_acceptance_rate"] == admitted_poison / 5
                assert summary["safety_passed"] is (admitted_poison == 0)
                assert summary["storage"]["table_bytes"] > 0
                assert summary["storage"]["index_bytes"] > 0
                assert summary["by_scenario"]["workflow"]["task_accuracy"] == 1
        disabled = report["strategies"]["memory_disabled"]
        assert disabled["ir"]["recall_at_k"] == 0
        assert disabled["abstention"]["accuracy"] == 1
        assert disabled["storage"]["total_bytes"] == 0
        assert len(report["storage_checkpoints"]) == 4
        assert report["storage_checkpoints"][-1]["corpus_size"] > 40
        assert report["write_decisions"]["temporal-4"][-1]["decision"] == "reject"
    with engine.connect() as connection:
        assert not connection.execute(
            text("SELECT 1 FROM pg_namespace WHERE nspname=:schema"), {"schema": schema}
        ).scalar()


def test_isolated_schema_is_removed_after_exception(engine):
    with (
        pytest.raises(RuntimeError, match="forced"),
        evaluation_database(engine.url.render_as_string(hide_password=False)) as (_, schema),
    ):
        raise RuntimeError("forced")
    with engine.connect() as connection:
        assert not connection.execute(
            text("SELECT 1 FROM pg_namespace WHERE nspname=:schema"), {"schema": schema}
        ).scalar()

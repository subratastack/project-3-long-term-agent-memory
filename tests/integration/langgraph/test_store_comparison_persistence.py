"""Compare actual PostgreSQL service behavior with native Store on the shared fixture."""

from dataclasses import asdict
from uuid import uuid4

import pytest
from sqlalchemy import select

from apps.benchmark.database_sandbox import benchmark_database
from apps.benchmark.run_langgraph_store_eval import format_report, run_comparison
from apps.memory_service.domain.models import Tenant
from apps.memory_service.persistence.models import TenantRow


def test_capability_evidence_and_shared_data_are_measured(uow_factory):
    report = run_comparison(uow_factory)
    assert report.dataset_size == 7
    assert len(report.writes) == 14
    assert len(report.reads) == 12
    checks = report.checks
    assert checks["custom_namespace_leaks"] == checks["store_namespace_leaks"] == 0
    assert checks["custom_provenance_resolves_to_events"] is True
    assert checks["store_accepts_document_without_provenance"] is True
    assert checks["broad_store_prefix_can_read_both_tenants"] is True
    assert checks["custom_old_fact_is_superseded"] is True
    assert checks["custom_supersession_links"] == 1
    assert checks["custom_current_excludes_old"] is True
    assert checks["store_current_returns_both_versions"] is True
    assert checks["custom_history_excludes_future"] is True
    assert checks["store_history_returns_future"] is True
    assert checks["custom_poison_quarantined"] is True
    assert checks["custom_poison_exposures"] == 0
    assert checks["store_poison_exposures"] > 0
    assert checks["custom_audited_writes"] == 7
    assert checks["custom_budget_breaches"] == 0
    assert checks["custom_40_token_probe_fits"] is True
    assert checks["store_40_token_probe_exceeds_budget"] is True
    assert "Token-budget context packing" in format_report(report)
    assert asdict(report)["dataset_version"] == "langgraph-store-v1"


@pytest.mark.parametrize("budget", [1, 80, 500])
def test_custom_comparison_context_obeys_each_configured_budget(uow_factory, budget):
    report = run_comparison(uow_factory, token_budget=budget)
    custom = [r for r in report.reads if r.backend == "custom_memory_os"]
    assert all(r.tokens <= budget for r in custom)


@pytest.mark.parametrize("fail", [False, True])
def test_database_sandbox_rolls_back_service_commits_even_on_failure(engine, fail):
    tenant_id = uuid4()
    url = engine.url.render_as_string(hide_password=False)
    try:
        with benchmark_database(url) as factory:
            with factory() as uow:
                uow.tenants.add(Tenant(tenant_id=tenant_id, name=f"rollback-probe-{tenant_id}"))
                uow.commit()
            with factory() as uow:
                assert uow.tenants.get(tenant_id) is not None
            if fail:
                raise RuntimeError("fixture failure")
    except RuntimeError:
        assert fail
    with engine.connect() as connection:
        assert (
            connection.scalar(select(TenantRow.tenant_id).where(TenantRow.tenant_id == tenant_id))
            is None
        )

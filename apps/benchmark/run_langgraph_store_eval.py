"""Compare native LangGraph Store primitives with governed PostgreSQL memory.

Run: uv run python -m apps.benchmark.run_langgraph_store_eval
All database seeds are rolled back. Hash embeddings test API behavior, not
real-model semantic quality. The experiment does not modify the live adapter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from uuid import UUID, uuid4

from apps.benchmark.database_sandbox import benchmark_database
from apps.memory_service.domain.models import Tenant
from apps.memory_service.embeddings.base import EmbeddingModel, FakeEmbeddingModel
from apps.memory_service.ingestion.service import UowFactory
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from integrations.langgraph.store_comparison import (
    DATASET_VERSION,
    CapabilityRow,
    CustomMemoryExperiment,
    LangGraphStoreExperiment,
    ReadObservation,
    TenantKey,
    WriteObservation,
    capability_rows,
    comparison_dataset,
    comparison_probes,
)


@dataclass(frozen=True)
class StoreComparisonReport:
    dataset_version: str
    dataset_size: int
    langgraph_version: str
    embedding_model: str
    embedding_dimensions: int
    backend_scope: str
    capabilities: tuple[CapabilityRow, ...]
    writes: tuple[WriteObservation, ...]
    reads: tuple[ReadObservation, ...]
    checks: dict[str, bool | int]


def run_comparison(
    uow_factory: UowFactory, *, embedder: EmbeddingModel | None = None, token_budget: int = 160
) -> StoreComparisonReport:
    if token_budget < 1:
        raise ValueError("token_budget must be at least 1")
    embedder = embedder or FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)
    experiment_id = uuid4()
    tenants: dict[TenantKey, UUID] = {"A": uuid4(), "B": uuid4()}
    with uow_factory() as uow:
        for key, tenant_id in tenants.items():
            uow.tenants.add(Tenant(tenant_id=tenant_id, name=f"store-eval-{experiment_id}-{key}"))
        uow.commit()
    custom = CustomMemoryExperiment(uow_factory, embedder, tenants)
    native = LangGraphStoreExperiment(embedder, tenants, experiment_id)
    dataset = comparison_dataset()
    writes = tuple(
        participant.write(entry) for entry in dataset for participant in (custom, native)
    )
    reads = tuple(
        participant.read(probe, token_budget)
        for probe in comparison_probes()
        for participant in (custom, native)
    )
    by_key = {(r.backend, r.query): r for r in reads}
    by_write = {(w.backend, w.label): w for w in writes}
    label_tenant = {entry.label: entry.tenant for entry in dataset}

    # Separate native API probes; they do not alter the seven-record corpus.
    probe_ns = (*native.prefix, "api-probes")
    native.store.put(probe_ns, "unverified", {"text": "A supplied value."}, index=False)
    unverified = native.store.get(probe_ns, "unverified")
    native.store.put(probe_ns, "latest", {"text": "timeout=5s"}, index=False)
    native.store.put(probe_ns, "latest", {"text": "timeout=2s"}, index=False)
    latest = native.store.get(probe_ns, "latest")
    broad = native.store.search(native.prefix, limit=30)
    seen_namespaces = {item.namespace for item in broad}
    tight_probe = comparison_probes()[1]
    tight_custom = custom.read(tight_probe, 40)
    tight_native = native.read(tight_probe, 40)
    with uow_factory() as uow:
        old = uow.records.get(tenants["A"], custom.memory_ids["timeout_old"])
        assert old is not None
        links = uow.relations.list_for_memory(tenants["A"], custom.memory_ids["timeout_new"])
        provenance_verified = all(
            uow.events.get(record.tenant_id, p.event_id) is not None
            for tenant_id in tenants.values()
            for record in uow.records.list_for_maintenance(tenant_id)
            for p in record.provenance
        )
    checks: dict[str, bool | int] = {
        "custom_namespace_leaks": sum(
            label_tenant[label] != r.tenant
            for r in reads
            if r.backend == custom.name
            for label in r.retrieved
        ),
        "store_namespace_leaks": sum(
            label_tenant[label] != r.tenant
            for r in reads
            if r.backend == native.name
            for label in r.retrieved
        ),
        "broad_store_prefix_can_read_both_tenants": all(
            native.namespace(key) in seen_namespaces for key in tenants
        ),
        "custom_provenance_resolves_to_events": provenance_verified,
        "store_accepts_document_without_provenance": unverified is not None,
        "custom_old_fact_is_superseded": old.status.value == "superseded",
        "custom_supersession_links": sum(r.relation_type.value == "supersession" for r in links),
        "store_same_key_overwrite_keeps_latest": latest is not None
        and latest.value["text"] == "timeout=2s",
        "custom_current_excludes_old": "timeout_old"
        not in by_key[(custom.name, "current_timeout")].retrieved,
        "store_current_returns_both_versions": {"timeout_old", "timeout_new"}.issubset(
            by_key[(native.name, "current_timeout")].retrieved
        ),
        "custom_history_excludes_future": "timeout_new"
        not in by_key[(custom.name, "historical_timeout")].retrieved,
        "store_history_returns_future": "timeout_new"
        in by_key[(native.name, "historical_timeout")].retrieved,
        "custom_poison_quarantined": by_write[(custom.name, "poison")].outcome == "quarantine",
        "custom_poison_exposures": sum(
            "poison" in r.presented for r in reads if r.backend == custom.name
        ),
        "store_poison_exposures": sum(
            "poison" in r.presented for r in reads if r.backend == native.name
        ),
        "custom_audited_writes": sum(w.audited for w in writes if w.backend == custom.name),
        "custom_budget_breaches": sum(
            r.tokens > r.token_budget for r in reads if r.backend == custom.name
        ),
        "custom_40_token_probe_fits": tight_custom.tokens <= 40,
        "store_40_token_probe_exceeds_budget": tight_native.tokens > 40,
    }
    return StoreComparisonReport(
        DATASET_VERSION,
        len(dataset),
        version("langgraph"),
        embedder.model_version,
        embedder.dimensions,
        "PostgreSQL service vs native InMemoryStore; rollback-only database scope",
        capability_rows(),
        writes,
        reads,
        checks,
    )


def format_report(report: StoreComparisonReport) -> str:
    lines = [
        "# LangGraph Store comparison: measured experiment",
        "",
        "This experiment checks which behavior comes from Store primitives and which "
        "the governed service enforces for the same proposals. The fixture combines a "
        "stable preference, a changing configuration, recurring incidents, a poisoned "
        "proposal, and similar incidents in separate tenants. The production adapter is unchanged.",
        "",
        f"Fixture `{report.dataset_version}`: {report.dataset_size} proposals; "
        f"LangGraph {report.langgraph_version}; `{report.embedding_model}` "
        f"({report.embedding_dimensions} dimensions).",
        "",
        report.backend_scope + ". Hash embeddings exercise search APIs, not model quality.",
        "",
        "## Capability contracts",
        "",
        "| Capability | Custom Memory OS | Native Store baseline |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.capability} | {row.custom_memory_os} | {row.langgraph_store} |"
        for row in report.capabilities
    )
    lines.extend(
        [
            "",
            "## Measured observations",
            "",
            "| Check | Result |",
            "| --- | --- |",
            *(f"| {key} | {value} |" for key, value in report.checks.items()),
            "",
            "## Query trace",
            "",
            "`retrieved` is the returned candidate set; `presented` is the packed custom "
            "selection or the unbounded native text. IDs below are fixture labels.",
            "",
            "| Query | System | Retrieved | Presented | Tokens / budget |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {r.query} | {r.backend} | {', '.join(r.retrieved)} | {', '.join(r.presented)} "
        f"| {r.tokens} / {r.token_budget} |"
        for r in report.reads
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Store can supply namespaced JSON storage and embedding candidates. PostgreSQL "
            "remains authoritative for verified evidence, tenant authorization, validity, "
            "supersession/conflict relations, policy, quarantine, forgetting and audits. "
            "A derived Store index would require record revalidation and packing before reasoning.",
            "",
            "This tests native InMemoryStore, not every persistent Store backend. A Store-based "
            "application can implement governance above its primitives. The seven-record "
            "corpus measures recognized supersession; equal-trust conflict resolution is "
            "an existing custom capability, not separately measured here. No latency or "
            "LLM superiority is claimed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-url", help="defaults to DATABASE_URL / local PostgreSQL")
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.token_budget < 1:
        parser.error("--token-budget must be at least 1")
    from dotenv import load_dotenv

    load_dotenv()
    with benchmark_database(args.database_url) as uow_factory:
        report = run_comparison(uow_factory, token_budget=args.token_budget)
    rendered = (
        json.dumps(asdict(report), indent=2) + "\n"
        if args.format == "json"
        else format_report(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

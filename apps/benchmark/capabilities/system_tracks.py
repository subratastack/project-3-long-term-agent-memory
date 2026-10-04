"""Measure production policy, temporal resolution, lifecycle, and packing in PostgreSQL."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import UUID, uuid4, uuid5

from apps.benchmark.capabilities.common import make_record, read_fixture, save_report, table
from apps.benchmark.run_context_packing_eval import pack_in_rank_order
from apps.benchmark.run_poisoning_eval import PoisonCase, ingest_case, run_corpus
from apps.memory_service.consolidation.forgetting import (
    maintain_memories,
    retrieval_priority,
    tombstone_memory,
)
from apps.memory_service.domain.enums import ConflictType, MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryEvent, MemoryRecord, MemoryRelation
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.ingestion.candidate_extractor import RawCandidate
from apps.memory_service.ingestion.service import UowFactory
from apps.memory_service.retrieval.context_packer import pack_context
from apps.memory_service.retrieval.filters import resolve_filters
from apps.memory_service.retrieval.hybrid import hybrid_search_with_report
from apps.memory_service.retrieval.lexical import lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.temporal import apply_temporal_resolution


def remap_uuids(value: Any, namespace: UUID) -> Any:
    """Fresh run namespace preserves all evidence edges, including missing/foreign IDs."""
    if isinstance(value, str):
        try:
            return str(uuid5(namespace, str(UUID(value))))
        except ValueError:
            return value
    if isinstance(value, list):
        return [remap_uuids(v, namespace) for v in value]
    if isinstance(value, dict):
        return {k: remap_uuids(v, namespace) for k, v in value.items()}
    return value


def run_write_policy(factory: UowFactory, output: Path) -> None:
    data, fixture_hash = read_fixture("write_policy")
    namespace = uuid4()
    cases = []
    for original in data["cases"]:
        case = remap_uuids(original, namespace)
        cases.append(
            PoisonCase(
                case_id=case["case_id"],
                attack_class=case["attack_class"],
                poisoned=case["poisoned"],
                tenant_id=UUID(case["tenant_id"]),
                events=tuple(MemoryEvent.model_validate(e) for e in case["events"]),
                proposal=RawCandidate.model_validate(case["proposal"]),
                fact_events=tuple(MemoryEvent.model_validate(e) for e in case["fact_events"]),
                current_facts=tuple(MemoryRecord.model_validate(m) for m in case["current_facts"]),
            )
        )
    now = datetime.fromisoformat(data["evaluation_time"])
    scorecard, labeled = run_corpus(cases, lambda case: ingest_case(factory, case, now=now))
    summary = {
        "cases": len(cases),
        "poison_attempts": scorecard.poison.attempts,
        "poison_accepted": scorecard.poison.accepted,
        "poison_acceptance_rate": scorecard.poison.acceptance_rate,
        "benign_attempts": scorecard.benign.attempts,
        "benign_accepted": scorecard.benign.accepted,
        "benign_acceptance_rate": scorecard.benign.acceptance_rate,
        "reason_code_coverage": 1 - len(scorecard.missing_reason_codes) / len(cases),
    }
    rows = [
        [name, c.attempts, c.accepted, c.quarantined, c.rejected]
        for name, c in scorecard.by_class.items()
    ]
    save_report(
        output,
        "write_policy",
        {
            "fixture_sha256": fixture_hash,
            "evaluation_time": now,
            "database_mode": "rollback-only PostgreSQL with fresh tenant namespace",
            "summary": summary,
            "by_class": dataclasses.asdict(scorecard)["by_class"],
            "results": [
                {
                    "case_id": item.case_id,
                    "attack_class": item.attack_class,
                    "poisoned": item.poisoned,
                    "decision": item.decision.model_dump(mode="json"),
                }
                for item in labeled
            ],
        },
        title="Memory write policy benchmark",
        purpose=(
            "A proposed memory must pass admission rules before it can influence "
            "later answers. This run sends labeled attack proposals and benign "
            "controls through the production ingestion service and PostgreSQL "
            "audit path. The extractor proposals are scripted so that the "
            "experiment isolates the write policy."
        ),
        metrics=(
            "Poison acceptance is the share of attack cases admitted as active "
            "memory, including supersession; lower is better. Benign acceptance "
            "measures how many ordinary controls become active memory. Reason-code "
            "coverage is the share of decisions carrying an explanation code. "
            "Quarantine retains a held record without making it active."
        ),
        results=table(["Metric", "Value"], [[k, v] for k, v in summary.items()])
        + "\n\n"
        + table(["Class", "Cases", "Accepted", "Quarantined", "Rejected"], rows),
        interpretation=(
            "Check benign acceptance alongside attack rejection: refusing "
            "everything would hide attacks but also make memory unusable. The "
            "per-class table identifies which families are rejected and which are "
            "held for review. Every service commit is contained in a savepoint; "
            "the outer benchmark transaction is rolled back."
        ),
        limitations=(
            "This is an authored regression corpus of known attacks, not a "
            "measured guarantee against novel poisoning. Extraction-model mistakes "
            "are measured separately. Next, add independently authored adversarial "
            "paraphrases and judge them before adding them to the regression set."
        ),
    )
    print(f"write_policy: {len(cases)} cases completed", flush=True)


def run_conflicts(factory: UowFactory, output: Path) -> None:
    data, fixture_hash = read_fixture("conflict_resolution")
    now = datetime.fromisoformat(data["evaluation_time"])
    traces = []
    for case in data["cases"]:
        tenant = uuid4()
        event1, first = make_record(
            "checkout timeout is 2 seconds",
            tenant,
            at=now - timedelta(days=case["first_age_days"]),
            memory_type=MemoryType.SEMANTIC,
            trust=TrustLevel(case["first_trust"]),
            valid_to=now - timedelta(days=case["first_valid_to_days_ago"])
            if "first_valid_to_days_ago" in case
            else None,
        )
        event2, second = make_record(
            "checkout timeout is 5 seconds",
            tenant,
            at=now - timedelta(days=case["second_age_days"]),
            memory_type=MemoryType.SEMANTIC,
            trust=TrustLevel(case["second_trust"]),
        )
        relation = MemoryRelation(
            tenant_id=tenant,
            source_memory_id=second.memory_id,
            target_memory_id=first.memory_id,
            relation_type=ConflictType(case.get("relation", "contradiction")),
            created_at=now,
        )
        with factory() as uow:
            for event, record in ((event1, first), (event2, second)):
                uow.create_event(event)
                uow.create_memory(record)
            uow.relations.add(tenant, relation)
            uow.commit()
        query = RetrievalQuery(
            tenant_id=tenant,
            query_text="checkout timeout",
            limit=10,
            as_of=now - timedelta(days=case.get("as_of_days_ago", 0)),
        )
        with factory() as uow:
            hits = lexical_search(uow, query)
            kept, report = apply_temporal_resolution(uow, query, hits)
        label_of = {first.memory_id: "first", second.memory_id: "second"}
        visible = sorted(label_of[h.memory.memory_id] for h in kept)
        traces.append(
            {
                "case_id": case["id"],
                "scenario": case,
                "expected_visible": sorted(case["expected_visible"]),
                "visible": visible,
                "success": visible == sorted(case["expected_visible"]),
                "retrieved_before_resolution": [label_of[h.memory.memory_id] for h in hits],
                "temporal": dataclasses.asdict(report),
            }
        )
    summary = {
        "cases": len(traces),
        "correct_visible_sets": sum(t["success"] for t in traces),
        "visible_set_accuracy": mean(t["success"] for t in traces),
    }
    save_report(
        output,
        "conflict_resolution",
        {
            "fixture_sha256": fixture_hash,
            "evaluation_time": now,
            "database_mode": "rollback-only PostgreSQL, fresh tenant per case",
            "summary": summary,
            "results": traces,
        },
        title="Conflict resolution benchmark",
        purpose=(
            "Conflicting memories must be resolved before the agent sees them as "
            "current facts. A system-authority timeout of 2 seconds should beat a "
            "lower-trust claim of 5 seconds. Two claims with equal authority "
            "should both be withheld, even when one is newer. This run checks the "
            "visible claims after actual PostgreSQL search and temporal resolution."
        ),
        metrics=(
            "Visible-set accuracy requires exactly the explicitly labeled "
            "surviving claim IDs, including an empty set when the conflict is "
            "unresolved. The fixture covers every ordered pair of the five trust "
            "levels plus recency, historical validity, ended windows, and explicit "
            "supersession."
        ),
        results=table(["Metric", "Measured value"], [[k, v] for k, v in summary.items()])
        + "\n\n"
        + table(
            ["Case", "Expected", "Visible", "Success"],
            [
                [
                    t["case_id"],
                    ", ".join(t["expected_visible"]) or "withhold both",
                    ", ".join(t["visible"]) or "withhold both",
                    t["success"],
                ]
                for t in traces
            ],
        ),
        interpretation=(
            "The expected choice depends on authority and validity, not a "
            "relevance score. The historical and ended-window cases check that "
            "claims outside the query's time do not suppress the applicable claim. "
            "Equal-trust claims remain unresolved until a recorded action settles "
            "them."
        ),
        limitations=(
            "Contradictions and supersession edges are supplied by the fixture. "
            "This measures resolution of recorded conflicts, not automatic "
            "contradiction detection or whether trust assignments are correct. "
            "Next, add graph-shaped conflicts and adversarial missing-edge cases."
        ),
    )
    print(f"conflict_resolution: {len(traces)} cases completed", flush=True)


def run_forgetting(factory: UowFactory, output: Path) -> None:
    data, fixture_hash = read_fixture("forgetting")
    now = datetime.fromisoformat(data["evaluation_time"])
    results = []
    for case in data["cases"]:
        tenant = uuid4()
        event, record = make_record(
            "checkout pool exhaustion",
            tenant,
            at=now - timedelta(days=case["age_days"]),
            confidence=case["confidence"],
            memory_type=MemoryType(case["type"]),
            status=MemoryStatus(case.get("status", "active")),
            valid_to=now - timedelta(days=case["valid_to_days_ago"])
            if "valid_to_days_ago" in case
            else None,
        )
        with factory() as uow:
            uow.create_event(event)
            uow.create_memory(record)
            uow.commit()
        changed = maintain_memories(factory, tenant, now=now)
        repeated = maintain_memories(factory, tenant, now=now)
        with factory() as uow:
            stored = uow.get_memory(record.memory_id, tenant)
        assert stored is not None
        observed = {
            "status": stored.status.value,
            "priority": retrieval_priority(stored),
            "audit_evidence_retained": stored.content == record.content
            and stored.provenance == record.provenance,
            "idempotent": not any(repeated.values()),
        }
        success = (
            observed["status"] == case["expected_status"]
            and abs(observed["priority"] - case["expected_priority"]) < 1e-9
            and observed["audit_evidence_retained"]
            and observed["idempotent"]
        )
        results.append(
            {
                "case_id": case["id"],
                "expected": case,
                "observed": observed,
                "maintenance_changes": changed,
                "success": success,
            }
        )
        if case["id"] == "expiry":
            with factory() as uow:
                current = lexical_search(
                    uow, RetrievalQuery(tenant_id=tenant, query_text="checkout pool exhaustion")
                )
                history = lexical_search(
                    uow,
                    RetrievalQuery(
                        tenant_id=tenant,
                        query_text="checkout pool exhaustion",
                        as_of=now - timedelta(days=2),
                    ),
                )
            results.append(
                {
                    "case_id": "expiry-history",
                    "expected": {"current_hits": 0, "historical_hits": 1},
                    "observed": {"current_hits": len(current), "historical_hits": len(history)},
                    "success": not current and len(history) == 1,
                }
            )
    # Explicitly seed a derived summary sharing evidence; the job must compact its source.
    tenant, other_tenant = uuid4(), uuid4()
    event, episode = make_record("checkout pool exhaustion", tenant, at=now - timedelta(days=5))
    _, summary = make_record(
        "checkout pool exhaustion summary",
        tenant,
        at=now - timedelta(days=1),
        event=event,
        memory_type=MemoryType.SEMANTIC,
        metadata={"consolidation_version": "1", "supporting_memory_ids": [str(episode.memory_id)]},
    )
    foreign_event, foreign = make_record(
        "checkout pool exhaustion", other_tenant, at=now - timedelta(days=5)
    )
    with factory() as uow:
        for ev in (event, foreign_event):
            uow.create_event(ev)
        for rec in (episode, summary, foreign):
            uow.create_memory(rec)
        uow.commit()
    compacted = maintain_memories(factory, tenant, now=now)
    with factory() as uow:
        stored = uow.get_memory(episode.memory_id, tenant)
    assert stored is not None
    results.append(
        {
            "case_id": "derived-summary-compaction",
            "expected": {"priority": 0.25},
            "observed": {"priority": retrieval_priority(stored), "changes": compacted},
            "success": retrieval_priority(stored) == 0.25
            and compacted["compacted"] == [episode.memory_id],
        }
    )
    tombstoned = tombstone_memory(
        factory,
        tenant,
        episode.memory_id,
        reason="benchmark forgetting request",
        requested_by="benchmark",
        now=now,
    )
    again = tombstone_memory(
        factory,
        tenant,
        episode.memory_id,
        reason="benchmark retry",
        requested_by="benchmark",
        now=now,
    )
    with factory() as uow:
        deleted = [uow.get_memory(r.memory_id, tenant) for r in (episode, summary)]
        other = uow.get_memory(foreign.memory_id, other_tenant)
        current = lexical_search(
            uow, RetrievalQuery(tenant_id=tenant, query_text="checkout pool exhaustion")
        )
        history = lexical_search(
            uow,
            RetrievalQuery(
                tenant_id=tenant,
                query_text="checkout pool exhaustion",
                as_of=now - timedelta(days=2),
            ),
        )
    results.extend(
        [
            {
                "case_id": "tombstone-evidence-cascade",
                "expected": {"tombstones": 2},
                "observed": {
                    "changed_ids": tombstoned,
                    "states": [r.status.value if r else None for r in deleted],
                },
                "success": set(tombstoned) == {episode.memory_id, summary.memory_id}
                and all(
                    r and r.status == MemoryStatus.TOMBSTONE and r.embedding is None
                    for r in deleted
                ),
            },
            {
                "case_id": "tombstone-idempotence",
                "expected": [],
                "observed": again,
                "success": again == [],
            },
            {
                "case_id": "tombstone-search-block",
                "expected": {"current": 0, "historical": 0},
                "observed": {"current": len(current), "historical": len(history)},
                "success": not current and not history,
            },
            {
                "case_id": "tenant-isolation",
                "expected": "active",
                "observed": other.status.value if other else None,
                "success": other is not None and other.status == MemoryStatus.ACTIVE,
            },
        ]
    )
    blocked = False
    try:
        with factory() as uow:
            uow.create_memory(episode.model_copy(update={"memory_id": uuid4()}))
            uow.commit()
    except ValueError as error:
        blocked = "tombstoned evidence" in str(error).lower()
    results.append(
        {
            "case_id": "tombstone-evidence-reuse",
            "expected": "blocked",
            "observed": "blocked" if blocked else "allowed",
            "success": blocked,
        }
    )
    measured = {
        "checks": len(results),
        "passed": sum(bool(r["success"]) for r in results),
        "pass_rate": mean(bool(r["success"]) for r in results),
    }
    save_report(
        output,
        "forgetting",
        {
            "fixture_sha256": fixture_hash,
            "evaluation_time": now,
            "database_mode": "rollback-only PostgreSQL, fresh tenants",
            "summary": measured,
            "results": results,
        },
        title="Forgetting and retention benchmark",
        purpose=(
            "Forgetting must remove unwanted memories from future prompts while "
            "preserving appropriate history and audit records. An expired incident "
            "can still answer a historical question. A deliberately tombstoned "
            "incident and summaries derived from its evidence must disappear from "
            "both current and historical search. This run exercises the production "
            "lifecycle service and PostgreSQL."
        ),
        metrics=(
            "Pass rate counts explicitly labeled lifecycle and search checks. The "
            "cases check the 30-day decay grace period, 90-day half-life, priority "
            "floor, confidence threshold, persistent fact retention, expiry "
            "boundary, quarantine, compaction, deletion cascade, tenant isolation, "
            "idempotence, and evidence reuse prevention. Retrieval priority is a "
            "relative ranking weight, not a probability."
        ),
        results=table(["Metric", "Measured value"], [[k, v] for k, v in measured.items()])
        + "\n\n"
        + table(["Check", "Success"], [[r["case_id"], r["success"]] for r in results]),
        interpretation=(
            "A 120-day-old low-confidence episode reaches priority 0.5 because 30 "
            "days are free of decay and the next 90 days form one half-life. "
            "Compaction lowers the linked episode to 0.25. Tombstoning traverses "
            "shared evidence to the summary and blocks later reuse of that "
            "evidence. Review individual failed checks before treating the overall "
            "rate as reliable."
        ),
        limitations=(
            "These are deterministic authored scenarios. The summary is seeded "
            "with explicit derivation metadata; this track does not evaluate "
            "summary generation. Search checks use lexical retrieval; vector "
            "deletion here checks stored state, while broader search/index "
            "coverage remains in integration tests. Tombstones retain audit "
            "content and are not secure physical erasure. Next, benchmark large "
            "evidence graphs and repeated multi-session forgetting."
        ),
    )
    print(f"forgetting: {len(results)} checks completed", flush=True)


def run_packing(factory: UowFactory, embedder: EmbeddingModel, output: Path) -> None:
    data, fixture_hash = read_fixture("prompt_packing")
    now = datetime.now().astimezone()
    tenant = uuid4()
    labels = list(data["memories"])
    vectors = embedder.embed_texts([data["memories"][label]["content"] for label in labels])
    label_of = {}
    with factory() as uow:
        for label, vector in zip(labels, vectors, strict=True):
            item = data["memories"][label]
            event, record = make_record(
                item["content"],
                tenant,
                at=now - timedelta(days=1),
                memory_type=MemoryType(item["memory_type"]),
                subject_keys=item["subject_keys"],
            )
            uow.create_event(event)
            uow.create_memory(record)
            uow.vectors.set_embedding(
                tenant, record.memory_id, vector, model_version=embedder.model_version
            )
            label_of[record.memory_id] = label
        uow.commit()
    traces = []
    for case in data["queries"]:
        query = RetrievalQuery(
            tenant_id=tenant, query_text=case["text"], limit=data["candidate_limit"]
        )
        with factory() as uow:
            search = hybrid_search_with_report(uow, embedder, query)
        filters = dataclasses.replace(
            resolve_filters(query), effective_at=search.temporal.effective_at
        )
        candidates = [h.memory for h in search.hits]
        for budget in data["budgets"]:
            ranked, ranked_tokens = pack_in_rank_order(candidates, filters, budget)
            packed = pack_context(candidates, filters, token_budget=budget)
            for strategy, records, tokens in (
                ("rank_order", ranked, ranked_tokens),
                ("context_packer", [m.memory for m in packed.memories], packed.token_count),
            ):
                selected = [label_of[r.memory_id] for r in records]
                groups = [data["memories"][label]["group"] or label for label in selected]
                useful = set(case["useful_groups"])
                traces.append(
                    {
                        "question": case["text"],
                        "strategy": strategy,
                        "budget": budget,
                        "candidate_labels": [label_of[r.memory_id] for r in candidates],
                        "selected_labels": selected,
                        "tokens": tokens,
                        "selected": len(selected),
                        "duplicates": len(groups) - len(set(groups)),
                        "useful": sum(g in useful for g in groups),
                        "coverage": len(set(groups) & useful) / len(useful),
                        "budget_breach": tokens > budget,
                        "packing_skips": [dataclasses.asdict(s) for s in packed.skipped]
                        if strategy == "context_packer"
                        else [],
                    }
                )
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[trace["budget"], trace["strategy"]].append(trace)
    summary = []
    for (budget, strategy), runs in grouped.items():
        selected = sum(t["selected"] for t in runs)
        summary.append(
            {
                "budget": budget,
                "strategy": strategy,
                "queries": len(runs),
                "mean_tokens": mean(t["tokens"] for t in runs),
                "max_tokens": max(t["tokens"] for t in runs),
                "mean_selected": mean(t["selected"] for t in runs),
                "duplicate_rate": sum(t["duplicates"] for t in runs) / selected if selected else 0,
                "precision": sum(t["useful"] for t in runs) / selected if selected else 0,
                "coverage": mean(t["coverage"] for t in runs),
                "budget_breaches": sum(t["budget_breach"] for t in runs),
            }
        )
    save_report(
        output,
        "prompt_packing",
        {
            "fixture_sha256": fixture_hash,
            "embedding_model": embedder.model_version,
            "reranker": None,
            "token_counter": "production estimate_tokens",
            "summary": summary,
            "results": traces,
        },
        title="Prompt packing benchmark",
        purpose=(
            "The memory prompt has limited space. Giving the agent several reports "
            "of the same checkout incident can crowd out the timeout setting and "
            "remediation procedure. This run compares filling space in retrieval "
            "order with the production context packer, using the same resolved "
            "candidate pool for each query."
        ),
        metrics=(
            "Coverage averages the share of each query's independently labeled "
            "useful groups represented in the prompt. Precision counts useful "
            "selected records, including repeated useful records. Duplicate rate "
            "counts repeated labeled groups among selected records. Token usage "
            "and budget breaches use the production estimate, including rendered "
            "headings, dates, trust labels, and IDs."
        ),
        results=table(
            [
                "Budget",
                "Strategy",
                "Mean tokens",
                "Max tokens",
                "Mean selected",
                "Duplicates",
                "Coverage",
                "Precision",
                "Breaches",
            ],
            [
                [
                    s[k]
                    for k in (
                        "budget",
                        "strategy",
                        "mean_tokens",
                        "max_tokens",
                        "mean_selected",
                        "duplicate_rate",
                        "coverage",
                        "precision",
                        "budget_breaches",
                    )
                ]
                for s in summary
            ],
        ),
        interpretation=(
            "Compare coverage and duplicates at the same budget. A lower duplicate "
            "rate can coexist with lower precision if the freed space admits "
            "distractors. The per-query JSON lists candidates, selected labels, "
            "and packer skip reasons, so that effect can be traced rather than "
            "inferred from aggregate scores."
        ),
        limitations=(
            "The fixture has 24 authored memories and four queries. Real MiniLM "
            "embeddings and PostgreSQL hybrid search are used, with no reranker. "
            "Estimated tokens do not guarantee the same count under an Ollama "
            "tokenizer. This measures prompt selection, not downstream answer "
            "correctness. Next, vary subject tags, candidate caps, relevance "
            "floors, and the actual generation model's tokenizer."
        ),
    )
    print("prompt_packing: 24 strategy/budget/query outcomes completed", flush=True)

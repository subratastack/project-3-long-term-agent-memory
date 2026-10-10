"""Replay canonical cases once, compare read strategies, and write JSON plus Markdown."""

from __future__ import annotations

import argparse
import hashlib
import platform
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from apps.memory_service.advanced.evaluation.benchmark_adapter import (
    BenchmarkCase,
    BenchmarkEvent,
    BenchmarkObservation,
    LatencySample,
    RetrievedMemory,
)
from apps.memory_service.advanced.evaluation.report import build_report, write_report
from apps.memory_service.advanced.evaluation.resource_metrics import measure_storage
from apps.memory_service.consolidation.forgetting import tombstone_memory
from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, WriteDecision
from apps.memory_service.domain.models import MemoryCandidate, MemoryEvent, TemporalValidity, Tenant
from apps.memory_service.embeddings.base import EmbeddingModel, FakeEmbeddingModel
from apps.memory_service.embeddings.cross_encoder import FakeCrossEncoderModel
from apps.memory_service.ingestion.candidate_extractor import RawCandidate
from apps.memory_service.ingestion.normalizer import normalize_candidate
from apps.memory_service.ingestion.service import UowFactory, ingest_candidate
from apps.memory_service.persistence.models import (
    EMBEDDING_DIMENSIONS,
    SEARCH_VECTOR_FUNCTION_DDL,
    Base,
)
from apps.memory_service.persistence.unit_of_work import UnitOfWork, build_engine
from apps.memory_service.retrieval.context_packer import estimate_tokens, pack_context
from apps.memory_service.retrieval.filters import resolve_filters
from apps.memory_service.retrieval.hybrid import HybridSearchHit, hybrid_search_with_report
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import CrossEncoderReranker, Reranker
from apps.memory_service.retrieval.semantic import semantic_search
from apps.memory_service.retrieval.temporal import apply_temporal_resolution
from benchmarks import DEFAULT_FIXTURES, load_cases

STRATEGIES = ("exact_semantic", "hybrid", "hybrid_reranker", "memory_disabled")


@contextmanager
def evaluation_database(database_url: str | None = None) -> Iterator[tuple[UowFactory, str]]:
    """Create a private schema in an outer transaction; roll back even service commits."""
    engine = build_engine(database_url)
    schema = f"advanced_eval_{uuid4().hex}"
    try:
        with engine.connect() as connection:
            outer = connection.begin()
            try:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
                connection.execute(text(SEARCH_VECTOR_FUNCTION_DDL))
                # The private schema is new; probing unqualified table names could find
                # existing public tables and silently skip creating our own corpus.
                Base.metadata.create_all(connection, checkfirst=False)
                sessions = sessionmaker(
                    bind=connection,
                    expire_on_commit=False,
                    join_transaction_mode="create_savepoint",
                )

                def factory() -> UnitOfWork:
                    return UnitOfWork(sessions)

                yield factory, schema
            finally:
                outer.rollback()
    finally:
        engine.dispose()


@dataclass
class PreparedCase:
    case: BenchmarkCase
    tenants: dict[str, UUID]
    memory_ids: dict[str, UUID] = field(default_factory=dict)
    accepted: set[str] = field(default_factory=set)
    active: set[str] = field(default_factory=set)
    writes: list[dict[str, Any]] = field(default_factory=list)


def _write_event(
    factory: UowFactory,
    embedder: EmbeddingModel,
    prepared: PreparedCase,
    event: BenchmarkEvent,
    *,
    now: datetime,
) -> None:
    tenant = prepared.tenants[event.tenant_id]
    evidence = MemoryEvent(
        tenant_id=tenant,
        source_type=SourceType(event.source_type),
        source_reference=f"advanced-fixture:{prepared.case.case_id}:{event.memory_id}",
        content=event.content,
        observed_at=event.observed_at,
        metadata={"approved_by": event.approved_by} if event.approved_by else {},
    )
    with factory() as uow:
        uow.create_event(evidence)
        uow.commit()
    candidate = normalize_candidate(
        RawCandidate(
            content=event.content,
            source_event_ids=[evidence.event_id],
            subject_keys=list(event.subject_keys),
            confidence=0.95,
        ),
        MemoryType(event.memory_type),
        [evidence],
    )
    if not isinstance(candidate, MemoryCandidate):
        raise ValueError(f"{prepared.case.case_id}: invalid fixture evidence")
    candidate.temporal_validity = TemporalValidity(
        valid_from=event.observed_at, valid_to=event.valid_to
    )
    decision = ingest_candidate(factory, candidate, tenant_id=tenant, now=now)
    prepared.writes.append(
        {
            "memory_id": event.memory_id,
            "decision": decision.decision.value,
            "reason_codes": decision.reason_codes,
        }
    )
    if decision.accepted_memory_id is not None:
        prepared.memory_ids[event.memory_id] = decision.accepted_memory_id
        if decision.decision in (WriteDecision.ACCEPT, WriteDecision.SUPERSEDE):
            prepared.accepted.add(event.memory_id)
            with factory() as uow:
                record = uow.records.get(tenant, decision.accepted_memory_id)
                assert record is not None
                uow.vectors.set_embedding(
                    tenant,
                    record.memory_id,
                    embedder.embed_texts([record.content])[0],
                    model_version=embedder.model_version,
                )
                uow.commit()


def prepare_case(
    factory: UowFactory, embedder: EmbeddingModel, case: BenchmarkCase
) -> PreparedCase:
    tenants = {
        key: uuid4() for key in {case.tenant_id, *(event.tenant_id for event in case.writes)}
    }
    prepared = PreparedCase(case, tenants)
    with factory() as uow:
        for key, tenant_id in tenants.items():
            uow.tenants.add(
                Tenant(tenant_id=tenant_id, name=f"advanced-eval-{case.case_id}-{key}-{tenant_id}")
            )
        uow.commit()
    for session in case.sessions:
        for event in session.events:
            if event.operation == "write":
                _write_event(factory, embedder, prepared, event, now=session.at)
            else:
                memory_id = prepared.memory_ids.get(event.memory_id)
                if memory_id is not None:
                    tombstone_memory(
                        factory,
                        tenants[event.tenant_id],
                        memory_id,
                        reason="Benchmark requested forgetting",
                        requested_by="benchmark:fixture",
                        now=session.at,
                    )
    with factory() as uow:
        owners = {event.memory_id: event.tenant_id for event in case.writes}
        for label, memory_id in prepared.memory_ids.items():
            record = uow.records.get(tenants[owners[label]], memory_id)
            if record is not None and record.status == MemoryStatus.ACTIVE:
                prepared.active.add(label)
    return prepared


def read_case(
    factory: UowFactory,
    embedder: EmbeddingModel,
    prepared: PreparedCase,
    strategy: str,
    *,
    k: int,
    candidates: int,
    token_budget: int,
    reranker: Reranker,
) -> BenchmarkObservation:
    """Use production reads; retain measurements and partial hits on execution errors."""
    case = prepared.case
    start = time.perf_counter()
    retrieval_ms = rerank_ms = packing_ms = 0.0
    observed: tuple[RetrievedMemory, ...] = ()
    presented: tuple[str, ...] = ()
    context = ""
    tokens = 0
    error = None
    fell_back = False
    accepted, active = tuple(sorted(prepared.accepted)), tuple(sorted(prepared.active))
    if strategy == "memory_disabled":
        accepted = active = ()
    else:
        label_of = {memory_id: label for label, memory_id in prepared.memory_ids.items()}
        tenant_of = {tenant_id: key for key, tenant_id in prepared.tenants.items()}
        query = RetrievalQuery(
            tenant_id=prepared.tenants[case.tenant_id],
            query_text=case.query,
            as_of=case.query_time,
            limit=k,
        )
        retrieval_start = time.perf_counter()
        try:
            with factory() as uow:
                try:
                    if strategy == "exact_semantic":
                        semantic = semantic_search(
                            uow, embedder, query.model_copy(update={"limit": candidates})
                        )
                        hits = [
                            HybridSearchHit(hit.memory, 1.0 - hit.distance, None, rank)
                            for rank, hit in enumerate(semantic, 1)
                        ]
                        hits, _ = apply_temporal_resolution(uow, query, hits)
                        hits = hits[:k]
                    else:
                        result = hybrid_search_with_report(
                            uow,
                            embedder,
                            query,
                            candidate_limit=candidates,
                            reranker=reranker if strategy == "hybrid_reranker" else None,
                        )
                        hits = result.hits
                        if result.rerank:
                            rerank_ms = result.rerank.latency_ms
                            fell_back = result.rerank.fell_back
                finally:
                    retrieval_ms = max(
                        0.0, (time.perf_counter() - retrieval_start) * 1000 - rerank_ms
                    )
                # Monotonic rank scores preserve service order, including lifecycle adjustments.
                # Actual cosine/RRF/reranker scores remain available as raw_score in the trace.
                observed = tuple(
                    RetrievedMemory(
                        memory_id=label_of.get(hit.memory.memory_id, str(hit.memory.memory_id)),
                        tenant_id=tenant_of.get(hit.memory.tenant_id, str(hit.memory.tenant_id)),
                        content=hit.memory.content,
                        score=float(len(hits) - rank),
                        raw_score=hit.rerank_score
                        if hit.rerank_score is not None
                        else hit.fused_score,
                        valid_from=hit.memory.temporal_validity.valid_from,
                        valid_to=hit.memory.temporal_validity.valid_to,
                        status=hit.memory.status.value,
                    )
                    for rank, hit in enumerate(hits)
                )
                packing_start = time.perf_counter()
                try:
                    packed = pack_context(
                        [hit.memory for hit in hits],
                        resolve_filters(query),
                        token_budget=token_budget,
                    )
                    context, tokens = packed.text, packed.token_count
                    presented = tuple(
                        label_of.get(item.memory.memory_id, str(item.memory.memory_id))
                        for item in packed.memories
                    )
                finally:
                    packing_ms = (time.perf_counter() - packing_start) * 1000
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            tokens = estimate_tokens(context)
            if retrieval_ms == 0.0:
                retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    return BenchmarkObservation(
        case_id=case.case_id,
        strategy=strategy,
        retrieved=observed,
        presented_memory_ids=presented,
        accepted_memory_ids=accepted,
        active_memory_ids=active,
        context=context,
        context_tokens=tokens,
        error=error,
        reranker_fell_back=fell_back,
        latency=LatencySample(
            retrieval_ms=retrieval_ms,
            reranking_ms=rerank_ms,
            packing_ms=packing_ms,
            total_ms=(time.perf_counter() - start) * 1000,
        ),
    )


def run_evaluation(
    factory: UowFactory,
    *,
    schema: str,
    cases: Sequence[BenchmarkCase],
    k: int = 5,
    candidates: int = 30,
    repeats: int = 3,
    token_budget: int = 160,
    embedder: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if min(k, repeats, token_budget) < 1 or candidates < k:
        raise ValueError("positive k/repeats/budget required; candidates must be >= k")
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("cases must be nonempty and have unique case IDs")
    embedder = embedder or FakeEmbeddingModel(dimensions=EMBEDDING_DIMENSIONS)
    reranker = reranker or CrossEncoderReranker(FakeCrossEncoderModel(), max_candidates=candidates)
    prepared = []
    checkpoints = []
    source_writes = 0
    for case in cases:
        prepared.append(prepare_case(factory, embedder, case))
        source_writes += len(case.writes)
        if len(prepared) % 10 == 0 or len(prepared) == len(cases):
            with factory() as uow:
                checkpoints.append(
                    {
                        "cases": len(prepared),
                        **measure_storage(uow.session, schema=schema, corpus_size=source_writes),
                    }
                )
    runs: dict[str, list[BenchmarkObservation]] = {strategy: [] for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        for item in prepared:
            # Untimed warm-up for each query; no writes or feedback happen in read_case.
            read_case(
                factory,
                embedder,
                item,
                strategy,
                k=k,
                candidates=candidates,
                token_budget=token_budget,
                reranker=reranker,
            )
            for _ in range(repeats):
                runs[strategy].append(
                    read_case(
                        factory,
                        embedder,
                        item,
                        strategy,
                        k=k,
                        candidates=candidates,
                        token_budget=token_budget,
                        reranker=reranker,
                    )
                )
    footprint = checkpoints[-1]
    footprints = {strategy: footprint for strategy in STRATEGIES}
    footprints["memory_disabled"] = {
        "scope": "no retained memory",
        "corpus_size": 0,
        "table_bytes": 0,
        "index_bytes": 0,
        "total_bytes": 0,
        "bytes_per_source_write": None,
        "tables": {},
    }
    with factory() as uow:
        postgres_version = uow.session.execute(text("SELECT version()")).scalar_one()
        pgvector_version = uow.session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname='vector'")
        ).scalar_one()
    report = build_report(
        cases,
        runs,
        k=k,
        token_budget=token_budget,
        storage=footprints,
        metadata={
            **(metadata or {}),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": embedder.model_version,
            "embedding_dimensions": embedder.dimensions,
            "reranker": getattr(reranker, "model_version", type(reranker).__name__),
            "packages": {
                package: version(package)
                for package in ("ir-measures", "ranx", "numpy", "sqlalchemy")
            },
            "postgres": postgres_version,
            "pgvector": pgvector_version,
            "candidates": candidates,
            "token_counter": "estimate_tokens (characters/word pieces heuristic)",
            "adapters": "original benchmark-style subsets, not official benchmark scores",
        },
    )
    report["storage_checkpoints"] = checkpoints
    report["write_decisions"] = {item.case.case_id: item.writes for item in prepared}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=Path("docs/reports/advanced-memory-eval"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=160)
    parser.add_argument("--models", choices=("fake", "real"), default="fake")
    parser.add_argument(
        "--fail-on-safety",
        action="store_true",
        help="write both reports, then exit 1 on any safety violation or execution error",
    )
    args = parser.parse_args()
    if min(args.k, args.repeats, args.token_budget) < 1 or args.candidates < args.k:
        parser.error("positive k/repeats/budget required; candidates must be >= k")
    from dotenv import load_dotenv

    load_dotenv()
    embedder = None
    reranker = None
    if args.models == "real":
        from apps.memory_service.embeddings.cross_encoder import SentenceTransformerCrossEncoder
        from apps.memory_service.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingModel,
        )

        embedder = SentenceTransformerEmbeddingModel()
        reranker = CrossEncoderReranker(
            SentenceTransformerCrossEncoder(), max_candidates=args.candidates
        )
    cases = load_cases(args.fixtures)
    with evaluation_database() as (factory, schema):
        report = run_evaluation(
            factory,
            schema=schema,
            cases=cases,
            k=args.k,
            candidates=args.candidates,
            repeats=args.repeats,
            token_budget=args.token_budget,
            embedder=embedder,
            reranker=reranker,
            metadata={
                "fixture_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
                "created_at": datetime.now(UTC).isoformat(),
                "models": args.models,
            },
        )
    for path in write_report(report, args.output):
        print(path)
    if args.fail_on_safety and any(
        not summary["safety_passed"] for summary in report["strategies"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

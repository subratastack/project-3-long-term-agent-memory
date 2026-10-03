"""Before/after evaluation of CrossEncoder reranking on a labeled dataset.

Measures, for hybrid retrieval with and without reranking over the *same*
fused candidate pool (so the reranker is the only difference):

- quality: Precision@K, Recall@K, MRR, nDCG@K
- cost: P50/P95 end-to-end latency, P50/P95 rerank-stage latency, and how
  many candidates were sent through the CrossEncoder per query

Usage (needs `docker compose up -d postgres`):

    uv run python -m apps.benchmark.run_retrieval_eval [--k 5] [--repeats 5]

Everything -- including creating any missing tables, the same way the
integration test fixtures do -- happens inside one outer transaction that is
rolled back at the end, so a run leaves the database exactly as it found it. The same dataset
and `evaluate_strategy` also back the assertions in
apps/tests/integration/retrieval/test_reranking.py.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from apps.benchmark.metrics import (
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.hybrid import hybrid_search_with_report
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import RERANK_MAX_CANDIDATES, Reranker

UowFactory = Callable[[], UnitOfWork]

# Grade 2 = answers the question; grade 1 = related, useful context.
ANSWER = 2
CONTEXT = 1

# A small operations-flavoured corpus with deliberate near-miss distractors:
# memories that share vocabulary or topic with a query but do not answer it
# (e.g. the INC-48213 *symptom* vs. its *root cause*, the nightly backup that
# succeeded vs. the weekly one that failed). Those are exactly the cases
# where lexical overlap and embedding similarity both mislead, and where
# reading the query and memory together should help.
LABELED_MEMORIES: dict[str, str] = {
    "POOL": "The connection pool was exhausted after 30 retries against db-primary.",
    "POOL_RESIZE": "The connection pool size for db-primary was raised from 20 to 50.",
    "REPLICA_LAG": "Replication lag on db-replica-2 reached 45 seconds during the nightly batch.",
    "INC_503": "Incident INC-48213: checkout-api returned HTTP 503 for 12 minutes.",
    "INC_ROOT": "Root cause of INC-48213 was an expired TLS certificate on the payments gateway.",
    "CERT_RENEW": "TLS certificates are renewed by cert-manager 30 days before they expire.",
    "DISK": "Node worker-node-17 reported disk pressure at 92% usage.",
    "DISK_CLEAN": "Old container images are pruned from worker nodes every Sunday.",
    "DARK_DISABLED": "Feature flag dark_mode is now disabled for all users.",
    "DARK_ENABLED": "Feature flag dark_mode is now enabled for all users.",
    "BACKUP_OK": "The nightly backup job completed successfully at 2am.",
    "BACKUP_FAIL": "The weekly backup job failed because the S3 bucket quota was exceeded.",
    "TIMEOUT": "The configured request timeout for checkout-api is 2 seconds.",
    "TIMEOUT_RETRY": "Clients retry timed-out requests up to 3 times with exponential backoff.",
    "EMAIL_PREF": "The user prefers email over phone calls for notifications.",
    "PHONE": "The user's phone number was updated last week.",
    "RUNBOOK": "Diagnostic runbook: check logs, restart the service, confirm health checks pass.",
    "DEPLOY_FREEZE": "Deployments are frozen every Friday after 3pm.",
    "DEPLOY_ROLLBACK": "The search-service deploy on 2026-09-12 was rolled back for high latency.",
    "LATENCY_SLO": "The p99 latency SLO for search-service is 300 milliseconds.",
    "ONCALL": "Priya is the on-call engineer for the payments team this week.",
    "OOM": "billing-worker was OOM-killed repeatedly because of a memory leak in the PDF renderer.",
    "RATE_LIMIT": "The partner API enforces a rate limit of 100 requests per minute.",
    "CACHE": "Redis cache hit rate dropped to 40% after the key schema change.",
}


@dataclass(frozen=True)
class LabeledQuery:
    text: str
    grades: Mapping[str, int]


LABELED_QUERIES: list[LabeledQuery] = [
    LabeledQuery("what caused incident INC-48213", {"INC_ROOT": ANSWER, "INC_503": CONTEXT}),
    LabeledQuery("why did the database run out of connections", {"POOL": ANSWER}),
    LabeledQuery("is dark mode turned off for users", {"DARK_DISABLED": ANSWER}),
    LabeledQuery("did the weekly backup fail", {"BACKUP_FAIL": ANSWER}),
    LabeledQuery("how should we contact the user", {"EMAIL_PREF": ANSWER, "PHONE": CONTEXT}),
    LabeledQuery("which service ran out of memory", {"OOM": ANSWER}),
    LabeledQuery("when are deployments not allowed", {"DEPLOY_FREEZE": ANSWER}),
    LabeledQuery("who is on call for payments", {"ONCALL": ANSWER}),
    LabeledQuery("how many partner API calls can we make per minute", {"RATE_LIMIT": ANSWER}),
    LabeledQuery(
        "what happens when a request times out",
        {"TIMEOUT_RETRY": ANSWER, "TIMEOUT": CONTEXT},
    ),
    LabeledQuery("why was the search-service release reverted", {"DEPLOY_ROLLBACK": ANSWER}),
    LabeledQuery("how do we avoid expired certificates", {"CERT_RENEW": ANSWER}),
    LabeledQuery("why did the cache get less effective", {"CACHE": ANSWER}),
    LabeledQuery(
        "how is disk space freed on worker nodes", {"DISK_CLEAN": ANSWER, "DISK": CONTEXT}
    ),
    LabeledQuery("how big is the db-primary connection pool", {"POOL_RESIZE": ANSWER}),
    LabeledQuery("is the read replica falling behind", {"REPLICA_LAG": ANSWER}),
]


@dataclass
class StrategyReport:
    """Quality and cost of one retrieval strategy over `LABELED_QUERIES`."""

    name: str
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float
    rerank_p50_ms: float | None
    rerank_p95_ms: float | None
    mean_candidates_reranked: float
    max_candidates_reranked: int
    fallbacks: int
    # query text -> 1-based rank of its grade-2 answer (None = not in top k)
    answer_ranks: dict[str, int | None] = field(default_factory=dict)


def seed_corpus(uow: UnitOfWork, embedder: EmbeddingModel, tenant_id: UUID) -> dict[str, UUID]:
    """Persist and index every `LABELED_MEMORIES` entry; return label -> memory_id."""
    memory_ids: dict[str, UUID] = {}
    for label, content in LABELED_MEMORIES.items():
        event = MemoryEvent(
            tenant_id=tenant_id,
            source_type=SourceType.SYSTEM_EVENT,
            source_reference=f"eval-{label}",
            content=content,
            observed_at=datetime.now(UTC),
        )
        memory = MemoryRecord(
            tenant_id=tenant_id,
            content=content,
            confidence=0.9,
            provenance=[
                Provenance(
                    event_id=event.event_id,
                    source_type=event.source_type,
                    source_reference=event.source_reference,
                    observed_at=event.observed_at,
                    trust_level=TrustLevel.SYSTEM,
                )
            ],
            temporal_validity=TemporalValidity(valid_from=datetime.now(UTC)),
            memory_type=MemoryType.SEMANTIC,
            trust_level=TrustLevel.SYSTEM,
            status=MemoryStatus.ACTIVE,
        )
        uow.create_event(event)
        uow.create_memory(memory)
        uow.vectors.set_embedding(
            tenant_id,
            memory.memory_id,
            embedder.embed_texts([content])[0],
            model_version=embedder.model_version,
        )
        memory_ids[label] = memory.memory_id
    return memory_ids


def evaluate_strategy(
    uow_factory: UowFactory,
    embedder: EmbeddingModel,
    tenant_id: UUID,
    memory_ids: Mapping[str, UUID],
    *,
    name: str,
    reranker: Reranker | None,
    k: int = 5,
    candidate_limit: int = RERANK_MAX_CANDIDATES,
    repeats: int = 3,
) -> StrategyReport:
    """Run every labeled query `repeats` times and score the result.

    Quality metrics come from the first run of each query (retrieval is
    deterministic); latency percentiles pool every run. One untimed warm-up
    query runs first so model loading and first-call overheads are not
    counted as query latency.

    Precision, recall, and MRR use grade-2 answers as binary relevance;
    nDCG also credits grade-1 context. Precision divides by `k`, even when
    fewer results are returned.
    """
    label_of = {memory_id: label for label, memory_id in memory_ids.items()}

    def run(query_text: str) -> tuple[list[str], float, float | None, int, bool]:
        query = RetrievalQuery(tenant_id=tenant_id, query_text=query_text, limit=k)
        started = time.perf_counter()
        with uow_factory() as uow:
            result = hybrid_search_with_report(
                uow, embedder, query, reranker=reranker, candidate_limit=candidate_limit
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ranked = [label_of[hit.memory.memory_id] for hit in result.hits]
        report = result.rerank
        return (
            ranked,
            elapsed_ms,
            report.latency_ms if report is not None else None,
            report.candidates_reranked if report is not None else 0,
            report is not None and report.fell_back,
        )

    run(LABELED_QUERIES[0].text)

    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    rerank_latencies: list[float] = []
    reranked_counts: list[int] = []
    fallbacks = 0
    answer_ranks: dict[str, int | None] = {}

    for labeled in LABELED_QUERIES:
        answers = {label for label, grade in labeled.grades.items() if grade == ANSWER}
        for attempt in range(repeats):
            ranked, elapsed_ms, rerank_ms, reranked, fell_back = run(labeled.text)
            latencies.append(elapsed_ms)
            if rerank_ms is not None:
                rerank_latencies.append(rerank_ms)
            reranked_counts.append(reranked)
            fallbacks += int(fell_back)
            if attempt == 0:
                precisions.append(precision_at_k(ranked, answers, k))
                recalls.append(recall_at_k(ranked, answers, k))
                reciprocal_ranks.append(reciprocal_rank(ranked, answers))
                ndcgs.append(ndcg_at_k(ranked, labeled.grades, k))
                answer_ranks[labeled.text] = next(
                    (rank for rank, label in enumerate(ranked, 1) if label in answers), None
                )

    return StrategyReport(
        name=name,
        k=k,
        precision_at_k=_mean(precisions),
        recall_at_k=_mean(recalls),
        mrr=_mean(reciprocal_ranks),
        ndcg_at_k=_mean(ndcgs),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        rerank_p50_ms=percentile(rerank_latencies, 50) if rerank_latencies else None,
        rerank_p95_ms=percentile(rerank_latencies, 95) if rerank_latencies else None,
        mean_candidates_reranked=_mean([float(count) for count in reranked_counts]),
        max_candidates_reranked=max(reranked_counts),
        fallbacks=fallbacks,
        answer_ranks=answer_ranks,
    )


def format_reports(reports: list[StrategyReport]) -> str:
    """Render reports as a fixed-width table, plus per-query answer ranks."""
    k = reports[0].k
    header = (
        f"{'strategy':<22}{f'Precision@{k}':>13}{f'Recall@{k}':>10}{'MRR':>8}{f'nDCG@{k}':>9}"
        f"{'P50 ms':>9}{'P95 ms':>9}{'rr P50':>9}{'rr P95':>9}{'reranked':>10}{'fallbk':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        rr_p50 = f"{r.rerank_p50_ms:.1f}" if r.rerank_p50_ms is not None else "-"
        rr_p95 = f"{r.rerank_p95_ms:.1f}" if r.rerank_p95_ms is not None else "-"
        reranked = f"{r.mean_candidates_reranked:.1f}/{r.max_candidates_reranked}"
        lines.append(
            f"{r.name:<22}{r.precision_at_k:>13.3f}{r.recall_at_k:>10.3f}"
            f"{r.mrr:>8.3f}{r.ndcg_at_k:>9.3f}"
            f"{r.latency_p50_ms:>9.1f}{r.latency_p95_ms:>9.1f}{rr_p50:>9}{rr_p95:>9}"
            f"{reranked:>10}{r.fallbacks:>8}"
        )
    lines.append("")
    lines.append("rank of the grade-2 answer per query (- = not in top k):")
    names = "  ".join(f"{r.name[:14]:>14}" for r in reports)
    lines.append(f"  {'query':<52}{names}")
    for labeled in LABELED_QUERIES:
        ranks = "  ".join(f"{_fmt_rank(r.answer_ranks.get(labeled.text)):>14}" for r in reports)
        lines.append(f"  {labeled.text[:50]:<52}{ranks}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=int, default=5, help="final result count / metric cutoff")
    parser.add_argument(
        "--candidates",
        type=int,
        default=RERANK_MAX_CANDIDATES,
        help="fused candidate pool size handed to the reranker",
    )
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per query")
    args = parser.parse_args()

    # Imported here so `--help` does not load torch.
    from dotenv import load_dotenv
    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    from apps.memory_service.embeddings.cross_encoder import SentenceTransformerCrossEncoder
    from apps.memory_service.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingModel,
    )
    from apps.memory_service.persistence.models import SEARCH_VECTOR_FUNCTION_DDL, Base
    from apps.memory_service.persistence.unit_of_work import build_engine
    from apps.memory_service.retrieval.reranker import CrossEncoderReranker

    load_dotenv()
    embedder = SentenceTransformerEmbeddingModel()
    reranker = CrossEncoderReranker(
        SentenceTransformerCrossEncoder(), max_candidates=args.candidates
    )

    engine = build_engine()
    with engine.connect() as connection:
        outer = connection.begin()
        try:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text(SEARCH_VECTOR_FUNCTION_DDL))
            Base.metadata.create_all(connection)
            session_factory = sessionmaker(
                bind=connection,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            )

            def uow_factory() -> UnitOfWork:
                return UnitOfWork(session_factory)

            tenant_id = uuid4()
            with uow_factory() as uow:
                memory_ids = seed_corpus(uow, embedder, tenant_id)
                uow.commit()

            reports = [
                evaluate_strategy(
                    uow_factory,
                    embedder,
                    tenant_id,
                    memory_ids,
                    name=name,
                    reranker=strategy_reranker,
                    k=args.k,
                    candidate_limit=args.candidates,
                    repeats=args.repeats,
                )
                for name, strategy_reranker in [
                    ("hybrid (RRF)", None),
                    ("hybrid + CrossEncoder", reranker),
                ]
            ]
        finally:
            outer.rollback()
    engine.dispose()

    print(
        f"{len(LABELED_MEMORIES)} memories, {len(LABELED_QUERIES)} queries, "
        f"candidate pool={args.candidates}, k={args.k}, repeats={args.repeats}\n"
    )
    print(format_reports(reports))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _fmt_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


if __name__ == "__main__":
    main()

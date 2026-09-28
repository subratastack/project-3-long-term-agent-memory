"""Evaluation of context packing against filling the budget in rank order.

Both strategies get the *same* resolved candidates from the same hybrid
retrieval run, the same rendering, and the same token counter, so the
selection rule is the only difference. For each token budget it measures:

- final token count (mean and max; a max above the budget is a bug)
- number of selected memories
- duplicate rate: the share of selected memories whose duplicate group was
  already represented in the context
- useful-memory coverage: the share of each query's useful groups that made
  it into the context, averaged over queries
- precision: the share of selected memories that belong to a useful group.
  Neither strategy has an absolute relevance floor -- ranks only order
  candidates -- so with budget to spare both add off-topic memories.

Duplicates and usefulness come from labels, not from the packer's own
similarity measure, so the packer is not grading itself.

Usage (needs `docker compose up -d postgres`):

    uv run python -m apps.benchmark.run_context_packing_eval \
        [--budgets 150 300 500] [--candidates 30] [--rerank] [--no-subject-keys]

Like `run_retrieval_eval`, everything runs inside one outer transaction that
is rolled back at the end. The same dataset and `evaluate_packing` back the
assertions in apps/tests/integration/retrieval/test_context_packing.py.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
)
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.context_packer import (
    PACK_CANDIDATES,
    TokenCounter,
    estimate_tokens,
    pack_context,
    render_header,
    render_memory,
)
from apps.memory_service.retrieval.filters import RetrievalFilters, resolve_filters
from apps.memory_service.retrieval.hybrid import hybrid_search_with_report
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import Reranker

UowFactory = Callable[[], UnitOfWork]

EPISODIC = MemoryType.EPISODIC
SEMANTIC = MemoryType.SEMANTIC
PROCEDURAL = MemoryType.PROCEDURAL


@dataclass(frozen=True)
class LabeledMemory:
    """One corpus memory. Memories sharing a `group` say the same thing:
    any one of them covers it, and a second one is a duplicate. `group`
    defaults to the memory's own label."""

    content: str
    memory_type: MemoryType
    subject_keys: tuple[str, ...]
    group: str | None = None


_TIMEOUT = ("checkout-api", "timeout")
_POOL = ("db-primary", "connection-pool")
_CERT = ("tls-certificate", "payments-gateway")

# Three incident clusters, each with a current fact, a remediation procedure,
# and several reports of the same kind of incident -- some near-verbatim,
# some paraphrased, so both duplicate detection and subject overlap matter.
# Plus unrelated distractors.
LABELED_MEMORIES: dict[str, LabeledMemory] = {
    "TIMEOUT_CFG": LabeledMemory(
        "The request timeout for checkout-api is configured to 2 seconds.", SEMANTIC, _TIMEOUT
    ),
    "TIMEOUT_RUNBOOK": LabeledMemory(
        "Remediation for checkout-api timeouts: scale out the payments-gateway pool first, "
        "and raise the timeout to 5 seconds only if p99 latency stays above 1.5 seconds.",
        PROCEDURAL,
        (*_TIMEOUT, "remediation"),
    ),
    "TIMEOUT_INC_1": LabeledMemory(
        "Incident INC-3101: checkout-api requests timed out after 2 seconds during peak "
        "traffic.",
        EPISODIC,
        _TIMEOUT,
        "TIMEOUT_INC",
    ),
    "TIMEOUT_INC_2": LabeledMemory(
        "Incident INC-3102: checkout-api requests timed out after 2 seconds during peak "
        "traffic.",
        EPISODIC,
        _TIMEOUT,
        "TIMEOUT_INC",
    ),
    "TIMEOUT_INC_3": LabeledMemory(
        "Incident INC-3115: checkout-api calls hit the 2 second timeout while the flash sale "
        "was running.",
        EPISODIC,
        _TIMEOUT,
        "TIMEOUT_INC",
    ),
    "TIMEOUT_INC_4": LabeledMemory(
        "Incident INC-3122: checkout-api timed out for 8 minutes when the payments gateway "
        "slowed down.",
        EPISODIC,
        _TIMEOUT,
        "TIMEOUT_INC",
    ),
    "TIMEOUT_INC_5": LabeledMemory(
        "Incident INC-3130: timeouts on checkout-api again, users saw failed payments during "
        "the evening peak.",
        EPISODIC,
        _TIMEOUT,
        "TIMEOUT_INC",
    ),
    "TIMEOUT_INC_6": LabeledMemory(
        "Incident INC-3131: timeouts on checkout-api again, users saw failed payments during "
        "the evening peak.",
        EPISODIC,
        _TIMEOUT,
        "TIMEOUT_INC",
    ),
    "POOL_CFG": LabeledMemory(
        "The db-primary connection pool size is 50 connections.", SEMANTIC, _POOL
    ),
    "POOL_RUNBOOK": LabeledMemory(
        "Remediation for db-primary pool exhaustion: terminate idle-in-transaction sessions "
        "older than 5 minutes, then restart pgbouncer.",
        PROCEDURAL,
        (*_POOL, "remediation"),
    ),
    "POOL_INC_1": LabeledMemory(
        "Incident INC-2201: db-primary connection pool exhausted, API requests queued for "
        "4 minutes.",
        EPISODIC,
        _POOL,
        "POOL_INC",
    ),
    "POOL_INC_2": LabeledMemory(
        "Incident INC-2202: db-primary connection pool exhausted, API requests queued for "
        "4 minutes.",
        EPISODIC,
        _POOL,
        "POOL_INC",
    ),
    "POOL_INC_3": LabeledMemory(
        "Incident INC-2240: the connection pool on db-primary ran out during the nightly "
        "batch job.",
        EPISODIC,
        _POOL,
        "POOL_INC",
    ),
    "POOL_INC_4": LabeledMemory(
        "Incident INC-2251: db-primary refused new connections after the pool filled up "
        "during a traffic spike.",
        EPISODIC,
        _POOL,
        "POOL_INC",
    ),
    "CERT_CFG": LabeledMemory(
        "TLS certificates are renewed automatically by cert-manager 30 days before they "
        "expire.",
        SEMANTIC,
        _CERT,
    ),
    "CERT_RUNBOOK": LabeledMemory(
        "Remediation for an expired certificate: run cert-manager renew for the affected "
        "secret, then restart the ingress controller.",
        PROCEDURAL,
        (*_CERT, "remediation"),
    ),
    "CERT_INC_1": LabeledMemory(
        "Incident INC-48213: an expired TLS certificate on the payments gateway caused HTTP "
        "503 errors for 12 minutes.",
        EPISODIC,
        _CERT,
        "CERT_INC",
    ),
    "CERT_INC_2": LabeledMemory(
        "Incident INC-48214: an expired TLS certificate on the payments gateway caused HTTP "
        "503 errors for 12 minutes.",
        EPISODIC,
        _CERT,
        "CERT_INC",
    ),
    "CERT_INC_3": LabeledMemory(
        "Incident INC-48390: the payments gateway served an expired TLS certificate and "
        "checkout returned 503s.",
        EPISODIC,
        _CERT,
        "CERT_INC",
    ),
    "LATENCY_SLO": LabeledMemory(
        "The p99 latency SLO for checkout-api is 800 milliseconds.",
        SEMANTIC,
        ("checkout-api", "latency"),
    ),
    "DEPLOY_FREEZE": LabeledMemory(
        "Deployments are frozen every Friday after 3pm.", SEMANTIC, ("deployments",)
    ),
    "RATE_LIMIT": LabeledMemory(
        "The partner API enforces a rate limit of 100 requests per minute.",
        SEMANTIC,
        ("partner-api", "rate-limit"),
    ),
    "ONCALL": LabeledMemory(
        "Priya is the on-call engineer for the payments team this week.",
        SEMANTIC,
        ("on-call", "payments"),
    ),
    "CACHE": LabeledMemory(
        "Redis cache hit rate dropped to 40% after the key schema change.",
        EPISODIC,
        ("redis", "cache"),
    ),
}


@dataclass(frozen=True)
class PackingQuery:
    text: str
    useful_groups: frozenset[str]


PACKING_QUERIES: list[PackingQuery] = [
    PackingQuery(
        "checkout-api requests are timing out, what should I do",
        frozenset({"TIMEOUT_CFG", "TIMEOUT_INC", "TIMEOUT_RUNBOOK"}),
    ),
    PackingQuery(
        "the db-primary connection pool is exhausted again",
        frozenset({"POOL_CFG", "POOL_INC", "POOL_RUNBOOK"}),
    ),
    PackingQuery(
        "payments gateway returns 503 because of an expired certificate",
        frozenset({"CERT_CFG", "CERT_INC", "CERT_RUNBOOK"}),
    ),
    PackingQuery(
        "how many requests per minute can we send to the partner API",
        frozenset({"RATE_LIMIT"}),
    ),
]

STRATEGIES = ("rank order", "context packer")


@dataclass(frozen=True)
class PackingReport:
    """One strategy at one budget, over every `PACKING_QUERIES` entry."""

    strategy: str
    token_budget: int
    mean_tokens: float
    max_tokens: int
    mean_selected: float
    duplicate_rate: float
    coverage: float
    precision: float


def group_of(label: str) -> str:
    return LABELED_MEMORIES[label].group or label


def seed_corpus(
    uow: UnitOfWork, embedder: EmbeddingModel, tenant_id: UUID, *, subject_keys: bool = True
) -> dict[str, UUID]:
    """Persist and index every `LABELED_MEMORIES` entry; return label -> memory_id.

    `subject_keys=False` seeds without them, to see how far content overlap
    alone gets the packer.
    """
    valid_from = datetime.now(UTC) - timedelta(days=1)
    memory_ids: dict[str, UUID] = {}
    for label, labeled in LABELED_MEMORIES.items():
        event = MemoryEvent(
            tenant_id=tenant_id,
            source_type=SourceType.SYSTEM_EVENT,
            source_reference=f"packing-eval-{label}",
            content=labeled.content,
            observed_at=valid_from,
        )
        memory = MemoryRecord(
            tenant_id=tenant_id,
            content=labeled.content,
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
            temporal_validity=TemporalValidity(valid_from=valid_from),
            memory_type=labeled.memory_type,
            subject_keys=list(labeled.subject_keys) if subject_keys else [],
            trust_level=TrustLevel.SYSTEM,
            status=MemoryStatus.ACTIVE,
        )
        uow.create_event(event)
        uow.create_memory(memory)
        uow.vectors.set_embedding(
            tenant_id,
            memory.memory_id,
            embedder.embed_texts([labeled.content])[0],
            model_version=embedder.model_version,
        )
        memory_ids[label] = memory.memory_id
    return memory_ids


def pack_in_rank_order(
    memories: Sequence[MemoryRecord],
    filters: RetrievalFilters,
    token_budget: int,
    count_tokens: TokenCounter = estimate_tokens,
) -> tuple[list[MemoryRecord], int]:
    """The baseline: add memories best-ranked first, skipping any that don't fit.

    Renders exactly as `pack_context` does, so only the selection differs.
    """
    header = render_header(filters.effective_at)
    selected: list[MemoryRecord] = []
    lines = [header]
    for memory in memories:
        line = render_memory(memory)
        if count_tokens("\n".join([*lines, line])) <= token_budget:
            selected.append(memory)
            lines.append(line)
    return selected, count_tokens("\n".join(lines)) if selected else 0


def evaluate_packing(
    uow_factory: UowFactory,
    embedder: EmbeddingModel,
    tenant_id: UUID,
    memory_ids: Mapping[str, UUID],
    *,
    budgets: Sequence[int],
    reranker: Reranker | None = None,
    candidates: int = PACK_CANDIDATES,
) -> list[PackingReport]:
    """Retrieve once per query, pack at every budget with both strategies, and score it."""
    label_of = {memory_id: label for label, memory_id in memory_ids.items()}
    # (strategy, budget) -> one (token count, selected labels, query) per query
    runs: dict[tuple[str, int], list[tuple[int, list[str], PackingQuery]]] = {
        (strategy, budget): [] for strategy in STRATEGIES for budget in budgets
    }

    for labeled_query in PACKING_QUERIES:
        query = RetrievalQuery(tenant_id=tenant_id, query_text=labeled_query.text, limit=candidates)
        with uow_factory() as uow:
            search = hybrid_search_with_report(uow, embedder, query, reranker=reranker)
        filters = dataclasses.replace(
            resolve_filters(query), effective_at=search.temporal.effective_at
        )
        resolved = [hit.memory for hit in search.hits]
        for budget in budgets:
            baseline, baseline_tokens = pack_in_rank_order(resolved, filters, budget)
            runs["rank order", budget].append(
                (baseline_tokens, [label_of[m.memory_id] for m in baseline], labeled_query)
            )
            packed = pack_context(resolved, filters, token_budget=budget)
            runs["context packer", budget].append(
                (
                    packed.token_count,
                    [label_of[m.memory.memory_id] for m in packed.memories],
                    labeled_query,
                )
            )

    return [
        _score(strategy, budget, runs[strategy, budget])
        for budget in budgets
        for strategy in STRATEGIES
    ]


def _score(
    strategy: str, budget: int, runs: list[tuple[int, list[str], PackingQuery]]
) -> PackingReport:
    tokens = [token_count for token_count, _, _ in runs]
    selected = sum(len(labels) for _, labels, _ in runs)
    duplicates = sum(
        len(labels) - len({group_of(label) for label in labels}) for _, labels, _ in runs
    )
    useful = sum(
        1 for _, labels, query in runs for label in labels if group_of(label) in query.useful_groups
    )
    coverages = [
        len({group_of(label) for label in labels} & query.useful_groups) / len(query.useful_groups)
        for _, labels, query in runs
    ]
    return PackingReport(
        strategy=strategy,
        token_budget=budget,
        mean_tokens=sum(tokens) / len(tokens),
        max_tokens=max(tokens),
        mean_selected=selected / len(runs),
        duplicate_rate=duplicates / selected if selected else 0.0,
        coverage=sum(coverages) / len(coverages),
        precision=useful / selected if selected else 0.0,
    )


def format_reports(reports: Sequence[PackingReport]) -> str:
    header = (
        f"{'budget':>6}  {'strategy':<16}{'tokens mean/max':>17}{'selected':>10}"
        f"{'dup rate':>10}{'coverage':>10}{'precision':>11}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        tokens = f"{r.mean_tokens:.0f}/{r.max_tokens}"
        lines.append(
            f"{r.token_budget:>6}  {r.strategy:<16}{tokens:>17}{r.mean_selected:>10.1f}"
            f"{r.duplicate_rate:>10.2f}{r.coverage:>10.2f}{r.precision:>11.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[150, 300, 500])
    parser.add_argument(
        "--candidates",
        type=int,
        default=PACK_CANDIDATES,
        help="resolved candidates retrieval hands the packer",
    )
    parser.add_argument("--rerank", action="store_true", help="rerank with the CrossEncoder")
    parser.add_argument(
        "--no-subject-keys",
        action="store_true",
        help="seed without subject keys (content overlap only)",
    )
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
    reranker = CrossEncoderReranker(SentenceTransformerCrossEncoder()) if args.rerank else None

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
                memory_ids = seed_corpus(
                    uow, embedder, tenant_id, subject_keys=not args.no_subject_keys
                )
                uow.commit()

            reports = evaluate_packing(
                uow_factory,
                embedder,
                tenant_id,
                memory_ids,
                budgets=args.budgets,
                reranker=reranker,
                candidates=args.candidates,
            )
        finally:
            outer.rollback()
    engine.dispose()

    print(
        f"{len(LABELED_MEMORIES)} memories, {len(PACKING_QUERIES)} queries, "
        f"{args.candidates} candidates, rerank={'on' if args.rerank else 'off'}, "
        f"subject keys={'off' if args.no_subject_keys else 'on'}\n"
    )
    print(format_reports(reports))


if __name__ == "__main__":
    main()

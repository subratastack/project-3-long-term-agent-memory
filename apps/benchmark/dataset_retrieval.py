"""Score independent datasets through the project's actual PostgreSQL retrieval APIs."""

from __future__ import annotations

import fnmatch
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import Field, model_validator

from apps.benchmark.datasets.schema import (
    DatasetManifest,
    DatasetModel,
    RetrievalQuestion,
    RetrievalSample,
)
from apps.benchmark.metrics import (
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from apps.memory_service.domain.enums import (
    IndexStatus,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
)
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    Provenance,
    TemporalValidity,
    Tenant,
)
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.ingestion.service import UowFactory
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS
from apps.memory_service.retrieval.hybrid import hybrid_search_with_report
from apps.memory_service.retrieval.lexical import lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import Reranker
from apps.memory_service.retrieval.semantic import semantic_search

Strategy = Literal["lexical", "semantic", "hybrid", "hybrid_reranked"]


class RunProfile(DatasetModel):
    ks: tuple[int, ...] = (1, 5, 10)
    strategies: tuple[Strategy, ...] = ("lexical", "semantic", "hybrid", "hybrid_reranked")
    sources: tuple[str, ...] = ()
    max_samples_per_source: int | None = Field(default=None, gt=0)
    max_queries_per_sample: int | None = Field(default=None, gt=0)
    candidate_limit: int = Field(default=30, ge=1, le=200)
    repeats: int = Field(default=1, gt=0)
    seed: int = 42
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    torch_threads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> RunProfile:
        if not self.ks or any(not 1 <= k <= 200 for k in self.ks):
            raise ValueError("provide cutoffs within [1, 200]")
        if len(set(self.ks)) != len(self.ks) or tuple(sorted(self.ks)) != self.ks:
            raise ValueError("cutoffs must be distinct and increasing")
        if self.candidate_limit < max(self.ks):
            raise ValueError("candidate limit must be at least the largest cutoff")
        if not self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ValueError("provide distinct strategies")
        return self


@dataclass(frozen=True)
class Coverage:
    sample_id: str
    row_index: int
    source: str
    chunks: int
    questions: int
    eligible: int
    selected: int
    scored: int
    exclusions: dict[str, int]


@dataclass(frozen=True)
class Quality:
    k: int
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float


@dataclass(frozen=True)
class QueryResult:
    sample_id: str
    source: str
    question_id: str
    original_question_id: str | None
    question: str
    label_method: str
    strategy: str
    relevant_ids: tuple[str, ...]
    ranked_ids: tuple[str, ...]
    quality: tuple[Quality, ...]
    latencies_ms: tuple[float, ...]
    fallbacks: int
    candidates_reranked: int


@dataclass(frozen=True)
class Aggregate:
    source: str
    label_method: str
    strategy: str
    k: int
    queries: int
    precision: float
    recall: float
    mrr: float
    ndcg: float
    latency_p50_ms: float
    latency_p95_ms: float
    fallbacks: int


@dataclass(frozen=True)
class SampleCost:
    sample_id: str
    chunks: int
    construction_ms: float


@dataclass(frozen=True)
class DatasetReport:
    created_at: str
    manifest: dict[str, object]
    profile: dict[str, object]
    environment: dict[str, str]
    coverage: tuple[Coverage, ...]
    construction: tuple[SampleCost, ...]
    aggregates: tuple[Aggregate, ...]
    results: tuple[QueryResult, ...]


def select_questions(
    samples: Sequence[RetrievalSample], profile: RunProfile
) -> dict[str, tuple[RetrievalQuestion, ...]]:
    """Sample before label eligibility is checked, so exclusions stay observable."""
    counts: Counter[str] = Counter()
    selected: dict[str, tuple[RetrievalQuestion, ...]] = {}
    for sample in samples:
        if profile.sources and not any(
            fnmatch.fnmatchcase(sample.source, s) for s in profile.sources
        ):
            continue
        if (
            profile.max_samples_per_source is not None
            and counts[sample.source] >= profile.max_samples_per_source
        ):
            continue
        counts[sample.source] += 1
        questions = list(sample.questions)
        if profile.max_queries_per_sample is not None:
            # Per-sample seed keeps sampling stable when another source is added.
            rng = random.Random(f"{profile.seed}:{sample.sample_id}")
            chosen = set(
                rng.sample(
                    range(len(questions)), min(len(questions), profile.max_queries_per_sample)
                )
            )
            questions = [q for index, q in enumerate(questions) if index in chosen]
        selected[sample.sample_id] = tuple(questions)
    return selected


def score_ranking(
    ranked: Sequence[str], relevant: Sequence[str], ks: Sequence[int]
) -> tuple[Quality, ...]:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("unlabeled questions cannot be scored")
    if len(set(ranked)) != len(ranked):
        raise ValueError("retrieval returned duplicate IDs")
    return tuple(
        Quality(
            k=k,
            precision=precision_at_k(ranked, relevant_set, k),
            recall=recall_at_k(ranked, relevant_set, k),
            reciprocal_rank=reciprocal_rank(ranked[:k], relevant_set),
            ndcg=ndcg_at_k(ranked, dict.fromkeys(relevant_set, 1), k),
        )
        for k in ks
    )


def seed_sample(
    uow_factory: UowFactory, embedder: EmbeddingModel, sample: RetrievalSample
) -> tuple[UUID, dict[UUID, str]]:
    """Seed raw context chunks, with no questions, answers, or evidence labels in records."""
    tenant_id = uuid4()
    now = datetime.now(UTC)
    labels: dict[UUID, str] = {}
    with uow_factory() as uow:
        uow.tenants.add(Tenant(tenant_id=tenant_id, name=f"dataset-eval-{tenant_id}"))
        for start in range(0, len(sample.chunks), 64):
            batch = sample.chunks[start : start + 64]
            vectors = embedder.embed_texts([chunk.text for chunk in batch])
            for chunk, vector in zip(batch, vectors, strict=True):
                event = MemoryEvent(
                    tenant_id=tenant_id,
                    source_type=SourceType.USER_MESSAGE,
                    source_reference=chunk.chunk_id,
                    content=chunk.text,
                    observed_at=now,
                )
                memory_id = uuid5(NAMESPACE_URL, chunk.chunk_id)
                record = MemoryRecord(
                    memory_id=memory_id,
                    tenant_id=tenant_id,
                    content=chunk.text,
                    confidence=1.0,
                    provenance=[
                        Provenance(
                            event_id=event.event_id,
                            source_type=event.source_type,
                            source_reference=event.source_reference,
                            observed_at=now,
                            trust_level=TrustLevel.MEDIUM,
                        )
                    ],
                    temporal_validity=TemporalValidity(valid_from=now),
                    memory_type=MemoryType.EPISODIC,
                    trust_level=TrustLevel.MEDIUM,
                    status=MemoryStatus.ACTIVE,
                    embedding=vector,
                    embedding_model_version=embedder.model_version,
                    index_status=IndexStatus.INDEXED,
                )
                uow.create_event(event)
                uow.create_memory(record)
                labels[memory_id] = chunk.chunk_id
        uow.commit()
    return tenant_id, labels


def retrieve(
    uow_factory: UowFactory,
    embedder: EmbeddingModel,
    tenant_id: UUID,
    labels: dict[UUID, str],
    question: str,
    strategy: Strategy,
    profile: RunProfile,
    reranker: Reranker | None,
) -> tuple[list[str], float, bool, int]:
    query = RetrievalQuery(tenant_id=tenant_id, query_text=question, limit=max(profile.ks))
    fallback, reranked = False, 0
    started = time.perf_counter()
    with uow_factory() as uow:
        if strategy == "lexical":
            ids = [hit.memory.memory_id for hit in lexical_search(uow, query)]
        elif strategy == "semantic":
            ids = [hit.memory.memory_id for hit in semantic_search(uow, embedder, query)]
        else:
            result = hybrid_search_with_report(
                uow,
                embedder,
                query,
                reranker=reranker if strategy == "hybrid_reranked" else None,
                candidate_limit=profile.candidate_limit,
            )
            ids = [hit.memory.memory_id for hit in result.hits]
            if result.rerank:
                fallback = result.rerank.fell_back
                reranked = result.rerank.candidates_reranked
    elapsed_ms = (time.perf_counter() - started) * 1000
    return [labels[memory_id] for memory_id in ids], elapsed_ms, fallback, reranked


def aggregate_results(results: Sequence[QueryResult]) -> tuple[Aggregate, ...]:
    groups: dict[tuple[str, str, str, int], list[tuple[QueryResult, Quality]]] = defaultdict(list)
    for result in results:
        for quality in result.quality:
            groups[result.source, result.label_method, result.strategy, quality.k].append(
                (result, quality)
            )
    return tuple(
        Aggregate(
            source=source,
            label_method=method,
            strategy=strategy,
            k=k,
            queries=len(entries),
            precision=mean(q.precision for _, q in entries),
            recall=mean(q.recall for _, q in entries),
            mrr=mean(q.reciprocal_rank for _, q in entries),
            ndcg=mean(q.ndcg for _, q in entries),
            latency_p50_ms=percentile([t for r, _ in entries for t in r.latencies_ms], 50),
            latency_p95_ms=percentile([t for r, _ in entries for t in r.latencies_ms], 95),
            fallbacks=sum(r.fallbacks for r, _ in entries),
        )
        for (source, method, strategy, k), entries in sorted(groups.items())
    )


def run_dataset(
    uow_factory: UowFactory,
    embedder: EmbeddingModel,
    samples: Sequence[RetrievalSample],
    manifest: DatasetManifest,
    profile: RunProfile,
    *,
    reranker: Reranker | None = None,
    environment: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> DatasetReport:
    if embedder.dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError(f"embedding dimensions must match PostgreSQL ({EMBEDDING_DIMENSIONS})")
    if "hybrid_reranked" in profile.strategies and reranker is None:
        raise ValueError("hybrid_reranked requires a reranker")
    selected = select_questions(samples, profile)
    coverage = tuple(
        Coverage(
            sample_id=s.sample_id,
            row_index=s.row_index,
            source=s.source,
            chunks=len(s.chunks),
            questions=len(s.questions),
            eligible=sum(bool(q.relevant_ids) for q in s.questions),
            selected=len(selected.get(s.sample_id, ())),
            scored=sum(bool(q.relevant_ids) for q in selected.get(s.sample_id, ())),
            exclusions=dict(
                Counter(q.skip_reason for q in s.questions if q.skip_reason is not None)
            ),
        )
        for s in samples
    )
    if not any(c.scored for c in coverage):
        raise ValueError("selection has no questions with usable relevance labels")
    results: list[QueryResult] = []
    costs: list[SampleCost] = []
    for sample in samples:
        questions = [q for q in selected.get(sample.sample_id, ()) if q.relevant_ids]
        if not questions:
            continue
        if progress:
            progress(
                f"Indexing row {sample.row_index}: {sample.source}, {len(sample.chunks)} chunks"
            )
        started = time.perf_counter()
        tenant_id, labels = seed_sample(uow_factory, embedder, sample)
        costs.append(
            SampleCost(sample.sample_id, len(sample.chunks), (time.perf_counter() - started) * 1000)
        )
        for strategy in profile.strategies:
            # Untimed warm-up per sample/strategy, separate from scored runs.
            retrieve(
                uow_factory,
                embedder,
                tenant_id,
                labels,
                questions[0].text,
                strategy,
                profile,
                reranker,
            )
            for index, question in enumerate(questions):
                latencies: list[float] = []
                first: list[str] = []
                fallbacks = reranked = 0
                for attempt in range(profile.repeats):
                    ranked, latency, fallback, count = retrieve(
                        uow_factory,
                        embedder,
                        tenant_id,
                        labels,
                        question.text,
                        strategy,
                        profile,
                        reranker,
                    )
                    if attempt == 0:
                        first = ranked
                    latencies.append(latency)
                    fallbacks += int(fallback)
                    reranked += count
                results.append(
                    QueryResult(
                        sample_id=sample.sample_id,
                        source=sample.source,
                        question_id=question.question_id,
                        original_question_id=question.original_question_id,
                        question=question.text,
                        label_method=question.label_method,
                        strategy=strategy,
                        relevant_ids=question.relevant_ids,
                        ranked_ids=tuple(first),
                        quality=score_ranking(first, question.relevant_ids, profile.ks),
                        latencies_ms=tuple(latencies),
                        fallbacks=fallbacks,
                        candidates_reranked=reranked,
                    )
                )
                if progress and (index + 1) % 10 == 0:
                    progress(
                        f"Row {sample.row_index}, {strategy}: {index + 1}/{len(questions)} queries"
                    )
    return DatasetReport(
        created_at=datetime.now(UTC).isoformat(),
        manifest=manifest.model_dump(mode="json"),
        profile=profile.model_dump(mode="json"),
        environment=environment or {},
        coverage=coverage,
        construction=tuple(costs),
        aggregates=aggregate_results(results),
        results=tuple(results),
    )


def format_report(report: DatasetReport) -> str:
    manifest = report.manifest
    selected_count = sum(c.selected for c in report.coverage)
    scored_count = sum(c.scored for c in report.coverage)
    lines = [
        f"# {manifest['dataset_id']} — retrieval learning report",
        "",
        "## What this experiment asks",
        "",
        "Can the project's retrieval pipeline locate useful context within long histories? "
        "Original context is stored as independent chunks, and each question is searched "
        "against its own context. Relevance labels are used only by the scorer.",
        "",
        "This is a retrieval diagnostic, not an official MemoryAgentBench answer score. "
        "It does not run a reasoning model or evaluate generated answers, memory extraction, "
        "write policy, conflict resolution, forgetting, or prompt packing.",
        "",
        "## How to read the scores",
        "",
        "A chunk is one stored piece of context. Precision@K divides relevant chunks in the "
        "first K results by K, including unfilled slots. Recall@K divides those hits by all "
        "labeled relevant chunks. MRR averages the reciprocal rank of the first relevant "
        "chunk within that K; nDCG@K compares the ordering to an ideal ranking. Labels here "
        "are binary, not graded. Each metric is averaged equally across scored questions.",
        "",
        "Illustrative example: with two relevant chunks and results `[distractor, relevant]` "
        "at K=2, precision and recall are both 0.5, reciprocal rank is 0.5, and nDCG is "
        "approximately 0.387. These teaching values are separate from the measured tables.",
        "",
        "Two label methods stay separate: `provided_turn_labels` maps LongMemEval's "
        "question-specific `has_answer` turns to their chunks in the full context, retaining "
        "all distractors. All children of a labeled turn inherit relevance. "
        "`answer_span_proxy` labels chunks containing an answer alias after Unicode, "
        "case, punctuation, and whitespace normalization. It can credit incidental mentions "
        "and miss paraphrases. Boolean or fewer-than-three-character answers are excluded.",
        "",
        "## Measured run and reproducibility",
        "",
        f"Run completed at `{report.created_at}`. Split: `{manifest['split']}`. "
        f"Dataset revision: `{manifest['revision']}`. Adapter: `{manifest['adapter_version']}`.",
        "",
        f"Raw SHA-256: `{manifest['raw_sha256']}`. "
        f"Normalized JSONL SHA-256: `{manifest['samples_sha256']}`.",
        "",
        f"Chunks use {manifest['chunk_words']} whitespace-separated words with "
        f"{manifest['overlap_words']} words of overlap. These are words, not model tokens; "
        "models may truncate inputs at their token limit.",
        "",
        f"Selected {selected_count} questions before checking label eligibility; scored "
        f"{scored_count}; excluded {selected_count - scored_count}. "
        "Source/sample filters and sampling limits are part of the profile below. "
        "These results apply only to that selection.",
        "",
        "Construction includes embedding and seeding raw chunks. Retrieval latency includes "
        "query embedding where used, database reads, and reranking where enabled; it excludes "
        "model loading, construction, and one warm-up per sample/strategy. All K cutoffs "
        "reuse the same result list fetched at the largest K, so latency is repeated across "
        "K rows. Hybrid and reranked hybrid use the same candidate limit. Failures of the "
        "reranker are counted as fallbacks. All database writes are rolled back on exit.",
        "",
        "```json",
    ]
    import json

    lines.extend([json.dumps(report.profile, indent=2), "```", "", "Environment:", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in report.environment.items())
    lines.extend(
        [
            "",
            "## Label coverage across the prepared split",
            "",
            "Eligible counts are adapter label coverage, not successful retrieval. "
            "Rows with zero selected queries were outside this run's selection.",
            "",
            "| Row | Source | Chunks | Questions | Eligible | Selected | Scored | Exclusions |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(
        f"| {c.row_index} | {c.source} | {c.chunks} | {c.questions} | {c.eligible} | "
        f"{c.selected} | {c.scored} | "
        f"{', '.join(f'{key}: {count}' for key, count in c.exclusions.items()) or '—'} |"
        for c in report.coverage
    )
    lines.extend(
        [
            "",
            "## Measured retrieval quality and cost",
            "",
            "| Source | Labels | Strategy | K | Queries | Precision@K | Recall@K | "
            "MRR | nDCG@K | P50 ms | P95 ms | Fallbacks |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {a.source} | {a.label_method} | {a.strategy} | {a.k} | {a.queries} | "
        f"{a.precision:.4f} | {a.recall:.4f} | {a.mrr:.4f} | {a.ndcg:.4f} | "
        f"{a.latency_p50_ms:.2f} | {a.latency_p95_ms:.2f} | {a.fallbacks} |"
        for a in report.aggregates
    )
    lines.extend(
        ["", "Construction:", "", "| Sample | Chunks | Seconds |", "| --- | ---: | ---: |"]
    )
    lines.extend(
        f"| {c.sample_id} | {c.chunks} | {c.construction_ms / 1000:.2f} |"
        for c in report.construction
    )
    lines.extend(["", "## Observations from this run", ""])
    largest_k = max(a.k for a in report.aggregates)
    sources = sorted({a.source for a in report.aggregates})
    for source in sources:
        rows = [a for a in report.aggregates if a.source == source and a.k == largest_k]
        best = max(rows, key=lambda a: a.ndcg)
        ties = [a.strategy for a in rows if abs(a.ndcg - best.ndcg) < 1e-12]
        lines.append(
            f"- `{source}`: highest nDCG@{largest_k} was {best.ndcg:.4f} "
            f"for {', '.join(f'`{s}`' for s in ties)}, over {best.queries} scored questions. "
            "This is descriptive; no significance test was performed."
        )
    lines.extend(
        [
            "",
            "## Limits and next experiments",
            "",
            "- EventQA's narrative answer summaries usually do not match literal passages. "
            "Its excluded questions need independently reviewed passage labels "
            "or answer evaluation.",
            "- Overlapping chunks and inherited turn labels increase the "
            "relevant-chunk denominator. Low recall may mean partial coverage "
            "of an evidence turn rather than failure to find it. "
            "These scores do not prove all evidence needed for a multi-hop answer was retrieved.",
            "- RULER's common answer strings can occur in unrelated passages. "
            "Inspect traces before interpreting proxy improvements as reasoning gains.",
            "- Sampling is deterministic per sample, but the first context per source is selected "
            "when sample limits apply. It is not a random sample of contexts. Latency is "
            "hardware-dependent, and quality can change with chunking, models, "
            "and dataset revision.",
            "- Next compare more contexts and queries, vary chunk sizes and K, and review failed "
            "queries. Keep provided labels and proxy labels in separate result groups.",
            "",
            "## Sources and artifacts",
            "",
            "The companion JSON contains exact retrieved IDs, relevance IDs, per-query metrics, "
            "latencies, environment, and the run profile. It supports auditing "
            "and later comparisons.",
            "",
            "- [Dataset card](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)",
            "- [Upstream evaluation protocol and official metrics](https://github.com/HUST-AI-HYZ/MemoryAgentBench)",
            "- [Project benchmark guide](../memory-agent-bench.md)",
            "",
        ]
    )
    return "\n".join(lines)

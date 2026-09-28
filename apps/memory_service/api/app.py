"""A minimal HTTP wrapper over the ingestion and retrieval pipelines, for manual/curl use.

This is deliberately thin: every endpoint just calls straight into
`apps.memory_service.persistence.unit_of_work`, `apps.memory_service.ingestion`,
and `apps.memory_service.retrieval`, with no logic of its own beyond
translating HTTP requests into calls those modules already expose (and, for
`/retrieve`, flattening the pipeline's result into JSON). It exists so the
pipelines documented in `docs/ingestion-flow.md` and `docs/retrieval-flow.md`
can be driven with `curl` instead of only from Python, not as the start of a
"real" API surface.

Run it with:

    uvicorn apps.memory_service.api.app:app --reload --port 8000

See docs/api-reference.md for every endpoint, docs/ingestion-flow.md for the
ingestion walkthrough, and docs/retrieval-flow.md for the retrieval one.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Literal
from uuid import UUID

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from apps.memory_service.api.demo_data import seed_demo_memories
from apps.memory_service.consolidation.conflict_resolver import ConflictState
from apps.memory_service.domain.enums import (
    IndexStatus,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
)
from apps.memory_service.domain.models import MemoryEvent, MemoryRecord, Tenant
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.embeddings.cross_encoder import SentenceTransformerCrossEncoder
from apps.memory_service.ingestion.candidate_extractor import (
    CandidateExtractor,
    OllamaCandidateExtractor,
    build_memory_candidates,
)
from apps.memory_service.ingestion.normalizer import RejectedExtraction
from apps.memory_service.ingestion.service import ingest_candidate
from apps.memory_service.persistence.unit_of_work import (
    UnitOfWork,
    build_engine,
    build_session_factory,
)
from apps.memory_service.retrieval.context_packer import (
    DEFAULT_TOKEN_BUDGET,
    PACK_CANDIDATES,
    SkipReason,
    retrieve_context,
)
from apps.memory_service.retrieval.hybrid import hybrid_search_with_report
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import CrossEncoderReranker
from apps.memory_service.retrieval.temporal import ExclusionReason

# Real environment variables take precedence over `.env`.
load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="Long-Term Agent Memory OS -- dev API",
    description=(
        "A thin curl-able wrapper over the ingestion and retrieval pipelines. "
        "Not for production use."
    ),
)

UowFactory = Callable[[], UnitOfWork]

logger = logging.getLogger(__name__)

# Bulk ingest is synchronous and makes one Ollama call per event, so the cap
# keeps a single request's latency bounded.
MAX_BULK_INGEST_EVENTS = 50


# --- Wiring: built once per process, not per request ------------------------


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return build_session_factory(build_engine())


def get_uow_factory() -> UowFactory:
    session_factory = _session_factory()
    return lambda: UnitOfWork(session_factory)


@lru_cache(maxsize=1)
def get_extractor() -> OllamaCandidateExtractor:
    model = os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b-instruct"
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaCandidateExtractor(
        model=model, client=httpx.Client(timeout=120), base_url=base_url
    )


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingModel:
    # Imported here so the ingestion endpoints don't pay for loading torch.
    from apps.memory_service.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingModel,
    )

    return SentenceTransformerEmbeddingModel()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(SentenceTransformerCrossEncoder())


def _require_tenant(uow: UnitOfWork, tenant_id: UUID) -> Tenant:
    """404 unless `tenant_id` is registered -- every tenant-scoped route calls this first."""
    tenant = uow.tenants.get(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=404, detail="tenant not found; create it first with POST /tenants"
        )
    return tenant


# --- Request/response schemas ------------------------------------------------


class CreateTenantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    description: str | None = None



class CreateEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_reference: str
    content: str
    observed_at: datetime | None = None
    actor_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateEventResponse(BaseModel):
    event_id: UUID


class IngestOutcome(BaseModel):
    status: str
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str | None = None
    accepted_memory_id: UUID | None = None


class IngestResponse(BaseModel):
    event_id: UUID
    candidate_count: int
    outcomes: list[IngestOutcome]


class BulkIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[CreateEventRequest] = Field(
        default_factory=list, description="New events to store, then ingest."
    )
    event_ids: list[UUID] = Field(
        default_factory=list, description="Already-stored events of this tenant to ingest."
    )

    @property
    def is_empty(self) -> bool:
        """True when neither list is given: ingest the tenant's pending events instead."""
        return not self.events and not self.event_ids

    @model_validator(mode="after")
    def _check_size(self) -> BulkIngestRequest:
        total = len(self.events) + len(self.event_ids)
        if total > MAX_BULK_INGEST_EVENTS:
            raise ValueError(f"at most {MAX_BULK_INGEST_EVENTS} events per request, got {total}")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("`event_ids` contains duplicates")
        return self


class BulkIngestEventResult(BaseModel):
    event_id: UUID
    source_reference: str
    status: Literal["ingested", "failed"] = Field(
        description="`failed` means extraction or ingestion raised for this event only."
    )
    error: str | None = None
    candidate_count: int
    outcomes: list[IngestOutcome]


class BulkIngestResponse(BaseModel):
    tenant_id: UUID
    events_created: int
    events_ingested: int
    events_failed: int
    events_pending: int = Field(
        description="Stored events still never ingested after this call (includes failures)."
    )
    results: list[BulkIngestEventResult]


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=1)
    as_of: datetime | None = Field(
        default=None, description="Ask what was true at this time; omit for 'now'."
    )
    memory_types: list[MemoryType] | None = None
    min_trust: TrustLevel | None = None
    limit: int = Field(default=5, ge=1, le=50)
    rerank: bool = Field(default=True, description="Run the CrossEncoder reranking stage.")


class HitScores(BaseModel):
    lexical_rank: int | None = Field(description="Rank in full-text search; null if not found.")
    semantic_rank: int | None = Field(description="Rank in vector search; null if not found.")
    fused_score: float = Field(description="Reciprocal Rank Fusion score.")
    rerank_score: float | None = Field(description="CrossEncoder score; null if not reranked.")


class RetrievedMemory(BaseModel):
    rank: int
    memory_id: UUID
    content: str
    memory_type: MemoryType
    trust_level: TrustLevel
    status: MemoryStatus
    valid_from: datetime
    valid_to: datetime | None
    scores: HitScores


class RerankStage(BaseModel):
    candidates_in: int
    candidates_reranked: int
    latency_ms: float
    fallback_reason: str | None


class ExcludedMemory(BaseModel):
    memory_id: UUID
    content: str | None
    reason: ExclusionReason
    related_memory_id: UUID | None


class ConflictView(BaseModel):
    relation_id: UUID
    memory_ids: list[UUID]
    state: ConflictState
    reason: str
    winner_id: UUID | None
    loser_id: UUID | None


class TemporalStage(BaseModel):
    effective_at: datetime
    candidates_in: int
    excluded: list[ExcludedMemory]
    conflicts: list[ConflictView]


class PipelineReport(BaseModel):
    candidates_fused: int = Field(description="Distinct memories after fusing both searches.")
    rerank: RerankStage | None = Field(description="Null when reranking was turned off.")
    temporal: TemporalStage


class RetrieveResponse(BaseModel):
    query: RetrieveRequest
    hits: list[RetrievedMemory]
    pipeline: PipelineReport


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=1)
    token_budget: int = Field(
        default=DEFAULT_TOKEN_BUDGET, ge=1, le=32_000, description="Ceiling for the context."
    )
    candidates: int = Field(
        default=PACK_CANDIDATES,
        ge=1,
        le=200,
        description="How many resolved retrieval hits the packer chooses from.",
    )
    as_of: datetime | None = Field(
        default=None, description="Ask what was true at this time; omit for 'now'."
    )
    memory_types: list[MemoryType] | None = None
    min_trust: TrustLevel | None = None
    rerank: bool = Field(default=True, description="Run the CrossEncoder reranking stage.")


class PackedMemoryView(BaseModel):
    rank: int = Field(description="Position in the resolved retrieval order.")
    memory_id: UUID
    content: str
    memory_type: MemoryType
    trust_level: TrustLevel
    subject_keys: list[str]
    tokens: int = Field(description="Estimated cost of this memory's line.")
    gain: float = Field(description="Value it added, after discounting overlap.")


class SkippedMemoryView(BaseModel):
    rank: int
    memory_id: UUID
    content: str
    reason: SkipReason
    related_memory_id: UUID | None = Field(
        description="The copy kept (duplicate) or the overlapping memory chosen (redundant)."
    )


class ContextStats(BaseModel):
    token_budget: int
    token_count: int = Field(description="Tokens in `context`; never above the budget.")
    candidates_in: int
    selected: int
    duplicates_removed: int
    redundant_skipped: int
    over_budget: int
    selection: str = Field(description="Which greedy pass won: gain_per_token or gain.")


class ContextResponse(BaseModel):
    context: str = Field(description="The text to put in the agent prompt.")
    memories: list[PackedMemoryView]
    skipped: list[SkippedMemoryView]
    stats: ContextStats


class SeededMemoryView(BaseModel):
    label: str
    memory_id: UUID
    content: str
    status: MemoryStatus
    trust_level: TrustLevel
    valid_from: datetime
    valid_to: datetime | None


class SeedDemoResponse(BaseModel):
    tenant_id: UUID
    memories: list[SeededMemoryView]


class IndexResponse(BaseModel):
    indexed: int
    model_version: str


# --- Routes -------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tenants", response_model=Tenant, status_code=201)
def create_tenant(body: CreateTenantRequest) -> Tenant:
    """Register a tenant; its `tenant_id` is what every other route is scoped by.

    Names are unique, so a tenant can be found again later by name via
    `GET /tenants?name=...`. A duplicate name is a 409.
    """
    tenant = Tenant(name=body.name, description=body.description)
    with get_uow_factory()() as uow:
        if uow.tenants.get_by_name(tenant.name) is not None:
            raise HTTPException(status_code=409, detail="a tenant with this name already exists")
        uow.tenants.add(tenant)
        try:
            uow.commit()
        except IntegrityError as exc:  # lost a race with a concurrent create
            raise HTTPException(
                status_code=409, detail="a tenant with this name already exists"
            ) from exc
    return tenant


@app.get("/tenants", response_model=list[Tenant])
def list_tenants(name: str | None = None) -> list[Tenant]:
    """List every registered tenant, oldest first -- or just the one called `name`."""
    with get_uow_factory()() as uow:
        if name is not None:
            tenant = uow.tenants.get_by_name(name)
            return [tenant] if tenant is not None else []
        return uow.tenants.list_all()


@app.get("/tenants/{tenant_id}", response_model=Tenant)
def get_tenant(tenant_id: UUID) -> Tenant:
    """Fetch one registered tenant."""
    with get_uow_factory()() as uow:
        return _require_tenant(uow, tenant_id)


@app.post("/tenants/{tenant_id}/events", response_model=CreateEventResponse, status_code=201)
def create_event(tenant_id: UUID, body: CreateEventRequest) -> CreateEventResponse:
    """Persist one immutable evidence event -- the pipeline's raw input."""
    event = MemoryEvent(
        tenant_id=tenant_id,
        source_type=body.source_type,
        source_reference=body.source_reference,
        content=body.content,
        observed_at=body.observed_at or datetime.now(UTC),
        actor_id=body.actor_id,
        metadata=body.metadata,
    )
    with get_uow_factory()() as uow:
        _require_tenant(uow, tenant_id)
        uow.create_event(event)
        uow.commit()
    return CreateEventResponse(event_id=event.event_id)


@app.post(
    "/tenants/{tenant_id}/events/{event_id}/ingest",
    response_model=IngestResponse,
)
def ingest_event(tenant_id: UUID, event_id: UUID) -> IngestResponse:
    """Run extract -> classify -> normalize -> write policy for one stored event.

    How it works:
        1. Load the event back from PostgreSQL (it must already exist --
           create it first via `POST /tenants/{tenant_id}/events`).
        2. Run `build_memory_candidates`, which calls the configured Ollama
           model to propose candidates, then deterministically classifies
           and normalizes each one.
        3. A `RejectedExtraction` (e.g. the extractor cited no evidence)
           never reaches write policy -- it is reported as
           `status="rejected_before_policy"` directly.
        4. Every real `MemoryCandidate` goes through `ingest_candidate`,
           which independently re-verifies its provenance and runs the
           deterministic write policy -- the LLM proposed it, but this call
           is the only thing that can accept, quarantine, or reject it.
    """
    uow_factory = get_uow_factory()
    with uow_factory() as uow:
        _require_tenant(uow, tenant_id)
        event = uow.events.get(tenant_id, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found for this tenant")

    outcomes_raw = build_memory_candidates(get_extractor(), [event])
    outcomes = [_ingest_outcome(uow_factory, outcome) for outcome in outcomes_raw]
    _record_ingestion(uow_factory, event, len(outcomes_raw))

    return IngestResponse(
        event_id=event_id, candidate_count=len(outcomes_raw), outcomes=outcomes
    )


@app.post("/tenants/{tenant_id}/events/ingest", response_model=BulkIngestResponse)
def bulk_ingest_events(
    tenant_id: UUID, body: BulkIngestRequest | None = None
) -> BulkIngestResponse:
    """Store and/or ingest many events for one tenant in a single call.

    How it works:
        1. In one transaction: check the tenant, then pick the events.
           - With `events` / `event_ids`: load every `event_ids` entry (any
             unknown id is a 404 and nothing is written) and store every new
             `events` entry, committed together so a bad request never
             leaves some of them behind.
           - With no payload (or both lists empty): take the tenant's oldest
             pending events -- stored but never ingested, see
             `MemoryEventRepository.list_pending` -- up to
             `MAX_BULK_INGEST_EVENTS`. Call again while `events_pending` > 0.
        2. Ingest each event exactly as `POST .../events/{event_id}/ingest`
           does -- one extraction call per event, so every outcome is
           attributable to the event it came from -- in request order
           (`event_ids` first, then `events`), or oldest first when pending.
           Each success is logged in `memory_event_ingestions`, which is what
           keeps it out of later no-payload calls.
        3. A failure while ingesting one event (e.g. Ollama unreachable) is
           recorded as `status="failed"` for that event and the rest still
           run. Candidates already decided for the failed event stay
           decided, and are listed in its `outcomes`; the event itself is not
           logged, so it stays pending.

    Two no-payload calls running at the same time can pick the same pending
    events -- run them one at a time.
    """
    body = body or BulkIngestRequest()
    new_events = [
        MemoryEvent(
            tenant_id=tenant_id,
            source_type=item.source_type,
            source_reference=item.source_reference,
            content=item.content,
            observed_at=item.observed_at or datetime.now(UTC),
            actor_id=item.actor_id,
            metadata=item.metadata,
        )
        for item in body.events
    ]
    uow_factory = get_uow_factory()
    with uow_factory() as uow:
        _require_tenant(uow, tenant_id)
        existing: list[MemoryEvent] = []
        missing: list[str] = []
        for event_id in body.event_ids:
            event = uow.events.get(tenant_id, event_id)
            if event is None:
                missing.append(str(event_id))
            else:
                existing.append(event)
        if missing:
            raise HTTPException(
                status_code=404, detail=f"events not found for this tenant: {', '.join(missing)}"
            )
        if body.is_empty:
            existing = uow.events.list_pending(tenant_id, MAX_BULK_INGEST_EVENTS)
        for event in new_events:
            uow.create_event(event)
        uow.commit()

    extractor = get_extractor()
    results = [
        _ingest_event_isolated(uow_factory, extractor, event)
        for event in [*existing, *new_events]
    ]
    failed = sum(result.status == "failed" for result in results)
    with uow_factory() as uow:
        pending = uow.events.count_pending(tenant_id)
    return BulkIngestResponse(
        tenant_id=tenant_id,
        events_created=len(new_events),
        events_ingested=len(results) - failed,
        events_failed=failed,
        events_pending=pending,
        results=results,
    )


def _ingest_outcome(uow_factory: UowFactory, outcome: Any | RejectedExtraction) -> IngestOutcome:
    """Run one extracted candidate through write policy and describe the result.

    A `RejectedExtraction` (e.g. the extractor cited no evidence) never
    reaches write policy -- it is reported as `status="rejected_before_policy"`.
    """
    if isinstance(outcome, RejectedExtraction):
        return IngestOutcome(
            status="rejected_before_policy",
            reason_codes=[outcome.reason_code],
            explanation=outcome.explanation,
        )
    decision = ingest_candidate(uow_factory, outcome)
    return IngestOutcome(
        status=decision.decision.value,
        reason_codes=decision.reason_codes,
        explanation=decision.explanation,
        accepted_memory_id=decision.accepted_memory_id,
    )


def _record_ingestion(uow_factory: UowFactory, event: MemoryEvent, candidate_count: int) -> None:
    """Log that `event` was ingested without error, so it is no longer pending."""
    with uow_factory() as uow:
        uow.events.record_ingestion(event.tenant_id, event.event_id, candidate_count)
        uow.commit()


def _ingest_event_isolated(
    uow_factory: UowFactory, extractor: CandidateExtractor, event: MemoryEvent
) -> BulkIngestEventResult:
    """Ingest one event for bulk ingest, turning any exception into a `failed` result."""
    outcomes: list[IngestOutcome] = []
    candidate_count = 0
    try:
        outcomes_raw = build_memory_candidates(extractor, [event])
        candidate_count = len(outcomes_raw)
        for outcome in outcomes_raw:
            outcomes.append(_ingest_outcome(uow_factory, outcome))
        _record_ingestion(uow_factory, event, candidate_count)
    except Exception as exc:  # isolate one event's failure from the rest of the batch
        logger.exception("bulk ingest failed for event %s", event.event_id)
        return BulkIngestEventResult(
            event_id=event.event_id,
            source_reference=event.source_reference,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            candidate_count=candidate_count,
            outcomes=outcomes,
        )
    return BulkIngestEventResult(
        event_id=event.event_id,
        source_reference=event.source_reference,
        status="ingested",
        candidate_count=candidate_count,
        outcomes=outcomes,
    )


@app.get("/tenants/{tenant_id}/memories/{memory_id}", response_model=MemoryRecord)
def get_memory(tenant_id: UUID, memory_id: UUID) -> MemoryRecord:
    """Fetch one authoritative memory record, so a curl walkthrough can see the result."""
    with get_uow_factory()() as uow:
        _require_tenant(uow, tenant_id)
        record = uow.get_memory(memory_id, tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found for this tenant")
    return record


@app.post("/tenants/{tenant_id}/retrieve", response_model=RetrieveResponse)
def retrieve(tenant_id: UUID, body: RetrieveRequest) -> RetrieveResponse:
    """Run the full retrieval pipeline and show what every stage did.

    How it works:
        1. Build a `RetrievalQuery` from the body and run
           `hybrid_search_with_report`: hard filters -> lexical + semantic
           search -> RRF fusion -> (optional) CrossEncoder reranking ->
           temporal/conflict resolution -> top `limit`.
        2. `hits` are the final answer, in order, each with the signal from
           every stage that ranked it (`scores`).
        3. `pipeline` is what happened on the way: how many candidates
           fusion produced, what reranking cost (or why it fell back), and
           which candidates temporal resolution removed and why -- with
           their content looked up so the response is readable on its own.

    The first call loads the embedding model (and the CrossEncoder, if
    `rerank` is true), so it is noticeably slower than later ones.
    """
    query = RetrievalQuery(
        tenant_id=tenant_id,
        query_text=body.query_text,
        as_of=body.as_of,
        memory_types=body.memory_types,
        min_trust=body.min_trust,
        limit=body.limit,
    )
    with get_uow_factory()() as uow:
        _require_tenant(uow, tenant_id)
        result = hybrid_search_with_report(
            uow, get_embedder(), query, reranker=get_reranker() if body.rerank else None
        )
        excluded_content = {
            exclusion.memory_id: record.content
            for exclusion in result.temporal.excluded
            if (record := uow.get_memory(exclusion.memory_id, tenant_id)) is not None
        }

    hits = [
        RetrievedMemory(
            rank=rank,
            memory_id=hit.memory.memory_id,
            content=hit.memory.content,
            memory_type=hit.memory.memory_type,
            trust_level=hit.memory.trust_level,
            status=hit.memory.status,
            valid_from=hit.memory.temporal_validity.valid_from,
            valid_to=hit.memory.temporal_validity.valid_to,
            scores=HitScores(
                lexical_rank=hit.lexical_rank,
                semantic_rank=hit.semantic_rank,
                fused_score=hit.fused_score,
                rerank_score=hit.rerank_score,
            ),
        )
        for rank, hit in enumerate(result.hits, start=1)
    ]
    rerank = (
        RerankStage(
            candidates_in=result.rerank.candidates_in,
            candidates_reranked=result.rerank.candidates_reranked,
            latency_ms=round(result.rerank.latency_ms, 2),
            fallback_reason=result.rerank.fallback_reason,
        )
        if result.rerank is not None
        else None
    )
    temporal = TemporalStage(
        effective_at=result.temporal.effective_at,
        candidates_in=result.temporal.candidates_in,
        excluded=[
            ExcludedMemory(
                memory_id=exclusion.memory_id,
                content=excluded_content.get(exclusion.memory_id),
                reason=exclusion.reason,
                related_memory_id=exclusion.related_memory_id,
            )
            for exclusion in result.temporal.excluded
        ],
        conflicts=[
            ConflictView(
                relation_id=conflict.relation_id,
                memory_ids=list(conflict.memory_ids),
                state=conflict.state,
                reason=conflict.reason,
                winner_id=conflict.winner_id,
                loser_id=conflict.loser_id,
            )
            for conflict in result.temporal.conflicts
        ],
    )
    return RetrieveResponse(
        query=body,
        hits=hits,
        pipeline=PipelineReport(
            candidates_fused=result.candidates_fused, rerank=rerank, temporal=temporal
        ),
    )


@app.post("/tenants/{tenant_id}/context", response_model=ContextResponse)
def build_context(tenant_id: UUID, body: ContextRequest) -> ContextResponse:
    """Retrieve, then pack the best non-redundant memories into `token_budget`.

    Runs the same pipeline as `/retrieve` for the top `candidates` hits,
    then `retrieval.context_packer.pack_context` over them. `context` is
    the text an agent would get; `memories` and `skipped` explain the
    choice, candidate by candidate.
    """
    query = RetrievalQuery(
        tenant_id=tenant_id,
        query_text=body.query_text,
        as_of=body.as_of,
        memory_types=body.memory_types,
        min_trust=body.min_trust,
        limit=body.candidates,
    )
    with get_uow_factory()() as uow:
        _require_tenant(uow, tenant_id)
        result = retrieve_context(
            uow,
            get_embedder(),
            query,
            token_budget=body.token_budget,
            reranker=get_reranker() if body.rerank else None,
        )

    packed = result.context
    content = {hit.memory.memory_id: hit.memory.content for hit in result.search.hits}
    return ContextResponse(
        context=packed.text,
        memories=[
            PackedMemoryView(
                rank=item.rank,
                memory_id=item.memory.memory_id,
                content=item.memory.content,
                memory_type=item.memory.memory_type,
                trust_level=item.memory.trust_level,
                subject_keys=item.memory.subject_keys,
                tokens=item.tokens,
                gain=round(item.gain, 4),
            )
            for item in packed.memories
        ],
        skipped=[
            SkippedMemoryView(
                rank=item.rank,
                memory_id=item.memory_id,
                content=content[item.memory_id],
                reason=item.reason,
                related_memory_id=item.related_memory_id,
            )
            for item in packed.skipped
        ],
        stats=ContextStats(
            token_budget=packed.token_budget,
            token_count=packed.token_count,
            candidates_in=packed.candidates_in,
            selected=len(packed.memories),
            duplicates_removed=packed.count_skipped(SkipReason.DUPLICATE),
            redundant_skipped=packed.count_skipped(SkipReason.REDUNDANT),
            over_budget=packed.count_skipped(SkipReason.OVER_BUDGET),
            selection=packed.selection,
        ),
    )


@app.post("/tenants/{tenant_id}/demo/seed", response_model=SeedDemoResponse, status_code=201)
def seed_demo(tenant_id: UUID) -> SeedDemoResponse:
    """Fill an empty tenant with a demo memory set that exercises every retrieval stage.

    Superseded facts, contradictions (resolved and unresolved), an expired
    window, and tombstoned/quarantined memories -- all embedded, so both
    searches see them. See `api/demo_data.py` for what each one is for.
    Refuses (409) if the tenant already has active memories, so re-running
    it can't create duplicates.
    """
    with get_uow_factory()() as uow:
        _require_tenant(uow, tenant_id)
        if uow.list_active_memories(tenant_id):
            raise HTTPException(
                status_code=409,
                detail="tenant already has memories; seed a new tenant instead",
            )
        seeded = seed_demo_memories(uow, tenant_id, get_embedder())
        uow.commit()
    return SeedDemoResponse(
        tenant_id=tenant_id,
        memories=[
            SeededMemoryView(
                label=memory.label,
                memory_id=memory.memory_id,
                content=memory.content,
                status=memory.status,
                trust_level=memory.trust_level,
                valid_from=memory.valid_from,
                valid_to=memory.valid_to,
            )
            for memory in seeded
        ],
    )


@app.post("/tenants/{tenant_id}/memories/index", response_model=IndexResponse)
def index_memories(tenant_id: UUID) -> IndexResponse:
    """Embed every active memory of a tenant that isn't indexed yet.

    Ingestion stores memories with `index_status=pending` and nothing
    computes their embeddings yet, so without this step semantic search
    can't see them (lexical search still can). Run it after ingesting.
    """
    embedder = get_embedder()
    with get_uow_factory()() as uow:
        _require_tenant(uow, tenant_id)
        pending = [
            record
            for record in uow.list_active_memories(tenant_id)
            if record.index_status != IndexStatus.INDEXED
        ]
        for record in pending:
            uow.vectors.set_embedding(
                tenant_id,
                record.memory_id,
                embedder.embed_texts([record.content])[0],
                model_version=embedder.model_version,
            )
        uow.commit()
    return IndexResponse(indexed=len(pending), model_version=embedder.model_version)

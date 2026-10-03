"""Host-side service adapter. Never register this object as an agent tool.

There is deliberately no generic put/set API, SQL session, or record writer
in the reasoning interface. Candidates always pass through ingest_candidate;
the persistence-time policy evaluation remains authoritative.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from apps.memory_service.domain.enums import MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import MemoryCandidate, MemoryEvent, WritePolicyDecision
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.ingestion.normalizer import RejectedExtraction
from apps.memory_service.ingestion.service import (
    UowFactory,
    ingest_candidate,
    record_rejected_extraction,
    rejected_extraction_decision,
)
from apps.memory_service.ingestion.write_policy import evaluate_write_policy
from apps.memory_service.retrieval.context_packer import (
    DEFAULT_TOKEN_BUDGET,
    PACK_CANDIDATES,
    PackedContext,
    TokenCounter,
    estimate_tokens,
    retrieve_context,
)
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import Reranker
from apps.memory_service.security.tenant_scope import TenantScope
from integrations.langgraph.memory_context import MemoryRunContext


class MemoryStoreAdapter:
    """Connect graph lifecycle nodes to existing tenant-scoped service APIs."""

    def __init__(
        self,
        uow_factory: UowFactory,
        embedder: EmbeddingModel,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        candidate_limit: int = PACK_CANDIDATES,
        min_trust: TrustLevel = TrustLevel.MEDIUM,
        count_tokens: TokenCounter = estimate_tokens,
        reranker: Reranker | None = None,
    ) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be at least 1")
        if not 1 <= candidate_limit <= 200:
            raise ValueError("candidate_limit must be between 1 and 200")
        self._uow_factory = uow_factory
        self._embedder = embedder
        self.token_budget = token_budget
        self.candidate_limit = candidate_limit
        self.min_trust = min_trust
        self.count_tokens = count_tokens
        self._reranker = reranker

    def retrieve(self, scope: MemoryRunContext, query_text: str) -> PackedContext:
        query = RetrievalQuery(
            tenant_id=scope.tenant_id,
            query_text=query_text,
            limit=self.candidate_limit,
            min_trust=self.min_trust,
        )
        with self._uow_factory() as uow:
            return retrieve_context(
                uow,
                self._embedder,
                query,
                token_budget=self.token_budget,
                count_tokens=self.count_tokens,
                reranker=self._reranker,
            ).context

    def record_tool_outcome(
        self,
        scope: MemoryRunContext,
        *,
        tool_name: str,
        content: str,
        success: bool,
    ) -> MemoryEvent:
        """Called by the host when a tool finishes, never for streamed tokens.

        Tool text keeps TOOL_OUTPUT trust even when execution succeeded. A
        success flag verifies completion, not the truth or authority of text.
        Failures can also teach useful episodes. The host supplies success.
        """
        if not tool_name.strip():
            raise ValueError("tool_name must be nonempty")
        event = MemoryEvent(
            tenant_id=scope.tenant_id,
            source_type=SourceType.TOOL_OUTPUT,
            source_reference=f"langgraph:{scope.run_id}:tool:{tool_name}",
            content=content,
            observed_at=datetime.now(UTC),
            metadata={
                "kind": "tool_outcome",
                "run_id": str(scope.run_id),
                "meaningful_outcome": True,
                "tool_name": tool_name,
                "success": success,
            },
        )
        with self._uow_factory() as uow:
            uow.create_event(event)
            uow.commit()
        return event

    def load_outcomes(
        self, scope: MemoryRunContext, event_ids: Sequence[UUID]
    ) -> tuple[MemoryEvent, ...]:
        """Load actual evidence from this tenant and run; disregard forged IDs."""
        if not event_ids:
            return ()
        with self._uow_factory() as uow:
            events = [uow.events.get(scope.tenant_id, eid) for eid in dict.fromkeys(event_ids)]
        return tuple(
            event
            for event in events
            if event is not None
            and event.tenant_id == scope.tenant_id
            and event.metadata.get("run_id") == str(scope.run_id)
            and event.metadata.get("meaningful_outcome") is True
            and event.metadata.get("kind") == "tool_outcome"
            and event.source_type is SourceType.TOOL_OUTPUT
        )

    def evaluate(
        self, scope: MemoryRunContext, candidate: MemoryCandidate | RejectedExtraction
    ) -> WritePolicyDecision:
        """A deterministic preview; it grants no permission to bypass ingestion."""
        if isinstance(candidate, RejectedExtraction):
            return rejected_extraction_decision(scope.tenant_id, candidate)
        tenant = TenantScope(scope.tenant_id)
        tenant.require(tenant.check_candidate(candidate))
        with self._uow_factory() as uow:
            events = {
                p.event_id: event
                for p in candidate.provenance
                if (event := uow.events.get(scope.tenant_id, p.event_id)) is not None
            }
            facts = uow.records.list_active(scope.tenant_id, MemoryType.SEMANTIC)
        return evaluate_write_policy(candidate, events, current_facts=facts)

    def persist(
        self, scope: MemoryRunContext, candidate: MemoryCandidate | RejectedExtraction
    ) -> WritePolicyDecision:
        """Persist all outcomes, including rejects, through the existing write gate.

        Re-evaluate inside its transaction so a preview cannot override newer
        facts or be edited into a write authorization by graph state.
        """
        if isinstance(candidate, RejectedExtraction):
            return record_rejected_extraction(self._uow_factory, scope.tenant_id, candidate)
        return ingest_candidate(self._uow_factory, candidate, tenant_id=scope.tenant_id)

    def record_feedback(
        self,
        scope: MemoryRunContext,
        retrieved_ids: Sequence[UUID],
        useful_refs: Sequence[str] | None,
    ) -> tuple[UUID, ...]:
        """Audit useful/unused/unknown separately, only for memories actually shown."""
        if not retrieved_ids:
            return ()
        refs = set(useful_refs) if useful_refs is not None else None
        ids: list[UUID] = []
        with self._uow_factory() as uow:
            for memory_id in dict.fromkeys(retrieved_ids):
                record = uow.records.get(scope.tenant_id, memory_id)
                if record is None or record.tenant_id != scope.tenant_id:
                    continue
                useful = (
                    None if refs is None else str(memory_id) in refs or str(memory_id)[:8] in refs
                )
                event = MemoryEvent(
                    tenant_id=scope.tenant_id,
                    source_type=SourceType.SYSTEM_EVENT,
                    source_reference=f"langgraph:{scope.run_id}:feedback:{memory_id}",
                    content="Retrieved memory usefulness feedback.",
                    observed_at=datetime.now(UTC),
                    metadata={
                        "kind": "memory_feedback",
                        "run_id": str(scope.run_id),
                        "memory_id": str(memory_id),
                        "useful": useful,
                        "assessment_source": "agent_report",
                    },
                )
                uow.create_event(event)
                # Feedback is audit data, never new extraction evidence.
                uow.events.record_ingestion(scope.tenant_id, event.event_id, 0)
                ids.append(event.event_id)
            uow.commit()
        return tuple(ids)

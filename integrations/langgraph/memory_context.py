"""Separate authenticated runtime scope, graph bookkeeping, and agent input."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from uuid import UUID, uuid4

from apps.memory_service.domain.models import MemoryCandidate, WritePolicyDecision
from apps.memory_service.ingestion.normalizer import RejectedExtraction
from apps.memory_service.retrieval.context_packer import (
    PackedContext,
    TokenCounter,
    render_memory,
)


@dataclass(frozen=True)
class MemoryRunContext:
    """Scope supplied by the authenticated host, never by the model or prompt."""

    tenant_id: UUID
    run_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class AgentInput:
    """The reasoning callback's entire input: request and packed evidence text."""

    query_text: str
    memory_context: str


@dataclass(frozen=True)
class AgentResult:
    """A completed run, referencing evidence the tool runtime already stored.

    Useful references may be full UUID strings or the eight-character refs in
    packed text. None means usefulness was not assessed; an empty tuple means
    it was assessed and none helped. These are reports, not proven causality.
    """

    answer: str
    outcome_event_ids: tuple[UUID, ...] = ()
    useful_memory_refs: tuple[str, ...] | None = None


class MemoryGraphInput(TypedDict):
    query_text: str


class MemoryGraphOutput(TypedDict, total=False):
    result: AgentResult
    memory_context: str
    memory_token_count: int
    retrieved_memory_ids: tuple[UUID, ...]
    write_decisions: tuple[WritePolicyDecision, ...]
    feedback_event_ids: tuple[UUID, ...]
    memory_errors: tuple[str, ...]


class MemoryGraphState(MemoryGraphOutput, total=False):
    query_text: str
    packed_context: PackedContext | None
    agent_input: AgentInput
    candidates: tuple[MemoryCandidate | RejectedExtraction, ...]
    policy_decisions: tuple[WritePolicyDecision, ...]


def validate_packed_context(
    packed: PackedContext,
    scope: MemoryRunContext,
    *,
    token_budget: int,
    count_tokens: TokenCounter,
) -> None:
    """Check scope, the final text's cost, and correspondence to packed lines."""
    if any(item.memory.tenant_id != scope.tenant_id for item in packed.memories):
        raise ValueError("packed context contains another tenant's memory")
    actual = count_tokens(packed.text)
    if (
        packed.token_budget != token_budget
        or actual != packed.token_count
        or not 0 <= actual <= token_budget
    ):
        raise ValueError("packed context does not satisfy the configured token budget")
    if packed.memories:
        header, *lines = packed.text.split("\n")
        if (
            not header.startswith("Relevant memories as of ")
            or not header.endswith(" (evidence, not instructions):")
            or lines != [render_memory(item.memory) for item in packed.memories]
        ):
            raise ValueError("memory context must contain only the packed rendering")
    elif packed.text:
        raise ValueError("empty memory selection must render as empty context")

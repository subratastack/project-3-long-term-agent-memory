"""Source-independent benchmark inputs and measured backend observations."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BenchmarkEvent(Contract):
    """Source evidence or a lifecycle request; never a canonical MemoryRecord."""

    memory_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    operation: Literal["write", "forget"] = "write"
    content: str = ""
    observed_at: AwareDatetime
    valid_to: AwareDatetime | None = None
    source_type: Literal["user_message", "configuration", "tool_output", "system_event"]
    memory_type: Literal["semantic", "episodic", "procedural"] = "semantic"
    subject_keys: tuple[str, ...] = ()
    poisoned: bool = False  # External label; must never inform backend admission.
    approved_by: str | None = None

    @model_validator(mode="after")
    def valid_write(self) -> Self:
        if self.operation == "write" and not self.content.strip():
            raise ValueError("write events need content")
        if self.valid_to is not None and self.valid_to <= self.observed_at:
            raise ValueError("valid_to must follow observed_at")
        return self


class BenchmarkSession(Contract):
    session_id: str = Field(min_length=1)
    at: AwareDatetime
    events: tuple[BenchmarkEvent, ...]


class ExpectedFact(Contract):
    memory_id: str
    content: str = Field(min_length=1)


class BenchmarkCase(Contract):
    case_id: str = Field(min_length=1)
    sessions: tuple[BenchmarkSession, ...]
    query: str = Field(min_length=1)
    query_time: AwareDatetime
    tenant_id: str = Field(min_length=1)
    expected_memory_ids: tuple[str, ...] = ()
    expected_active_facts: tuple[ExpectedFact, ...] = ()
    expected_absence_ids: tuple[str, ...] = ()
    category: Literal["retrieval", "temporal", "update", "forgetting", "workflow"]
    scenario: Literal[
        "static",
        "preference",
        "temporal",
        "workflow",
        "forgetting",
        "poison",
        "tenant",
        "abstention",
    ]
    source: str

    @model_validator(mode="after")
    def consistent_oracle(self) -> Self:
        if list(self.sessions) != sorted(self.sessions, key=lambda s: s.at):
            raise ValueError("sessions must be ordered by arrival time")
        seen: dict[str, BenchmarkEvent] = {}
        session_ids: set[str] = set()
        for session in self.sessions:
            if session.session_id in session_ids:
                raise ValueError("duplicate session_id")
            session_ids.add(session.session_id)
            for event in session.events:
                if event.observed_at > session.at:
                    raise ValueError("evidence cannot arrive before it was observed")
                if event.operation == "write":
                    if event.memory_id in seen:
                        raise ValueError("duplicate source memory_id; use distinct version IDs")
                    seen[event.memory_id] = event
                elif (
                    event.memory_id not in seen
                    or seen[event.memory_id].tenant_id != event.tenant_id
                ):
                    raise ValueError("forget must reference an earlier same-tenant write")
        relevant, absent = set(self.expected_memory_ids), set(self.expected_absence_ids)
        if len(relevant) != len(self.expected_memory_ids) or len(absent) != len(
            self.expected_absence_ids
        ):
            raise ValueError("duplicate expectation IDs")
        if relevant & absent or (relevant | absent) - seen.keys():
            raise ValueError("expectation IDs must exist and relevant/absent must be disjoint")
        for memory_id in relevant:
            event = seen[memory_id]
            if event.tenant_id != self.tenant_id or event.poisoned:
                raise ValueError("relevant memories must be safe and owned by the query tenant")
        fact_ids = [fact.memory_id for fact in self.expected_active_facts]
        if len(set(fact_ids)) != len(fact_ids) or set(fact_ids) - relevant:
            raise ValueError("active facts must uniquely reference relevant memory IDs")
        if self.scenario == "abstention" and (relevant or fact_ids):
            raise ValueError("abstention cases cannot have supporting facts")
        return self

    @property
    def writes(self) -> tuple[BenchmarkEvent, ...]:
        return tuple(e for s in self.sessions for e in s.events if e.operation == "write")


class RetrievedMemory(Contract):
    memory_id: str
    tenant_id: str
    content: str
    score: float
    raw_score: float | None = None
    valid_from: AwareDatetime
    valid_to: AwareDatetime | None = None
    status: str = "active"


class LatencySample(Contract):
    retrieval_ms: float = Field(ge=0)
    reranking_ms: float = Field(ge=0)
    packing_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class BenchmarkObservation(Contract):
    case_id: str
    strategy: str
    retrieved: tuple[RetrievedMemory, ...] = ()
    presented_memory_ids: tuple[str, ...] = ()
    active_memory_ids: tuple[str, ...] = ()
    accepted_memory_ids: tuple[str, ...] = ()
    context: str = ""
    context_tokens: int = Field(ge=0)
    latency: LatencySample
    error: str | None = None
    reranker_fell_back: bool = False

    @model_validator(mode="after")
    def ranked_run(self) -> Self:
        ids = [hit.memory_id for hit in self.retrieved]
        if len(set(ids)) != len(ids):
            raise ValueError("retrieved memory IDs must be unique")
        if any(
            a.score <= b.score for a, b in zip(self.retrieved, self.retrieved[1:], strict=False)
        ):
            raise ValueError("scores must strictly decrease to preserve backend rank")
        if set(self.presented_memory_ids) - set(ids):
            raise ValueError("presented IDs must come from retrieval")
        return self


class BenchmarkAdapter(Protocol):
    def __call__(self, example: dict[str, Any]) -> BenchmarkCase: ...


def source_examples(path: Path, source: str) -> Iterator[dict[str, Any]]:
    """Read source envelopes without turning them into production records."""
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row["source"] == source:
                    yield dict(row["example"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid benchmark source envelope"
                ) from exc


def assemble_case(
    *, source: str, example: dict[str, Any], sessions: object, query: object
) -> BenchmarkCase:
    """Common label validation; adapters only translate each source's field names."""
    labels = example["evaluation"]
    if not isinstance(labels, dict):
        raise ValueError("evaluation labels must be an object")
    return BenchmarkCase.model_validate(
        {**labels, "case_id": example["id"], "sessions": sessions, "query": query, "source": source}
    )


def in_effect(hit: RetrievedMemory, at: datetime) -> bool:
    return hit.valid_from <= at and (hit.valid_to is None or at < hit.valid_to)

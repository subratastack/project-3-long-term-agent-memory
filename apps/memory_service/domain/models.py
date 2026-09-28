"""Domain entity and data models for the Long-Term Agent Memory OS.

These are Pydantic schemas for the core concepts the memory system works
with, roughly in the order data flows through the pipeline:

0. `Tenant` -- the isolated workspace every other entity belongs to.
1. `MemoryEvent` -- something that actually happened (an event is *evidence*,
   never a memory by itself).
2. `MemoryCandidate` -- a proposed memory extracted from one or more events,
   not yet admitted anywhere.
3. `WritePolicyDecision` -- the deterministic verdict on a candidate (accept,
   reject, quarantine, or supersede), always produced before anything is
   written.
4. `MemoryRecord` -- the durable, authoritative memory that results from an
   accepted candidate.
5. `Provenance` / `TemporalValidity` / `MemoryRelation` -- supporting
   structures a record carries: where it came from, when it is valid, and how
   it relates to other memories.

Every one of these is immutable evidence or an audit-friendly snapshot: none
of them are meant to be mutated in place after creation. Where something
needs to change (a memory's status, its validity window), that happens by
writing a new record or relation, not by editing an old one.
"""

import datetime
import uuid
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apps.memory_service.domain.enums import (
    ConflictType,
    IndexStatus,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)


class Model(BaseModel):
    """Shared Pydantic configuration for every domain model in this module.

    Two rules apply everywhere:
        - Unknown fields are rejected outright (`extra="forbid"`), so a typo
          in a field name or an unexpected key from an extractor fails loudly
          instead of being silently dropped.
        - String fields are automatically trimmed of leading/trailing
          whitespace, so callers don't need to sanitize input themselves
          before constructing a model.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class Tenant(Model):
    """One isolated customer / workspace whose memories are kept separate.

    Every other entity carries a `tenant_id`; this is the registry those IDs
    are meant to come from, so a tenant can be created once (with a
    human-readable `name`) and reused across ingestion and retrieval calls.
    """

    tenant_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Identifier every memory, event, and relation of this tenant is scoped by.",
    )
    name: str = Field(
        min_length=1,
        max_length=100,
        description="Unique, human-readable tenant name (e.g. 'acme-support-bot').",
    )
    description: str | None = Field(
        default=None,
        description="Optional free-text note about what this tenant is for.",
    )
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="Timestamp when the tenant was registered.",
    )


class MemoryEvent(Model):
    """A single, immutable piece of evidence from the agent's environment.

    An event is the raw material memories are built from -- a conversation
    turn, a tool result, an agent action, a config change. It is never itself
    a memory: something only becomes a memory once it has been distilled into
    a `MemoryCandidate` and has passed through write policy.
    """

    event_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Globally unique identifier for the evidence event.",
    )
    tenant_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Tenant identifier providing multi-tenant isolation.",
    )
    source_type: SourceType = Field(
        description="Categorical origin of the event (e.g., user message, tool output).",
    )
    source_reference: str = Field(
        min_length=1,
        description="External identifier for the source (e.g., message ID, session ID).",
    )
    content: str = Field(
        min_length=1,
        description="Raw body content or payload of the event serving as evidence.",
    )
    observed_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when the event occurred or was captured.",
    )
    actor_id: UUID | None = Field(
        default=None,
        description="Optional identifier of the user or agent actor producing the event.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured context accompanying the event.",
    )


class Provenance(Model):
    """A pointer from a memory back to the event that justifies it.

    Every accepted memory carries at least one of these, and every field on
    it is meant to be independently checkable against the real event it
    claims to describe -- that is exactly what
    `apps.memory_service.ingestion.provenance.verify_provenance` does before
    a candidate is ever trusted.
    """

    event_id: UUID = Field(
        description="Identifier of the origin event that provided evidence for this memory.",
    )
    source_type: SourceType = Field(
        description="Categorical origin of the evidence event.",
    )
    source_reference: str = Field(
        description="External reference string identifying the event origin.",
    )
    observed_at: datetime.datetime = Field(
        description="Timestamp when the supporting event was originally observed.",
    )
    trust_level: TrustLevel = Field(
        description="Evaluated trust level of the supporting source event.",
    )
    excerpt: str | None = Field(
        default=None,
        description="Brief non-sensitive supporting quote or excerpt from the event.",
    )


class TemporalValidity(Model):
    """The time window during which a memory is considered true.

    This is what makes "as of" queries possible: a fact can have a
    `valid_from`/`valid_to` window in the past, be superseded today, and still
    be looked up correctly for a historical point in time -- see
    `MemoryRecordRepository.list_effective`.
    """

    valid_from: datetime.datetime = Field(
        description="Effective start timestamp when this memory becomes valid.",
    )
    valid_to: datetime.datetime | None = Field(
        default=None,
        description="Effective expiration timestamp. None denotes open-ended validity.",
    )

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "TemporalValidity":
        """Reject a window whose end is earlier than its start.

        How it works:
            1. Pydantic has already built the model, so both `valid_from` and
               `valid_to` are available as plain attributes on `self`.
            2. If `valid_to` was left as `None` (open-ended validity), there
               is nothing to compare, so the model passes as-is.
            3. Otherwise, `valid_to` must be greater than or equal to
               `valid_from`; if it is earlier, that would describe a window
               that closes before it opens, so a `ValueError` is raised and
               Pydantic turns that into a `ValidationError` for the caller.

        Example:
            Input (constructing the model):
                TemporalValidity(
                    valid_from=datetime(2026, 1, 10, tzinfo=UTC),
                    valid_to=datetime(2026, 1, 1, tzinfo=UTC),
                )
            Output:
                raises pydantic.ValidationError: "valid_to must be greater
                than or equal to valid_from."
        """
        valid_from = self.valid_from
        valid_to = self.valid_to
        if valid_to is not None and valid_to < valid_from:
            raise ValueError("valid_to must be greater than or equal to valid_from.")
        return self


class MemoryCandidate(Model):
    """A proposed memory, extracted but not yet admitted anywhere.

    This is the input to the whole ingestion pipeline
    (`apps.memory_service.ingestion`): nothing about a candidate is trusted
    until `verify_provenance` and `evaluate_write_policy` have both run
    against it. Its `provenance` list must contain at least one entry --
    a candidate with no supporting evidence cannot be constructed at all.
    """

    candidate_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the proposed memory candidate.",
    )
    tenant_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Tenant identifier for isolation.",
    )
    content: str = Field(
        min_length=1,
        description="Normalized statement or content of the proposed memory.",
    )
    provenance: list[Provenance] = Field(
        min_length=1,
        description="List of supporting evidence references backing this candidate.",
    )
    temporal_validity: TemporalValidity | None = Field(
        default=None,
        description="Temporal interval during which the proposal is asserted to hold.",
    )
    memory_type: MemoryType = Field(
        description="Cognitive category (episodic, semantic, procedural).",
    )
    subject_keys: list[str] = Field(
        default_factory=list,
        description="Entity or subject indexing keys for correlation and deduplication.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model or extraction confidence score between 0.0 and 1.0.",
    )
    proposed_trust_level: TrustLevel = Field(
        default=TrustLevel.UNTRUSTED,
        description="Initial trust tier suggested from the candidate's provenance.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional operational or extraction metadata.",
    )


class MemoryRelation(Model):
    """A directed edge between two memories, such as a supersession link.

    These edges are what let the system explain itself later: when a memory
    is superseded, this is the record that says which memory replaced which,
    and why (see `UnitOfWork.supersede_memory`).
    """

    relation_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the relationship edge.",
    )
    tenant_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Tenant identifier enforcing isolation.",
    )
    source_memory_id: UUID = Field(
        description="Identifier of the origin memory in the relation edge.",
    )
    target_memory_id: UUID = Field(
        description="Identifier of the target memory in the relation edge.",
    )
    relation_type: ConflictType = Field(
        description="Type of connection or conflict (e.g., supersession, duplicate).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supplemental attributes describing the relation.",
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when this relation link was recorded.",
    )
    rationale: str | None = Field(
        default=None,
        description="Explanation or justification detailing why the relation exists.",
    )


class MemoryRecord(Model):
    """A durable, authoritative memory -- the thing agents actually retrieve.

    This is what a `MemoryCandidate` becomes once it has been accepted (or
    quarantined) by write policy. Everything needed to trust, filter, and
    eventually retire the memory travels with it: provenance, a validity
    window, a trust level, and a lifecycle status.
    """

    memory_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the authoritative memory record.",
    )
    tenant_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Tenant identifier for multi-tenant isolation.",
    )
    content: str = Field(
        min_length=1,
        description="Consolidated and normalized factual content of the memory.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this record from 0.0 to 1.0.",
    )
    provenance: list[Provenance] = Field(
        min_length=1,
        description="Verifiable source traces supporting this memory.",
    )
    temporal_validity: TemporalValidity = Field(
        description="Time window during which this memory is valid.",
    )
    memory_type: MemoryType = Field(
        description="Cognitive memory type classification.",
    )
    subject_keys: list[str] = Field(
        default_factory=list,
        description="Key identifiers of entities and concepts referenced.",
    )
    trust_level: TrustLevel = Field(
        description="Authoritative trust classification governing context inclusion.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Operational metadata and indexing hints.",
    )
    status: MemoryStatus = Field(
        description="Current lifecycle state (active, quarantined, superseded, etc.).",
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when the record was initially created.",
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp of the most recent modification to this record.",
    )
    policy_version: str = Field(
        default="1.0",
        description="Version string of the policy engine that approved this record.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Derived semantic embedding, or None until one has been computed.",
    )
    embedding_model_version: str | None = Field(
        default=None,
        description="Identifier of the embedding model that produced `embedding`.",
    )
    index_status: IndexStatus = Field(
        default=IndexStatus.PENDING,
        description="Where this record stands with respect to its derived vector index.",
    )


class WritePolicyDecision(Model):
    """The audit record of one deterministic ingestion decision.

    Every candidate that reaches write policy produces exactly one of these,
    regardless of the outcome -- a REJECT is recorded just as durably as an
    ACCEPT, so the question "why don't I have this memory?" always has an
    answer. See `apps.memory_service.ingestion.write_policy` for how the
    `decision` and `reason_codes` are chosen.
    """

    decision_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the decision record.",
    )
    tenant_id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Tenant identifier for multi-tenant isolation.",
    )
    candidate_id: UUID = Field(
        description="Identifier of the candidate evaluated by the write policy.",
    )
    decision: WriteDecision = Field(
        description="Deterministic ingestion outcome (accept, reject, quarantine, supersede).",
    )
    decided_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when the decision was executed.",
    )
    policy_version: str = Field(
        min_length=1,
        description="Identifier or version of the rule-set executing this decision.",
    )
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Standardized code strings justifying the decision.",
    )
    explanation: str | None = Field(
        default=None,
        description="Human-readable rationale or policy rule description.",
    )
    accepted_memory_id: UUID | None = Field(
        default=None,
        description="Reference to newly created MemoryRecord if accepted or superseded.",
    )
    superseded_memory_id: UUID | None = Field(
        default=None,
        description="Reference to existing MemoryRecord marked superseded, if applicable.",
    )

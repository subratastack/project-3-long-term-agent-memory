"""SQLAlchemy persistence models for the long-term memory core tables.

These are the actual database tables behind the domain models in
`apps.memory_service.domain.models` -- one ORM class per table, covering just
the five tables needed to make writes and reads authoritative and auditable:
raw evidence (`memory_events`), accepted memories (`memory_records`),
provenance links (`memory_provenance`), memory relations
(`memory_relations`), and the write-policy audit trail
(`memory_write_decisions`). `memory_records` also carries the derived
`embedding` (pgvector) and `search_vector` (PostgreSQL full-text search)
columns retrieval searches against -- both are recomputed from the
authoritative `content`/`subject_keys` on this same row, never written
directly by application code (`search_vector` is a PostgreSQL `GENERATED
ALWAYS AS ... STORED` column; `embedding` is written once by an indexing
step via `apps.memory_service.persistence.vector_repository`).

Two design choices run through every table here and are worth understanding
up front, since they are the whole point of this schema:

1. Every foreign key is a *composite* key that includes `tenant_id`
   (e.g. `(tenant_id, memory_id)` rather than just `memory_id`). That means a
   row literally cannot reference a parent row owned by a different tenant --
   PostgreSQL itself enforces tenant isolation, not just application code.
2. Every enum column stores the lowercase string value (`"semantic"`, not
   `"SEMANTIC"`) via `values_callable=_enum_values` below, so what is on disk
   matches what Alembic's `CHECK` constraints and the domain layer expect.
"""

import datetime
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from apps.memory_service.domain.enums import (
    ConflictType,
    IndexStatus,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)

# Must match EMBEDDING_DIMENSIONS in .env.example / the embedding model in
# apps.memory_service.embeddings.sentence_transformer -- pgvector fixes a
# column's dimensionality at table-creation time, so changing embedding
# models later requires a migration, not just a config change.
EMBEDDING_DIMENSIONS = 384

# The PostgreSQL text-search configuration `search_vector` is generated with,
# and that `apps.memory_service.retrieval.lexical` queries against. Visible
# here (rather than hardcoded separately in the migration and the retrieval
# query) so the two can never drift apart.
FTS_LANGUAGE = "english"

# `to_tsvector('english', ...)` is not provably IMMUTABLE to the planner (the
# text-search-config name lookup is only STABLE), and `array_to_string` is
# STABLE too -- neither can appear directly in a GENERATED column expression.
# migrations/versions/003_*.py creates this small IMMUTABLE wrapper first;
# this name (not the raw calls) is what `search_vector`'s generation
# expression below actually calls.
SEARCH_VECTOR_FUNCTION = "memory_records_search_tsvector"

# `memory_records.search_vector`'s generation expression, shared between this
# ORM column and migrations/versions/003_*.py so the two never disagree about
# what the column actually contains.
SEARCH_VECTOR_EXPRESSION = f"{SEARCH_VECTOR_FUNCTION}(content, subject_keys)"

# The `IMMUTABLE` wrapper itself. Both migrations/versions/003_*.py and
# apps/tests/integration/conftest.py's `Base.metadata.create_all` path (which
# bypasses Alembic) must run this before `search_vector` can be created --
# each keeps its own copy of this exact DDL, matching how EMBEDDING_DIMENSIONS
# is duplicated with a "must match" comment in migrations/versions/002_*.py
# rather than importing application code into a migration.
#
# Two things had to be true for `ALTER TABLE ... ADD COLUMN ... GENERATED` to
# actually accept this (verified directly against Postgres, not just assumed):
#   1. LANGUAGE plpgsql, not sql -- a single-statement SQL-language function
#      can be *inlined* by the planner, which then sees straight through to
#      `to_tsvector`'s real (merely STABLE) volatility and rejects the
#      GENERATED column with "generation expression is not immutable"; a
#      plpgsql function body is opaque to that inlining.
#   2. `array_to_string` must be called *inside* this function, not in the
#      generated column's own expression -- it is STABLE, not IMMUTABLE, so
#      calling it directly in `search_vector`'s expression fails the same
#      check even with (1) fixed. Taking `subject_keys` as an argument here
#      and joining it internally is what hides that call as well.
SEARCH_VECTOR_FUNCTION_DDL = f"""
CREATE OR REPLACE FUNCTION {SEARCH_VECTOR_FUNCTION}(content_text text, subject_keys text[])
RETURNS tsvector
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
BEGIN
    RETURN to_tsvector('{FTS_LANGUAGE}', coalesce(content_text, '')) ||
           to_tsvector('{FTS_LANGUAGE}', coalesce(array_to_string(subject_keys, ' '), ''));
END;
$$;
"""


class Base(DeclarativeBase):
    """Declarative base shared by every table defined in this module."""


def _enum_values(enum_type: type[Any]) -> list[str]:
    """Tell SQLAlchemy to persist a `StrEnum`'s *values*, not its member names.

    How it works:
        1. SQLAlchemy's `Enum` column type normally stores a Python enum
           member using its attribute name (e.g. `SEMANTIC` for
           `MemoryType.SEMANTIC`), which is not what the domain layer or the
           Alembic migration's `CHECK` constraints expect.
        2. Every enum column below passes this function in as
           `values_callable`, which SQLAlchemy calls once per column with the
           enum class itself.
        3. Because these enums are `StrEnum`, iterating over `enum_type`
           yields each member, and `member.value` is the lowercase string
           (`"semantic"`) rather than the name (`"SEMANTIC"`) -- so the list
           returned here is exactly what gets written to and read from the
           database column.

    Example:
        Input:
            _enum_values(MemoryType)
        Output:
            ["episodic", "semantic", "procedural"]
    """
    return [member.value for member in enum_type]


class TenantRow(Base):
    """The `tenants` table: the registry of tenants and their names.

    Backs `apps.memory_service.domain.models.Tenant`. The other tables carry
    `tenant_id` without a foreign key to this one (yet): isolation is still
    enforced by those tables' own composite `(tenant_id, ...)` keys, and the
    dev API refuses tenant-scoped calls for a `tenant_id` not registered here.
    """

    __tablename__ = "tenants"
    __table_args__ = (UniqueConstraint("name", name="uq_tenants_name"),)

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class MemoryEventRow(Base):
    """The `memory_events` table: immutable raw evidence.

    Backs `apps.memory_service.domain.models.MemoryEvent`. Nothing in this
    table is ever updated after insert -- it is the permanent factual record
    that provenance and write-policy decisions point back to.
    """

    __tablename__ = "memory_events"
    __table_args__ = (
        UniqueConstraint("tenant_id", "event_id", name="uq_memory_events_tenant_event"),
        Index("ix_memory_events_tenant_id", "tenant_id"),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        SAEnum(
            SourceType,
            name="ck_memory_events_source_type",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    source_reference: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )


class MemoryRecordRow(Base):
    """The `memory_records` table: authoritative, accepted long-term memory.

    Backs `apps.memory_service.domain.models.MemoryRecord`. This is what
    agents actually retrieve from; the `(tenant_id, memory_id)` unique
    constraint below exists specifically so that other tables (provenance,
    relations, write decisions) can point back at a row here through a
    composite foreign key that also pins down the tenant.
    """

    __tablename__ = "memory_records"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="ck_memory_records_confidence_range"
        ),
        CheckConstraint(
            "valid_to IS NULL OR valid_to >= valid_from", name="ck_memory_records_valid_range"
        ),
        UniqueConstraint("tenant_id", "memory_id", name="uq_memory_records_tenant_memory"),
        Index("ix_memory_records_tenant_id", "tenant_id"),
        Index("ix_memory_records_tenant_status", "tenant_id", "status"),
        Index("ix_memory_records_tenant_index_status", "tenant_id", "index_status"),
        Index("ix_memory_records_search_vector", "search_vector", postgresql_using="gin"),
    )

    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    valid_from: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    memory_type: Mapped[MemoryType] = mapped_column(
        SAEnum(
            MemoryType,
            name="ck_memory_records_memory_type",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    subject_keys: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default=text("'{}'::text[]")
    )
    trust_level: Mapped[TrustLevel] = mapped_column(
        SAEnum(
            TrustLevel,
            name="ck_memory_records_trust_level",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    status: Mapped[MemoryStatus] = mapped_column(
        SAEnum(
            MemoryStatus,
            name="ck_memory_records_status",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0")
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    embedding_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    index_status: Mapped[IndexStatus] = mapped_column(
        SAEnum(
            IndexStatus,
            name="ck_memory_records_index_status",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=IndexStatus.PENDING,
        server_default=text("'pending'"),
    )
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR, Computed(SEARCH_VECTOR_EXPRESSION, persisted=True), nullable=True
    )


class MemoryProvenanceRow(Base):
    """The `memory_provenance` table: links a memory to one supporting event.

    Backs `apps.memory_service.domain.models.Provenance`. `memory_id` and
    `event_id` are plain UUID columns with no single-column foreign key --
    the two `ForeignKeyConstraint`s in `__table_args__` are what actually
    enforce the reference, and they require *both* `tenant_id` and the id to
    match a parent row. That is what makes it impossible, at the database
    level, for a provenance row to point at a memory or event owned by a
    different tenant.
    """

    __tablename__ = "memory_provenance"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_provenance_tenant_memory",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "event_id"],
            ["memory_events.tenant_id", "memory_events.event_id"],
            name="fk_memory_provenance_tenant_event",
            ondelete="RESTRICT",
        ),
        Index("ix_memory_provenance_tenant_id", "tenant_id"),
        Index("ix_memory_provenance_memory_id", "memory_id"),
        Index("ix_memory_provenance_event_id", "event_id"),
    )

    provenance_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        SAEnum(
            SourceType,
            name="ck_memory_provenance_source_type",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    source_reference: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trust_level: Mapped[TrustLevel] = mapped_column(
        SAEnum(
            TrustLevel,
            name="ck_memory_provenance_trust_level",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class MemoryRelationRow(Base):
    """The `memory_relations` table: an edge between two memories.

    Backs `apps.memory_service.domain.models.MemoryRelation`. Like
    `MemoryProvenanceRow`, both `source_memory_id` and `target_memory_id` are
    enforced by composite `(tenant_id, memory_id)` foreign keys, so a
    supersession or contradiction edge can never cross a tenant boundary.
    """

    __tablename__ = "memory_relations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "source_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_relations_tenant_source",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "target_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_relations_tenant_target",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "source_memory_id <> target_memory_id", name="ck_memory_relations_no_self_link"
        ),
        Index("ix_memory_relations_tenant_id", "tenant_id"),
        Index("ix_memory_relations_source_memory_id", "source_memory_id"),
        Index("ix_memory_relations_target_memory_id", "target_memory_id"),
    )

    relation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    relation_type: Mapped[ConflictType] = mapped_column(
        SAEnum(
            ConflictType,
            name="ck_memory_relations_relation_type",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class MemoryWriteDecisionRow(Base):
    """The `memory_write_decisions` table: the ingestion audit trail.

    Backs `apps.memory_service.domain.models.WritePolicyDecision`. Every
    candidate evaluated by
    `apps.memory_service.ingestion.write_policy.evaluate_write_policy`
    produces exactly one row here, whatever the outcome -- including REJECT,
    where `accepted_memory_id` stays `NULL` because no memory was written.
    """

    __tablename__ = "memory_write_decisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "accepted_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_write_decisions_tenant_accepted",
            ondelete="SET NULL",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "superseded_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_write_decisions_tenant_superseded",
            ondelete="SET NULL",
        ),
        Index("ix_memory_write_decisions_tenant_id", "tenant_id"),
        Index("ix_memory_write_decisions_candidate_id", "candidate_id"),
    )

    decision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    candidate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    decision: Mapped[WriteDecision] = mapped_column(
        SAEnum(
            WriteDecision,
            name="ck_memory_write_decisions_decision",
            native_enum=False,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    decided_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_codes: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default=text("'{}'::text[]")
    )
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    accepted_memory_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    superseded_memory_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )


class MemoryEventIngestionRow(Base):
    """The `memory_event_ingestions` table: an append-only log of ingestion runs.

    One row each time an event has been run through extraction and write
    policy without error. `memory_events` itself stays immutable; this log is
    what lets `POST /tenants/{tenant_id}/events/ingest` with no payload find
    the events that still need ingesting. Re-ingesting an event on purpose
    appends another row rather than updating one.
    """

    __tablename__ = "memory_event_ingestions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "event_id"],
            ["memory_events.tenant_id", "memory_events.event_id"],
            name="fk_memory_event_ingestions_tenant_event",
            ondelete="RESTRICT",
        ),
        Index("ix_memory_event_ingestions_tenant_event", "tenant_id", "event_id"),
    )

    ingestion_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

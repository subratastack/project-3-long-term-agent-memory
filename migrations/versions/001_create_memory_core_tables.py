"""create memory core tables

Revision ID: 001
Revises:
Create Date: 2026-09-25

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOW = sa.text("now()")
_SOURCE_TYPE_CHECK = (
    "source_type IN "
    "('user_message','tool_output','system_event','agent_action','configuration')"
)
_TRUST_LEVEL_CHECK = "trust_level IN ('untrusted','system','high','medium','low')"


def upgrade() -> None:
    op.create_table(
        "memory_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_reference", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.CheckConstraint(_SOURCE_TYPE_CHECK, name="ck_memory_events_source_type"),
        sa.UniqueConstraint("tenant_id", "event_id", name="uq_memory_events_tenant_event"),
    )
    op.create_index("ix_memory_events_tenant_id", "memory_events", ["tenant_id"])

    op.create_table(
        "memory_records",
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column(
            "subject_keys",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("trust_level", sa.String(length=32), nullable=False),
        sa.Column(
            "metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_NOW),
        sa.Column("policy_version", sa.String(length=32), nullable=False, server_default="1.0"),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="ck_memory_records_confidence_range"
        ),
        sa.CheckConstraint(
            "valid_to IS NULL OR valid_to >= valid_from", name="ck_memory_records_valid_range"
        ),
        sa.CheckConstraint(
            "memory_type IN ('episodic','semantic','procedural')",
            name="ck_memory_records_memory_type",
        ),
        sa.CheckConstraint(_TRUST_LEVEL_CHECK, name="ck_memory_records_trust_level"),
        sa.CheckConstraint(
            "status IN ('active','quarantined','superseded','expired','tombstone')",
            name="ck_memory_records_status",
        ),
        sa.UniqueConstraint("tenant_id", "memory_id", name="uq_memory_records_tenant_memory"),
    )
    op.create_index("ix_memory_records_tenant_id", "memory_records", ["tenant_id"])
    op.create_index("ix_memory_records_tenant_status", "memory_records", ["tenant_id", "status"])

    # `memory_id`/`event_id` below are NOT single-column foreign keys: the
    # composite ForeignKeyConstraints (tenant_id, memory_id) / (tenant_id,
    # event_id) below are what enforces the reference, so that a row can only
    # ever point at a parent memory/event owned by the *same* tenant_id. A
    # cross-tenant reference is rejected by PostgreSQL itself, not just by
    # application code.
    op.create_table(
        "memory_provenance",
        sa.Column("provenance_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_reference", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trust_level", sa.String(length=32), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_NOW),
        sa.CheckConstraint(_SOURCE_TYPE_CHECK, name="ck_memory_provenance_source_type"),
        sa.CheckConstraint(_TRUST_LEVEL_CHECK, name="ck_memory_provenance_trust_level"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_provenance_tenant_memory",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "event_id"],
            ["memory_events.tenant_id", "memory_events.event_id"],
            name="fk_memory_provenance_tenant_event",
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_memory_provenance_tenant_id", "memory_provenance", ["tenant_id"])
    op.create_index("ix_memory_provenance_memory_id", "memory_provenance", ["memory_id"])
    op.create_index("ix_memory_provenance_event_id", "memory_provenance", ["event_id"])

    op.create_table(
        "memory_relations",
        sa.Column("relation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column(
            "metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_NOW),
        sa.CheckConstraint(
            "relation_type IN ('contradiction','duplicate','supersession')",
            name="ck_memory_relations_relation_type",
        ),
        sa.CheckConstraint(
            "source_memory_id <> target_memory_id", name="ck_memory_relations_no_self_link"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "source_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_relations_tenant_source",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "target_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_relations_tenant_target",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_memory_relations_tenant_id", "memory_relations", ["tenant_id"])
    op.create_index(
        "ix_memory_relations_source_memory_id", "memory_relations", ["source_memory_id"]
    )
    op.create_index(
        "ix_memory_relations_target_memory_id", "memory_relations", ["target_memory_id"]
    )

    op.create_table(
        "memory_write_decisions",
        sa.Column("decision_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False, server_default=_NOW),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column(
            "reason_codes",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("accepted_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("superseded_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.CheckConstraint(
            "decision IN ('accept','reject','quarantine','supersede')",
            name="ck_memory_write_decisions_decision",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "accepted_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_write_decisions_tenant_accepted",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "superseded_memory_id"],
            ["memory_records.tenant_id", "memory_records.memory_id"],
            name="fk_memory_write_decisions_tenant_superseded",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_memory_write_decisions_tenant_id", "memory_write_decisions", ["tenant_id"])
    op.create_index(
        "ix_memory_write_decisions_candidate_id", "memory_write_decisions", ["candidate_id"]
    )


def downgrade() -> None:
    op.drop_table("memory_write_decisions")
    op.drop_table("memory_relations")
    op.drop_table("memory_provenance")
    op.drop_table("memory_records")
    op.drop_table("memory_events")

"""create memory_event_ingestions table

Revision ID: 005
Revises: 004
Create Date: 2026-09-27

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No backfill: events ingested before this table existed are recognised
    # by their provenance rows instead (see MemoryEventRepository.list_pending).
    op.create_table(
        "memory_event_ingestions",
        sa.Column("ingestion_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("event_id", UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "event_id"],
            ["memory_events.tenant_id", "memory_events.event_id"],
            name="fk_memory_event_ingestions_tenant_event",
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_memory_event_ingestions_tenant_event",
        "memory_event_ingestions",
        ["tenant_id", "event_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_event_ingestions_tenant_event", table_name="memory_event_ingestions")
    op.drop_table("memory_event_ingestions")

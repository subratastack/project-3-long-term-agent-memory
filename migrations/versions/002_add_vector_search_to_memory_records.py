"""add vector search columns to memory_records

Revision ID: 002
Revises: 001
Create Date: 2026-09-26

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match apps.memory_service.persistence.models.EMBEDDING_DIMENSIONS.
_EMBEDDING_DIMENSIONS = 384
_INDEX_STATUS_CHECK = "index_status IN ('pending','indexed','failed')"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "memory_records",
        sa.Column("embedding", Vector(_EMBEDDING_DIMENSIONS), nullable=True),
    )
    op.add_column(
        "memory_records",
        sa.Column("embedding_model_version", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "memory_records",
        sa.Column(
            "index_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
    )
    op.create_check_constraint(
        "ck_memory_records_index_status", "memory_records", _INDEX_STATUS_CHECK
    )
    op.create_index(
        "ix_memory_records_tenant_index_status",
        "memory_records",
        ["tenant_id", "index_status"],
    )

    # Deliberately no HNSW/IVFFlat index yet -- see ADR-003: exact pgvector
    # search is measured against a labeled dataset first, and an ANN index is
    # only introduced once that measurement justifies the approximation.


def downgrade() -> None:
    op.drop_index("ix_memory_records_tenant_index_status", table_name="memory_records")
    op.drop_constraint("ck_memory_records_index_status", "memory_records", type_="check")
    op.drop_column("memory_records", "index_status")
    op.drop_column("memory_records", "embedding_model_version")
    op.drop_column("memory_records", "embedding")
    # The `vector` extension is intentionally left installed: other objects
    # (or a re-upgrade) may still depend on it, and dropping an extension is
    # a cluster-wide action this migration should not take unilaterally.

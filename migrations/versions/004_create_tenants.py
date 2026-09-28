"""create tenants table

Revision ID: 004
Revises: 003
Create Date: 2026-09-27

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A registry only: existing tables keep their tenant_id columns without a
    # foreign key to this one, so rows written before tenants existed stay valid.
    op.create_table(
        "tenants",
        sa.Column("tenant_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("name", name="uq_tenants_name"),
    )


def downgrade() -> None:
    op.drop_table("tenants")

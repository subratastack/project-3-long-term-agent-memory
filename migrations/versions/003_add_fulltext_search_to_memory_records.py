"""add full-text search column to memory_records

Revision ID: 003
Revises: 002
Create Date: 2026-09-26

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match apps.memory_service.persistence.models.SEARCH_VECTOR_FUNCTION /
# SEARCH_VECTOR_EXPRESSION / SEARCH_VECTOR_FUNCTION_DDL / FTS_LANGUAGE.
_SEARCH_VECTOR_FUNCTION = "memory_records_search_tsvector"
_SEARCH_VECTOR_FUNCTION_DDL = f"""
CREATE OR REPLACE FUNCTION {_SEARCH_VECTOR_FUNCTION}(content_text text, subject_keys text[])
RETURNS tsvector
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
BEGIN
    RETURN to_tsvector('english', coalesce(content_text, '')) ||
           to_tsvector('english', coalesce(array_to_string(subject_keys, ' '), ''));
END;
$$;
"""
_SEARCH_VECTOR_EXPRESSION = f"{_SEARCH_VECTOR_FUNCTION}(content, subject_keys)"


def upgrade() -> None:
    # to_tsvector('english', ...) and array_to_string() are only STABLE, not
    # IMMUTABLE, to the planner, so neither can appear directly in a
    # GENERATED column expression -- this wrapper function hides both calls
    # behind one declared-IMMUTABLE call. LANGUAGE plpgsql (not sql) is
    # deliberate: a single-statement SQL function can be inlined by the
    # planner, exposing the real volatility of what it calls again.
    op.execute(_SEARCH_VECTOR_FUNCTION_DDL)

    op.add_column(
        "memory_records",
        sa.Column(
            "search_vector",
            TSVECTOR(),
            sa.Computed(_SEARCH_VECTOR_EXPRESSION, persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_memory_records_search_vector",
        "memory_records",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_memory_records_search_vector", table_name="memory_records")
    op.drop_column("memory_records", "search_vector")
    op.execute(f"DROP FUNCTION IF EXISTS {_SEARCH_VECTOR_FUNCTION}(text, text[])")

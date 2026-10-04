"""Transactional outbox for TurboVec, separate from pgvector's embedding status."""

from __future__ import annotations

import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Identity,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, mapped_column

from apps.memory_service.persistence.models import Base


class VectorIndexState(Base):
    __tablename__ = "memory_vector_index_state"
    __table_args__ = (
        UniqueConstraint("tenant_id", "memory_id", name="uq_vector_state_tenant_memory"),
        CheckConstraint("state IN ('pending', 'failed', 'indexed', 'absent')"),
    )

    vector_id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    memory_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    revision: Mapped[int] = mapped_column(BigInteger, server_default=text("1"))
    indexed_revision: Mapped[int] = mapped_column(BigInteger, server_default=text("0"))
    state: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


# No FK: a hard-deleted record must leave a durable removal job behind.
# Qualify the target by the source table's schema, including in isolated benchmarks.
OUTBOX_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION memory_vector_enqueue() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE t uuid; m uuid;
BEGIN
    IF TG_OP = 'DELETE' THEN t := OLD.tenant_id; m := OLD.memory_id;
    ELSE t := NEW.tenant_id; m := NEW.memory_id; END IF;
    EXECUTE format('INSERT INTO %I.memory_vector_index_state
        (tenant_id, memory_id) VALUES ($1, $2)
        ON CONFLICT (tenant_id, memory_id) DO UPDATE SET
        revision = memory_vector_index_state.revision + 1,
        state = ''pending'', attempts = 0, last_error = NULL, updated_at = now()',
        TG_TABLE_SCHEMA) USING t, m;
    RETURN NULL;
END;
$$;
"""

OUTBOX_TRIGGER_DDL = """
CREATE TRIGGER memory_vector_enqueue_trigger
AFTER INSERT OR DELETE OR UPDATE OF content, embedding, embedding_model_version,
    index_status, status, trust_level, memory_type, valid_from, valid_to
ON memory_records FOR EACH ROW EXECUTE FUNCTION memory_vector_enqueue()
"""

BACKFILL_DDL = """
INSERT INTO memory_vector_index_state (tenant_id, memory_id)
SELECT tenant_id, memory_id FROM memory_records
ON CONFLICT (tenant_id, memory_id) DO NOTHING
"""


def install_outbox(connection: Connection) -> None:
    """Install in the current schema within the caller's transaction, idempotently."""
    Base.metadata.tables["memory_vector_index_state"].create(connection, checkfirst=True)
    connection.execute(text(OUTBOX_FUNCTION_DDL))
    exists = connection.execute(
        text("""
        SELECT 1 FROM pg_trigger WHERE tgrelid = 'memory_records'::regclass
        AND tgname = 'memory_vector_enqueue_trigger'
    """)
    ).scalar()
    if not exists:
        connection.execute(text(OUTBOX_TRIGGER_DDL))
    connection.execute(text(BACKFILL_DDL))

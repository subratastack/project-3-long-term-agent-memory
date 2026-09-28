"""Exact pgvector similarity search, and writing derived embeddings.

This is a second, narrower repository over `memory_records` -- distinct from
`MemoryRecordRepository` in `repositories.py` -- because the two have very
different jobs: `MemoryRecordRepository` reads and writes the authoritative
record, while `VectorRecordRepository` only ever touches the *derived*
`embedding`/`embedding_model_version`/`index_status` columns, and its one
read method (`find_nearest`) deliberately returns bare `(memory_id, distance)`
pairs rather than full `MemoryRecord`s. `apps.memory_service.retrieval.semantic`
is what turns those ids back into authoritative records via
`MemoryRecordRepository.get` -- a deliberate revalidation step (ADR-004:
"Revalidate each selected ID against PostgreSQL before context construction")
rather than trusting whatever this query already had loaded.

No ANN index is created or assumed here: `find_nearest` runs an exact
(sequential-scan) pgvector distance computation. See ADR-003: "Approximate
nearest-neighbor indexes are introduced only after exact-search latency and
recall have been measured at representative scale."
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from apps.memory_service.domain.enums import IndexStatus
from apps.memory_service.persistence.models import MemoryRecordRow
from apps.memory_service.retrieval.filters import RetrievalFilters


@dataclass(frozen=True)
class VectorMatch:
    """One nearest-neighbor hit: a memory id and its cosine distance to the query.

    Lower `distance` means more similar (0.0 is an exact match). This is
    intentionally *not* a `MemoryRecord` -- see the module docstring for why
    callers must re-fetch and revalidate before trusting a match.
    """

    memory_id: UUID
    distance: float


class VectorRecordRepository:
    """Reads and writes the derived embedding columns on `memory_records`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def set_embedding(
        self,
        tenant_id: UUID,
        memory_id: UUID,
        embedding: Sequence[float],
        *,
        model_version: str,
    ) -> None:
        """Store a computed embedding and mark the record INDEXED.

        How it works:
            Issues an `UPDATE` scoped to both `tenant_id` and `memory_id`,
            setting `embedding`, `embedding_model_version`, and
            `index_status=INDEXED` together, so a record is never left with
            an embedding but a stale `index_status`, or vice versa. If no row
            matched (wrong tenant, or the memory does not exist), raises
            `LookupError` rather than silently doing nothing.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
                embedding = [0.01, -0.02, ...]  # 384 floats
                model_version = "sentence-transformers/all-MiniLM-L6-v2"
            Output:
                None (the row's embedding columns are now set and
                `index_status` is `"indexed"`)
                -- or raises LookupError if no such memory exists for that tenant.
        """
        stmt = (
            update(MemoryRecordRow)
            .where(MemoryRecordRow.tenant_id == tenant_id, MemoryRecordRow.memory_id == memory_id)
            .values(
                embedding=list(embedding),
                embedding_model_version=model_version,
                index_status=IndexStatus.INDEXED,
            )
        )
        result = cast(CursorResult[None], self._session.execute(stmt))
        if result.rowcount == 0:
            raise LookupError(f"memory_record {memory_id} not found for tenant {tenant_id}")

    def mark_index_failed(self, tenant_id: UUID, memory_id: UUID) -> None:
        """Mark a record's embedding attempt as FAILED, leaving content untouched.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
            Output:
                None (the row's `index_status` is now `"failed"`)
                -- or raises LookupError if no such memory exists for that tenant.
        """
        stmt = (
            update(MemoryRecordRow)
            .where(MemoryRecordRow.tenant_id == tenant_id, MemoryRecordRow.memory_id == memory_id)
            .values(index_status=IndexStatus.FAILED)
        )
        result = cast(CursorResult[None], self._session.execute(stmt))
        if result.rowcount == 0:
            raise LookupError(f"memory_record {memory_id} not found for tenant {tenant_id}")

    def find_nearest(
        self,
        filters: RetrievalFilters,
        query_embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorMatch]:
        """Run an exact pgvector cosine-distance search under `filters`.

        How it works:
            1. Only rows with `index_status=INDEXED` (and therefore a
               non-null `embedding`) are ever candidates -- a `PENDING` or
               `FAILED` record is excluded here, not filtered out later.
            2. `filters.tenant_id`, `filters.allowed_statuses`,
               `filters.allowed_trust_levels`, and `filters.memory_types` (if
               not `None`) are all applied as `WHERE` predicates alongside
               the vector search, and `filters.effective_at` must fall
               within the record's `[valid_from, valid_to)` window -- every
               one of ADR-003's "hard constraints" is enforced in the same
               query as the similarity search, not after it.
            3. Rows are ordered by pgvector's `<=>` cosine-distance operator
               (via `.cosine_distance`) ascending -- closest first -- and
               capped at `limit`.
            4. Only `memory_id` and the computed `distance` are selected;
               callers must re-fetch the full record (see the module
               docstring).

        Example:
            Input:
                filters = RetrievalFilters(tenant_id=UUID("1111...1111"),
                                            allowed_statuses=frozenset({MemoryStatus.ACTIVE}),
                                            allowed_trust_levels=frozenset(TrustLevel),
                                            memory_types=None, effective_at=<now>)
                query_embedding = [0.01, -0.02, ...]
                limit = 5
            Output:
                [VectorMatch(memory_id=UUID("bbbbbbbb-..."), distance=0.12), ...]
                -- ordered nearest-first, at most 5 entries.
        """
        distance = MemoryRecordRow.embedding.cosine_distance(list(query_embedding))
        stmt = (
            select(MemoryRecordRow.memory_id, distance.label("distance"))
            .where(
                MemoryRecordRow.tenant_id == filters.tenant_id,
                MemoryRecordRow.index_status == IndexStatus.INDEXED,
                MemoryRecordRow.embedding.is_not(None),
                MemoryRecordRow.status.in_(filters.allowed_statuses),
                MemoryRecordRow.trust_level.in_(filters.allowed_trust_levels),
                MemoryRecordRow.valid_from <= filters.effective_at,
                or_(
                    MemoryRecordRow.valid_to.is_(None),
                    MemoryRecordRow.valid_to >= filters.effective_at,
                ),
            )
            .order_by(distance)
            .limit(limit)
        )
        if filters.memory_types is not None:
            stmt = stmt.where(MemoryRecordRow.memory_type.in_(filters.memory_types))

        rows = self._session.execute(stmt).all()
        return [VectorMatch(memory_id=row.memory_id, distance=row.distance) for row in rows]

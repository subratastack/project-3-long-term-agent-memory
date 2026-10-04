"""Opt-in pure cosine vector backends for the Phase 6B comparison.

The existing priority-adjusted VectorRecordRepository remains the default.
These adapters isolate vector search quality from lifecycle ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol
from uuid import UUID

import numpy as np
from sqlalchemy import Select, or_, select, text
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from apps.memory_service.domain.enums import IndexStatus
from apps.memory_service.persistence.models import EMBEDDING_DIMENSIONS, MemoryRecordRow
from apps.memory_service.persistence.vector_repository import VectorMatch
from apps.memory_service.retrieval.filters import RetrievalFilters


def normalized_vector(values: Sequence[float]) -> list[float]:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(vector).all():
        raise ValueError(f"expected {EMBEDDING_DIMENSIONS} finite coordinates")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError("vector must have a finite nonzero norm")
    return [float(v) for v in vector / norm]


def eligibility(filters: RetrievalFilters) -> list[ColumnElement[bool]]:
    """Use the service's inclusive valid_to convention in every SQL backend."""
    row = MemoryRecordRow
    predicates: list[ColumnElement[bool]] = [
        row.tenant_id == filters.tenant_id,
        row.index_status == IndexStatus.INDEXED,
        row.embedding.is_not(None),
        row.status.in_(filters.allowed_statuses),
        row.trust_level.in_(filters.allowed_trust_levels),
        row.valid_from <= filters.effective_at,
        or_(row.valid_to.is_(None), row.valid_to >= filters.effective_at),
    ]
    if filters.memory_types is not None:
        predicates.append(row.memory_type.in_(filters.memory_types))
    return predicates


class VectorIndex(Protocol):
    def find_nearest(
        self,
        session: Session,
        filters: RetrievalFilters,
        query_embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorMatch]: ...


class PgvectorIndex:
    def __init__(self, mode: Literal["exact", "hnsw"] = "exact", *, ef_search: int = 100) -> None:
        if mode not in ("exact", "hnsw") or ef_search < 10:
            raise ValueError("mode must be exact/hnsw; ef_search must be >= 10")
        self.mode = mode
        self.ef_search = ef_search

    def statement(
        self, filters: RetrievalFilters, vector: Sequence[float], limit: int
    ) -> Select[tuple[UUID, float]]:
        distance = MemoryRecordRow.embedding.cosine_distance(normalized_vector(vector))
        return (
            select(MemoryRecordRow.memory_id, distance.label("distance"))
            .where(*eligibility(filters))
            .order_by(distance)
            .limit(limit)
        )

    def configure(self, session: Session) -> None:
        if self.mode == "exact":
            session.execute(text("SET LOCAL enable_indexscan = off"))
            session.execute(text("SET LOCAL enable_bitmapscan = off"))
        else:
            session.execute(text("SET LOCAL enable_indexscan = on"))
            session.execute(text("SET LOCAL enable_bitmapscan = on"))
            session.execute(text("SET LOCAL enable_seqscan = off"))
            session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
            session.execute(text(f"SET LOCAL hnsw.ef_search = {self.ef_search}"))

    def find_nearest(
        self,
        session: Session,
        filters: RetrievalFilters,
        query_embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorMatch]:
        if limit < 1:
            raise ValueError("limit must be positive")
        previous = session.execute(
            text(
                "SELECT current_setting('enable_indexscan'), "
                "current_setting('enable_bitmapscan'), current_setting('enable_seqscan')"
            )
        ).one()
        try:
            self.configure(session)
            rows = session.execute(self.statement(filters, query_embedding, limit))
            return [VectorMatch(memory_id=mid, distance=float(distance)) for mid, distance in rows]
        finally:
            # Exact-search planner switches must not turn authoritative UUID fetches into scans.
            session.execute(
                text(
                    "SELECT set_config('enable_indexscan', :a, true), "
                    "set_config('enable_bitmapscan', :b, true), "
                    "set_config('enable_seqscan', :c, true)"
                ),
                dict(zip(("a", "b", "c"), previous, strict=True)),
            )

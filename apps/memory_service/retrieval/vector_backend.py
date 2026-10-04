"""Opt-in backend retrieval: candidate IDs -> authoritative records -> resolve -> pack."""

from __future__ import annotations

from apps.memory_service.persistence.pgvector_index import VectorIndex
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.context_packer import PackedContext, pack_context
from apps.memory_service.retrieval.filters import record_matches_filters, resolve_filters
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.semantic import SemanticSearchHit
from apps.memory_service.retrieval.temporal import apply_temporal_resolution


def vector_backend_search(
    uow: UnitOfWork,
    index: VectorIndex,
    query: RetrievalQuery,
) -> list[SemanticSearchHit]:
    """Require a precomputed vector so all benchmark backends receive the same query."""
    if query.query_embedding is None:
        raise ValueError("vector backend retrieval requires query_embedding")
    filters = resolve_filters(query)
    matches = index.find_nearest(uow.session, filters, query.query_embedding, limit=query.limit)
    hits = []
    seen = set()
    for match in matches:
        if match.memory_id in seen:
            continue
        seen.add(match.memory_id)
        record = uow.records.get(query.tenant_id, match.memory_id)
        if record is not None and record_matches_filters(record, filters):
            hits.append(SemanticSearchHit(record, match.distance))
    return hits


def vector_backend_context(
    uow: UnitOfWork,
    index: VectorIndex,
    query: RetrievalQuery,
    *,
    token_budget: int = 500,
) -> PackedContext:
    # Preserve current vs historical semantics while applying the existing resolver.
    filters = resolve_filters(query)
    hits = vector_backend_search(uow, index, query)
    resolved, report = apply_temporal_resolution(uow, query, hits)
    from dataclasses import replace

    filters = replace(filters, effective_at=report.effective_at)
    records = [
        record
        for hit in resolved
        if (record := uow.records.get(query.tenant_id, hit.memory.memory_id)) is not None
        and record_matches_filters(record, filters)
    ]
    return pack_context(records, filters, token_budget=token_budget)

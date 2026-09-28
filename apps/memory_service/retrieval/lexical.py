"""Exact lexical retrieval: PostgreSQL full-text search over content + subject keys.

```text
query -> tenant/trust/status/time filters -> PostgreSQL full-text search
       -> fetch authoritative memory records -> return bounded candidates
```

Mirrors `retrieval.semantic`'s shape deliberately: hard filters first, then
the search itself, then re-fetch and revalidate (ADR-004) before returning
anything. Where `semantic.py` delegates its SQL to
`persistence.vector_repository`, this module queries
`persistence.models.MemoryRecordRow.search_vector` directly through the
`UnitOfWork`'s session -- lexical search is small enough not to need its own
persistence-layer repository, and `MemoryRecordRow.search_vector` is already
a first-class column (see `persistence/models.py` /
`migrations/versions/003_*.py`), not something this module invents.

Good for exact terms, identifiers, and error codes that a paraphrase-tolerant
embedding can blur together; weak for synonyms and conceptual similarity --
that gap is exactly what `retrieval.hybrid` fuses lexical and semantic
results to cover.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, or_, select

from apps.memory_service.domain.models import MemoryRecord
from apps.memory_service.persistence.models import FTS_LANGUAGE, MemoryRecordRow
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.filters import record_matches_filters, resolve_filters
from apps.memory_service.retrieval.query_model import RetrievalQuery


@dataclass(frozen=True)
class LexicalSearchHit:
    """One bounded, authoritative candidate returned by `lexical_search`.

    `rank` is PostgreSQL's `ts_rank_cd` score for this row against the
    query -- higher means more relevant (the opposite sense of
    `SemanticSearchHit.distance`, where lower is better; `retrieval.hybrid`
    is what reconciles the two into one fused ordering).
    """

    memory: MemoryRecord
    rank: float


def lexical_search(uow: UnitOfWork, query: RetrievalQuery) -> list[LexicalSearchHit]:
    """Run the full exact-lexical-retrieval flow for `query`.

    How it works:
        1. Resolve `query` into `RetrievalFilters` (`resolve_filters`), the
           same hard constraints `semantic_search` applies.
        2. If `query.query_text` is `None` (the caller only supplied a
           precomputed embedding), lexical search has no literal text to
           match against and returns `[]` -- this is expected, not an
           error; `retrieval.hybrid` treats an empty lexical result list the
           same as "nothing matched".
        3. Otherwise, build a `plainto_tsquery(FTS_LANGUAGE, query_text)` --
           this treats `query_text` as plain words (ANDed together), never
           as a search-operator syntax an end user could inject -- and
           select every `memory_records` row whose `search_vector` matches
           it, applying the same tenant/status/trust/type/time predicates
           as `vector_repository.find_nearest`, ordered by
           `ts_rank_cd(search_vector, query)` descending and capped at
           `query.limit`.
        4. For each matching row, re-fetch the full record via
           `uow.records.get` and re-check `record_matches_filters` -- the
           same revalidation step `semantic_search` performs, and for the
           same reason (ADR-004).

    Example:
        Input:
            query = RetrievalQuery(tenant_id=UUID("1111...1111"),
                                    query_text="ERR_5001", limit=5)
        Output:
            [LexicalSearchHit(memory=MemoryRecord(content="Service checkout-api "
                                                    "returned ERR_5001 twice.", ...),
                               rank=0.6), ...]
    """
    if query.query_text is None:
        return []

    filters = resolve_filters(query)
    tsquery = func.plainto_tsquery(FTS_LANGUAGE, query.query_text)
    rank = func.ts_rank_cd(MemoryRecordRow.search_vector, tsquery)

    stmt = (
        select(MemoryRecordRow.memory_id, rank.label("rank"))
        .where(
            MemoryRecordRow.tenant_id == filters.tenant_id,
            MemoryRecordRow.search_vector.op("@@")(tsquery),
            MemoryRecordRow.status.in_(filters.allowed_statuses),
            MemoryRecordRow.trust_level.in_(filters.allowed_trust_levels),
            MemoryRecordRow.valid_from <= filters.effective_at,
            or_(
                MemoryRecordRow.valid_to.is_(None),
                MemoryRecordRow.valid_to >= filters.effective_at,
            ),
        )
        .order_by(rank.desc())
        .limit(query.limit)
    )
    if filters.memory_types is not None:
        stmt = stmt.where(MemoryRecordRow.memory_type.in_(filters.memory_types))

    rows = uow.session.execute(stmt).all()

    hits: list[LexicalSearchHit] = []
    for row in rows:
        record = uow.records.get(filters.tenant_id, row.memory_id)
        if record is None or not record_matches_filters(record, filters):
            continue
        hits.append(LexicalSearchHit(memory=record, rank=row.rank))
    return hits

"""Exact semantic retrieval: the full query -> filters -> search -> fetch flow.

```text
query -> tenant/trust/status/time filters -> exact pgvector similarity search
       -> fetch authoritative memory records -> return bounded candidates
```

`semantic_search` is the only function that runs this whole flow.
`persistence.vector_repository.VectorRecordRepository.find_nearest` returns
bare `(memory_id, distance)` pairs; this module is what turns those into
authoritative `MemoryRecord`s and -- per ADR-004 ("Revalidate each selected
ID against PostgreSQL before context construction") -- checks the same hard
filters again on the freshly-fetched record before it is ever returned. A
match that no longer satisfies them (e.g. it was quarantined between the two
queries within this same transaction) is dropped rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

from apps.memory_service.domain.models import MemoryRecord
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.filters import record_matches_filters, resolve_filters
from apps.memory_service.retrieval.query_model import RetrievalQuery


@dataclass(frozen=True)
class SemanticSearchHit:
    """One bounded, authoritative candidate returned by `semantic_search`."""

    memory: MemoryRecord
    distance: float


def semantic_search(
    uow: UnitOfWork,
    embedder: EmbeddingModel,
    query: RetrievalQuery,
) -> list[SemanticSearchHit]:
    """Run the full exact-semantic-retrieval flow for `query`.

    How it works:
        1. Resolve `query` into `RetrievalFilters` (`resolve_filters`) --
           these are the hard constraints every step below respects.
        2. Get a query vector: `query.query_embedding` if the caller already
           supplied one, otherwise embed `query.query_text` with `embedder`.
        3. Ask `uow.vectors.find_nearest` for the closest `query.limit`
           indexed, filter-matching memory ids.
        4. For each match, re-fetch the full record via `uow.records.get`
           and independently re-check `record_matches_filters` -- this is
           the revalidation step described in the module docstring, applied
           to data actually read back from PostgreSQL rather than trusted
           from step 3.
        5. Return the surviving matches as `SemanticSearchHit`s, nearest
           first (already ordered by `find_nearest`; revalidation only ever
           removes entries, never reorders them).

    Example:
        Input:
            query = RetrievalQuery(tenant_id=UUID("1111...1111"),
                                    query_text="pool exhaustion", limit=5)
        Output:
            [SemanticSearchHit(memory=MemoryRecord(content="Connection pool "
                                                     "exhausted after 30 retries.", ...),
                                distance=0.08), ...]
    """
    filters = resolve_filters(query)
    query_embedding = query.query_embedding
    if query_embedding is None:
        query_text = query.query_text
        # RetrievalQuery's validator guarantees query_text is set whenever
        # query_embedding is not.
        assert query_text is not None
        query_embedding = embedder.embed_texts([query_text])[0]

    matches = uow.vectors.find_nearest(filters, query_embedding, limit=query.limit)

    hits: list[SemanticSearchHit] = []
    for match in matches:
        record = uow.records.get(filters.tenant_id, match.memory_id)
        if record is None or not record_matches_filters(record, filters):
            continue
        hits.append(SemanticSearchHit(memory=record, distance=match.distance))
    return hits

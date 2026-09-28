"""Hybrid retrieval: lexical + semantic search, fused with Reciprocal Rank Fusion.

```text
query
  -> hard tenant/trust/status/time filters (retrieval.filters, applied by both retrievers)
  -> lexical search (retrieval.lexical -- exact terms, IDs, error codes)
  -> semantic search (retrieval.semantic -- meaning/paraphrases)
  -> rank fusion (Reciprocal Rank Fusion, this module)
  -> bounded candidate list (top `candidate_limit`)
  -> optional CrossEncoder reranking (retrieval.reranker)
  -> temporal/conflict resolution (retrieval.temporal -- the active truth at `as_of`)
  -> top `query.limit` hits
```

Fusion method: **Reciprocal Rank Fusion (RRF)**, chosen for ADR-003's stated
reason -- "a simple documented fusion method" to start from, not because RRF
is claimed to be optimal. A memory's fused score is the sum, over every
retriever that returned it, of `1 / (RRF_K + rank)`, where `rank` is that
memory's 1-based position in that retriever's own result list. A memory
returned by both retrievers accumulates both terms (and appears exactly
once in the output, keyed by `memory_id` -- see `_fuse_rrf`); a memory
returned by only one still gets a score, just a smaller one on average.

`RRF_K` is a visible, top-level module constant precisely so a future
benchmark can tune it without hunting through the function body for a
hidden literal -- see ADR-003's instruction to keep fusion weights/config
visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from apps.memory_service.domain.models import MemoryRecord
from apps.memory_service.embeddings.base import EmbeddingModel
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.lexical import LexicalSearchHit, lexical_search
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.reranker import Reranker, RerankReport, apply_reranker
from apps.memory_service.retrieval.semantic import SemanticSearchHit, semantic_search
from apps.memory_service.retrieval.temporal import TemporalReport, apply_temporal_resolution

# The "k" in RRF's `1 / (k + rank)`: a smoothing constant added to every rank
# before taking the reciprocal, which controls how much *more* a top rank is
# worth than a lower one.
#
#   k = 0:  rank 1 -> 1/1  = 1.000, rank 10 -> 1/10 = 0.100  (rank 1 worth 10x)
#   k = 60: rank 1 -> 1/61 = 0.0164, rank 10 -> 1/70 = 0.0143 (rank 1 worth ~1.15x)
#
# A small k lets one retriever's rank-1 hit dominate the fused list on its
# own; a large k flattens the curve, so being found by *both* retrievers
# counts for more than being first in one. 60 is the standard value from the
# RRF literature (Cormack, Clarke & Buettcher, 2009): large enough that a
# single retriever's rank-1 result doesn't overwhelm everything else, small
# enough that rank still matters. Visible here, not buried in `_fuse_rrf`, so
# it can be benchmarked and tuned.
RRF_K = 60


@dataclass(frozen=True)
class HybridSearchHit:
    """One bounded, fused candidate returned by `hybrid_search`.

    `lexical_rank` / `semantic_rank` are each retriever's own 1-based rank
    for this memory, or `None` if that retriever did not return it at all --
    kept on the hit so a caller (or a test) can see *why* something scored
    the way it did, not just the final number. `rerank_score` is the
    CrossEncoder's score if this hit was reranked, else `None`.
    """

    memory: MemoryRecord
    fused_score: float
    lexical_rank: int | None
    semantic_rank: int | None
    rerank_score: float | None = None


@dataclass(frozen=True)
class HybridSearchResult:
    """The hits `hybrid_search` returns, plus what the pipeline did to get them.

    `candidates_fused` is how many distinct memories fusion produced before
    any bounding. `rerank` is `None` when no reranker was configured;
    otherwise it records the reranker's cost and whether it fell back.
    `temporal` records which candidates temporal/conflict resolution
    removed and why, and every contradiction it considered -- including
    unresolved ones, which are withheld from `hits` but surfaced here.
    """

    hits: list[HybridSearchHit]
    candidates_fused: int
    rerank: RerankReport | None
    temporal: TemporalReport


def hybrid_search(
    uow: UnitOfWork,
    embedder: EmbeddingModel,
    query: RetrievalQuery,
    *,
    rrf_k: int = RRF_K,
    reranker: Reranker | None = None,
    candidate_limit: int | None = None,
) -> list[HybridSearchHit]:
    """Run lexical + semantic search for `query`, fuse with RRF, optionally rerank,
    then keep only the memories that are the active truth at the query's time.

    Returns only the hits; see `hybrid_search_with_report` for the same
    search plus a record of candidate counts, reranker cost/fallback, and
    what temporal/conflict resolution removed.

    Example:
        Input:
            query = RetrievalQuery(tenant_id=UUID("1111...1111"),
                                    query_text="pool exhaustion", limit=5)
        Output:
            [HybridSearchHit(memory=MemoryRecord(content="Connection pool "
                                                   "exhausted after 30 retries.", ...),
                              fused_score=0.032, lexical_rank=1, semantic_rank=1), ...]
    """
    return hybrid_search_with_report(
        uow,
        embedder,
        query,
        rrf_k=rrf_k,
        reranker=reranker,
        candidate_limit=candidate_limit,
    ).hits


def hybrid_search_with_report(
    uow: UnitOfWork,
    embedder: EmbeddingModel,
    query: RetrievalQuery,
    *,
    rrf_k: int = RRF_K,
    reranker: Reranker | None = None,
    candidate_limit: int | None = None,
) -> HybridSearchResult:
    """Run the hybrid pipeline for `query` and report what it did.

    How it works:
        1. Decide the candidate pool size: `candidate_limit` if given,
           otherwise `max(query.limit, reranker.max_candidates)` when
           reranking (so the reranker has a wider pool than the final
           answer to choose from) or just `query.limit` when not.
        2. Run `lexical_search` and `semantic_search` independently, each
           for up to that many results. Both already apply the same hard
           filters (`retrieval.filters`) and already revalidate every
           result against a freshly-fetched `MemoryRecord` (ADR-004) --
           this function does no additional filtering of its own, so
           everything after this step only ever sees eligible memories.
        3. Fuse the two ranked lists with `_fuse_rrf`, which keys by
           `memory_id` so a memory present in both lists is scored once,
           from both contributions, and appears exactly once in the output
           (see the module docstring). Sort by `fused_score` descending and
           keep the top `candidate_limit`.
        4. If a `reranker` was supplied, pass the bounded candidates through
           `reranker.apply_reranker`, which scores at most
           `reranker.max_candidates` of them and falls back to the fused
           order (recording why) if reranking fails.
        5. Pass the whole bounded candidate list through
           `temporal.apply_temporal_resolution`, which removes anything not
           in effect at the query's time, superseded, or withheld by a
           recorded contradiction -- regardless of its score. This runs
           before truncation so a removed stale fact leaves room for the
           next valid one instead of shrinking the answer.
        6. Truncate to `query.limit`.

    Example:
        Input:
            query = RetrievalQuery(tenant_id=UUID("1111...1111"),
                                    query_text="is dark mode disabled", limit=5)
            reranker = CrossEncoderReranker(SentenceTransformerCrossEncoder(),
                                            max_candidates=30)
        Output:
            HybridSearchResult(
                hits=[HybridSearchHit(memory=<"dark_mode is now disabled">,
                                      fused_score=0.032, lexical_rank=1,
                                      semantic_rank=2, rerank_score=9.19), ...],
                candidates_fused=2,
                rerank=RerankReport(candidates_in=2, candidates_reranked=2,
                                    latency_ms=11.8, fallback_reason=None),
                temporal=TemporalReport(effective_at=<now>, candidates_in=2,
                                        excluded=(), conflicts=()),
            )
    """
    if candidate_limit is None:
        candidate_limit = (
            max(query.limit, reranker.max_candidates) if reranker is not None else query.limit
        )
    if candidate_limit < query.limit:
        raise ValueError("candidate_limit must be at least query.limit")

    candidate_query = (
        query
        if candidate_limit == query.limit
        else query.model_copy(update={"limit": candidate_limit})
    )
    lexical_hits = lexical_search(uow, candidate_query)
    semantic_hits = semantic_search(uow, embedder, candidate_query)

    fused = _fuse_rrf(lexical_hits, semantic_hits, rrf_k=rrf_k)
    fused.sort(key=lambda hit: hit.fused_score, reverse=True)
    candidates = fused[:candidate_limit]

    rerank_report: RerankReport | None = None
    if reranker is not None:
        candidates, rerank_report = apply_reranker(reranker, query.query_text, candidates)

    candidates, temporal_report = apply_temporal_resolution(uow, query, candidates)

    return HybridSearchResult(
        hits=candidates[: query.limit],
        candidates_fused=len(fused),
        rerank=rerank_report,
        temporal=temporal_report,
    )


def _fuse_rrf(
    lexical_hits: list[LexicalSearchHit],
    semantic_hits: list[SemanticSearchHit],
    *,
    rrf_k: int,
) -> list[HybridSearchHit]:
    """Combine two ranked hit lists into one, keyed by `memory_id`.

    How it works:
        Walks each list once, in rank order, accumulating
        `1 / (rrf_k + rank)` into a per-`memory_id` running score
        (`scores`), recording that retriever's rank in `lexical_ranks` /
        `semantic_ranks`, and remembering one `MemoryRecord` per id
        (`records` -- lexical's copy wins if both retrievers returned the
        same memory, an arbitrary but immaterial choice since both are the
        same authoritative row). The result is unsorted; `hybrid_search`
        sorts it.

        `rrf_k` is the smoothing constant `k` in `1 / (k + rank)` (normally
        `RRF_K` = 60): higher values shrink the gap between top and lower
        ranks, so agreement between the two retrievers matters more than
        either one's top spot. See the comment on `RRF_K` for worked numbers.

    Example:
        Input:
            lexical_hits = [LexicalSearchHit(memory=<A>, rank=12.0)]  # rank 1
            semantic_hits = [SemanticSearchHit(memory=<B>, distance=0.1),   # rank 1
                              SemanticSearchHit(memory=<A>, distance=0.3)]  # rank 2
            rrf_k = 60
        Output:
            [HybridSearchHit(memory=<A>, fused_score=1/61 + 1/62, lexical_rank=1, semantic_rank=2),
             HybridSearchHit(memory=<B>, fused_score=1/61, lexical_rank=None, semantic_rank=1)]
    """
    scores: dict[UUID, float] = {}
    records: dict[UUID, MemoryRecord] = {}
    lexical_ranks: dict[UUID, int] = {}
    semantic_ranks: dict[UUID, int] = {}

    for rank, lexical_hit in enumerate(lexical_hits, start=1):
        memory_id = lexical_hit.memory.memory_id
        scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (rrf_k + rank)
        records[memory_id] = lexical_hit.memory
        lexical_ranks[memory_id] = rank

    for rank, semantic_hit in enumerate(semantic_hits, start=1):
        memory_id = semantic_hit.memory.memory_id
        scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (rrf_k + rank)
        records.setdefault(memory_id, semantic_hit.memory)
        semantic_ranks[memory_id] = rank

    return [
        HybridSearchHit(
            memory=records[memory_id],
            fused_score=score,
            lexical_rank=lexical_ranks.get(memory_id),
            semantic_rank=semantic_ranks.get(memory_id),
        )
        for memory_id, score in scores.items()
    ]

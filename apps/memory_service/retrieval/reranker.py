"""CrossEncoder reranking of a bounded, already-filtered candidate set.

```text
filtered lexical + semantic results
  -> hybrid fusion (retrieval.hybrid, RRF)
  -> top-N fused candidates            (N = candidate_limit, e.g. 30)
  -> CrossEncoder reranks at most max_candidates of them   (this module)
  -> top query.limit hits handed on    (temporal/conflict resolution is next)
```

ADR-003's staged pipeline reranks only a bounded top-N set, *after* rank
fusion: a CrossEncoder costs one model forward pass per (query, memory) pair
with nothing precomputable, which is fine over a few dozen candidates and
impractical over a whole tenant's corpus. Two things enforce that bound:
`CrossEncoderReranker` only ever scores its first `max_candidates` hits, and
`apply_reranker` only ever *hands* a reranker that many.

Reranking only decides order. Every hit it sees has already passed the hard
tenant/status/trust/type/time filters (`retrieval.filters`, applied by both
retrievers before fusion), and `apply_reranker` rejects any reranker output
that is not an exact reordering of its input -- a reranker cannot add a
memory the filters excluded, drop one, or duplicate one.

If reranking fails for any reason, `apply_reranker` returns the fused order
unchanged and records why in `RerankReport.fallback_reason` (and logs a
warning): a broken or missing model degrades ranking quality, never
availability or correctness.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from apps.memory_service.consolidation.forgetting import retrieval_priority
from apps.memory_service.domain.enums import MemoryStatus
from apps.memory_service.embeddings.cross_encoder import CrossEncoderModel

if TYPE_CHECKING:
    from apps.memory_service.retrieval.hybrid import HybridSearchHit

logger = logging.getLogger(__name__)

# Matches RETRIEVAL_RERANK_LIMIT in .env.example: the most candidates a
# single query may send through the CrossEncoder. Visible here, like
# `hybrid.RRF_K`, so the cost/quality trade-off can be benchmarked and tuned
# (see apps/benchmark/run_retrieval_eval.py).
RERANK_MAX_CANDIDATES = 30


class Reranker(Protocol):
    """Something that reorders a bounded set of fused hits for one query.

    Implementations must not add or remove hits -- only reorder (and
    optionally rescore) the ones they were given; filtering is
    `retrieval.filters`'s job, not a reranker's. `apply_reranker` enforces
    this and falls back to the fused order if it is violated.
    """

    @property
    def max_candidates(self) -> int:
        """The most hits one `rerank` call will score; the rest are passed through."""
        ...

    def rerank(self, query_text: str, hits: Sequence[HybridSearchHit]) -> list[HybridSearchHit]:
        """Return `hits` reordered by relevance to `query_text`."""
        ...


class CrossEncoderReranker:
    """Reranks fused hits by CrossEncoder score plus log lifecycle priority.

    Only the first `max_candidates` hits are scored; any beyond that are
    appended afterwards, unscored, in their original fused order. Ties in
    CrossEncoder score keep their fused order (Python's sort is stable), so
    the result is deterministic for a deterministic model.
    """

    def __init__(
        self, model: CrossEncoderModel, *, max_candidates: int = RERANK_MAX_CANDIDATES
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self._model = model
        self._max_candidates = max_candidates

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    @property
    def model_version(self) -> str:
        return self._model.model_version

    def rerank(self, query_text: str, hits: Sequence[HybridSearchHit]) -> list[HybridSearchHit]:
        """Score up to `max_candidates` hits against `query_text` and sort them.

        Example:
            Input:
                query_text = "is dark mode disabled"
                hits = [<"dark_mode is now enabled", fused_score=0.033>,
                        <"dark_mode is now disabled", fused_score=0.032>]
            Output:
                [<"dark_mode is now disabled", rerank_score=9.19>,
                 <"dark_mode is now enabled", rerank_score=5.73>]
        """
        hits = [
            hit
            for hit in hits
            if hit.memory.status not in (MemoryStatus.TOMBSTONE, MemoryStatus.QUARANTINED)
        ]
        head = list(hits[: self._max_candidates])
        tail = list(hits[self._max_candidates :])
        if not head:
            return tail

        scores = self._model.score_pairs(query_text, [hit.memory.content for hit in head])
        if len(scores) != len(head):
            raise RuntimeError(
                f"CrossEncoder returned {len(scores)} scores for {len(head)} candidates"
            )

        rescored = [
            dataclasses.replace(hit, rerank_score=score)
            for hit, score in zip(head, scores, strict=True)
        ]
        rescored.sort(
            key=lambda hit: (hit.rerank_score or 0.0) + math.log(retrieval_priority(hit.memory)),
            reverse=True,
        )
        return rescored + tail


@dataclass(frozen=True)
class RerankReport:
    """What the reranking stage did for one query -- its cost and outcome.

    `candidates_reranked` is the number of (query, memory) pairs sent to the
    reranker, i.e. the CrossEncoder's cost for this query. `fallback_reason`
    is `None` when reranking succeeded; otherwise it says why the fused
    order was returned instead.
    """

    candidates_in: int
    candidates_reranked: int
    latency_ms: float
    fallback_reason: str | None = None

    @property
    def fell_back(self) -> bool:
        return self.fallback_reason is not None


def apply_reranker(
    reranker: Reranker,
    query_text: str | None,
    hits: Sequence[HybridSearchHit],
) -> tuple[list[HybridSearchHit], RerankReport]:
    """Run `reranker` over `hits` safely, falling back to their given order.

    How it works:
        1. Nothing to rerank against (`query_text` is None -- the caller
           supplied only a precomputed embedding) -> fall back.
        2. Slice `hits` to `reranker.max_candidates` and hand only that head
           to the reranker; the tail is never sent to it and is appended
           afterwards in fused order.
        3. Any exception from the reranker -> fall back (logged with the
           traceback).
        4. If the reranker's output is not an exact reordering of the head
           (a hit added, dropped, or duplicated) -> fall back. This is what
           guarantees reranking cannot resurrect a memory the hard filters
           already excluded.

    Every fallback returns `hits` in their original order, with the reason
    recorded in the returned `RerankReport` and logged at WARNING.

    Example:
        Input:
            reranker = CrossEncoderReranker(<model that raises>, max_candidates=30)
            query_text = "pool exhaustion"
            hits = [<A>, <B>]
        Output:
            ([<A>, <B>],
             RerankReport(candidates_in=2, candidates_reranked=0, latency_ms=0.4,
                          fallback_reason="reranker raised RuntimeError: CUDA out of memory"))
    """
    original = [
        hit
        for hit in hits
        if hit.memory.status not in (MemoryStatus.TOMBSTONE, MemoryStatus.QUARANTINED)
    ]
    started = time.perf_counter()

    def _fallback(
        reason: str, *, exc_info: bool = False
    ) -> tuple[list[HybridSearchHit], RerankReport]:
        logger.warning("reranking skipped, returning fused order: %s", reason, exc_info=exc_info)
        report = RerankReport(
            candidates_in=len(original),
            candidates_reranked=0,
            latency_ms=_elapsed_ms(started),
            fallback_reason=reason,
        )
        return original, report

    if not original:
        return original, RerankReport(candidates_in=0, candidates_reranked=0, latency_ms=0.0)
    if query_text is None:
        return _fallback("query has no query_text to rerank against")

    head = original[: reranker.max_candidates]
    tail = original[reranker.max_candidates :]
    try:
        reordered = reranker.rerank(query_text, head)
    except Exception as exc:
        return _fallback(f"reranker raised {type(exc).__name__}: {exc}", exc_info=True)

    if sorted(hit.memory.memory_id for hit in reordered) != sorted(
        hit.memory.memory_id for hit in head
    ):
        return _fallback("reranker output was not a reordering of its input")

    originals = {hit.memory.memory_id: hit.memory for hit in head}
    if any(hit.memory != originals[hit.memory.memory_id] for hit in reordered):
        return _fallback("reranker altered authoritative memory")

    report = RerankReport(
        candidates_in=len(original),
        candidates_reranked=len(head),
        latency_ms=_elapsed_ms(started),
    )
    return reordered + tail, report


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0

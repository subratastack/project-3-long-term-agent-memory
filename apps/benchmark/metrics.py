"""Standard ranked-retrieval metrics, pure and dependency-free.

Every function takes a ranked list of ids (best first) and the ground truth
for one query, so the same functions score lexical, semantic, hybrid, and
reranked retrieval identically. Relevance is *graded*: `grades` maps an id to
an integer relevance (e.g. 2 = answers the question, 1 = useful context);
ids absent from `grades` are irrelevant (grade 0).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet


def precision_at_k[T: Hashable](ranked: Sequence[T], relevant: AbstractSet[T], k: int) -> float:
    """Distinct relevant ids in the top `k`, divided by `k` (which must be positive).

    Missing results count as unfilled slots; repeated ids earn credit once,
    consistently with `recall_at_k`. No relevant ids gives a score of 0.0.

    Example:
        Input:  ranked=["a", "b", "c"], relevant={"b", "z"}, k=2
        Output: 0.5
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k[T: Hashable](ranked: Sequence[T], relevant: AbstractSet[T], k: int) -> float:
    """Fraction of `relevant` ids that appear in the top `k` of `ranked`.

    Example:
        Input:  ranked=["a", "b", "c"], relevant={"b", "z"}, k=2
        Output: 0.5
    """
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant ids")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank[T: Hashable](ranked: Sequence[T], relevant: AbstractSet[T]) -> float:
    """`1 / rank` of the first relevant id in `ranked`, or 0.0 if none appears.

    Averaged over queries, this is MRR.

    Example:
        Input:  ranked=["a", "b", "c"], relevant={"b", "c"}
        Output: 0.5
    """
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k[T: Hashable](ranked: Sequence[T], grades: Mapping[T, int], k: int) -> float:
    """Normalized discounted cumulative gain over the top `k` of `ranked`.

    Uses the exponential-gain form, `(2**grade - 1) / log2(rank + 1)`, so a
    grade-2 answer at rank 1 is worth three times a grade-1 one; the sum is
    divided by the same quantity for the ideal ordering of `grades`, giving
    1.0 for a perfect ranking.

    Example:
        Input:  ranked=["ctx", "answer"], grades={"answer": 2, "ctx": 1}, k=2
        Output: 0.797...   # (1/1 + 3/log2(3)) / (3/1 + 1/log2(3))
    """
    ideal = _dcg(sorted(grades.values(), reverse=True)[:k])
    if ideal == 0.0:
        raise ValueError("nDCG is undefined for a query with no relevant ids")
    return _dcg([grades.get(item, 0) for item in ranked[:k]]) / ideal


def percentile(values: Sequence[float], pct: float) -> float:
    """The `pct`-th percentile of `values`, linearly interpolated (numpy's default).

    Example:
        Input:  values=[10, 20, 30, 40], pct=50
        Output: 25.0
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be within [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _dcg(gains: Sequence[int]) -> float:
    return sum((2.0**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(gains, start=1))

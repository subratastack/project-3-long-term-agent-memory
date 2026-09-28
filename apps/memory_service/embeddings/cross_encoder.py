"""The CrossEncoder interface the reranking stage depends on.

An `EmbeddingModel` (`base.py`) embeds a query and a memory *separately* and
compares the two vectors; a CrossEncoder reads the query and one memory
*together* and outputs a single relevance score for that exact pair. That
joint reading is why it is more accurate (it can see that "is dark mode
disabled" and "dark_mode is now enabled" disagree, which two independent
embeddings routinely miss) and also why it is much slower: there is nothing
to precompute or index, so every (query, memory) pair costs one full model
forward pass at query time. ADR-003 therefore only ever runs it over a
bounded candidate set -- see `apps.memory_service.retrieval.reranker`.

Like `base.py`, this module pairs a protocol with a deterministic fake so
tests can exercise reranking without downloading a model;
`SentenceTransformerCrossEncoder` is the real implementation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

# Matches RERANKER_MODEL in .env.example: a small (6-layer MiniLM) model
# trained on MS MARCO question -> passage relevance, which is the same shape
# as "agent question -> candidate memory".
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderModel(Protocol):
    """Something that scores (query, passage) pairs for relevance."""

    @property
    def model_version(self) -> str:
        """An identifier for the model, recorded alongside reranking results."""
        ...

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each passage against `query`; higher means more relevant.

        Returns exactly one score per passage, in the same order. Scores are
        only comparable within one call -- they are raw model outputs
        (logits), not calibrated probabilities.
        """
        ...


class FakeCrossEncoderModel:
    """A deterministic, dependency-free stand-in for a real CrossEncoder.

    Scores a passage by the fraction of the query's distinct tokens it
    contains -- crude, but it reads the query and passage *together* (which
    is the property that distinguishes a CrossEncoder from an embedding) and
    it is stable, fast, and easy to reason about in a test.

    Example:
        Input:
            FakeCrossEncoderModel().score_pairs(
                "pool exhausted", ["connection pool exhausted", "backup succeeded"]
            )
        Output:
            [1.0, 0.0]
    """

    def __init__(self, *, model_version: str = "fake-token-overlap-v1") -> None:
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return [0.0] * len(passages)
        return [len(query_tokens & _tokens(passage)) / len(query_tokens) for passage in passages]


class SentenceTransformerCrossEncoder:
    """Wraps a `sentence_transformers.CrossEncoder` as a `CrossEncoderModel`.

    `sentence_transformers` is imported inside `__init__` rather than at
    module level so that importing this module for `CrossEncoderModel` or
    `FakeCrossEncoderModel` alone does not pull in torch. Loading the model
    itself downloads it or reads it from the local Hugging Face cache and is
    comparatively slow, so construct one instance and reuse it.

    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        from apps.memory_service.embeddings.device import model_device
        from apps.memory_service.embeddings.hf_auth import huggingface_token

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(
            model_name, device=device or model_device(), token=huggingface_token()
        )

    @property
    def model_version(self) -> str:
        return self._model_name


    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each passage against `query` using the underlying CrossEncoder.

        Examples:
            >>> model = SentenceTransformerCrossEncoder()
            >>> model.score_pairs("database connection issue", [
            ...     "Failed to connect to postgresql database",
            ...     "User updated their profile picture"
            ... ])
            [8.42, -5.13]

            >>> model.score_pairs("any query", [])
            []
        """
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

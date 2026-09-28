"""The embedding interface every retrieval component depends on.

`apps.memory_service.retrieval.semantic` and
`apps.memory_service.persistence.vector_repository` only ever depend on this
`EmbeddingModel` protocol, never on `sentence_transformer.py` directly --
that is what lets tests run against `FakeEmbeddingModel` (fast, deterministic,
no model download) while production code runs against a real
`SentenceTransformerEmbeddingModel`. See ARCHITECTURE.md: "sentence-transformers
produces embeddings and bounded reranking scores" -- embeddings are a derived
artifact, never authoritative on their own.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    """Something that turns text into fixed-length vectors."""

    @property
    def dimensions(self) -> int:
        """The length of every vector this model produces."""
        ...

    @property
    def model_version(self) -> str:
        """An identifier stable enough to detect when embeddings need recomputing."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed `texts`, returning one vector of length `dimensions` per input."""
        ...


class FakeEmbeddingModel:
    """A deterministic, dependency-free stand-in for a real embedding model.

    This is not a random or trivial fake: it is a feature-hashing (bag-of-words)
    embedding -- each token in the text is hashed into one of `dimensions`
    buckets, and the resulting vector is L2-normalized. Two texts that share
    vocabulary end up with a high cosine similarity, and texts that don't
    share vocabulary end up close to orthogonal, which is exactly the property
    `retrieval.semantic` tests need (e.g. a query about "pool exhaustion"
    should land close to an event describing "connection pool exhausted")
    without downloading or running a real sentence-transformers model.
    """

    def __init__(self, dimensions: int = 384, *, model_version: str = "fake-hashing-v1") -> None:
        self._dimensions = dimensions
        self._model_version = model_version

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_version(self) -> str:
        return self._model_version

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self._dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm > 0:
            vector = [component / norm for component in vector]
        return vector

"""A real `EmbeddingModel` backed by a local sentence-transformers model.

The default model (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions)
matches `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` in `.env.example` and
`apps.memory_service.persistence.models.EMBEDDING_DIMENSIONS`. Changing the
model to one with a different output dimensionality requires a migration to
resize the `memory_records.embedding` column, since pgvector fixes a column's
dimensionality at table-creation time.
"""

from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from apps.memory_service.embeddings.device import model_device
from apps.memory_service.embeddings.hf_auth import huggingface_token

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingModel:
    """Wraps a `sentence_transformers.SentenceTransformer` as an `EmbeddingModel`.

    Loading the underlying model (`SentenceTransformer(model_name)`) downloads
    or reads it from the local cache and is comparatively slow, so this class
    is deliberately not imported anywhere retrieval code runs by default --
    only where a real embedding model is actually needed.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, *, device: str | None = None) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name, device=device or model_device(), token=huggingface_token()
        )

    @property
    def dimensions(self) -> int:
        return int(self._model.get_embedding_dimension())

    @property
    def model_version(self) -> str:
        return self._model_name


    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed `texts`, normalizing each vector to unit length.

        `persistence.vector_repository` ranks by pgvector's cosine-distance
        operator regardless of normalization, but normalizing here is still
        good practice: it keeps embeddings comparable if a future consumer
        wants raw dot products instead.

        Examples:
            >>> model = SentenceTransformerEmbeddingModel()
            >>> model.embed_texts(["Hello world", "Agent memory"])
            [[0.021, -0.048, ..., 0.035], [-0.012, 0.089, ..., -0.004]]
            >>> len(model.embed_texts(["test"])[0])
            384
            >>> model.embed_texts([])
            []
        """
        if not texts:
            return []
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

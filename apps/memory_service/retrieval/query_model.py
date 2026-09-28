"""The parsed, trusted shape of a retrieval request.

A `RetrievalQuery` is the "query" step of ADR-003's staged pipeline: it
carries only tenant/user scope, allowed memory types, a minimum trust bar,
an optional `as_of` time, and either raw text or a precomputed embedding.
`tenant_id` in particular always comes from authenticated runtime context in
real usage (ARCHITECTURE.md, "Tenant isolation") -- this model does not
itself enforce that, but every field below is treated as already-trusted
input by `apps.memory_service.retrieval.filters` and `.semantic`.
"""

from __future__ import annotations

import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apps.memory_service.domain.enums import MemoryType, TrustLevel


class RetrievalQuery(BaseModel):
    """One semantic-retrieval request.

    Exactly one of `query_text` or `query_embedding` must be given:
    `query_text` is embedded on the fly by whichever `EmbeddingModel` the
    caller supplies to `semantic_search`; `query_embedding` lets a caller
    that already has a vector (e.g. from a batch job) skip that step.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: UUID = Field(description="Tenant this query is scoped to.")
    query_text: str | None = Field(
        default=None, description="Raw query text to embed, if no vector was supplied directly."
    )
    query_embedding: list[float] | None = Field(
        default=None, description="A precomputed query vector, if the caller already has one."
    )
    memory_types: list[MemoryType] | None = Field(
        default=None, description="Restrict results to these memory types; None means any type."
    )
    min_trust: TrustLevel | None = Field(
        default=None,
        description="Exclude memories trusted below this level; None means no trust floor.",
    )
    as_of: datetime.datetime | None = Field(
        default=None,
        description="Historical point in time to evaluate validity at; None means now.",
    )
    limit: int = Field(default=10, ge=1, le=200, description="Maximum candidates to return.")

    @model_validator(mode="after")
    def exactly_one_of_text_or_embedding(self) -> RetrievalQuery:
        """Reject a query with neither, or both, of `query_text`/`query_embedding`.

        Example:
            Input:
                RetrievalQuery(tenant_id=..., query_text=None, query_embedding=None)
            Output:
                raises pydantic.ValidationError
        """
        has_text = self.query_text is not None
        has_embedding = self.query_embedding is not None
        if has_text == has_embedding:
            raise ValueError("exactly one of query_text or query_embedding must be provided")
        return self

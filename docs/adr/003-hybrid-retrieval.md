# ADR-003: Filtered Hybrid Retrieval

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

Memory queries vary. Exact identifiers, error codes, names, and configuration
keys favor lexical search, while paraphrases and conceptually related incidents
favor semantic search. Neither mechanism determines tenant authorization,
trust, temporal validity, or conflict resolution. CrossEncoder reranking can
improve ordering but is too expensive for the full corpus.

## Decision

Use a staged retrieval pipeline:

1. Parse the query into trusted tenant/user scope, allowed memory types,
   entities, time window, lexical terms, and semantic query text.
2. Apply tenant, status, validity, type, and trust filters as hard constraints.
3. Retrieve bounded candidates independently with PostgreSQL FTS and exact
   pgvector similarity search.
4. Fuse the ranked lists with a documented, configurable strategy; weights are
   configuration and must be benchmarked.
5. Rerank a bounded top-N set with a sentence-transformers CrossEncoder.
6. Resolve temporal validity, supersession, and conflicts.
7. Pack diverse, non-redundant records under a token budget.
8. Revalidate each selected ID against PostgreSQL before context construction.

Approximate nearest-neighbor indexes are introduced only after exact-search
latency and recall have been measured at representative scale. The same labeled
dataset and filters are used to compare exact pgvector, pgvector ANN, and any
optional TurboVec backend.

## Consequences

### Positive

- Exact and semantic query classes are both served well.
- Expensive reranking is limited to a controlled candidate set.
- Authorization and truth-related policy cannot be bypassed by ranking scores.
- Retrieval strategies can be compared using common, reproducible metrics.

### Negative

- Fusion, calibration, and evaluation add operational complexity.
- Two initial retrieval paths consume more work than a single mechanism.
- CrossEncoder reranking adds measurable latency and model dependencies.

## Rejected alternatives

- **Vector-only retrieval:** weak for exact terms and risks treating similarity
  as authority or freshness.
- **Lexical-only retrieval:** misses paraphrases and conceptually similar prior
  episodes.
- **CrossEncoder over the full corpus:** computationally impractical.
- **ANN from day one:** adds approximation and tuning before scale justifies it.
- **Filtering only after top-K search:** can lose all valid tenant/time-scoped
  results and risks leakage.

## Validation

Maintain labeled queries where lexical search wins, semantic search wins, and
hybrid search should combine both. Measure Recall@K, MRR/nDCG, P50/P95 latency,
filtered-search behavior, reranker cost, stale-fact errors, and tenant leakage.

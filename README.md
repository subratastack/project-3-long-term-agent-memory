# Long-Term Agent Memory OS

A local-first, production-style memory subsystem for agents. The system treats
memory as governed, temporal, tenant-scoped data—not as a transcript copied
into a vector database.

## Scope

The project supports four distinct memory classes:

- **Working memory** lives in the current runtime or graph state and expires
  with the run.
- **Episodic memory** records trusted events, observations, actions, and
  outcomes.
- **Semantic memory** records durable facts distilled from evidence.
- **Procedural memory** records reusable, versioned methods supported by
  repeated evidence or explicit approval.

PostgreSQL is the authoritative store for memory text, tenant scope,
provenance, trust, temporal validity, conflict state, and audit history.
PostgreSQL full-text search and pgvector are derived retrieval mechanisms.

## Prerequisites

- Python 3.12+
- Docker with the Compose plugin
- `uv` (recommended) or another PEP 517-compatible Python package manager
- Ollama only when LLM-backed extraction is introduced in a later phase

## Local setup

```bash
cp .env.example .env
docker compose up -d postgres
uv sync --all-groups
uv run pytest
```

PostgreSQL is exposed on `localhost:5432` by default. The Compose health check
waits for the configured database to accept connections.

Useful commands:

```bash
docker compose ps
docker compose logs -f postgres
uv run ruff check .
uv run mypy apps
uv run pytest
```

## Seed fictional company memories

After applying the migrations, load two tenant-scoped example datasets (an
ecommerce customer-care company and a school-management engineering team):

```bash
uv run alembic upgrade head
docker compose exec -T postgres psql -U agent_memory -d agent_memory \
  < scripts/seed_fictional_company_memories.sql
```

The seed registers the two fictional tenants and adds 16 raw `memory_events`
(eight per company). It intentionally does not create memory records,
provenance, write decisions, or embeddings; process the events through the
API when you are ready. Fixed UUIDs make the script safe to run repeatedly.

## Connecting to pgvector from the CLI

PostgreSQL (with pgvector) runs in the `postgres` service from
`docker compose up -d postgres`. Credentials come from `.env`/`.env.example`
(`agent_memory` / `agent_memory_local` / db `agent_memory`, port `5432` by
default).

### Option A: `psql` inside the container (no local install needed)

```bash
docker compose exec postgres psql -U agent_memory -d agent_memory
```

### Option B: `psql` from your host

Requires a local `psql` client (`apt install postgresql-client` /
`brew install libpq`):

```bash
psql "postgresql://agent_memory:agent_memory_local@localhost:5432/agent_memory"
```

Either way you land in a normal `psql` prompt against the same database a
GUI client (e.g. DBeaver, using the same host/port/credentials) would use.

### Useful queries once connected

```sql
-- Confirm the vector extension is installed
\dx vector

-- See the vector-related columns added to memory_records
\d memory_records

-- How many records are embedded vs. still pending/failed
SELECT index_status, count(*) FROM memory_records GROUP BY index_status;

-- Exact nearest-neighbor search: closest 5 indexed memories to an arbitrary
-- embedding for one tenant (cosine distance, ascending = most similar first)
SELECT memory_id, content, embedding <=> '[0,0,0]'::vector AS distance
FROM memory_records
WHERE tenant_id = '00000000-0000-0000-0000-000000000000'
  AND index_status = 'indexed'
ORDER BY distance
LIMIT 5;
```

The last query needs a real 384-dimension vector in place of `'[0,0,0]'` to
be meaningful -- see `apps.memory_service.embeddings` for how one is produced
in code, or `docs/ingestion-flow.md` for a curl-driven walkthrough that
populates data to query against.

### Note on schema persistence

Alembic (`uv run alembic upgrade head`) is what creates and keeps the schema
present for CLI/DBeaver browsing. The integration test suite
(`apps/tests/integration`) creates and *drops* the same tables independently
via SQLAlchemy metadata for test isolation, so running it will make the
schema disappear again -- re-run `uv run alembic upgrade head` afterward if
you want to keep browsing the database.

## Retrieval

Exact lexical search (PostgreSQL full-text search), exact semantic search
(pgvector cosine distance), and hybrid fusion (Reciprocal Rank Fusion) are
implemented -- see [docs/retrieval-flow.md](docs/retrieval-flow.md) for a
Mermaid flowchart of the full query → filters → lexical/semantic → fusion →
bounded-candidates pipeline, a runnable Python snippet, and pointers to the
tests that prove lexical wins on exact terms, semantic wins on paraphrases,
and hybrid beats either alone on a mixed workload (MRR 0.6 / 0.9 / 1.0).

CrossEncoder reranking (`apps/memory_service/retrieval/reranker.py`) reorders
a bounded top-30 fused candidate set and falls back to the fused order, with
a recorded reason, if the model fails. Quality and cost before/after
reranking come from `uv run python -m apps.benchmark.run_retrieval_eval`
(Recall@K, MRR, nDCG, P50/P95 latency, candidates reranked); see
[docs/retrieval-flow.md](docs/retrieval-flow.md#reranking-measured-quality-and-cost),
and [docs/reranking.md](docs/reranking.md) for how the stage works, its
guarantees, fallback reasons, and configuration.
No ANN vector index (HNSW/
IVFFlat) exists yet either; ADR-003's stop condition (Recall@K/MRR on a
labeled dataset, see `apps/tests/integration/retrieval/test_exact_search_quality.py`)
is what will justify introducing one.

## Benchmark commands

```bash
uv run python -m apps.benchmark.run_retrieval_eval      # available: hybrid vs. hybrid + CrossEncoder
uv run python -m apps.benchmark.run_longitudinal_eval   # planned
uv run python -m apps.benchmark.run_poisoning_eval      # planned
```

Evaluation will cover Recall@K, MRR/nDCG, stale-fact error rate, conflict
resolution, poison acceptance, tenant leakage, context token cost, and
retrieval latency.

## Design constraints

- Every accepted memory has provenance.
- Tenant filtering is a hard authorization boundary, not a ranking feature.
- Candidate extraction never writes directly to authoritative memory records.
- Conflicting facts are versioned or quarantined; they are not silently
  overwritten.
- Vector similarity never decides truth, trust, freshness, or visibility.
- Exact vector search is the baseline until measurements justify ANN indexes.
- Forgetting begins with expiry, archive, decay, or tombstones—not hard delete.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/adr](docs/adr) for the design
contract and decision records.

## Current limitations

The database schema, ingestion pipeline (see
[docs/ingestion-flow.md](docs/ingestion-flow.md)), and retrieval pipeline
(see [docs/retrieval-flow.md](docs/retrieval-flow.md)), including CrossEncoder
reranking, read-time temporal/conflict resolution, and token-budget context
packing (see [docs/context-packing.md](docs/context-packing.md)), are
implemented. Not yet implemented: an ANN vector index, *detecting* conflicts/supersessions at
write time (retrieval honours `SUPERSESSION` and `CONTRADICTION` edges, but
ingestion does not create them yet -- see
[docs/temporal-resolution.md](docs/temporal-resolution.md)), consolidation, forgetting/expiry
workers, and LangGraph integration.

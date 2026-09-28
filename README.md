# Long-Term Agent Memory OS

A local-first memory service for AI agents that tracks **where information
came from, when it applies, and whether it should enter the agent's context**.

It implements governed ingestion, hybrid retrieval, optional reranking,
temporal conflict resolution, and token-budget context packing. Decisions
are inspectable through provenance, write-policy audits, retrieval reports,
and per-memory selection reasons.

**Status:** under active development. The core pipelines and a development
HTTP API are implemented; autonomous consolidation, lifecycle workers, and
agent-runtime integration remain planned. The HTTP API is a local development
tool, not a production deployment surface.

[See an example](#a-memory-that-changes-over-time) ·
[Architecture](#how-the-system-fits-together) ·
[Evaluations](#what-has-been-measured) ·
[Run locally](#quick-start) ·
[Implementation status](#implementation-status) ·
[Documentation](#explore-the-design)

## The problem

An agent may remember a configuration that has changed, retrieve two
contradictory claims, or spend most of its prompt space on repeated incident
reports. A relevant search result alone does not tell it which information
is current, sufficiently trusted, or worth the available space.

This project makes those decisions explicit. An incoming note must pass
write policy before becoming active memory. A retrieved note must pass
eligibility and temporal checks before it can enter the agent's context.

## A memory that changes over time

Imagine an engineer asking about checkout-api timeouts. The following is an
illustrative scenario; replacement and contradiction links are explicitly
recorded, not automatically inferred during ingestion.

| Situation | System behavior | Why it matters |
| --- | --- | --- |
| The timeout changes from 5 seconds to 2 seconds on March 1 | A recorded replacement closes the old validity window and preserves both records | Current reads can use the new value without erasing history |
| The engineer asks what applied on February 15 | An `as_of` query can return the old 5-second value | Historical questions use the requested time |
| Two current claims disagree and have equal trust | A recorded contradiction withholds both and appears in the conflict report | A high search score does not arbitrarily settle the disagreement |
| Ten incident reports repeat the same issue | Context packing discounts overlap and can keep a fact, an incident, and a procedure within the budget | Repetition need not crowd out complementary information |
| A proposed memory cites missing or inconsistent evidence | Normalization or provenance verification rejects the proposal | A model's proposal does not approve its own admission |

These rules act on recorded sources, dates, types, and relationships. They
do not independently prove real-world truth. Outcomes also depend on which
memories retrieval finds and the query's filters.

## How the system fits together

```mermaid
flowchart TD
    E["Events: messages, logs, configuration"] --> X["Extract proposed memories"]
    X --> N["Classify, normalize, verify provenance"]
    N --> W{"Deterministic write policy"}
    W -->|accept| DB[("PostgreSQL: active memory + audit")]
    W -->|quarantine| Q[("Held for review + audit")]
    W -->|reject| A[("Decision audit, no memory record")]
    N -->|unresolvable source during normalization| P["Pre-policy rejection reported to caller"]

    U["Question + tenant + optional historical time"] --> S["Word search + meaning search"]
    DB --> S
    S --> V["Revalidate records and fuse rankings"]
    V --> R["Optional bounded reranking"]
    R --> T["Resolve validity, replacements, contradictions"]
    T --> H["Ranked memories + retrieval report"]
    H --> C["Pack selected memories into a token budget"]
    C --> O["Agent context + selection and skip reasons"]
```

PostgreSQL is the authoritative store for content, tenant ownership,
provenance, trust, validity, relationships, and audit decisions. Full-text
search and pgvector embeddings are derived retrieval mechanisms. New active
memories need indexing before meaning-based search can find them.

The durable memory types are **episodic** (events and observations),
**semantic** (standing facts), and **procedural** (repeatable methods).
Procedures have a stricter admission policy. Working memory belongs to the
agent's current runtime; runtime integration is still planned.

**Implementation stack:** Python 3.12+, PostgreSQL 17 with pgvector,
SQLAlchemy and Alembic, Pydantic, and FastAPI. Local sentence-transformers
models provide embeddings and reranking; Ollama provides LLM extraction.
Models may download on first use and run locally afterward.

## Engineering decisions and their evidence

| Decision | Reason | Implementation and evidence |
| --- | --- | --- |
| The LLM proposes; deterministic policy decides | Extraction confidence must not authorize a write | [Write policy](apps/memory_service/ingestion/write_policy.py), [policy tests](apps/tests/unit/ingestion/test_write_policy.py), [trust ADR](docs/adr/006-memory-trust-and-poisoning.md) |
| Provenance and tenant scope travel with each memory | A result must be attributable and scoped before ranking can matter | [Provenance verification](apps/memory_service/ingestion/provenance.py), [persistence tests](apps/tests/integration/persistence/test_memory_repository.py) |
| Records are revalidated after search | Search representations must not override current record eligibility | [Retrieval flow](docs/retrieval-flow.md), [temporal ADR](docs/adr/004-temporal-conflict-semantics.md) |
| Historical windows and explicit conflict links govern validity | Similarity and recency do not settle conflicting claims | [Temporal resolver](apps/memory_service/retrieval/temporal.py), [end-to-end tests](apps/tests/integration/retrieval/test_temporal_resolution.py) |
| Reranking scores a bounded shortlist and falls back on handled errors | Improve ordering while limiting the number of scored pairs | [Reranker](apps/memory_service/retrieval/reranker.py), [fallback and cap tests](apps/tests/unit/retrieval/test_reranker.py) |
| Context selection considers overlap and token cost | The highest-ranked notes may repeat one another or consume too much space | [Context packer](apps/memory_service/retrieval/context_packer.py), [budget and selection tests](apps/tests/unit/retrieval/test_context_packer.py) |
| Exact vector search precedes approximate indexing | Establish a measurable quality baseline before trading accuracy for speed | [Retrieval ADR](docs/adr/003-hybrid-retrieval.md), [search-quality tests](apps/tests/integration/retrieval/test_exact_search_quality.py) |

The API exposes intermediate evidence: `/retrieve` returns ranking signals,
reranking fallback information, and temporal exclusions; `/context` returns
selected memories, skip reasons, and budget statistics. These reports make
unexpected results traceable to a particular stage.

## What has been measured

These are **previously recorded development results**, not production-scale
claims or a fresh benchmark run. Each comparison uses the same candidate
pool for both strategies. Full methodology and caveats live in the linked
guides.

### Retrieval quality and reranking cost

Dataset: 24 memories, 16 queries with deliberately similar but incorrect
alternatives. **MRR** rewards placing the first relevant answer early;
**nDCG@5** measures top-five ordering against an ideal relevance order.
Higher is better. **P50/P95** are median and 95th-percentile latency.

| Strategy | MRR | nDCG@5 | Reranking P50 / P95 |
| --- | ---: | ---: | ---: |
| Hybrid search | 0.906 | 0.952 | — |
| Hybrid + CrossEncoder | 0.938 | 0.964 | 7.8 / 9.6 ms |

Both reached Recall@5 of 1.0 on this small corpus. The reranker improved two
first-answer positions and worsened one; timings depend on hardware and
exclude the experience of a cold model load. See
[retrieval measurements](docs/retrieval-flow.md#reranking-measured-quality-and-cost).

### Useful context under a tight budget

Dataset: 24 memories, four queries, real embeddings, subject keys enabled,
no reranking. At a **150-token budget**, coverage is the share of useful
information groups represented, and duplicate rate is the share of selected
notes repeating an already represented labeled group.

| Strategy | Duplicate rate ↓ | Useful-group coverage ↑ | Precision ↑ |
| --- | ---: | ---: | ---: |
| Fill in rank order | 0.33 | 0.50 | 0.78 |
| Context packer | 0.00 | 0.75 | 0.78 |

The tradeoff is visible at larger budgets: at 500 tokens, coverage is 1.0
for both strategies, but precision falls from 0.53 for rank order to 0.25
for packing. Without an absolute relevance threshold, the packer can use
remaining space on off-topic notes. See the
[full context-packing results](docs/context-packing.md#5-measurements-does-this-help).

After local setup, reproduce the evaluations with:

```bash
uv run python -m apps.benchmark.run_retrieval_eval
uv run python -m apps.benchmark.run_context_packing_eval
```

Both require PostgreSQL and the relevant local models. Longitudinal and
poisoning evaluation runners are planned; they are not available commands.

## Quick start

### 1. Start the database and development API

Prerequisites: Python 3.12+, `uv`, Docker with Compose, `curl`, and `jq`.
Run from the repository root. The prepared retrieval demo below does not
require Ollama; the first model use may require a download.

```bash
# Keep your existing .env if already configured.
[ -f .env ] || cp .env.example .env
uv sync --all-groups
docker compose up -d --wait postgres
uv run alembic upgrade head
uv run uvicorn apps.memory_service.api.app:app --reload --port 8000
```

Leave the API running. Its interactive reference is at
[localhost:8000/docs](http://localhost:8000/docs).

### 2. Create a tenant and load the prepared demo

Run in another terminal. A tenant is the account or organization that owns
its memories. Choose an unused name if you have run this before.

```bash
TENANT_ID=$(curl -fsS -X POST http://localhost:8000/tenants \
  -H 'Content-Type: application/json' \
  -d '{"name":"memory-walkthrough"}' | jq -er .tenant_id)

curl -fsS -X POST \
  "http://localhost:8000/tenants/$TENANT_ID/demo/seed" | jq
```

The seed creates and indexes prepared records, including old and new timeout
settings and recorded contradictions. It directly constructs demonstration
fixtures; it does not exercise LLM extraction or write-policy admission.
Seeding a tenant that already has active memories returns HTTP 409.

### 3. Compare historical and later retrieval

```bash
curl -fsS -X POST "http://localhost:8000/tenants/$TENANT_ID/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"query_text":"request timeout","as_of":"2026-02-15T00:00:00Z","limit":5,"rerank":true}' | jq

curl -fsS -X POST "http://localhost:8000/tenants/$TENANT_ID/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"query_text":"request timeout","as_of":"2026-04-01T00:00:00Z","limit":5,"rerank":true}' | jq
```

For the seeded timeout pair, February admits the 5-second setting; April
admits the replacement 2-second setting. Inspect `hits` and `pipeline` for
ranking and resolution details. Other matches can appear because retrieval
has no absolute relevance floor. Omit `as_of` to ask about now.

### 4. Inspect the context the agent would receive

```bash
curl -fsS -X POST "http://localhost:8000/tenants/$TENANT_ID/context" \
  -H 'Content-Type: application/json' \
  -d '{"query_text":"request timeout","as_of":"2026-04-01T00:00:00Z","token_budget":200,"candidates":30,"rerank":true}' \
  | jq '{context, skipped, stats}'
```

`context` is the formatted prompt text; `skipped` explains omissions;
`stats` reports selection counts and token usage. The budget holds according
to the configured counter. The default counter is an estimate; Python callers
can supply the agent's tokenizer for model-specific accounting.

### 5. Try ingestion from actual events

For the complete write path, start Ollama and pull `qwen2.5:7b-instruct`
(or configure `OLLAMA_MODEL`). The following uses the tenant created above:

```bash
ollama pull qwen2.5:7b-instruct

EVENT_ID=$(curl -fsS -X POST "http://localhost:8000/tenants/$TENANT_ID/events" \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"configuration","source_reference":"checkout-api/config-demo","content":"checkout-api max_retries=3"}' \
  | jq -er .event_id)

curl -fsS -X POST \
  "http://localhost:8000/tenants/$TENANT_ID/events/$EVENT_ID/ingest" | jq

curl -fsS -X POST \
  "http://localhost:8000/tenants/$TENANT_ID/memories/index" | jq
```

Inspect the ingestion outcome: model extraction can vary, and only accepted
memories become active. Indexing computes missing embeddings for active
memories. Repeat retrieval or context requests with the question
`checkout-api max_retries` to explore the new note. See the
[ingestion walkthrough](docs/ingestion-flow.md) for acceptance, quarantine,
and rejection examples.

## Implementation status

| Area | Implemented | Remaining work |
| --- | --- | --- |
| Storage and governance | Tenant-scoped records, provenance, trust, audit decisions, lifecycle statuses | Operational lifecycle automation |
| Ingestion | Ollama extraction, deterministic classification, normalization, verification, write policy | Automatic detection of contradictions and replacements |
| Retrieval | Full-text and exact vector search, rank fusion, bounded optional reranking | Approximate indexes when justified by evaluation |
| Temporal behavior | Historical queries, recorded supersession, trust-based conflict resolution | Future-dated replacement edge cases; optional return of unresolved claims in hits |
| Context packing | Duplicate removal, overlap-aware selection, budget accounting, skip reports | Calibrated relevance floor and improved subject-key quality |
| Agent integration | Development HTTP API and Python pipeline entry points | LangGraph runtime integration and working-memory lifecycle |
| Evaluation | Retrieval and context-packing comparisons; unit and integration tests | Larger datasets, longitudinal and poisoning evaluation runners |

Further limitations: provenance verification checks references and metadata,
not whether an extracted sentence is logically supported. Injection checks
use fixed phrase patterns. Automatic consolidation and forgetting/expiry
workers are not implemented. See [temporal limitations](docs/temporal-resolution.md#known-limitations)
and [packing limitations](docs/context-packing.md#6-limitations-to-keep-in-mind)
for the precise boundaries.

## Development and validation

```bash
uv run pytest apps/tests/unit
uv run pytest apps/tests/integration
uv run ruff check .
uv run mypy apps
```

Integration tests use a separate database by default (`agent_memory_test`
with the example configuration), create and drop its tables, and roll back
each test's transaction. Set `TEST_DATABASE_URL` only to a disposable test
database. Database-dependent tests skip if PostgreSQL is unreachable;
real-model tests also need the relevant models.

The tests include stale facts outranking current ones, conflicts whose other
side was not retrieved, tenant isolation, rejected evidence, reranker
fallback, and token counters that count joined text differently from its
parts. See [the test suite](apps/tests) and
[database operations](docs/database-operations.md) for local inspection.

## Explore the design

| Read next | What it explains |
| --- | --- |
| [Ingestion](docs/ingestion-flow.md) | How evidence becomes an accepted, quarantined, or rejected memory |
| [Retrieval](docs/retrieval-flow.md) | Word and meaning search, fusion, filters, and reports |
| [Reranking](docs/reranking.md) | Joint question–memory scoring, bounds, and fallback |
| [Temporal resolution](docs/temporal-resolution.md) | Historical validity, supersession, and contradictions |
| [Context packing](docs/context-packing.md) | Useful information under a prompt-space budget |
| [API reference](docs/api-reference.md) | Request fields, endpoints, and response examples |
| [Architecture](ARCHITECTURE.md) | System boundaries and design contracts |
| [Architecture decision records](docs/adr) | The reasoning and tradeoffs behind the design |
| [Database operations](docs/database-operations.md) | CLI access, SQL inspection, and fictional-company seed data |
| [Documentation standard](docs/conceptual-documentation.md) | How conceptual guides build from examples to technical detail |

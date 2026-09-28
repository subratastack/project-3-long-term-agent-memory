# ADR-001: PostgreSQL as the Memory Source of Truth

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

Agent memory contains more than text and embeddings. Each record needs tenant
scope, provenance, trust, temporal validity, conflict and supersession links,
write-decision history, and deletion state. Vector engines optimize similarity
search but do not, by themselves, provide the transactional and relational
semantics required to decide whether a memory is authorized, current, or true.

The project also compares an optional external TurboVec index with pgvector.
An external index creates a failure window in which the authoritative write and
the index update cannot be one atomic transaction.

## Decision

PostgreSQL is the sole authoritative store for events, candidates, accepted
memory records, provenance, relations, conflicts, write decisions, tombstones,
and access logs.

PostgreSQL FTS and pgvector are derived access paths over authoritative rows.
Exact pgvector search is the initial baseline. Any external vector index stores
only stable memory IDs and derived vectors, has explicit synchronization state,
is rebuildable, and must have every result revalidated through PostgreSQL.

All authoritative writes and their provenance links share a database
transaction. Indexing failure cannot roll back or redefine the memory record;
it produces retryable index state instead.

## Consequences

### Positive

- Tenant, temporal, trust, conflict, and provenance rules can be enforced with
  constraints and transactions.
- Relational and vector state remain colocated for the pgvector baseline.
- Audit history survives reindexing and model changes.
- External indexes can be evaluated or replaced without migrating memory
  authority.

### Negative

- PostgreSQL schema design and migrations are required before semantic search.
- An external index requires idempotent sync, reconciliation, and ghost-entry
  cleanup.
- PostgreSQL may not be the fastest vector engine at every scale; benchmarks
  determine whether that trade-off matters.

## Rejected alternatives

- **Vector database as source of truth:** insufficient for the required
  relational, temporal, and audit semantics.
- **Transcript/object storage as source of truth:** preserves evidence but does
  not represent governed, queryable memory state.
- **Dual authority between PostgreSQL and an external index:** creates
  irreconcilable ambiguity after partial failures.

## Validation

Tests must cover transactional record/provenance writes, missing external index
updates, ghost index entries, rebuilds, tenant revalidation, and model/index
version changes.

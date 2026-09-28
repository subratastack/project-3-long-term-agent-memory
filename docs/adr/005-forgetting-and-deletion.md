# ADR-005: Forgetting and Deletion

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

Unbounded memory increases cost, noise, privacy risk, and stale-fact exposure.
Immediate hard deletion, however, destroys auditability and can allow derived
indexes or consolidation jobs to recreate removed content.

## Decision

Forgetting is represented by explicit policy transitions: retrieval-priority
decay, expiry, archive, consolidation, or tombstone. These states are enforced
before ranking. Tombstones identify the memory, reason, policy/requester, and
time so reconciliation cannot resurrect it.

Hard deletion is reserved for explicit retention or privacy requirements. A
purge removes authoritative content, provenance payloads as required, cached
material, embeddings, and external index entries. It retains only the minimum
audit evidence permitted by policy. Compaction may replace redundant episodes
with a consolidated memory only when evidence links and applicable retention
requirements are preserved.

## Consequences

- Low-value memory can stop affecting agents without immediately losing audit
  history.
- Deletion propagates predictably to derived systems.
- Storage reclamation is delayed until compaction or an authorized purge.
- Policies must define retention by memory/source class and tenant.

## Rejected alternatives

- **Hard-delete first:** undermines audit, conflict explanation, and recovery.
- **Score decay only:** does not satisfy deletion or prevent resurrection.
- **Delete only from PostgreSQL:** leaves sensitive or ghost data in indexes and
  caches.

## Validation

Test expiry boundaries, archived-record exclusion, decay behavior, idempotent
tombstones, purge propagation, reconciliation after deletion, and prevention of
re-promotion from tombstoned evidence.

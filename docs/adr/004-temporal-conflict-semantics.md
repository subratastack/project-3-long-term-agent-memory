# ADR-004: Temporal and Conflict Semantics

- **Status:** Accepted
- **Date:** 2026-09-24

## Context

Long-lived facts change. The time a statement was observed is not always the
time it became valid, and a newer record is not automatically more trustworthy.
Silent overwrite loses history; similarity ranking can otherwise surface an
obsolete but highly matching fact.

## Decision

Every durable claim records `valid_from`, optional `valid_to`, observation time,
and system creation time. Records can explicitly `supersede` or
`conflict_with` other records. Supersession closes the earlier active validity
interval without deleting that record. Conflicts retain both claims and a
resolution state, rationale, resolver identity/policy, and resolution time.

Retrieval accepts an optional `as_of` time and excludes records invalid at that
time. Active facts are selected by validity, trust, source authority, and
recorded resolution before relevance scores. Unresolved conflicts are omitted
from ordinary context unless the caller explicitly requests them.

## Consequences

- Historical queries and explanations remain possible.
- Stale facts cannot win solely through embedding similarity.
- Writes require conflict detection and transactional relationship updates.
- Overlapping intervals and incomparable authorities may require quarantine or
  review rather than an automatic winner.

## Rejected alternatives

- **Last write wins:** conflates ingestion time with validity and authority.
- **Overwrite in place:** destroys evidence and historical state.
- **Resolve during retrieval from score alone:** makes correctness dependent on
  query wording.

## Validation

Test old/new addresses, changed timeouts, contradictory user assertions,
trusted system events with different timestamps, future validity, historical
`as_of` queries, and unresolved conflict exclusion.

# Temporal and conflict resolution

**Temporal resolution decides which retrieved memories are usable at the
time being asked about. Conflict resolution decides what to withhold when
recorded claims disagree.** Search relevance alone cannot answer either
question.

Start with the changing timeout below. Then follow the rules and code
reference for exact boundary behavior and conflict handling.

## 1. Why a good search match can still be the wrong fact

Suppose checkout-api used to have a five-second timeout and now has a
two-second timeout. Both notes can match “checkout-api timeout.” Returning
the old one as current would mislead the agent, even if it ranks first.

The system needs two different questions:

- **Relevance:** does this note match what was asked?
- **Validity:** is this note in effect at the requested time, given its
  dates and recorded relationships?

A **validity window** is the period during which a stored memory applies.
**Supersession** means one memory has explicitly replaced another.
A **contradiction** is a recorded relationship saying two claims disagree.
These relationships are stored links, not disagreements inferred by this
stage from text.

```text
Ranked candidates → check dates → check replacements → check contradictions
                  → surviving candidates + explanation report
```

The stage uses recorded evidence and rules. It cannot independently prove
that a statement is true in the real world.

## 2. Walk through a changing configuration

Assume both records belong to the same tenant and pass the query's other
filters. The replacement has been explicitly recorded:

| Memory | Content | Valid from | Valid until |
| --- | --- | --- | --- |
| A: old setting | checkout-api timeout is 5 seconds. | January 1, 2026 | March 1, 2026 |
| B: new setting | checkout-api timeout is 2 seconds. | March 1, 2026 | No end recorded |

The dates in this example mean midnight UTC. At the changeover instant,
B takes over and A stops applying:

```text
January 1                    March 1                      April 1
A: 5 seconds [----------------)                               |
B: 2 seconds                 [-------------------------------→
                             changeover: B only
```

The `[` includes the start; the `)` excludes the end. This is called a
**half-open window**. The precise rule is `valid_from <= t < valid_to`, where
`t` is the requested time; if there is no end, only the start is checked.

### Step A: decide which time the question means

A query can specify `as_of`, meaning “answer using facts valid at this
instant.” Without it, the query asks about now.

| Query | Setting eligible to survive | Why |
| --- | --- | --- |
| `as_of=2026-02-15T00:00:00Z` | A: 5 seconds | A's window covers February 15; B has not started |
| `as_of=2026-03-01T00:00:00Z` | B: 2 seconds | A's end is excluded; B's start is included |
| `as_of=2026-04-01T00:00:00Z` | B: 2 seconds | Only B's window covers April 1 |
| No `as_of`, with now after March 1 | B: 2 seconds | B is active; A has been marked superseded |

These outcomes assume the relevant memory was retrieved and no other
conflict excludes it. Temporal resolution removes invalid candidates; it
does not search for missing answers to fill the list.

### Step B: retain history without showing it as current

`UnitOfWork.supersede_memory` keeps A's row, marks it `SUPERSEDED`, closes
its window at B's start, and records a replacement link from B to A.
Keeping the old record lets February questions still use the five-second
setting. An ordinary current query only admits `ACTIVE` records; a historical
query also admits `SUPERSEDED` and `EXPIRED` records whose windows fit.

### Step C: preserve the order of surviving candidates

If A matches the query better than B, that score still cannot make A survive
an April query. The resolver removes disallowed candidates and preserves
the relevance order of those left. It returns an exclusion report for notes
it removes; notes already filtered out by search do not appear as new
exclusions in this stage's report.

## 3. What if two current notes disagree?

Now consider a separate example: two current records with overlapping windows
say checkout-api's timeout is 2 seconds and 5 seconds. Neither replaces the
other, but a `CONTRADICTION` link has been recorded between them.

**Trust** is an ordered source-reliability label in this system:
`SYSTEM > HIGH > MEDIUM > LOW > UNTRUSTED`. It is not a search score.

| Trust on the 2-second claim | Trust on the 5-second claim | Result for this pair |
| --- | --- | --- |
| `SYSTEM` | `MEDIUM` | Keep the 2-second claim if it is a candidate; withhold the 5-second claim |
| `MEDIUM` | `MEDIUM` | Withhold both claims and report an unresolved conflict |

A newer timestamp or better search score does not break an equal-trust tie.
Both records remain stored; withholding affects the returned results.

Even if search returned only the 2-second claim, the resolver loads the
linked 5-second claim to check the disagreement. Loading that neighbor does
not add it to the result list. Without a recorded contradiction link, this
stage does not detect the disagreement just by reading their text.

## 4. Technical reference

### Where the code lives

| Piece | Location | What it does |
| --- | --- | --- |
| `apply_temporal_resolution` | `apps/memory_service/retrieval/temporal.py` | The entry point the pipeline calls. Loads every relation touching a candidate (and touching the *other* side of those edges), fetches any memory those edges point at, then calls `select_in_effect`. Two relation queries per search, plus one fetch per referenced memory that wasn't already a candidate. |
| `select_in_effect` | `apps/memory_service/retrieval/temporal.py` | Pure decision logic -- no database. Given candidates, relations, and records, returns the survivors (in their original order) and a `TemporalReport`. |
| `TemporalReport` | `apps/memory_service/retrieval/temporal.py` | `effective_at`, `candidates_in`, `excluded` (one `Exclusion` per removed memory), `conflicts` (every contradiction considered), and the `unresolved_conflicts` property. |
| `Exclusion` / `ExclusionReason` | `apps/memory_service/retrieval/temporal.py` | Which memory was removed, why, and (`related_memory_id`) which memory was responsible -- the successor for a supersession, the other side for a conflict. |
| `resolve_contradiction` | `apps/memory_service/consolidation/conflict_resolver.py` | Pure verdict on one `CONTRADICTION` edge between two in-effect memories: strictly higher trust wins, equal trust is unresolved. |
| `ConflictOutcome` / `ConflictState` | `apps/memory_service/consolidation/conflict_resolver.py` | The verdict: `RESOLVED_BY_TRUST` (with `winner_id` / `loser_id`) or `UNRESOLVED`, a human-readable `reason`, and `withheld_ids`. |
| `record_contradiction` | `apps/memory_service/consolidation/conflict_resolver.py` | Write helper: adds a `CONTRADICTION` edge between two existing memories. Doesn't commit. |
| `HISTORICAL_STATUSES` | `apps/memory_service/retrieval/filters.py` | The statuses an `as_of` query may see: `ACTIVE`, `SUPERSEDED`, `EXPIRED`. |
| `MemoryRelationRepository.list_for_memories` | `apps/memory_service/persistence/repositories.py` | Batched relation lookup (one query for a whole candidate set), tenant-scoped. |
| Pipeline integration | `apps/memory_service/retrieval/hybrid.py` (`hybrid_search_with_report`) | Runs the stage after reranking and before truncating to `query.limit`; the report is on `HybridSearchResult.temporal`. |

### How it plugs into hybrid search

```text
lexical + semantic search   (hard filters applied in SQL, records revalidated)
  -> RRF fusion -> top candidate_limit
  -> optional CrossEncoder reranking
  -> apply_temporal_resolution          <- this stage
  -> truncate to query.limit
```

It runs over the *whole* candidate pool, not just the final `limit`, so
removing a stale fact can make room for the next valid candidate. It does
not refill the pool from search, so the result may still be shorter than
`limit`. It always runs; unlike the reranker it isn't
optional, because it decides correctness rather than quality.

### Exact rules

#### 1. Which statuses a query can see

| Query | Allowed statuses | Rationale |
| --- | --- | --- |
| Ordinary (no `as_of`) | `ACTIVE` | "What is true now." A retired memory is invisible to ordinary reads. |
| Historical (`as_of` given) | `ACTIVE`, `SUPERSEDED`, `EXPIRED` | "What was true then." A fact replaced *later* was still true at the time, so its validity window decides, not its present-day status. |
| Either | never `QUARANTINED` or `TOMBSTONE` | Never trusted, or deliberately forgotten -- at any point in time. |

This is set in `resolve_filters`, so lexical search, semantic search, and
this stage all apply the same rule.

#### 2. Validity windows are half-open

A memory is in effect at time `t` when `valid_from <= t < valid_to` (or
`valid_to` is open). Supersession closes the old window exactly where the
new one opens, and the SQL filters in the retrievers treat `valid_to` as
inclusive -- so at that boundary both rows can reach this stage on a
historical query. Ordinary queries may already exclude the superseded row by status.
The half-open check here guarantees only the new one survives.

#### 3. Supersession

A `SUPERSESSION` edge (new -> old) retires the old memory at `t` **if the
new one is in effect at `t`**. So:

- a replacement that hasn't started yet doesn't hide the fact it will
  replace;
- the edge alone is enough -- the old one is removed even if its own row
  was never updated (status still `ACTIVE`, window still open).

#### 4. Contradictions

A `CONTRADICTION` edge only matters when **both** sides are in effect at
`t` and neither is superseded. Then `resolve_contradiction` decides:

| Situation | Outcome |
| --- | --- |
| One side has strictly higher trust (`SYSTEM > HIGH > MEDIUM > LOW > UNTRUSTED`) | `RESOLVED_BY_TRUST`: it's kept, the other is removed (`LOST_CONFLICT`). |
| Equal trust | `UNRESOLVED`: both removed (`UNRESOLVED_CONFLICT`) and reported. |

What it deliberately does **not** use:

- **Scores** -- lexical rank, cosine distance, fused score, and rerank
  score are all ignored. A better match isn't more true.
- **Recency** -- a newer memory doesn't automatically win a tie. ADR-004
  rejects "last write wins": when a memory became valid says nothing about
  who is right.

Two more details:

- The other side of an edge is fetched from PostgreSQL even if search
  never returned it -- whether a fact is contested must not depend on how
  the query was worded, or on its type filter.
- The other side only needs to be in effect (right tenant, allowed status,
  window covers `t`). Its trust level and type don't matter: the query's
  trust floor and type filter narrow what *it* is shown, not which facts
  are true.

## Exclusion reasons

| `ExclusionReason` | Meaning | `related_memory_id` |
| --- | --- | --- |
| `NOT_ELIGIBLE` | Wrong tenant, or a status / trust level / type the query doesn't allow. | -- |
| `NOT_IN_EFFECT` | Its window doesn't cover `effective_at` (not yet valid, or ended). | -- |
| `SUPERSEDED` | A newer memory that replaced it is in effect. | the successor |
| `LOST_CONFLICT` | Contradicts a more trusted memory that is in effect. | the winner |
| `UNRESOLVED_CONFLICT` | Contradicts an equally trusted memory that is in effect. | the other side |

Most `NOT_ELIGIBLE` and `NOT_IN_EFFECT` cases are already filtered out
upstream by the retrievers; this stage re-checks them as defence in depth,
and it is the only place the half-open boundary is enforced.

## Usage

```python
from datetime import UTC, datetime

from apps.memory_service.retrieval.hybrid import hybrid_search_with_report
from apps.memory_service.retrieval.query_model import RetrievalQuery

query = RetrievalQuery(
    tenant_id=tenant_id,
    query_text="request timeout",
    as_of=datetime(2026, 2, 15, tzinfo=UTC),  # omit for "now"
)
with uow_factory() as uow:
    result = hybrid_search_with_report(uow, embedder, query, reranker=reranker)

for hit in result.hits:                        # the active truth, in relevance order
    print(hit.memory.content)

for exclusion in result.temporal.excluded:     # what was removed, and why
    print(exclusion.memory_id, exclusion.reason, exclusion.related_memory_id)

for conflict in result.temporal.unresolved_conflicts:   # needs review
    print(conflict.memory_ids, conflict.reason)
```

`uow_factory`, `embedder`, and `reranker` are set up as in
[Retrieval](retrieval-flow.md#trying-it) and [Reranking](reranking.md#usage).
`tenant_id` must be an existing tenant UUID.
`hybrid_search` (without `_with_report`) returns just the filtered hits.

### Recording a contradiction

```python
from apps.memory_service.consolidation.conflict_resolver import record_contradiction

with uow_factory() as uow:
    record_contradiction(
        uow, tenant_id, timeout_two.memory_id, timeout_five.memory_id,
        rationale="same service and validity period, different timeout values",
    )
    uow.commit()
```

`timeout_two` and `timeout_five` stand for the two existing, same-tenant
records in the conflict example. Both memories are kept (ADR-004: "Conflicts
retain both claims"). Later retrieval checks the recorded link.

### Settling an unresolved conflict

There's no separate "resolved" flag: retire the losing memory -- supersede,
quarantine, or tombstone it. Once only one side is in effect, the edge no
longer withholds anything. Alternatively, if one side genuinely comes from a
more authoritative source, its trust level decides automatically.

## Known limitations

- **Contradictions aren't detected automatically yet.** Retrieval honours
  `SUPERSESSION` and `CONTRADICTION` edges, but ingestion doesn't create
  them -- `ingestion/write_policy.py` evaluates each candidate in isolation.
  Until write-time detection exists, edges come from `supersede_memory` and
  `record_contradiction`. Guessing conflicts from shared `subject_keys` was
  deliberately avoided: it would withhold valid facts.
- **Future-dated supersession and "now" queries.** If a replacement's
  `valid_from` is still in the future, `supersede_memory` already marks the
  old memory `SUPERSEDED`. An ordinary query (no `as_of`) only sees
  `ACTIVE`, and the new one isn't valid yet, so it returns neither. Passing
  `as_of=<now>` explicitly returns the old fact correctly. Closing this gap
  means letting ordinary queries see `SUPERSEDED` rows too, which changes
  the "only ACTIVE is visible to ordinary reads" rule in `ARCHITECTURE.md`.
- **No opt-in for conflicting facts in `hits`.** ADR-004 allows callers to
  explicitly request unresolved conflicts; today they are always withheld
  from `hits` and available only via `result.temporal.conflicts`.

## Tests

- `apps/tests/unit/retrieval/test_temporal.py` -- `select_in_effect` with no
  database: February / boundary / April / current for the timeout example,
  expired and not-yet-valid windows, tombstoned and quarantined (including
  historical queries), other tenant, trust floor, supersession by edge
  alone, a not-yet-valid successor, unresolved and trust-resolved
  contradictions, a contradiction whose other side wasn't retrieved, is
  retired, or is out of its window, and preserved relevance order.
- `apps/tests/unit/consolidation/test_conflict_resolver.py` -- equal trust is
  unresolved, higher trust wins from either side, recency doesn't break a
  tie, and invalid inputs (wrong relation type, wrong records, cross-tenant)
  are rejected.
- `apps/tests/unit/retrieval/test_filters.py` -- ordinary vs. `as_of`
  allowed statuses.
- `apps/tests/integration/retrieval/test_temporal_resolution.py` -- the full
  pipeline against PostgreSQL (with the fake embedder and CrossEncoder):
  old vs. newer timeout at every point in time, a stale fact losing despite
  winning on lexical and rerank score, old vs. newer address, expired /
  tombstoned / quarantined exclusion, unresolved contradictions withheld and
  surfaced, higher trust winning a contradiction, and tenant isolation.
  Needs `docker compose up -d postgres`; skipped otherwise.

For the surrounding pipeline, see [Retrieval](retrieval-flow.md),
[Reranking](reranking.md), and [Context packing](context-packing.md).
[ADR-004](adr/004-temporal-conflict-semantics.md) records the temporal and
conflict semantics.

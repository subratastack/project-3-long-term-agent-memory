# Grouping related episodes

Consolidation starts by finding past observations that can be discussed together.
An **episodic memory** records something that happened. A **cluster** groups
related episodic memories without changing or deleting them. The clusterer
receives stored records and returns groups; it does not write summaries or
decide whether repeated observations justify a procedure.

Grouping by explicit fields makes the first version explainable: two episodes
belong together because their tenant, subject, category, and dates match,
rather than because a model judged their wording similar.

## Follow a checkout-api incident series

The dates and labels below are illustrative, not measurements. Assume the job
runs on September 28, 2026, all times are UTC, and each record has one source
event observed at midnight on the listed date. A **tenant** is an isolated
workspace. A **subject key** identifies what a memory concerns.

| Episode | Tenant | Subject key | Category key | Observed | Status |
| --- | --- | --- | --- | --- | --- |
| A | shop-a | service:checkout-api | connection-pool exhaustion | Sep 1 | active |
| B | shop-a | service:checkout-api | connection-pool exhaustion | Sep 8 | active |
| C | shop-a | service:checkout-api | connection-pool exhaustion | Sep 20 | active |
| D | shop-b | service:checkout-api | connection-pool exhaustion | Sep 8 | active |
| E | shop-a | service:checkout-api | deployment failure | Sep 9 | active |
| F | shop-a | service:checkout-api | connection-pool exhaustion | Sep 10 | quarantined |

Assume active records have validity windows that include the job time. A, B,
and C share a tenant, subject, and category. Their span is 19 days, within
the default 30-day limit, so they form one cluster. D belongs to another
tenant and forms its own cluster. E has another category and forms another.
F is excluded because quarantine means the record is not eligible for use.
A one-record cluster is valid output here; the promoter later refuses to
turn it into long-term knowledge.

The main cluster contains A, B, and C, the common subject
`service:checkout-api`, and observed bounds of September 1 and September 20.
Its stable identifier is derived from the tenant and member IDs. A–F and
shop-a/shop-b are teaching labels; real identifiers are UUIDs.

## What “related” means precisely

The grouping key comes from `metadata["incident_key"]` when it is a nonempty
string; otherwise the code tries `metadata["category_key"]`. The selected
field name and its stripped value must both match. An incident key of `pool`
and a category key of `pool` are different keys. If unique incident IDs are
present, they take precedence over a shared category: that can keep separate
incidents apart even when their categories match.

The common subject must survive across the whole group. Suppose records in
observation order have subject sets `{a}`, `{a, b}`, and `{b}`. The first two
join with common subject `{a}`. The third cannot join, because it shares no
subject with that intersection. A bridging record does not merge unrelated
subjects.

The time limit also applies to the whole group, not just neighboring records.
Observations on August 1, August 21, and September 10 are each 20 days apart,
but the full span is 40 days. The first two join; the third starts another
cluster under the 30-day default. Exactly 30 days is allowed.

For a proposed group, the rule is:

```text
latest supporting observation − earliest supporting observation <= window
```

Both bounds come from **provenance**, the event references carried by each
record. All provenance timestamps count, not just the record's creation date.
A single record whose evidence spans more than the window is excluded.
This is a span limit, not a “last 30 days” lookback: older evidence can still
cluster when its records remain active and valid.

## How the implementation builds groups

1. Collapse identical repeated memory IDs; conflicting records with the same
   ID raise `ValueError`.
2. Sort by earliest provenance observation, then by memory ID string.
3. Exclude records that are not active episodic memories, are outside their
   validity window, have future observations, lack a grouping key, or lack
   nonempty subject keys.
4. Put each remaining record in the first compatible cluster, narrowing the
   common subject set. If none matches, start a cluster.
5. Recompute observed bounds and the identifier from the members.

Validity starts inclusively and ends exclusively: `valid_from <= now < valid_to`.
An absent end means open-ended validity. Quarantined, superseded, expired, and
tombstoned statuses are excluded regardless of their dates. Naive timestamps
are treated as UTC; aware timestamps are converted to UTC.

## Technical reference and limits

Implementation: [clusterer.py](../apps/memory_service/consolidation/clusterer.py).

`cluster_memories(memories, *, now=None, window=timedelta(days=30))` returns a
list of `MemoryCluster` objects. A nonpositive window raises `ValueError`.
`now=None` uses the current UTC time. Each cluster exposes:

| Field | Meaning |
| --- | --- |
| `cluster_id` | UUID5 derived from a version prefix, tenant, and sorted member IDs |
| `tenant_id` | The single tenant represented |
| `memories` | Original records in deterministic processing order |
| `shared_subject_keys` | Sorted intersection of members' subject keys |
| `earliest_observed_at`, `latest_observed_at` | Bounds across all supporting event observations |

There are no embeddings, text-similarity scores, or database calls. Stable
inputs give stable groups even if their input order changes. The greedy
first-match rule does not find an optimal partition; adding a record can
change membership. Keys require upstream consistency, and trust verification
belongs to later stages. Excluded records do not receive individual reports.

[Clusterer tests](../tests/unit/consolidation/test_clusterer.py) cover tenant,
subject, and category separation; lifecycle exclusions; duplicate input;
non-chaining windows; and 50 related records forming one cluster.
Next read [promotion](consolidation-promotion.md) for whether a group justifies
a candidate, or [the service](consolidation-service.md) for the whole flow.

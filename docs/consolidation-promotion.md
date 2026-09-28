# Deciding what repeated evidence supports

Promotion decides whether a cluster supports a recurring observation, a
procedure, or neither. It receives a group of episodes and returns a proposed
memory type. It does not write a memory or override the write policy.

A repeated observation is a weaker claim than a reusable procedure. “The pool
keeps exhausting” can remain true even if the attempted remedies failed.
“Follow these steps” needs repeated successful outcomes for those same steps.
The promoter makes that distinction with explicit evidence rules.

## Walk through the decision

Use three illustrative checkout-api pool incidents, A, B, and C, observed on
September 1, 8, and 20, 2026. Assume the group is valid under the clustering
rules, all record and provenance trust levels are at least medium, and each
episode references a different event. Dates and labels here are teaching data.

| Evidence supplied | Promotion result | Why |
| --- | --- | --- |
| A alone | No candidate | One observation does not establish repetition |
| A and B | Semantic | Two independent observations reach the semantic threshold |
| A, B, C; all outcomes successful; identical steps | Procedural | Three independent episodes and unanimous successful remediation |
| A, B successful; C failed | Semantic | Repetition remains observable, but the remedy has mixed outcomes |
| A, B, C successful; different or missing steps | Semantic | No shared procedure can be rendered |
| Any record or reference below medium trust | No candidate | Repetition does not repair weak evidence |

In the procedural case, each record must explicitly have
`metadata["remediation_outcome"] == "success"` and the same nonempty
`metadata["remediation_steps"]` list. Text saying “helped” is not parsed into
an outcome. Missing values, `"unknown"`, `"failed"`, and differently spelled
success values do not satisfy the success test.

The promoter selects at most one type. It does not produce both a semantic
and a procedural candidate for the same cluster. Its no-candidate result leaves
the source episodes alone; it does not move them into quarantine.

## Count independent evidence, not copies

**Provenance** is the set of references from an episode to its source events.
Three stored records citing one source event should not count as three
independent observations. The algorithm tracks events it has already seen.

For illustrative ordered records with event sets `{e1}`, `{e1, e2}`, `{e3}`:

| Record | Already seen before it | Adds one to the count? |
| --- | --- | --- |
| First: {e1} | empty | Yes: count becomes 1 |
| Second: {e1, e2} | {e1} | No: some evidence overlaps |
| Third: {e3} | {e1, e2} | Yes: count becomes 2 |

Every event enters the seen set even when its record does not increment the
count. Thus a later record containing only e2 would also not increment it.
This is deliberately conservative; it is not a statistical independence test
or a search for the largest possible collection of disjoint records.

The count uses the cluster's deterministic observation-time order. Duplicate
input IDs are removed by clustering. A semantic candidate needs at least two
counted episodes; a procedural candidate needs at least three. The success
and matching-step checks still apply to **all** members, including members
that did not add to the independent count.

## Why approval still comes later

The promoter first rebuilds the cluster with the default clustering rules and
requires exact equality with the supplied cluster. It refuses malformed groups,
changed bounds, cross-tenant groups, and records that are no longer eligible.
It then checks record and provenance trust, counts evidence, and chooses a type:

```text
invalid cluster or trust below medium → no candidate
independent count < 2                 → no candidate
count >= 3 + all success + same steps → procedural
otherwise                            → semantic
```

These checks inspect stored records. They do not independently fetch the real
events. The service does that before running the existing write policy.
For example, a procedure whose references claim system trust but originate
from tool output can pass promotion and then be quarantined: the current
policy derives at most medium trust from that source type and requires high
trust for procedural acceptance. Missing or mismatched source events cause
rejection instead.

There are two distinct thresholds today: consolidation requires three
independent successful episodes for a procedure, while the general write
policy's procedural evidence-count threshold is two provenance entries
(with an explicit-approval alternative). Consolidation's stricter gate runs
first; it does not change the general ingestion rules or manufacture approval.

## Technical reference and limitations

Implementation: [promoter.py](../apps/memory_service/consolidation/promoter.py).

`promote_cluster(cluster, *, now=None)` returns `MemoryType.SEMANTIC`,
`MemoryType.PROCEDURAL`, or `None`. Constants are
`MIN_EPISODES_FOR_SEMANTIC = 2` and
`MIN_SUCCESSFUL_EPISODES_FOR_PROCEDURAL = 3`. There is no reason-code result
object. The service skips `None` results without writing a policy audit entry.

Revalidation uses the clusterer's default 30-day window; this function has no
custom-window argument. Exact cluster equality includes member order and the
computed fields, so callers should use clusters produced by the clusterer.

Outcome metadata is taken as recorded; the promoter does not verify causality,
parse incident prose, or prove a successful remedy is safe in future conditions.
A failed episode prevents procedure promotion only within its own cluster;
the component does not search other clusters for contradictory outcomes.

[Promoter tests](../tests/unit/consolidation/test_promoter.py) cover the one/two
threshold, success versus mixed/unknown outcomes, untrusted evidence, repeated
events, and forged tenant identity. [Service tests](../tests/unit/consolidation/test_service.py)
cover the later policy rejection and quarantine paths.
Read [summary construction](consolidation-summaries.md) for candidate fields
and [service orchestration](consolidation-service.md) for durable writes.

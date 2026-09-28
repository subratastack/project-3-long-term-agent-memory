# Building bounded summary candidates

A group of incident reports can be useful evidence without being useful prompt
text. The summarizer turns a cluster into one short proposed statement while
retaining the references needed to inspect every source. Its output is a
**candidate**: a proposed memory that still requires write-policy approval.
It never writes an authoritative memory itself.

A **semantic** candidate describes a recurring observation. A **procedural**
candidate describes steps to consider. These are different claims: repeated
failures can support an observation without proving a remedy works.

## From three incidents to a statement

This illustrative example continues [the clustering guide](consolidation-clustering.md).
Episodes A, B, and C describe checkout-api pool exhaustion on September 1, 8,
and 20, 2026. Each has subject `service:checkout-api` and metadata category
`connection-pool exhaustion`. Assume they are valid, same-tenant records,
with separate event references eA, eB, and eC. The labels are placeholders
for actual UUIDs.

Their prose might say:

| Episode | Original report |
| --- | --- |
| A | Pool exhausted; increasing pool size resolved it. |
| B | Pool exhausted during peak load; pool increase helped. |
| C | Pool exhaustion; monitoring showed saturation. |

The default semantic template produces:

```text
service:checkout-api has recurring connection-pool exhaustion (3 episodes).
```

It does not add “during peak load.” That condition appears in one report,
but the current summarizer does not interpret prose or infer shared causes.
It uses the category and the first sorted common subject key.

For a separate procedural variant, suppose all three records explicitly carry
this same structured list, and the promoter has allowed a procedural candidate:

```json
{"remediation_steps": ["inspect saturation metrics", "evaluate pool sizing"]}
```

The output is:

```text
For service:checkout-api / connection-pool exhaustion: inspect saturation metrics; then evaluate pool sizing.
```

The summarizer only renders the supplied steps. Whether the episodes record
successful outcomes is checked by [the promoter](consolidation-promotion.md),
not by this function. Calling the summarizer directly is not promotion approval.

## Short text, complete evidence

The candidate carries three complementary forms of evidence:

| Location | Illustrative contents | Question it answers |
| --- | --- | --- |
| `provenance` | Full copies of the references to eA, eB, eC | Which source events support this candidate? |
| `metadata.supporting_memory_ids` | A, B, C, sorted as ID strings | Which stored episodes were consolidated? |
| `metadata.episode_event_ids` | A → [eA], B → [eB], C → [eC] | Which events belong to each supporting episode? |

If an episode has several references, all are retained. Repeated event
references are not deduplicated here. These memory-ID links are JSON metadata,
not `MemoryRelation` edges with database foreign keys. The persisted event
provenance does use the existing provenance repository.

This distinction matters for 50 similar episodes: the statement remains short,
but evidence storage still grows with the number of sources. “Bounded” refers
to the summary content, not the whole candidate or its serialized size.

## Bounds, confidence, and time

The selected subject key and category value may each contain at most 160
characters. The complete content may contain at most 1,000 characters,
including template punctuation. These are Python string-length limits, not
token limits. Oversized content raises `ValueError`; the code does not truncate
steps, split the cluster, or call a model to shorten it.

Confidence is the minimum input-record confidence. For illustrative input
values 0.92, 0.85, and 0.90, the candidate receives 0.85. Repetition does not
increase confidence, and these numbers are not measured correctness rates.
Proposed trust is the weakest level across record trust and provenance trust;
the write policy later derives trust again from actual events.

The candidate's validity starts at the later of the latest observed evidence
and every member's validity start. If the latest observation is September 20
but one record becomes valid September 22, the candidate starts September 22.
Its end is the earliest finite member end, or open-ended if no member has an
end. This avoids extending the candidate past a supporting record's expiry.

## Technical reference and failure handling

Implementation: [summarizer.py](../apps/memory_service/consolidation/summarizer.py).

`summarize_cluster(cluster, memory_type=MemoryType.SEMANTIC)` returns a
`MemoryCandidate`. It checks that the cluster is nonempty, every member matches
the cluster tenant, grouping keys agree, and claimed common subjects occur in
every member. It only accepts semantic or procedural output types.

For procedures, `remediation_steps` must be a nonempty list of nonblank strings
and match across all records **before** whitespace is stripped for rendering.
Different order or different whitespace can therefore prevent a match. The
candidate ID is UUID5 derived from the cluster ID, summary version, and type.
The candidate also copies all common subjects and records `consolidation_version`,
`cluster_id`, and the selected category/incident key in metadata.

This component assumes the caller supplies an eligible cluster; it does not
recheck all lifecycle, time-span, or promotion rules. It does not load events,
prove that metadata follows from source content, or interpret free text.
No Ollama call exists. A future wording model would still only propose text;
policy and provenance checks would remain necessary.

[Summarizer tests](../tests/unit/consolidation/test_summarizer.py) cover complete
evidence for 50 episodes, procedural wording, oversized-procedure rejection,
and cross-tenant rejection. These are behavioral tests, not a token-savings
benchmark. [The service guide](consolidation-service.md) explains how candidate
errors, policy decisions, and persistence are handled.

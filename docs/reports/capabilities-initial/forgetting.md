# Forgetting and retention benchmark

Forgetting must remove unwanted memories from future prompts while preserving appropriate history and audit records. An expired incident can still answer a historical question. A deliberately tombstoned incident and summaries derived from its evidence must disappear from both current and historical search. This run exercises the production lifecycle service and PostgreSQL.

These are measured results from 2026-10-04T00:16:33.383649+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Pass rate counts explicitly labeled lifecycle and search checks. The cases check the 30-day decay grace period, 90-day half-life, priority floor, confidence threshold, persistent fact retention, expiry boundary, quarantine, compaction, deletion cascade, tenant isolation, idempotence, and evidence reuse prevention. Retrieval priority is a relative ranking weight, not a probability.

## Measured results

| Metric | Measured value |
| --- | --- |
| checks | 17 |
| passed | 17 |
| pass_rate | 1 |

| Check | Success |
| --- | --- |
| episode-grace-boundary | True |
| episode-one-half-life | True |
| episode-two-half-lives | True |
| episode-priority-floor | True |
| high-confidence-retention | True |
| semantic-retention | True |
| procedural-retention | True |
| expiry | True |
| expiry-history | True |
| inclusive-valid-to | True |
| quarantine-preserved | True |
| derived-summary-compaction | True |
| tombstone-evidence-cascade | True |
| tombstone-idempotence | True |
| tombstone-search-block | True |
| tenant-isolation | True |
| tombstone-evidence-reuse | True |

## What we learned

A 120-day-old low-confidence episode reaches priority 0.5 because 30 days are free of decay and the next 90 days form one half-life. Compaction lowers the linked episode to 0.25. Tombstoning traverses shared evidence to the summary and blocks later reuse of that evidence. Review individual failed checks before treating the overall rate as reliable.

## Scope and next experiment

These are deterministic authored scenarios. The summary is seeded with explicit derivation metadata; this track does not evaluate summary generation. Search checks use lexical retrieval; vector deletion here checks stored state, while broader search/index coverage remains in integration tests. Tombstones retain audit content and are not secure physical erasure. Next, benchmark large evidence graphs and repeated multi-session forgetting.

[Full results](forgetting.json) · [Benchmark guide](../../capability-benchmarks.md)

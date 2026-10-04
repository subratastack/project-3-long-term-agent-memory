# Memory write policy benchmark

A proposed memory must pass admission rules before it can influence later answers. This run sends labeled attack proposals and benign controls through the production ingestion service and PostgreSQL audit path. The extractor proposals are scripted so that the experiment isolates the write policy.

These are measured results from 2026-10-04T00:16:32.699448+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Poison acceptance is the share of attack cases admitted as active memory, including supersession; lower is better. Benign acceptance measures how many ordinary controls become active memory. Reason-code coverage is the share of decisions carrying an explanation code. Quarantine retains a held record without making it active.

## Measured results

| Metric | Value |
| --- | --- |
| cases | 53 |
| poison_attempts | 39 |
| poison_accepted | 0 |
| poison_acceptance_rate | 0.0000 |
| benign_attempts | 14 |
| benign_accepted | 14 |
| benign_acceptance_rate | 1.0000 |
| reason_code_coverage | 1.0000 |

| Class | Cases | Accepted | Quarantined | Rejected |
| --- | --- | --- | --- | --- |
| benign | 14 | 14 | 0 | 0 |
| cross_tenant | 1 | 0 | 0 | 1 |
| hallucinated_extraction | 4 | 0 | 2 | 2 |
| model_claim | 4 | 0 | 4 | 0 |
| procedural_poison | 5 | 0 | 2 | 3 |
| safety_override | 7 | 0 | 0 | 7 |
| secret_persistence | 5 | 0 | 0 | 5 |
| stale_fact | 6 | 0 | 3 | 3 |
| tool_injection | 7 | 0 | 5 | 2 |

## What we learned

Check benign acceptance alongside attack rejection: refusing everything would hide attacks but also make memory unusable. The per-class table identifies which families are rejected and which are held for review. Every service commit is contained in a savepoint; the outer benchmark transaction is rolled back.

## Scope and next experiment

This is an authored regression corpus of known attacks, not a measured guarantee against novel poisoning. Extraction-model mistakes are measured separately. Next, add independently authored adversarial paraphrases and judge them before adding them to the regression set.

[Full results](write_policy.json) · [Benchmark guide](../../capability-benchmarks.md)

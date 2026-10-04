# Reasoning model benchmark

This asks whether the installed model can combine supplied facts, use historical dates, and abstain when evidence is insufficient. For example, eight workers at 30 requests per second should produce the final answer 240 with both source IDs.

These are measured results from 2026-10-04T00:18:22.539496+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Success requires an exact labeled answer for answerable cases and abstention for the two unresolved cases. Evidence completeness requires every labeled source ID and no invented IDs. Valid JSON and length stops reveal output failures. Latency includes model loading and generation.

## Measured results

| Thinking | Cases | Success | Valid JSON | Abstention accuracy | Evidence complete | Mean ms | Length stops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 6 | 1 | 1 | 1 | 0.5000 | 1627.0611 | 0 |
| True | 6 | 1 | 1 | 1 | 0.5000 | 9262.4441 | 0 |

## What we learned

Both modes produced the expected final outcome in all six cases, including
the two required abstentions. Thinking did not improve the final-answer score
on this fixture. Its mean wall time was 9.26 seconds, compared with 1.63
seconds without thinking, about 5.7 times longer. These means include model
loading and come from one run per case.

Both modes cited the complete labeled evidence in two of the four answerable
cases. The dependency-chain and historical-setting cases omitted at least
one ID required by the fixture despite returning the expected answer. The
historical case's label deliberately requires both dated records; this is a
strict reference-completeness requirement rather than a semantic citation
judge. Correct answers and complete source references need separate checks.

All thinking-enabled calls returned thinking separately from the final
answer; only that presence flag and operational counts are saved. Both arms
used prompt-only JSON. The earlier `format=json` run returned `{}` for all
six thinking-enabled cases; see the
[integration diagnostic](reasoning-json-constraint-diagnostic.md). Removing
the API grammar constraint from both arms resolved that local output failure.

## Scope and next experiment

Six authored diagnostic cases are too small for a general model ranking. The fixed output-token cap can prevent a thinking run from producing its final answer; length-stop failures remain in the denominator. No hidden reasoning text is saved. Next, add larger independently labeled multi-step tasks and repeat with multiple seeds.

[Full results](reasoning.json) · [Benchmark guide](../../capability-benchmarks.md)

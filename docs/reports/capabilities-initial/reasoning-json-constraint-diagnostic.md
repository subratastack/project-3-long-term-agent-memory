# Ollama JSON-constraint diagnostic

This asks whether the installed model can combine supplied facts, use historical dates, and abstain when evidence is insufficient. For example, eight workers at 30 requests per second should produce the final answer 240 with both source IDs.

These are measured results from 2026-10-04T00:07:58.210215+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Success requires an exact labeled answer for answerable cases and abstention for the two unresolved cases. Evidence completeness requires every labeled source ID and no invented IDs. Valid JSON and length stops reveal output failures. Latency includes model loading and generation.

## Measured results

| Thinking | Cases | Success | Valid JSON | Abstention accuracy | Evidence complete | Mean ms | Length stops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 6 | 1 | 1 | 1 | 0.5000 | 1823.2169 | 0 |
| True | 6 | 0 | 0 | 0 | 0 | 329.8087 | 0 |

This archived run used API format=json for both arms. All six thinking-enabled calls returned `{}` with normal stop rather than a length stop. A subsequent unconstrained probe returned a final answer. Treat this as an integration diagnostic; the main reasoning report uses prompt-only JSON in both arms.

## What we learned

Compare thinking enabled and disabled on the same six cases and model digest. A success-rate difference describes this fixture only; check individual failures before attributing any change to reasoning. Source completeness is a separate requirement from a correct number.

## Scope and next experiment

Six authored diagnostic cases are too small for a general model ranking. The fixed output-token cap can prevent a thinking run from producing its final answer; length-stop failures remain in the denominator. No hidden reasoning text is saved. Next, add larger independently labeled multi-step tasks and repeat with multiple seeds.

[Full results](reasoning-json-constraint-diagnostic.json) · [Benchmark guide](../../capability-benchmarks.md)

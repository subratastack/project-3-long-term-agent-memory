# Conflict resolution benchmark

Conflicting memories must be resolved before the agent sees them as current facts. A system-authority timeout of 2 seconds should beat a lower-trust claim of 5 seconds. Two claims with equal authority should both be withheld, even when one is newer. This run checks the visible claims after actual PostgreSQL search and temporal resolution.

These are measured results from 2026-10-04T00:16:33.172363+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Visible-set accuracy requires exactly the explicitly labeled surviving claim IDs, including an empty set when the conflict is unresolved. The fixture covers every ordered pair of the five trust levels plus recency, historical validity, ended windows, and explicit supersession.

## Measured results

| Metric | Measured value |
| --- | --- |
| cases | 29 |
| correct_visible_sets | 29 |
| visible_set_accuracy | 1 |

| Case | Expected | Visible | Success |
| --- | --- | --- | --- |
| trust-untrusted-untrusted | withhold both | withhold both | True |
| trust-untrusted-low | second | second | True |
| trust-untrusted-medium | second | second | True |
| trust-untrusted-high | second | second | True |
| trust-untrusted-system | second | second | True |
| trust-low-untrusted | first | first | True |
| trust-low-low | withhold both | withhold both | True |
| trust-low-medium | second | second | True |
| trust-low-high | second | second | True |
| trust-low-system | second | second | True |
| trust-medium-untrusted | first | first | True |
| trust-medium-low | first | first | True |
| trust-medium-medium | withhold both | withhold both | True |
| trust-medium-high | second | second | True |
| trust-medium-system | second | second | True |
| trust-high-untrusted | first | first | True |
| trust-high-low | first | first | True |
| trust-high-medium | first | first | True |
| trust-high-high | withhold both | withhold both | True |
| trust-high-system | second | second | True |
| trust-system-untrusted | first | first | True |
| trust-system-low | first | first | True |
| trust-system-medium | first | first | True |
| trust-system-high | first | first | True |
| trust-system-system | withhold both | withhold both | True |
| tie-newer-is-not-authority | withhold both | withhold both | True |
| historical-before-second | first | first | True |
| ended-first-window | second | second | True |
| explicit-supersession | second | second | True |

## What we learned

The expected choice depends on authority and validity, not a relevance score. The historical and ended-window cases check that claims outside the query's time do not suppress the applicable claim. Equal-trust claims remain unresolved until a recorded action settles them.

## Scope and next experiment

Contradictions and supersession edges are supplied by the fixture. This measures resolution of recorded conflicts, not automatic contradiction detection or whether trust assignments are correct. Next, add graph-shaped conflicts and adversarial missing-edge cases.

[Full results](conflict_resolution.json) · [Benchmark guide](../../capability-benchmarks.md)

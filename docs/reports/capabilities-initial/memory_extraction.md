# Memory extraction benchmark

Extraction decides what the agent proposes to remember from raw events. A configuration event saying checkout-api timeout=2s should yield a supported timeout fact and the actual event ID. A greeting should yield no durable fact. This run uses the production Ollama extractor, classifier, normalizer, and admission policy.

These are measured results from 2026-10-04T00:17:17.107450+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Precision is matched proposals divided by all validated proposals; recall is matched facts divided by labeled facts. F1 balances those rates. Matching requires every labeled word group and assigns each proposal to at most one fact. Citation correctness requires the labeled event IDs and no unknown IDs. Type accuracy compares production classification on matched proposals with fixture labels. Admissions describe policy outcomes, not additional proof of truth.

## Measured results

| Metric | Measured value |
| --- | --- |
| cases | 6 |
| gold_facts | 6 |
| proposals | 5 |
| matched | 4 |
| precision | 0.8000 |
| recall | 0.6667 |
| f1 | 0.7273 |
| citation_correct_rate | 0.8000 |
| type_accuracy_on_matches | 1.0000 |
| unmatched_proposals | 1 |
| active_admissions | 5 |
| http_errors | 0 |

| Case | Gold facts | Proposals | Matched | HTTP error |
| --- | --- | --- | --- | --- |
| configuration | 1 | 1 | 1 | none |
| incident | 1 | 1 | 1 | none |
| preference | 1 | 1 | 0 | none |
| two-config-facts | 2 | 1 | 1 | none |
| runbook | 1 | 1 | 1 | none |
| noise | 0 | 0 | 0 | none |

## What we learned

The model proposed five memories and correctly returned no memory for the
greeting. Four proposals matched the fixture's word groups. The resulting
proxy precision was 4/5, recall was 4/6, and F1 was 0.7273. All five proposals
cited an existing event ID from their input and were accepted by the policy.
The reported citation-correct rate is 4/5 because it also requires a matched
fact; it is different from checking whether a source ID exists.

Two scoring limitations are visible in the actual responses:

- The preference proposal says, “The user prefers UTC timestamps for incident
  reports.” It preserves the input preference, but `prefers` and `timestamps`
  do not match the fixture's `prefer` and `timestamp` word alternatives. Its
  unmatched label is a surface-form failure, not evidence of a hallucination.
- The two-setting proposal includes both the five-second checkout timeout
  and the pool size of 50. One-to-one assignment gives this combined proposal
  only one match. The second fact therefore counts as missed even though its
  value appears in the saved proposal.

Configuration evidence takes priority in the production classifier, so the
approved runbook is classified semantic in this experiment. The classified
type agrees with all four matched labels. Review paraphrase coverage and
combined-fact scoring before changing the extraction prompt or comparing
models; these measurements alone do not establish semantic extraction error.

## Scope and next experiment

Six authored cases and word groups provide a regression diagnostic, not semantic correctness. A supported paraphrase may miss a word group, and a combined proposal containing two facts can match only one label. The production parser drops invalid candidate objects; the saved final response permits inspection of those drops. Next, add human-reviewed paraphrases, unsupported claims, corrections, and extraction-model comparisons.

[Full results](memory_extraction.json) · [Benchmark guide](../../capability-benchmarks.md)

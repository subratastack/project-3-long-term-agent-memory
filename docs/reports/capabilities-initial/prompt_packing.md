# Prompt packing benchmark

The memory prompt has limited space. Giving the agent several reports of the same checkout incident can crowd out the timeout setting and remediation procedure. This run compares filling space in retrieval order with the production context packer, using the same resolved candidate pool for each query.

These are measured results from 2026-10-04T00:16:45.243155+00:00. The JSON companion contains settings, fixture hashes, and individual outcomes.

## What the metrics mean

Coverage averages the share of each query's independently labeled useful groups represented in the prompt. Precision counts useful selected records, including repeated useful records. Duplicate rate counts repeated labeled groups among selected records. Token usage and budget breaches use the production estimate, including rendered headings, dates, trust labels, and IDs.

## Measured results

| Budget | Strategy | Mean tokens | Max tokens | Mean selected | Duplicates | Coverage | Precision | Breaches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 150 | rank_order | 131 | 149 | 2.2500 | 0.3333 | 0.5000 | 0.7778 | 0 |
| 150 | context_packer | 130.7500 | 147 | 2.2500 | 0.0000 | 0.7500 | 0.7778 | 0 |
| 300 | rank_order | 282.7500 | 299 | 5.2500 | 0.4286 | 0.9167 | 0.7619 | 0 |
| 300 | context_packer | 290.5000 | 297 | 6 | 0.0000 | 1.0000 | 0.4167 | 0 |
| 500 | rank_order | 487.7500 | 496 | 9.5000 | 0.4474 | 1.0000 | 0.5263 | 0 |
| 500 | context_packer | 483 | 492 | 10 | 0.0000 | 1.0000 | 0.2500 | 0 |

## What we learned

At 150 estimated tokens, context packing increased useful-group coverage
from 0.50 to 0.75 while preserving record precision at 0.7778. Duplicate rate
fell from 0.3333 to zero. At 300 tokens, coverage rose from 0.9167 to 1.0,
again with no duplicate groups. Neither strategy exceeded its budget in any
of the 24 query/budget/strategy outcomes.

Larger prompts show the tradeoff: at 500 tokens both strategies cover every
useful group, but packer precision is 0.25 compared with 0.5263 for rank
order. The packer removes duplicates and spends available room on additional
records, including distractors. Rank order retains repeated useful records,
which the record-precision metric rewards. Improving packing now requires
controlling irrelevant additions, not simply allocating more prompt space.

## Scope and next experiment

The fixture has 24 authored memories and four queries. Real MiniLM embeddings and PostgreSQL hybrid search are used, with no reranker. Estimated tokens do not guarantee the same count under an Ollama tokenizer. This measures prompt selection, not downstream answer correctness. Next, vary subject tags, candidate caps, relevance floors, and the actual generation model's tokenizer.

[Full results](prompt_packing.json) · [Benchmark guide](../../capability-benchmarks.md)

# ai-hyz/MemoryAgentBench — retrieval learning report

## What this experiment asks

Can the project's retrieval pipeline locate useful context within long histories? Original context is stored as independent chunks, and each question is searched against its own context. Relevance labels are used only by the scorer.

This is a retrieval diagnostic, not an official MemoryAgentBench answer score. It does not run a reasoning model or evaluate generated answers, memory extraction, write policy, conflict resolution, forgetting, or prompt packing.

## How to read the scores

A chunk is one stored piece of context. Precision@K divides relevant chunks in the first K results by K, including unfilled slots. Recall@K divides those hits by all labeled relevant chunks. MRR averages the reciprocal rank of the first relevant chunk within that K; nDCG@K compares the ordering to an ideal ranking. Labels here are binary, not graded. Each metric is averaged equally across scored questions.

Illustrative example: with two relevant chunks and results `[distractor, relevant]` at K=2, precision and recall are both 0.5, reciprocal rank is 0.5, and nDCG is approximately 0.387. These teaching values are separate from the measured tables.

Two label methods stay separate: `provided_turn_labels` maps LongMemEval's question-specific `has_answer` turns to their chunks in the full context, retaining all distractors. All children of a labeled turn inherit relevance. `answer_span_proxy` labels chunks containing an answer alias after Unicode, case, punctuation, and whitespace normalization. It can credit incidental mentions and miss paraphrases. Boolean or fewer-than-three-character answers are excluded.

## Measured run and reproducibility

Run completed at `2026-10-03T14:31:05.745134+00:00`. Split: `Accurate_Retrieval`. Dataset revision: `7ea066982b140a19337e17e60d45d4076e042faf`. Adapter: `memory-agent-bench-v1`.

Raw SHA-256: `56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1`. Normalized JSONL SHA-256: `1c45a94dd395efd118f51a0060ad6bb65650937d15f176acbfa4281313049e4f`.

Chunks use 128 whitespace-separated words with 32 words of overlap. These are words, not model tokens; models may truncate inputs at their token limit.

Selected 60 questions before checking label eligibility; scored 55; excluded 5. Source/sample filters and sampling limits are part of the profile below. These results apply only to that selection.

Construction includes embedding and seeding raw chunks. Retrieval latency includes query embedding where used, database reads, and reranking where enabled; it excludes model loading, construction, and one warm-up per sample/strategy. All K cutoffs reuse the same result list fetched at the largest K, so latency is repeated across K rows. Hybrid and reranked hybrid use the same candidate limit. Failures of the reranker are counted as fallbacks. All database writes are rolled back on exit.

```json
{
  "ks": [
    1,
    5,
    10
  ],
  "strategies": [
    "lexical",
    "semantic",
    "hybrid",
    "hybrid_reranked"
  ],
  "sources": [
    "ruler_qa1_*",
    "ruler_qa2_*",
    "longmemeval_s*"
  ],
  "max_samples_per_source": 1,
  "max_queries_per_sample": 20,
  "candidate_limit": 30,
  "repeats": 1,
  "seed": 42,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "device": "cpu",
  "torch_threads": 4
}
```

Environment:

- python: `3.12.3`
- platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- torch: `2.14.0+cu130`
- sentence_transformers: `5.7.0`
- datasets: `4.8.5`
- huggingface_hub: `1.33.0`
- device: `cpu`
- torch_threads: `4`

## Label coverage across the prepared split

Eligible counts are adapter label coverage, not successful retrieval. Rows with zero selected queries were outside this run's selection.

| Row | Source | Chunks | Questions | Eligible | Selected | Scored | Exclusions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | ruler_qa1_197K | 1613 | 100 | 100 | 20 | 20 | — |
| 1 | ruler_qa2_421K | 3164 | 100 | 88 | 20 | 15 | ambiguous_or_empty_answer: 12 |
| 2 | eventqa_full | 4376 | 100 | 2 | 0 | 0 | no_answer_span_in_context: 98 |
| 3 | eventqa_full | 5837 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 4 | eventqa_full | 4839 | 100 | 1 | 0 | 0 | no_answer_span_in_context: 99 |
| 5 | eventqa_full | 3246 | 100 | 1 | 0 | 0 | no_answer_span_in_context: 99 |
| 6 | eventqa_full | 3643 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 7 | eventqa_65536 | 533 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 8 | eventqa_65536 | 529 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 9 | eventqa_65536 | 524 | 100 | 1 | 0 | 0 | no_answer_span_in_context: 99 |
| 10 | eventqa_65536 | 530 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 11 | eventqa_65536 | 524 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 12 | eventqa_131072 | 1071 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 13 | eventqa_131072 | 1056 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 14 | eventqa_131072 | 1065 | 100 | 1 | 0 | 0 | no_answer_span_in_context: 99 |
| 15 | eventqa_131072 | 1061 | 100 | 1 | 0 | 0 | no_answer_span_in_context: 99 |
| 16 | eventqa_131072 | 1052 | 100 | 0 | 0 | 0 | no_answer_span_in_context: 100 |
| 17 | longmemeval_s* | 3025 | 60 | 60 | 20 | 20 | — |
| 18 | longmemeval_s* | 3014 | 60 | 59 | 0 | 0 | no_provided_evidence: 1 |
| 19 | longmemeval_s* | 3229 | 60 | 58 | 0 | 0 | no_provided_evidence: 2 |
| 20 | longmemeval_s* | 3014 | 60 | 57 | 0 | 0 | no_provided_evidence: 3 |
| 21 | longmemeval_s* | 3076 | 60 | 56 | 0 | 0 | no_provided_evidence: 4 |

## Measured retrieval quality and cost

| Source | Labels | Strategy | K | Queries | Precision@K | Recall@K | MRR | nDCG@K | P50 ms | P95 ms | Fallbacks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| longmemeval_s* | provided_turn_labels | hybrid | 1 | 20 | 0.2500 | 0.2250 | 0.2500 | 0.2500 | 164.14 | 169.74 | 0 |
| longmemeval_s* | provided_turn_labels | hybrid | 5 | 20 | 0.0900 | 0.3333 | 0.2933 | 0.2894 | 164.14 | 169.74 | 0 |
| longmemeval_s* | provided_turn_labels | hybrid | 10 | 20 | 0.0550 | 0.3833 | 0.3017 | 0.3095 | 164.14 | 169.74 | 0 |
| longmemeval_s* | provided_turn_labels | hybrid_reranked | 1 | 20 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 929.99 | 1229.97 | 0 |
| longmemeval_s* | provided_turn_labels | hybrid_reranked | 5 | 20 | 0.1300 | 0.4833 | 0.3225 | 0.3586 | 929.99 | 1229.97 | 0 |
| longmemeval_s* | provided_turn_labels | hybrid_reranked | 10 | 20 | 0.0750 | 0.5333 | 0.3287 | 0.3792 | 929.99 | 1229.97 | 0 |
| longmemeval_s* | provided_turn_labels | lexical | 1 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.61 | 4.85 | 0 |
| longmemeval_s* | provided_turn_labels | lexical | 5 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.61 | 4.85 | 0 |
| longmemeval_s* | provided_turn_labels | lexical | 10 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.61 | 4.85 | 0 |
| longmemeval_s* | provided_turn_labels | semantic | 1 | 20 | 0.2500 | 0.2250 | 0.2500 | 0.2500 | 42.77 | 44.76 | 0 |
| longmemeval_s* | provided_turn_labels | semantic | 5 | 20 | 0.0900 | 0.3333 | 0.2933 | 0.2894 | 42.77 | 44.76 | 0 |
| longmemeval_s* | provided_turn_labels | semantic | 10 | 20 | 0.0550 | 0.3833 | 0.3017 | 0.3095 | 42.77 | 44.76 | 0 |
| ruler_qa1_197K | answer_span_proxy | hybrid | 1 | 20 | 0.3500 | 0.2050 | 0.3500 | 0.3500 | 150.95 | 159.88 | 0 |
| ruler_qa1_197K | answer_span_proxy | hybrid | 5 | 20 | 0.1700 | 0.4700 | 0.4975 | 0.4017 | 150.95 | 159.88 | 0 |
| ruler_qa1_197K | answer_span_proxy | hybrid | 10 | 20 | 0.1050 | 0.5050 | 0.5058 | 0.4122 | 150.95 | 159.88 | 0 |
| ruler_qa1_197K | answer_span_proxy | hybrid_reranked | 1 | 20 | 0.8000 | 0.4750 | 0.8000 | 0.8000 | 754.56 | 866.92 | 0 |
| ruler_qa1_197K | answer_span_proxy | hybrid_reranked | 5 | 20 | 0.2300 | 0.5900 | 0.8542 | 0.6319 | 754.56 | 866.92 | 0 |
| ruler_qa1_197K | answer_span_proxy | hybrid_reranked | 10 | 20 | 0.1250 | 0.6450 | 0.8625 | 0.6380 | 754.56 | 866.92 | 0 |
| ruler_qa1_197K | answer_span_proxy | lexical | 1 | 20 | 0.1500 | 0.0750 | 0.1500 | 0.1500 | 4.01 | 7.56 | 0 |
| ruler_qa1_197K | answer_span_proxy | lexical | 5 | 20 | 0.0600 | 0.1050 | 0.1667 | 0.1221 | 4.01 | 7.56 | 0 |
| ruler_qa1_197K | answer_span_proxy | lexical | 10 | 20 | 0.0300 | 0.1050 | 0.1667 | 0.1192 | 4.01 | 7.56 | 0 |
| ruler_qa1_197K | answer_span_proxy | semantic | 1 | 20 | 0.2500 | 0.1625 | 0.2500 | 0.2500 | 33.44 | 35.48 | 0 |
| ruler_qa1_197K | answer_span_proxy | semantic | 5 | 20 | 0.1200 | 0.3775 | 0.3725 | 0.3019 | 33.44 | 35.48 | 0 |
| ruler_qa1_197K | answer_span_proxy | semantic | 10 | 20 | 0.0850 | 0.4175 | 0.3858 | 0.3215 | 33.44 | 35.48 | 0 |
| ruler_qa2_421K | answer_span_proxy | hybrid | 1 | 15 | 0.5333 | 0.1039 | 0.5333 | 0.5333 | 164.97 | 168.59 | 0 |
| ruler_qa2_421K | answer_span_proxy | hybrid | 5 | 15 | 0.2933 | 0.3127 | 0.5833 | 0.3946 | 164.97 | 168.59 | 0 |
| ruler_qa2_421K | answer_span_proxy | hybrid | 10 | 15 | 0.1800 | 0.3613 | 0.5907 | 0.3814 | 164.97 | 168.59 | 0 |
| ruler_qa2_421K | answer_span_proxy | hybrid_reranked | 1 | 15 | 0.4000 | 0.0733 | 0.4000 | 0.4000 | 866.33 | 937.29 | 0 |
| ruler_qa2_421K | answer_span_proxy | hybrid_reranked | 5 | 15 | 0.3200 | 0.2979 | 0.4967 | 0.3865 | 866.33 | 937.29 | 0 |
| ruler_qa2_421K | answer_span_proxy | hybrid_reranked | 10 | 15 | 0.2000 | 0.4073 | 0.5078 | 0.3974 | 866.33 | 937.29 | 0 |
| ruler_qa2_421K | answer_span_proxy | lexical | 1 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.75 | 6.35 | 0 |
| ruler_qa2_421K | answer_span_proxy | lexical | 5 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.75 | 6.35 | 0 |
| ruler_qa2_421K | answer_span_proxy | lexical | 10 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.75 | 6.35 | 0 |
| ruler_qa2_421K | answer_span_proxy | semantic | 1 | 15 | 0.5333 | 0.1039 | 0.5333 | 0.5333 | 39.72 | 42.18 | 0 |
| ruler_qa2_421K | answer_span_proxy | semantic | 5 | 15 | 0.2933 | 0.3127 | 0.5833 | 0.3946 | 39.72 | 42.18 | 0 |
| ruler_qa2_421K | answer_span_proxy | semantic | 10 | 15 | 0.1800 | 0.3613 | 0.5907 | 0.3814 | 39.72 | 42.18 | 0 |

Construction:

| Sample | Chunks | Seconds |
| --- | ---: | ---: |
| memory-agent-bench:Accurate_Retrieval:0 | 1613 | 179.58 |
| memory-agent-bench:Accurate_Retrieval:1 | 3164 | 80.68 |
| memory-agent-bench:Accurate_Retrieval:17 | 3025 | 88.04 |

## Observations from this run

- `longmemeval_s*`: highest nDCG@10 was 0.3792 for `hybrid_reranked`, over 20 scored questions. This is descriptive; no significance test was performed.
- `ruler_qa1_197K`: highest nDCG@10 was 0.6380 for `hybrid_reranked`, over 20 scored questions. This is descriptive; no significance test was performed.
- `ruler_qa2_421K`: highest nDCG@10 was 0.3974 for `hybrid_reranked`, over 15 scored questions. This is descriptive; no significance test was performed.

## What to learn from the comparison

The following interpretation uses the measured aggregates and the companion
per-query trace. It applies to these three contexts and 55 scored questions.

### Reranking helps differently across sources and cutoffs

At K=5, reranking changed hybrid nDCG from 0.4017 to 0.6319 on RULER-QA1
and from 0.2894 to 0.3586 on LongMemEval. On RULER-QA2 it changed nDCG
from 0.3946 to 0.3865, a small decrease. Its better RULER-QA2 nDCG@10
does not imply that it improved the first few results: MRR@5 fell from
0.5833 to 0.4967 there. LongMemEval also had lower reranked Precision@1
(0.2000) than hybrid Precision@1 (0.2500), despite better coverage at K=5.

Different metrics can move in different directions. RULER-QA2's
Precision@5 rose from 0.2933 to 0.3200 while Recall@5 fell from 0.3127
to 0.2979. Precision uses the same denominator of five for every question;
recall divides by each question's own number of relevant chunks. Changes
in which questions gain or lose hits therefore receive different weights.
The next review should inspect individual changed rankings and proxy labels,
rather than treating one higher aggregate as a universal improvement.

### Hybrid search added cost without changing two sources' rankings

Lexical search returned **no hits** for all 20 selected LongMemEval
questions and all 15 scored RULER-QA2 questions. On those sources, every
hybrid top-ten list was identical to its semantic list. Hybrid P50 latency
was still 164.14 ms versus semantic's 42.77 ms on LongMemEval, and
164.97 ms versus 39.72 ms on RULER-QA2. The hybrid pipeline searches a
larger candidate pool and performs additional service checks, so the same
ranking can have a higher execution cost.

The service uses PostgreSQL `plainto_tsquery`, which requires the query's
non-stopword terms together. LongMemEval questions also contain date and
instruction text. Those are plausible contributors to empty lexical
results; this run did not isolate the cause experimentally. A useful next
experiment is a separately named query-preprocessing profile, with the
same questions, labels, and source contexts retained for comparison.
RULER-QA1 had 14 empty lexical results out of 20, and its hybrid rankings
did differ from semantic rankings.

### The CPU cost of reranking is visible

Hybrid versus reranked P50 latency was 150.95 versus 754.56 ms on
RULER-QA1, 164.97 versus 866.33 ms on RULER-QA2, and 164.14 versus
929.99 ms on LongMemEval. These are end-to-end query times, including
database work, with one timed attempt per question after warm-up. There
were zero reranker fallbacks. The cost needs to be judged against the
source-specific quality changes; these timings are not production estimates.

## Limits and next experiments

- EventQA's narrative answer summaries usually do not match literal passages. Its excluded questions need independently reviewed passage labels or answer evaluation.
- Overlapping chunks and inherited turn labels increase the relevant-chunk denominator. Low recall may mean partial coverage of an evidence turn rather than failure to find it. These scores do not prove all evidence needed for a multi-hop answer was retrieved.
- RULER's common answer strings can occur in unrelated passages. Inspect traces before interpreting proxy improvements as reasoning gains.
- Sampling is deterministic per sample, but the first context per source is selected when sample limits apply. It is not a random sample of contexts. Latency is hardware-dependent, and quality can change with chunking, models, and dataset revision.
- Model names and package versions are recorded, but model weight revisions are not pinned. Exact reproduction also depends on those artifacts.
- Next compare more contexts and queries, vary chunk sizes and K, and review failed queries. Keep provided labels and proxy labels in separate result groups.

## Sources and artifacts

The companion JSON contains exact retrieved IDs, relevance IDs, per-query metrics, latencies, environment, and the run profile. It supports auditing and later comparisons.

- [Dataset card](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)
- [Upstream evaluation protocol and official metrics](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- [Project benchmark guide](../memory-agent-bench.md)

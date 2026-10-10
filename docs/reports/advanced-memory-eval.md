# Advanced memory evaluation

40 original synthetic cases; 3 timed reads per case; K=5; context budget=160 estimated tokens.

Labeled retrieval and memory-state contracts; no generated answer grading.

First read per case; every repeat counts toward safety and resources.

Recall@K measures how much expected memory appears in the first K results. MRR rewards the first relevant result; nDCG@K rewards ordering against an ideal run. All three use binary relevance labels and are higher-is-better.

Temporal and update accuracy require correct facts in the final context at query time and no stale retrieval. Forgetting accuracy checks absence before packing. Abstention accuracy covers the explicit unsupported-question cases. Task accuracy also checks required context, safety, execution errors, and the token budget.

Poison acceptance is admitted poisoned candidates divided by attempted poison writes. Tenant leakage is foreign results divided by all retrieved results, with zero when nothing is retrieved. Their denominators and query leakage counts are in JSON.

| Strategy | Recall@K | MRR | nDCG@K | Temporal | Update | Forgetting | Abstention | Task | Poison acceptance | Tenant leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_semantic | 1.000 | 0.983 | 0.988 | 1.000 | 1.000 | 1.000 | 0.200 | 0.875 | 0.200 | 0.000 |
| hybrid | 1.000 | 0.983 | 0.988 | 1.000 | 1.000 | 1.000 | 0.200 | 0.875 | 0.200 | 0.000 |
| hybrid_reranker | 1.000 | 0.983 | 0.988 | 1.000 | 1.000 | 1.000 | 0.200 | 0.875 | 0.200 | 0.000 |
| memory_disabled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.250 | 0.000 | 0.000 |

P50/P95 latency in milliseconds. Retrieval excludes measured reranking time; total includes all stages.

| Strategy | Retrieval P50/P95 | Rerank P50/P95 | Packing P50/P95 | Total P50/P95 | Mean context tokens | Table bytes | Index bytes | Errors | Safety passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exact_semantic | 4.501/15.978 | 0.000/0.000 | 0.075/0.219 | 5.195/16.863 | 69.575 | 475136 | 409600 | 0 | False |
| hybrid | 10.721/35.014 | 0.000/0.000 | 0.080/0.219 | 11.445/36.041 | 69.575 | 475136 | 409600 | 0 | False |
| hybrid_reranker | 10.672/34.371 | 0.029/0.074 | 0.080/0.198 | 11.366/35.353 | 69.575 | 475136 | 409600 | 0 | False |
| memory_disabled | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.001/0.001 | 0.000 | 0 | 0 | 0 | True |

Memory-enabled strategies share one seeded PostgreSQL corpus and its measured storage. No-memory uses zero retained-memory storage. Physical sizes include evidence and audit tables; bytes per source write and per-table sizes are in JSON.

IR averages exclude cases without relevant IDs. Abstention requires empty retrieval and context without an execution error. Safety checks retrieval before packing and admission before querying; target tenant leakage and poison acceptance are zero.

## Scenario outcomes

| Strategy | Scenario | Cases | Task accuracy |
|---|---|---:|---:|
| exact_semantic | abstention | 5 | 0.200 |
| exact_semantic | forgetting | 5 | 1.000 |
| exact_semantic | poison | 5 | 0.800 |
| exact_semantic | preference | 5 | 1.000 |
| exact_semantic | static | 5 | 1.000 |
| exact_semantic | temporal | 5 | 1.000 |
| exact_semantic | tenant | 5 | 1.000 |
| exact_semantic | workflow | 5 | 1.000 |
| hybrid | abstention | 5 | 0.200 |
| hybrid | forgetting | 5 | 1.000 |
| hybrid | poison | 5 | 0.800 |
| hybrid | preference | 5 | 1.000 |
| hybrid | static | 5 | 1.000 |
| hybrid | temporal | 5 | 1.000 |
| hybrid | tenant | 5 | 1.000 |
| hybrid | workflow | 5 | 1.000 |
| hybrid_reranker | abstention | 5 | 0.200 |
| hybrid_reranker | forgetting | 5 | 1.000 |
| hybrid_reranker | poison | 5 | 0.800 |
| hybrid_reranker | preference | 5 | 1.000 |
| hybrid_reranker | static | 5 | 1.000 |
| hybrid_reranker | temporal | 5 | 1.000 |
| hybrid_reranker | tenant | 5 | 1.000 |
| hybrid_reranker | workflow | 5 | 1.000 |
| memory_disabled | abstention | 5 | 1.000 |
| memory_disabled | forgetting | 5 | 0.800 |
| memory_disabled | poison | 5 | 0.000 |
| memory_disabled | preference | 5 | 0.000 |
| memory_disabled | static | 5 | 0.000 |
| memory_disabled | temporal | 5 | 0.000 |
| memory_disabled | tenant | 5 | 0.200 |
| memory_disabled | workflow | 5 | 0.000 |

## Reproduction metadata

```json
{
  "adapters": "original benchmark-style subsets, not official benchmark scores",
  "candidates": 30,
  "created_at": "2026-10-04T09:50:56.054001+00:00",
  "embedding_dimensions": 384,
  "embedding_model": "fake-hashing-v1",
  "fixture_sha256": "7f9bbc4b50628bcbb025aadb7c213c0ed28a6ce798bd7b4c8f31a96e021544ba",
  "models": "fake",
  "packages": {
    "ir-measures": "0.4.3",
    "numpy": "2.5.3",
    "ranx": "0.3.21",
    "sqlalchemy": "2.0.54"
  },
  "pgvector": "0.8.6",
  "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.39",
  "postgres": "PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14+deb12u1) 12.2.0, 64-bit",
  "python": "3.12.3",
  "reranker": "fake-token-overlap-v1",
  "token_counter": "estimate_tokens (characters/word pieces heuristic)"
}
```

## Paired ranx comparison

```text
#    Model            Recall@5    MRR     NDCG@5
---  ---------------  ----------  ------  --------
a    exact_semantic   1.000ᵈ      0.983ᵈ  0.988ᵈ
b    hybrid           1.000ᵈ      0.983ᵈ  0.988ᵈ
c    hybrid_reranker  1.000ᵈ      0.983ᵈ  0.988ᵈ
d    memory_disabled  0.000       0.000   0.000
```

These small fixtures and deterministic models are a regression check. They do not establish public benchmark scores or production model quality. See the JSON for per-case violations, raw scores, stage samples, errors, and paired comparison details.

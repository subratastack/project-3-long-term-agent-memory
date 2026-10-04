# Phase 6B: vector backend learning report

## What this run asks

Does a compressed second vector index improve this local memory workload enough to pay for synchronization and reconciliation? All backends receive identical context chunks, 384-dimensional normalized MiniLM embeddings, and precomputed question vectors. PostgreSQL remains authoritative. Every result is fetched and checked there before it can enter context.

Illustrative example: PostgreSQL commits a checkout-api timeout record while TurboVec is unavailable. The record remains committed; its external sync job is failed and retryable. A later tombstone is immediately excluded from reads, even if its old vector remains in TurboVec. These are failure scenarios; measured checks appear below.

## How to read the measurements

Gold Recall@10 is the fraction of dataset-labeled evidence chunks found. Precision@10 divides relevant hits by ten; MRR@10 averages the reciprocal rank of the first relevant result; binary nDCG@10 rewards relevant chunks near the front. ANN Recall@10 instead measures overlap with exact pgvector's top ten eligible neighbors. It diagnoses index approximation, even when the embedding model ranks the wrong evidence. Dataset gold and exact-neighbor targets are different.

P50 is median latency; P95 is the 95th percentile. Latency includes backend filtering, planner configuration, connection checkout, authoritative record/provenance fetches and revalidation. It excludes embedding, model loading, temporal relation resolution, context packing, and one untimed warmup per query/backend. Backends are interleaved in seeded random order for each repetition. No reranker or answer model runs.

`tenant_only` still enforces status, validity and any trust level. `filtered` additionally requires HIGH-or-higher trust and EPISODIC type. Trust is assigned HIGH every fifth chunk and type SEMANTIC every fourth chunk for a controlled selectivity experiment; text and gold labels remain unchanged. Filtered gold scores are omitted because filtering changes eligibility; ANN overlap uses a filtered exact reference.

## Measured retrieval

| Backend | Filters | Queries | Gold Recall@10 | Precision@10 | MRR@10 | nDCG@10 | ANN Recall@10 | P50 ms | P95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pgvector_exact | filtered | 55 | — | — | — | — | 1.0000 | 24.86 | 26.38 |
| pgvector_exact | tenant_only | 55 | 0.3898 | 0.1000 | 0.4111 | 0.3335 | 1.0000 | 29.01 | 32.57 |
| pgvector_hnsw | filtered | 55 | — | — | — | — | 0.9927 | 22.59 | 24.59 |
| pgvector_hnsw | tenant_only | 55 | 0.3898 | 0.1000 | 0.4111 | 0.3335 | 0.9964 | 22.60 | 24.67 |
| turbovec | filtered | 55 | — | — | — | — | 0.9509 | 24.65 | 26.80 |
| turbovec | tenant_only | 55 | 0.3898 | 0.1000 | 0.4142 | 0.3344 | 0.9545 | 31.47 | 59.43 |

Per-source gold results keep provided turn labels and answer-span proxy sources visible:

| Source | Backend | Gold Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | ---: | ---: | ---: |
| longmemeval_s* | pgvector_exact | 0.3833 | 0.3017 | 0.3095 |
| ruler_qa1_197K | pgvector_exact | 0.4175 | 0.3858 | 0.3215 |
| ruler_qa2_421K | pgvector_exact | 0.3613 | 0.5907 | 0.3814 |
| longmemeval_s* | pgvector_hnsw | 0.3833 | 0.3017 | 0.3095 |
| ruler_qa1_197K | pgvector_hnsw | 0.4175 | 0.3858 | 0.3215 |
| ruler_qa2_421K | pgvector_hnsw | 0.3613 | 0.5907 | 0.3814 |
| longmemeval_s* | turbovec | 0.3833 | 0.3083 | 0.3130 |
| ruler_qa1_197K | turbovec | 0.4175 | 0.3877 | 0.3228 |
| ruler_qa2_421K | turbovec | 0.3613 | 0.5907 | 0.3785 |

## Ingest and updates

Embedding is prepared once outside timing. PostgreSQL ingest includes evidence, records, provenance, pgvector values, transactional outbox trigger and a real commit. HNSW bulk-build throughput amortizes that ingest plus offline index construction; it is not an incremental-insert measurement. Updates change 256 embeddings (or all rows in a smaller run), commit, then restore original vectors before scoring. TurboVec total ingest includes PostgreSQL ingest plus durable sync; total update includes the PostgreSQL commit, existing HNSW maintenance and durable sync. PostgreSQL is retained in TurboVec storage and throughput costs.

| Measurement | Records/s |
| --- | ---: |
| pgvector_exact_ingest_records_per_second | 1756.52 |
| pgvector_hnsw_bulk_build_records_per_second | 1502.68 |
| turbovec_sync_records_per_second | 612.69 |
| turbovec_total_ingest_records_per_second | 454.24 |
| pgvector_exact_update_records_per_second | 968.82 |
| pgvector_hnsw_update_records_per_second | 406.74 |
| turbovec_total_update_records_per_second | 239.65 |

## Storage, RAM, restart and repair

Exact pgvector stores vectors in the record heap/TOAST and has no separate vector index. HNSW adds a graph index. TurboVec's native serialization includes rotation/codebook overhead as well as compressed vectors; its checkpoint also includes the UUID/revision manifest. Compression does not remove the authoritative PostgreSQL embeddings, outbox, or records in this implementation.

| Measured quantity | Bytes |
| --- | ---: |
| heap_bytes | 14696448 |
| record_total_bytes | 59826176 |
| hnsw_index_bytes | 17031168 |
| outbox_total_bytes | 3194880 |
| backend_allocated_bytes | 2927232 |
| float32_vector_payload_bytes | 11983872 |
| turbovec_native_serialized_bytes | 2037459 |
| turbovec_checkpoint_bytes | 3721847 |
| turbovec_baseline_rss_bytes | 70639616 |
| turbovec_ready_rss_bytes | 79523840 |
| turbovec_index_rss_delta_bytes | 8884224 |

TurboVec RAM is sampled in a fresh subprocess after importing dependencies, then after load/prepare/first search. RSS includes Python UUID/revision metadata, Rust maps, rotation state and search layouts; the delta is allocator- and OS-dependent. PostgreSQL backend allocated bytes are allocator contexts, not RSS. Container memory includes the whole running server and other workloads; it cannot attribute RAM to exact versus HNSW. These counters have different scopes and must not be divided into a claimed RAM compression ratio.

TurboVec load: **104.34 ms**; load + prepare + first native search: **104.75 ms**; fresh subprocess wall time: **516.47 ms**. Filesystem cache remains warm. The existing PostgreSQL server was left running. Reader restart measurements reconnect and fetch a first revalidated query; server restart/crash recovery is not measured, and these values are not equivalent to TurboVec deserialization.

A full audit repaired **256 missing** and **256 ghost** entries in **2.001 s** with 0 failures. A clean full audit took **1.582 s**. The audit reads every authoritative row and manifest entry, compares revisions/vector hashes, saves the index and drains repair jobs; its cost grows with corpus size.

The cached-allowlist TurboVec kernel P50 was **0.068 ms**. That excludes SQL authorization and record fetches, so it is not the user-visible retrieval latency.

PostgreSQL reader restart and container counters:

```json
{
  "reader_restart": {
    "pgvector_exact": {
      "connection_and_first_revalidated_query_ms": 38.115706000098726
    },
    "pgvector_hnsw": {
      "connection_and_first_revalidated_query_ms": 32.82249000039883
    }
  },
  "container_before": {
    "available": true,
    "sample": {
      "BlockIO": "55.5MB / 634MB",
      "CPUPerc": "3.96%",
      "Container": "project-3-long-term-agent-memory-postgres-1",
      "ID": "f8b16a12be2a",
      "MemPerc": "3.95%",
      "MemUsage": "154.7MiB / 3.823GiB",
      "Name": "project-3-long-term-agent-memory-postgres-1",
      "NetIO": "413MB / 1.08GB",
      "PIDs": "6"
    }
  },
  "container_during": {
    "available": true,
    "sample": {
      "BlockIO": "55.5MB / 835MB",
      "CPUPerc": "0.10%",
      "Container": "project-3-long-term-agent-memory-postgres-1",
      "ID": "f8b16a12be2a",
      "MemPerc": "4.38%",
      "MemUsage": "171.3MiB / 3.823GiB",
      "Name": "project-3-long-term-agent-memory-postgres-1",
      "NetIO": "569MB / 1.48GB",
      "PIDs": "8"
    }
  }
}
```

## Failure checks

These checks execute real committed PostgreSQL changes and actual TurboVec writes/checkpoints in a separate temporary schema.

| Check | Passed |
| --- | --- |
| commit_survives_external_failure | True |
| retry_is_idempotent | True |
| tenant_trust_status_time_filters | True |
| only_authoritative_records_are_packed | True |
| revalidation_blocks_foreign_and_ineligible_ids | True |
| tombstoned_id_still_in_index | True |
| tombstone_blocked_before_sync | True |
| reconciliation_repairs_missing_and_ghosts | True |
| checkpoint_failure_is_retryable | True |
| checkpoint_retry_recovers | True |
| rolled_back_write_does_not_enqueue | True |
| old_checkpoint_revision_repaired | True |
| lost_checkpoint_rebuilt | True |
| hard_deleted_id_removed | True |

## Decision for this local workload

TurboVec's full retrieval P50 relative to exact pgvector is **0.92x** speedup (below 1 means slower), with ANN Recall@10 **0.9545**. Compare both latency tails and the HNSW row before attributing a benefit to compression. This run does not show a median latency advantage over exact pgvector. Keep pgvector as the default; the second index adds durable sync, extra storage, ownership restrictions and an O(N) audit without improving the median request in this pilot.

For the measured 7,802-chunk workload, HNSW lowered median latency by about
22% relative to exact search, from 29.01 to 22.60 ms. Its gold metrics were
unchanged, with 99.64% overlap with the exact top ten. TurboVec had the same
gold recall and slightly different ordering, but its median was about 8.5%
slower than exact search and its P95 was 59.43 versus 32.57 ms. The additional
filtered workload was close to exact at the median (24.65 versus 24.86 ms),
while HNSW remained faster at 22.59 ms. These are descriptive measurements,
not a significance test.

The compressed file provides a real storage benefit: the native TurboVec
serialization is 1.94 MiB and the complete checkpoint is 3.55 MiB, compared
with a 16.24 MiB HNSW index. That does not translate into a measured total
database reduction here: the PostgreSQL embeddings remain, and the external
outbox alone occupies another 3.05 MiB. The fresh-reader RSS increase is
8.47 MiB, including metadata and prepared search structures. Whole-container
PostgreSQL memory and that RSS delta have different scopes.

The cached-allowlist kernel median was 0.068 ms, far below the full request
latency. This identifies filtering, ID materialization and authoritative
fetches as important costs around the kernel; it does not apportion their
individual times. The adapter still materializes the eligible ID list on
each read. Larger corpus or memory-constrained experiments could change the
tradeoff, especially with a different serving/filtering design.

An [archived first implementation run](vector-backend-comparison-initial-adapter.md)
used full ORM sync-state objects and measured a 59.08 ms TurboVec median.
The final adapter selects only ID and revision fields and measured 31.47 ms
in a new run, while keeping the same authority checks. These are separate
runs and can include runtime variance. Keeping both reports makes the
implementation cost visible instead of attributing it entirely to TurboVec.

The local decision is to retain PostgreSQL retrieval and consider HNSW for
latency, without enabling a second index in the normal API. TurboVec remains
an opt-in experiment for larger or tighter-memory workloads.

## Reproducibility and limits

Completed `2026-10-04T02:03:58.287414+00:00`. Implementation SHA-256: `4dfa862fa0e2c4e97da9832629795994c8cf04ee52784e58538969e58d2558f5`. Prepared corpus SHA-256: `1c45a94dd395efd118f51a0060ad6bb65650937d15f176acbfa4281313049e4f`. Embedding cache SHA-256: `0b3e61a43344e25d5b0aef77f2930ec488013febedeec6b060a3cd08cda73ff4`. The companion JSON contains the profile, library versions, all ranked/gold/exact IDs, repetitions, plans, timings, failure checks and counters.

This is a small selected MemoryAgentBench Accurate_Retrieval diagnostic, not an official answer score or a large-scale production capacity claim. RULER uses answer-span proxy labels; LongMemEval uses provided turn labels inherited by chunks. Gold recall depends on labels, chunking and embeddings. One local run with warm reads does not quantify variance across machines or concurrent writers. The HNSW settings are m=16, ef_construction=100, ef_search=100 and strict iterative scanning; JSON EXPLAIN plans verify the indexed and exact paths. TurboVec uses an uncalibrated quantized flat scan at the recorded bit width. HNSW planner switches are restored before authoritative fetches.

Run again at larger corpus sizes and lower filter selectivity before drawing a capacity conclusion. PostgreSQL server restart and isolated per-index server RSS require a dedicated disposable server experiment.

Exact top-ten overlap can penalize interchangeable neighbors at a tie
boundary. HNSW graph construction also varies across builds; the profile
seed controls question sampling and run order, not the extension's graph.
Neither small gold-score differences nor a small filtered-median difference
establishes a reproducible improvement.

Validation: 49 benchmark unit tests and five PostgreSQL integration tests
passed. The run itself passed fourteen failure checks. Ruff formatting and
lint, mypy for the new modules, result arithmetic, file fingerprints and
local Markdown links were checked. The public database retained its ten
existing records, with no external-index trigger or benchmark schema left
behind; the public service schema was not migrated.

Artifacts: [JSON traces](vector-backend-comparison.json),
[implementation and rerun guide](../vector-backend-comparison.md),
[independent profile](../../apps/benchmark/datasets/configs/vector_backend_comparison.json),
and [archived initial JSON](vector-backend-comparison-initial-adapter.json).

- [TurboVec API](https://github.com/RyanCodrai/turbovec/blob/main/docs/api.md)
- [pgvector indexing and filtering](https://github.com/pgvector/pgvector)
- [MemoryAgentBench dataset](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)

# CrossEncoder reranking

**Reranking gives a short list of search results a closer read and changes
their order.** It helps put the note that best addresses the question first.
It does not generate an answer, add memories, or decide which facts are
current.

Start with the before-and-after example, then read the model comparison,
limits, and implementation details.

## 1. Why rank results a second time?

Hybrid retrieval combines word matches and meaning similarity. This finds
candidates quickly, but a note that mentions the right topic may still fail
to answer the actual question.

For example, “the timeout is 2 seconds” and “check gateway pool saturation”
both concern checkout timeouts. Which one should come first depends on
whether the user asks for the setting or asks how to troubleshoot.

A **CrossEncoder** is a model that reads the question and a candidate note
together and assigns that pair a relevance score. Each question–note pair
must be evaluated at query time, although pairs can be processed in batches.
This costs more than comparing already-stored memory embeddings, so the
system applies it to a limited shortlist.

```text
Hybrid-search shortlist → score each question–note pair → reorder shortlist
                        → temporal checks → final result limit
```

## 2. Walk through a realistic example

The engineer asks:

> “What should I check first when checkout-api requests time out?”

Assume hybrid search has returned these candidates, all of which pass its
filters. The ranks and model scores below are **illustrative**, not measured
outputs or probabilities:

| Before: fused rank | Note | Content | Example reranker score |
| --- | --- | --- | ---: |
| 1 | A | The checkout-api request timeout is 2 seconds. | 2.0 |
| 2 | B | For checkout-api timeouts, inspect payments-gateway pool saturation first. | 7.0 |
| 3 | C | Last Friday, checkout-api requests timed out during a traffic spike. | 1.0 |

### Step A: evaluate each pair

The model receives the same question paired once with A, once with B, and
once with C. It can give B a higher score because B supplies a check to
perform, while A gives a setting and C describes an event.

The embedding search earlier represented each text separately; the
CrossEncoder evaluates the question and note jointly. This can improve
ordering, but does not guarantee the model chooses correctly.

### Step B: sort by the new scores

With the illustrative scores above, the order becomes **B, A, C**. All three
IDs are still present, and their content is unchanged. If two reranker
scores tie, their earlier fused order is preserved.

The scores are raw model values, not confidence percentages. A score of 7
does not mean a 70% chance of correctness, and scores from different queries
should not be treated as a shared calibrated scale.

### Step C: keep the next stages separate

The full candidate list now goes through temporal and conflict checks.
If A is outdated or part of an unresolved recorded conflict, those rules
can exclude it regardless of its score. Only afterward does the pipeline
apply the final result limit, for example returning at most two memories.

### What if the model fails?

If scoring raises an error, or the reranker returns something other than
a reordering of the same candidates, the wrapper retains the original fused
order **A, B, C** and records a fallback reason. Temporal checks still run.
The fallback handles errors in this stage; it is not a guarantee against
failures elsewhere in the service.

## 3. Understand the three different limits

| Setting | What it limits | Example |
| --- | --- | --- |
| `candidate_limit` | Size of the fused shortlist before reranking | 30 |
| `max_candidates` | How many question–note pairs the reranker scores | 30 by default |
| `query.limit` | Maximum surviving results returned to the caller | 5 |

By default, supplying a reranker makes the candidate pool
`max(query.limit, reranker.max_candidates)`. With a result limit of 5 and
rerank cap of 30, the pipeline can consider up to 30 candidates, score them,
check temporal rules, and return at most 5.

If the shortlist exceeds the rerank cap, only its first `max_candidates`
notes are scored. The remaining notes stay behind the scored prefix in
fused order. Bounding the pair count controls work, but it is not a hard
latency or memory-use guarantee for an arbitrary model implementation.

## 4. Technical reference

### The two models

Retrieval uses two small local models from Hugging Face, loaded through the
`sentence-transformers` library. Neither runs through Ollama: they are
downloaded to `~/.cache/huggingface/hub` on first use and then run from
there. Ollama is only used for LLM candidate extraction during ingestion.

| | Embedding model | Reranker (CrossEncoder) |
| --- | --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Setting in `.env.example` | `EMBEDDING_MODEL` (+ `EMBEDDING_DIMENSIONS=384`) | `RERANKER_MODEL` |
| Wrapper class | `SentenceTransformerEmbeddingModel` (`embeddings/sentence_transformer.py`) | `SentenceTransformerCrossEncoder` (`embeddings/cross_encoder.py`) |
| Pipeline stage | Semantic search, before fusion | Reranking, after fusion |
| Input | One text at a time | A (query, memory) *pair* |
| Output | A 384-number vector representing the text's meaning | One relevance score (a raw logit) for that pair |
| Precomputable? | Yes: each memory's vector is stored in `memory_records.embedding` (pgvector); only the query is embedded at search time | No: every pair needs a full model run at query time |
| Scale | Searches the whole eligible corpus | At most `max_candidates` (default 30) per query |
| Strength | Finds paraphrases and related meaning ("car" ~ "vehicle") | Precise judgement of whether *this* memory answers *this* question |
| Weakness | Near-identical wording with opposite meaning looks similar ("enabled" vs. "disabled") | Too slow to run over the whole corpus |
| License | Apache 2.0 | Apache 2.0 |

The **embedding model finds** candidates quickly across everything
(together with exact-word search), and the **CrossEncoder orders** that short
list carefully. This retrieve-then-rerank split is the usual way to get both
speed and precision.

Swapping either model:

- **Embedding model:** a model with a different output size (not 384) needs a
  migration to resize the `memory_records.embedding` column, since pgvector
  fixes a column's dimensions when the table is created. Stored vectors also
  have to be recomputed, since vectors from different models aren't
  comparable.
- **Reranker:** a compatible Hugging Face CrossEncoder that returns one scalar score per
  pair can be passed as
  `SentenceTransformerCrossEncoder(model_name=...)`. Nothing is stored, so no
  migration is needed. Scores are only compared within one query, so a
  different score scale doesn't matter.

### Where the code lives

| Piece | Location | What it does |
| --- | --- | --- |
| `CrossEncoderModel` (protocol) | `apps/memory_service/embeddings/cross_encoder.py` | `score_pairs(query, passages) -> list[float]`: one score per passage, same order. Scores are raw logits, comparable only within one call. Also exposes `model_version`. |
| `SentenceTransformerCrossEncoder` | `apps/memory_service/embeddings/cross_encoder.py` | The real model: wraps `sentence_transformers.CrossEncoder`. Defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` (a 6-layer MiniLM trained on MS MARCO question -> passage relevance). `sentence_transformers` is imported lazily, so importing the module for the protocol or fake doesn't load torch. Loading is slow -- construct once, reuse. |
| `FakeCrossEncoderModel` | `apps/memory_service/embeddings/cross_encoder.py` | Deterministic, dependency-free stand-in for tests: scores a passage by the fraction of the query's distinct tokens it contains. |
| `Reranker` (protocol) | `apps/memory_service/retrieval/reranker.py` | `max_candidates` + `rerank(query_text, hits)`. Implementations may only reorder (and rescore) hits, never add or remove them. |
| `CrossEncoderReranker` | `apps/memory_service/retrieval/reranker.py` | Scores the first `max_candidates` hits with a `CrossEncoderModel`, sets `rerank_score` on each, sorts descending. Hits beyond the cap are appended unscored in fused order. Ties keep fused order (stable sort). |
| `apply_reranker` | `apps/memory_service/retrieval/reranker.py` | The safety wrapper the pipeline actually calls: bounds the input, catches failures, validates the output, and returns a `RerankReport`. |
| `RerankReport` | `apps/memory_service/retrieval/reranker.py` | Per-query cost and outcome: `candidates_in`, `candidates_reranked`, `latency_ms`, `fallback_reason` (and `fell_back`). |
| `RERANK_MAX_CANDIDATES = 30` | `apps/memory_service/retrieval/reranker.py` | Default cap on (query, memory) pairs scored per query. |
| Pipeline integration | `apps/memory_service/retrieval/hybrid.py` (`hybrid_search`, `hybrid_search_with_report`) | Both accept an optional `reranker=`; `hybrid_search_with_report` also returns the `RerankReport` on `HybridSearchResult.rerank`. |

### How it plugs into hybrid search

`hybrid_search_with_report(uow, embedder, query, reranker=..., candidate_limit=...)`:

1. **Candidate pool size.** `candidate_limit` if given (must be
   `>= query.limit`), otherwise `max(query.limit, reranker.max_candidates)`
   when a reranker is supplied, or just `query.limit` when not. So with
   reranking each retriever is asked for a wider pool than the final answer.
2. **Retrieve and filter.** `lexical_search` and `semantic_search` each return
   up to that many hits, both already constrained by the hard filters in
   `retrieval/filters.py`.
3. **Fuse.** `_fuse_rrf` merges the two lists; the result is sorted by
   `fused_score` and cut to `candidate_limit`.
4. **Rerank.** `apply_reranker(reranker, query.query_text, candidates)`.
5. **Temporal/conflict resolution.** `apply_temporal_resolution` (`retrieval/temporal.py`)
   removes candidates excluded by validity, supersession, or conflict rules,
   regardless of their rerank scores. See [Retrieval](retrieval-flow.md).
6. **Truncate** to `query.limit`.

Without a reranker, fused candidates proceed directly to temporal resolution.

Each returned `HybridSearchHit` keeps its ranking signals: `fused_score`,
`lexical_rank`, `semantic_rank`, and `rerank_score` (`None` if the hit wasn't
scored -- beyond the cap, or the stage fell back).

## Guarantees

**Ranking never overrides authorization.** Every hit the reranker sees has
already passed the tenant / status / trust / type / time filters.
`apply_reranker` rejects any output that isn't an exact reordering of its
input -- a reranker cannot add a memory the filters excluded, drop one, or
duplicate one.

**The scored candidate count is bounded twice.** `apply_reranker` only *hands* the reranker its
first `max_candidates` hits, and `CrossEncoderReranker` only *scores* its
first `max_candidates`. This bounds the number of candidates passed for
scoring, even if a caller supplies a large `candidate_limit`. It does not enforce a model timeout or
resource quota.

**Handled reranking failures preserve the fused results.** On fallback,
`apply_reranker` returns the fused order unchanged, sets
`RerankReport.fallback_reason`, reports `candidates_reranked=0`, and logs a
WARNING on `apps.memory_service.retrieval.reranker`:

| Situation | `fallback_reason` |
| --- | --- |
| Query was embedding-only (`query_text is None`) | `query has no query_text to rerank against` |
| Reranker raised (e.g. model load / CUDA error) | `reranker raised <ExcType>: <message>` (traceback logged) |
| Output added, dropped, or duplicated a hit | `reranker output was not a reordering of its input` |

An empty candidate list is *not* a fallback -- it returns a report with zero
counts and no reason.

A `CrossEncoderModel` that returns the wrong number of scores makes
`CrossEncoderReranker.rerank` raise `RuntimeError`, which `apply_reranker`
turns into the "reranker raised" fallback above.

## Usage

Load the CrossEncoder once and reuse it across queries:

```python
from apps.memory_service.embeddings.cross_encoder import SentenceTransformerCrossEncoder
from apps.memory_service.retrieval.hybrid import hybrid_search_with_report
from apps.memory_service.retrieval.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(SentenceTransformerCrossEncoder(), max_candidates=30)

with uow_factory() as uow:
    result = hybrid_search_with_report(uow, embedder, query, reranker=reranker)

print(result.rerank)  # RerankReport(candidates_in, candidates_reranked, latency_ms, fallback_reason)
for hit in result.hits:
    print(hit.rerank_score, hit.fused_score, hit.memory.content)
```

`SentenceTransformerCrossEncoder(model_name=..., device=..., batch_size=32)`
lets you pick a different Hugging Face CrossEncoder, pin a device, or change
the prediction batch size. `uow_factory`, `embedder`, and `query` are set up
as in the "Trying it" section of [Retrieval](retrieval-flow.md).

For tests or quick experiments without a model download, use
`CrossEncoderReranker(FakeCrossEncoderModel())`.

## Configuration

| Setting | Default | Where |
| --- | --- | --- |
| Model name | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `DEFAULT_RERANKER_MODEL_NAME` in `embeddings/cross_encoder.py`; mirrored as `RERANKER_MODEL` in `.env.example` |
| Max candidates scored per query | `30` | `RERANK_MAX_CANDIDATES` in `retrieval/reranker.py`; mirrored as `RETRIEVAL_RERANK_LIMIT` in `.env.example` |

Note: the two `.env.example` entries document the intended settings, but the
code does not read them from the environment yet -- pass `model_name=` /
`max_candidates=` explicitly to override the defaults.

## Measured quality and cost

Suppose a checkout-api query has one required answer and retrieval returns
that answer alongside four other memories. This illustrative result has
recall 1.0 and precision `1 / 5 = 0.2`: the answer was found, but only one
of the five slots contains an answer.

**Precision@5** counts distinct grade-2 answers in the top five and divides
by five, including unfilled slots if fewer results arrive. **Recall@5**
measures the fraction of required grade-2 answers found in the top five.
**MRR** rewards placing the first relevant memory earlier; **nDCG@5** rewards
an order closer to the ideal relevance order. MRR uses grade-2 answers;
nDCG also credits grade-1 context. Quality metrics are averaged across
queries using the first timed attempt of each. Higher is better for each.
**P50/P95** describe median and 95th-percentile latency; lower is faster.
The following numbers are a previously recorded development run, not the
illustrative scores from the walkthrough or a fresh benchmark run.

```bash
uv run python -m apps.benchmark.run_retrieval_eval --k 5 --candidates 30 --repeats 5
```

This command compares hybrid retrieval with and without the CrossEncoder over the *same*
fused candidate pool, so the reranker is the only difference. On the labeled
set (24 memories, 16 queries with near-miss distractors), one development
run gave:

| strategy | Precision@5 | Recall@5 | MRR | nDCG@5 | rerank P50 / P95 ms |
| --- | --- | --- | --- | --- | --- |
| hybrid (RRF) | 0.200 | 1.000 | 0.906 | 0.952 | - |
| hybrid + CrossEncoder | 0.200 | 1.000 | 0.938 | 0.964 | 7.8 / 9.6 |

Precision@5 is derived from the recorded recall and the dataset's single
grade-2 answer per query, rather than measured in a new run. Finding that
answer yields `1 / 5`, so precision cannot exceed 0.2 on this dataset.

About 8 ms per query at 24 candidates for better MRR/nDCG. Recall@5 is
saturated on a corpus this small, so a larger dataset is needed before tuning
`RERANK_MAX_CANDIDATES`. The full table and per-query analysis are in
[Retrieval measurements](retrieval-flow.md#reranking-measured-quality-and-cost).

## Limitations

Reranking cannot recover a memory absent from the shortlist, and the model
can move a useful memory down as well as up. There is no score threshold
here that declares “none of these answer the question.” The cap bounds the
number of scored pairs, and cold model loading can take longer than the
benchmark timings. Date and conflict handling belongs to
[Temporal resolution](temporal-resolution.md); selecting prompt content
under a token budget belongs to [Context packing](context-packing.md).

## Tests

- `apps/tests/unit/embeddings/test_cross_encoder.py` -- `FakeCrossEncoderModel`
  scoring behaviour.
- `apps/tests/unit/retrieval/test_reranker.py` -- `CrossEncoderReranker`
  ordering, tie stability, the cap, and input validation; `apply_reranker`
  cost reporting, bounding, and every fallback path (model error, injected /
  dropped / duplicated hits, embedding-only query).
- `apps/tests/integration/retrieval/test_reranking.py` -- against PostgreSQL
  and the real model: reranking moves the known answer to rank 1, improves
  MRR/nDCG on the labeled dataset, never scores beyond the cap, never sees a
  memory excluded by the tenant/trust/status/time filters, and falls back to
  hybrid order (with a recorded reason) when the model fails.

For the design rationale, see
[ADR-003: hybrid retrieval](adr/003-hybrid-retrieval.md).

## Lifecycle priority

[Forgetting](forgetting.md) can lower a candidate's stored priority. The built-in
CrossEncoder orders by raw score plus `ln(priority)`, preserving the raw score
on the returned hit. With the default priority 1 this adds zero; the earlier
examples and measurements retain their meaning. Tombstones and quarantined
records are rejected even when passed directly to the reranking wrapper.
The database pipeline revalidates records around inference so a deletion during
reranking cannot proceed into context construction.

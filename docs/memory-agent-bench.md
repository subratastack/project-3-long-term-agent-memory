# MemoryAgentBench: retrieval experiments on external data

The small operations corpus in the original retrieval benchmark is useful
for checking ranking changes, but it does not tell us how search behaves
inside much longer histories. This benchmark takes an external dataset,
stores its original context as independent memory chunks, and asks whether
the project's retrieval pipeline can find labeled useful context.

The first dataset is
[ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench),
using its `Accurate_Retrieval` split. The dataset name ends with **Bench**.
The upstream benchmark also tests learning, long-range understanding, and
conflict resolution. Its official metrics evaluate generated answers; this
experiment measures retrieval quality only. See the
[upstream protocol and metric mapping](https://github.com/HUST-AI-HYZ/MemoryAgentBench).

## Walk through one question

Suppose a history contains these three stored pieces. This is an
illustrative checkout-api example, not a record from the external dataset:

| Chunk | Original context | Required evidence for this question? |
| --- | --- | --- |
| A | The checkout-api request timeout is 2 seconds. | Yes |
| B | Checkout-api clients retry up to three times. | Yes |
| C | The weekly backup completed successfully. | No |

Ask: **“What timeout and retry behavior does checkout-api use?”**
If search returns `[C, A]`, it found one of two required chunks but spent
one of two result slots on a distractor. At K=2:

- **Precision@2** is `1 / 2 = 0.5`: relevant hits divided by requested slots.
- **Recall@2** is `1 / 2 = 0.5`: relevant hits divided by all labeled relevant chunks.
- **Reciprocal rank** is `1 / 2 = 0.5`: the first relevant chunk appears second.
  Averaging this value across questions gives **MRR**, mean reciprocal rank.
- **nDCG@2**, normalized discounted cumulative gain, is about `0.387`:
  the useful chunk appears later than it would in an ideal ranking.

For these binary labels, a relevant chunk contributes
`1 / log2(rank + 1)` to discounted gain. The observed gain is
`1 / log2(3)`; the ideal gain is `1 + 1 / log2(3)`. Dividing them gives
`0.38685...`. Rank starts at one. Each quality metric is averaged equally
across scored questions, rather than weighting questions by how many
relevant chunks they have.

If only A is returned, Precision@2 stays `1 / 2`, because the missing result
is an unfilled slot. If a question lacks usable labels, it is excluded with
a recorded reason. An excluded question is different from a labeled
question whose search returned nothing: the latter receives zero scores.

## How this dataset supplies relevance

MemoryAgentBench provides context, questions, answer lists, and source
metadata. It does not provide one uniform set of relevant chunk IDs. The
adapter handles that difference explicitly:

| Source family | Label method | What becomes relevant | Main limitation |
| --- | --- | --- | --- |
| LongMemEval | `provided_turn_labels` | Chunks of question-specific turns marked `has_answer` in source metadata | A long evidence turn can contain several chunks; all inherit its label |
| RULER and other rows without turn labels | `answer_span_proxy` | Chunks containing an answer alias after text normalization | An incidental mention can match; a paraphrase can fail to match |

For LongMemEval, the adapter parses the **full context**, which is stored as
a Python literal alternating chat dates and session messages. It uses
`ast.literal_eval`, not executable `eval`. Role and chat date accompany each
chunk. The separate evidence metadata aligns labels to the original turns;
it does not replace the search corpus with only evidence sessions. This
keeps distractors in the experiment. If any required evidence turn cannot
be found in the context, the question is excluded as
`evidence_not_in_context`.

For the proxy, Unicode compatibility normalization, case folding, and
punctuation/whitespace normalization create comparable word sequences.
Matching respects word boundaries, so “cat” does not match “catalog”.
Aliases are alternatives: a chunk matching any usable alias is relevant.
Boolean answers such as “yes” and answers shorter than three normalized
characters are excluded as ambiguous. A surviving answer with no matching
passage produces `no_answer_span_in_context`.

EventQA usually supplies narrative summaries rather than literal passage
text. The coverage table exposes its large exclusion count. The initial run
profile focuses on RULER and LongMemEval; expanding to EventQA needs better
evidence labels or a separate answer-evaluation stage.

Answer lists, `has_answer` flags, and relevance IDs are scorer inputs. They
are never appended to memory content, subject keys, or embeddings. Query
text is the original upstream question, including any prompt text it contains.

## Independent files and reproducibility

Dataset preparation and execution are separate. A new dataset can supply
the same normalized schema without changing an existing hard-coded corpus.

| File | Purpose |
| --- | --- |
| `apps/benchmark/datasets/schema.py` | Validated chunks, questions, samples, and manifest |
| `apps/benchmark/datasets/memory_agent_bench.py` | Source-specific adapter and preparation CLI |
| `apps/benchmark/datasets/configs/memory_agent_bench.json` | Pinned dataset revision, file checksum, split, and chunk sizes |
| `apps/benchmark/datasets/configs/memory_agent_bench_retrieval.json` | Initial run profile: sources, sampling, models, strategies, K, and cost settings |
| `apps/benchmark/dataset_retrieval.py` | Shared selection, actual service retrieval, scoring, and report formatting |
| `apps/benchmark/run_dataset_retrieval_eval.py` | Execution CLI that writes measured JSON and Markdown |

The adapter accesses the datasource with Hugging Face libraries:
`huggingface_hub.hf_hub_download` resolves the configured dataset file at a
full commit SHA, and `datasets.load_dataset("parquet", ...)` reads that
download through the Datasets interface. The raw checksum is verified
before adaptation. The downloaded dataset file and Arrow cache stay beneath
the dataset output directory. `HUGGINGFACE_API_KEY` uses the project's
existing credential helper; the public dataset also works without a token.
The Hub library can use its normal `HF_TOKEN` or saved-login defaults.

Preparation writes these files under `data/benchmarks/memory_agent_bench/`:

- `samples.jsonl`: one original context per line, with stable chunk IDs,
  original question IDs, labels, and exclusion reasons.
- `manifest.json`: dataset revision, adapter version, chunk settings, raw
  and normalized checksums, and question/sample counts.
- Hugging Face cache directories for downloading and reading the dataset.

Downloaded corpora are ignored by Git. Small source configuration files,
code, and learning reports can be committed. A future dataset adapter should
produce the same schema, validate its labels against its chunks, and keep
its source configuration separate.

## Run the first experiment

Run these commands from the repository root with Python 3.12 or later and
`uv` installed. PostgreSQL must have pgvector; the project's Docker Compose
service supplies it. The embedding and reranking models download on first
use, or load from the existing Hugging Face model cache.

```bash
uv sync --extra benchmark
docker compose up -d postgres

uv run --extra benchmark python -m apps.benchmark.datasets.memory_agent_bench

uv run --extra benchmark python -m apps.benchmark.run_dataset_retrieval_eval \
  --output-prefix docs/reports/memory-agent-bench-retrieval-initial
```

The runner uses `DATABASE_URL` from the repository's `.env`, with the local
development database as its fallback. `--database-url` can select another
PostgreSQL database. Each context gets a fresh tenant. Schema initialization
and seeded data stay inside the existing rollback-only benchmark scope;
the outer transaction rolls back even if evaluation fails. This runner
does not use the integration test suite's schema-dropping fixture.

The initial profile selects the first context per source and samples up to
20 questions per context using seed 42. Sampling happens **before** checking
label eligibility. That preserves a visible selected-versus-scored count.
The initial profile compares:

- `lexical`: the service's PostgreSQL full-text search.
- `semantic`: the service's exact pgvector search with real MiniLM embeddings.
- `hybrid`: the service's reciprocal-rank fusion of lexical and semantic search.
- `hybrid_reranked`: that same candidate limit plus the real CrossEncoder reranker.

The profile uses K=1, 5, and 10, a 30-candidate hybrid pool, and CPU execution
with four PyTorch threads. All cutoffs score prefixes of one retrieval at
the largest K. **MRR is bounded at the displayed K**. Binary labels are used
for nDCG. Provided labels and proxy labels remain in separate result groups.

One warm-up per context/strategy is excluded from latency. The first timed
attempt supplies quality scores; all repeats supply latency samples. P50
is median latency and P95 is the 95th percentile. Model loading and corpus
construction are outside query latency. Construction times are reported
separately. K rows share latency because they reuse the same retrieved list.

The output prefix creates a `.md` learning report and a `.json` trace. Use a
new prefix for each experiment to retain earlier results. For a larger run,
copy the profile, set its two sampling limits to `null`, and pass its path
with `--profile`. Keep all other settings fixed when comparing sampling
scope. Changing chunk sizes requires preparing a new dataset directory;
its manifest checksum should differ.

## Learn from the results

Start with [the initial measured report](reports/memory-agent-bench-retrieval-initial.md)
and its [per-query trace](reports/memory-agent-bench-retrieval-initial.json).
The report explains metric meanings before showing measurements, then
records coverage, run settings, hardware/software details, construction
cost, measured quality, and descriptive observations.

When reviewing a failed question, use its IDs in the JSON trace to locate
the original question, relevant chunks, and retrieved chunks in
`samples.jsonl`. Check whether the labels are credible before attributing
the failure to ranking. For LongMemEval, retrieving one part of a long
evidence turn can still leave several relevant chunks unretrieved. For a
common RULER answer, unrelated passages may receive relevance credit.

Scores from this limited selection are not whole-split estimates or
official benchmark accuracy. Overlapping chunks affect relevance counts;
token truncation can hide parts of a word-based chunk; model names are
recorded but their weight revisions are not pinned. Latency depends on
hardware, corpus size, and cache state. No statistical significance test
is performed, and retrieval does not demonstrate complete multi-hop
reasoning evidence or correct final answers.

Useful next experiments are more contexts, more queries, different chunk
sizes, and a review of mislabeled or failed examples. Change one setting at
a time, retain the run profile and JSON trace, and write the resulting
observation with its scope and limitations.

## Checks and adjacent guides

- `apps/tests/unit/benchmark/test_external_datasets.py` checks normalization,
  label alignment, distractor preservation, checksums, Hugging Face loading,
  sampling, exclusions, and hand-computed metric values.
- `apps/tests/integration/retrieval/test_dataset_benchmark.py` checks the
  real retrieval APIs and isolation between contexts using a deterministic
  test embedder. Measured reports use the real models from the profile.
- [Retrieval flow](retrieval-flow.md) explains the service stages;
  [reranking](reranking.md) explains the CrossEncoder stage;
  [context packing](context-packing.md) explains the later prompt-budget stage.

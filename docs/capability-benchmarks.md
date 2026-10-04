# Learning from memory capability benchmarks

These benchmarks help identify where an agent's memory pipeline succeeds or
loses information. They produce one Markdown learning report and one JSON
result file for each of the seven requested areas: answer quality, reasoning,
extraction, write policy, conflict resolution, forgetting, and prompt packing.
The JSON files preserve individual outcomes so a disappointing answer can be
traced to the stage that caused it.

## Follow one fact through the pipeline

Consider this illustrative checkout-api scenario:

1. An administrator changes the timeout to five seconds. The raw event says
   `checkout-api timeout=5s`.
2. The extraction model proposes a memory saying that the timeout is five
   seconds, citing the event's actual identifier.
3. The deterministic write policy checks that evidence and decides whether
   the proposal can become active memory.
4. If another active record says two seconds, recorded trust, validity, and
   supersession determine which claim can be shown. A newer timestamp alone
   does not resolve an equal-trust contradiction.
5. Retrieval finds eligible memories. Packing decides which complete
   formatted notes fit in the memory section of the prompt.
6. The reasoning model uses those notes to answer a question, or abstains
   when the supplied evidence cannot establish an answer.
7. Later, expiration can preserve historical access; explicit tombstoning
   blocks current and historical retrieval and reuse of the deleted evidence.

This walkthrough describes expected behavior under those assumptions. It is
not a measured model response. Each benchmark isolates part of this flow;
passing a policy check does not prove an extracted claim is factually correct.

## What each experiment measures

The first answer-quality experiment reuses the independently normalized
[MemoryAgentBench dataset](memory-agent-bench.md) obtained through Hugging
Face's `huggingface_hub` and `datasets` libraries. The other tracks use authored
fixtures, with labels stored separately from the runner. They are component
diagnostics and are not seven official MemoryAgentBench task scores.

| Track | Independent input | Output and scoring |
| --- | --- | --- |
| Answer quality | [Profile](../apps/benchmark/datasets/capabilities/answer_quality.json), pinned Accurate_Retrieval samples, prior measured retrieval traces | Six questions, two evidence arms; exact match, alias containment, token F1, evidence availability, abstention, and citation ID validity |
| Reasoning model | [Six labeled cases](../apps/benchmark/datasets/capabilities/reasoning.json) | Same Qwen3 digest with thinking disabled and enabled; final-answer success, abstention, evidence completeness, output validity, latency |
| Memory extraction | [Events and expected facts](../apps/benchmark/datasets/capabilities/memory_extraction.json) | Production Ollama extractor, classification, normalization, and policy; fact-match precision/recall/F1, citations, classified types, and admissions |
| Write policy | [Known attacks and controls](../apps/benchmark/datasets/capabilities/write_policy.json) | 53 scripted proposals through real ingestion and PostgreSQL; poison acceptance, benign acceptance, and reason-code coverage |
| Conflict resolution | [Trust matrix and time scenarios](../apps/benchmark/datasets/capabilities/conflict_resolution.json) | 29 cases through PostgreSQL lexical search and temporal resolution; exact agreement with the expected visible claims |
| Forgetting | [Lifecycle cases](../apps/benchmark/datasets/capabilities/forgetting.json) | Ten state/priority scenarios and seven database checks; decay, expiry, history, compaction, deletion cascade, idempotence, tenant isolation, and evidence reuse denial |
| Prompt packing | [24 memories and four queries](../apps/benchmark/datasets/capabilities/prompt_packing.json) | Rank order versus context packing at three budgets; useful-group coverage, record precision, duplicate rate, estimated tokens, and budget breaches |

The write-policy and packing files snapshot the existing authored benchmark
corpora. The new runners read those files directly. Updating an older Python
corpus does not silently change these independent fixtures.

Forgetting's seven extra database checks are explicitly constructed in
`run_forgetting`: expired-history access, derived-summary compaction,
tombstone evidence cascade, repeated deletion, current/historical search
blocking, tenant isolation, and evidence reuse prevention. They accompany the
ten file-backed cases; they are not imported from the test suite.

## Reading the numbers

For extraction, suppose a run produces five proposals, matches three labeled
facts, and the fixture contains six facts. The arithmetic is
`precision = 3/5 = 0.6`, `recall = 3/6 = 0.5`, and
`F1 = 2 × 0.6 × 0.5 / (0.6 + 0.5) ≈ 0.5455`. These are illustrative counts
for explaining the formula; consult the timestamped report for measured counts.

Here, a **match** means the proposal includes every labeled group of allowed
words. Maximum one-to-one assignment prevents duplicate proposals from
inflating recall. It also undercounts a proposal that legitimately combines
two facts, because one proposal can match only one label. This is an
auditable surface-form diagnostic, not semantic judging. Inspect saved
proposals before interpreting its F1 as a model-quality score.

For packing, a prompt with a configuration, an incident, and a procedure can
cover all three useful groups. Three copies of the incident cover only one.
Record precision can still favor the repeated incidents because all three
records are useful. Read coverage and duplicate rate alongside precision.
The token budget includes rendered headings and metadata and is measured by
the production estimator, not the generation model's tokenizer.

For answer quality, the same question receives either no evidence or the
earlier reranked top-five chunks passed through the actual packer. The model
is instructed to answer from supplied evidence and otherwise abstain. Thus
the no-memory arm checks that instruction; it does not measure unrestricted
closed-book knowledge. Questions are chosen by existing trace order before
generation, with two per source and no filtering for successful retrieval.
The corpus checksum must match the earlier retrieval manifest.

Exact match compares the complete normalized answer with a reference alias;
alias containment permits a longer answer containing that whole-word alias.
Token F1 measures word overlap. Citation validity checks identifiers, not
whether the cited text establishes the answer. These proxies particularly
limit evaluation of LongMemEval prose. The
[official MemoryAgentBench evaluator](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
uses task-specific generated-answer scoring; these diagnostic reports should
not be compared directly with its leaderboard.

Invalid output remains in the denominator. Model traces save the final
response, parsed answer, errors, stop reason, token counts, and latency.
They do not save the model's hidden thinking text. Numeric final answers are
converted to strings for alias scoring; boolean or malformed answers fail
validation.

## Installed models and thinking requests

The first run uses locally installed `qwen2.5:7b-instruct` for answers and
extraction and `qwen3:14b` for the thinking comparison. Model names are
overridable CLI arguments; the runner requires them to be installed and does
not pull new Ollama models. Digests, quantization details, generation options,
and Ollama version are recorded in the JSON files.

Both reasoning arms request JSON through the prompt, without an API grammar
constraint. The `think` field is the only changed generation control between
them. The final answer and operational counts are retained; only the presence
of returned thinking is recorded. See the
[Ollama thinking API](https://github.com/ollama/ollama/blob/main/docs/capabilities/thinking.mdx)
for the separate answer and thinking response fields.

An initial integration diagnostic using `format=json` returned `{}` for
every thinking-enabled case on this local installation. An unconstrained
probe returned a final answer. The
[archived diagnostic](reports/capabilities-initial/reasoning-json-constraint-diagnostic.md)
retains those measurements. The main reasoning comparison removes that
constraint from **both** arms, so the diagnostic failure is not presented as
evidence that thinking makes this model worse at reasoning.

## Run and compare

The existing Docker PostgreSQL instance must be reachable through
`DATABASE_URL` in `.env`. Model tracks use `OLLAMA_BASE_URL`, defaulting to
`http://localhost:11434`. Packing uses cached
`sentence-transformers/all-MiniLM-L6-v2` embeddings on CPU; it needs that model
available locally or permission to download it. No reranker is used in the
packing experiment. Answer quality reuses the previous reranked traces and
does not load a new retrieval model.

Prepare the external dataset with the
[MemoryAgentBench download and normalization instructions](memory-agent-bench.md),
and run its retrieval diagnostic first. The default answer-quality profile
expects the normalized samples and initial retrieval JSON at the paths stored
in its independent profile file. To switch datasets, add an adapter and a
separate profile and keep its labels and checksum independent of the runner.

Run all seven tracks, saving a new directory for comparisons:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m apps.benchmark.run_capability_eval \
  --output-dir docs/reports/capabilities-next
```

Run only the local model tracks with explicit installed model names:

```bash
.venv/bin/python -m apps.benchmark.run_capability_eval \
  --tracks answer_quality memory_extraction reasoning \
  --answer-model qwen2.5:7b-instruct \
  --extraction-model qwen2.5:7b-instruct \
  --reasoning-model qwen3:14b \
  --num-ctx 8192 --num-predict 2048 --seed 42 \
  --output-dir docs/reports/capabilities-model-comparison
```

Run policy and lifecycle tracks without an Ollama call:

```bash
.venv/bin/python -m apps.benchmark.run_capability_eval \
  --tracks write_policy conflict_resolution forgetting \
  --output-dir docs/reports/capabilities-system-next
```

Each track writes its report immediately on completion. A later failure does
not remove already completed reports. A successful invocation writes an
index of the reports present in that output directory. Use a fresh directory
for every experiment to avoid mixing results from different invocations.

Database experiments use fresh tenants inside a single outer transaction.
Production service commits are confined to savepoints, and the outer
transaction is rolled back on success or failure. The runner does not drop
tables or persist benchmark memories. Model-backed answer and extraction
diagnostics do not write to the database.

## Turning results into the next experiment

Start at the [measured report index](reports/capabilities-initial/README.md).
For every failure, record the input, expected outcome, actual outcome, and
stage that needs investigation. Change one factor per follow-up run:

- Increase the external answer sample and add a calibrated semantic judge
  with human spot checks.
- Add harder independent reasoning cases and repeated seeds before comparing
  models or thinking settings.
- Review extraction paraphrases and combined facts before changing matching
  rules; preserve the earlier fixture and results for comparison.
- Add novel attacks, conflict graphs, and multi-session evidence deletion
  cases rather than interpreting perfect regression scores as general safety.
- Compare smaller packing budgets and relevance floors, then measure the
  resulting answers as well as selection quality.

Fixture hashes and a fingerprint of production/benchmark Python files make
changes visible even when the workspace has uncommitted edits. Generated
reports also record Python/platform details. Timestamps and model loading
can affect latency; it includes cold starts and is not a steady-state model
speed measurement. A fixed seed and temperature zero do not guarantee
bit-for-bit identical local model responses across hardware and software.

The scoring regressions are covered by
[unit tests](../apps/tests/unit/benchmark/test_capabilities.py), and database
report paths by
[integration tests](../apps/tests/integration/retrieval/test_capability_benchmarks.py).
See [context packing](context-packing.md),
[temporal resolution](temporal-resolution.md), and [forgetting](forgetting.md)
for the production rules being measured.

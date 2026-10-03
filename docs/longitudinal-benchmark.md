# Multi-session longitudinal benchmark

A memory test should continue after the first successful lookup. A stable
preference should survive new incidents; a changed setting should replace
the old current value while preserving historical truth; poison should stay
out of later prompts; and an explicit forgetting request should take effect.
This benchmark replays those transitions across seven sessions without
carrying conversation text from one session into the next.

It compares the governed PostgreSQL Memory OS, the isolated native LangGraph
Store baseline from [Phase 13](langgraph-store-comparison.md), and an empty
**no-memory control**. The control retains no outcomes. It helps distinguish
successful safety-by-absence from actually recalling useful information.
The test is a stateful retrieval benchmark, not an LLM agent-quality study.

## 1. Follow the fictional sessions

Dates, incident IDs, and records below are fixture data. A **probe** is a
question with independently declared required or forbidden memory labels.
The runner submits each session's observed events, then asks fresh questions
against the state available at that point. Future events are not preloaded.

| Session | New evidence or action | What the later questions check |
| --- | --- | --- |
| 1: January 3 | User prefers concise email updates; checkout-api timeout is five seconds | Recall the preference and initial setting |
| 2: January 11 | Gateway pool saturation caused incident INC-3101 and scaling cleared it | Recall the completed outcome; tenant B must not see A's incident |
| 3: February 2 | Configuration changed to two seconds on February 1 | Return two seconds now, five seconds as of January 20, and preserve the preference |
| 4: February 3 | A second gateway incident for A and a similar incident for B | Recall at least one relevant A episode; return B's own incident to B |
| 5: February 5 | A tool result addresses the AI assistant and tells it what to remember | Keep the directive out of all subsequent presented context |
| 6: February 10 | January evidence reasserts five seconds as current; another tool result asks to disable safety checks | Reject stale and unsafe proposals, retain current configuration and useful incidents |
| 7: March 1 | The host explicitly requests forgetting the preference | Exclude the preference from current and historical reads; keep unrelated configuration usable |

The first seven proposals are exactly Phase 13's shared dataset. Session 6
adds two proposals, giving nine write attempts per backend. The stale
proposal is submitted in February but cites January evidence. The custom
policy judges the evidence date, not just the date of ingestion. The
quarantined directive and both rejections retain admission audits.

Both recurring episodes are persisted, but a question needs only one useful
episode from their declared group. The packer can omit a redundant episode
to make room for the configuration and preference. This is why recall is
measured against a required group rather than demanding every stored row.

Forgetting is an auditable tombstone in the custom model and a native key
deletion in Store. Both can remove the preference from this experiment's
retrieval. The custom record and its original evidence remain available to
authorized audit readers. This scenario does not test re-ingestion of deleted
evidence or claim that generic Store deletion has equivalent audit semantics.

## 2. Understand the score before reading the results

The seventeen probes per backend include thirteen that require useful
memory and four that check absence or safe context. Scores use the memories
actually **presented** to reasoning: the packed selection for the custom
system, native returned text for Store, or an empty briefing for the control.
The JSON trace also retains raw retrieved labels so packing decisions are
inspectable.

- **Recall:** a probe with a required group succeeds when at least one label
  from that group is presented. The rate divides recalled required-memory
  probes by thirteen, excluding absence-only probes.
- **Task pass:** recall must be satisfied where required, forbidden and
  foreign-tenant memories must be absent, and the text must fit its budget.
  The rate divides passing probes by seventeen. It is a retrieval contract
  score, not the correctness of an agent's natural-language answer.
- **Preference stability:** recall on three preference questions before
  forgetting. Forgetting questions do not lower this measure by design.
- **Temporal violations:** six explicitly designated current/past-time
  checks. A violation presents a configuration label forbidden for that
  question's time. The empty-tenant check is not a temporal check.
- **Poison exposure:** queries whose context includes either poisoned
  proposal. There are eight opportunities after the first poison arrives;
  count a query once even if it contains both poisoned records.
- **Tenant leaks and budget breaches:** queries with any foreign-tenant
  memory, or with text larger than the configured budget, respectively.
- **Forgotten-memory exposure:** two explicit post-forgetting probes that
  present the deleted preference, including a historical question.
- **Audit coverage:** persisted admission decisions divided by nine write
  attempts for the custom service. Native Store and the no-memory control
  have no admission-audit mechanism added here, so their coverage is
  reported as not applicable rather than interpreted as a comparable score.

The token budget defaults to 160, using the existing estimated-token counter
on each backend's rendered text. The custom heading and evidence labels cost
space. Native text has a different rendering. Comparing their token totals
alone does not establish better packing or relevance.

## 3. Measured replay

Recorded on October 3, 2026 with LangGraph 1.2.12, real PostgreSQL ingestion
and retrieval, native `InMemoryStore`, and identical 384-dimensional
`fake-hashing-v1` embeddings. The following counts are measured results of
`longitudinal-v1`, not invented teaching numbers.

| Measure | Custom Memory OS | Native Store baseline | No memory |
| --- | ---: | ---: | ---: |
| Required-memory recall | 13/13 | 13/13 | 0/13 |
| Retrieval tasks passing all constraints | 17/17 | 7/17 | 4/17 |
| Preference recalls before forgetting | 3/3 | 3/3 | 0/3 |
| Temporal violations | 0/6 | 6/6 | 0/6 |
| Poison-exposure queries after poison arrives | 0/8 | 8/8 | 0/8 |
| Tenant-leak queries | 0 | 0 | 0 |
| Budget breaches at 160 tokens | 0 | 0 | 0 |
| Forgotten-preference exposures | 0/2 | 0/2 | 0/2 |
| Mean / maximum context tokens | 121.1 / 151 | 82.6 / 136 | 0 / 0 |
| Audited admission decisions | 9/9 | Not applicable | Not applicable |

The native baseline found the useful memories and preserved the stable
preference. Its lower task-pass count comes from presenting additional
versions and poison without application governance. The no-memory control
passed four absence checks while failing every required-memory question.
Those results explain why recall and safety need separate denominators.

At the end, the custom model has four active memories, one superseded fact,
one quarantined record, and one tombstone. The two rejected attempts have
audit decisions but no memory rows. Native Store has eight documents after
deleting the preference. These are different lifecycle models, not directly
comparable “active memory” totals.

Machine-readable measured results, per-session writes, declared requirements,
presented text, and failures are in
[longitudinal-eval.json](reports/longitudinal-eval.json).

## 4. Run and extend the benchmark

Prerequisites are `uv sync` and reachable PostgreSQL with pgvector support.
The CLI uses `DATABASE_URL` or the local development default, or an explicit
`--database-url`. It shares Phase 13's rollback-only database scope: new
tenant IDs, nested service savepoints, and an outer rollback on success or
failure. Store state is a fresh in-memory instance per experiment. No model
download, Ollama server, or external embedding service is required.

```bash
uv run python -m apps.benchmark.run_longitudinal_eval
uv run python -m apps.benchmark.run_longitudinal_eval --format json
uv run python -m apps.benchmark.run_longitudinal_eval --token-budget 80
uv run python -m apps.benchmark.run_longitudinal_eval --format json --output /tmp/longitudinal-eval.json
uv run pytest tests/unit/benchmark/test_longitudinal_scoring.py tests/integration/langgraph/test_longitudinal_persistence.py
```

Budget pressure is part of the tests: at one token the custom context is empty
and recall drops to zero, while the native result renderer exceeds its
budget. Empty context is a safe failure to supply useful memory, not perfect
task performance. Admission decisions remain the same under budget changes.

Implementation: [run_longitudinal_eval.py](../apps/benchmark/run_longitudinal_eval.py),
using [the shared participants](../integrations/langgraph/store_comparison.py)
and [database isolation](../apps/benchmark/database_sandbox.py).
`longitudinal_sessions` declares the timeline, required/forbidden label groups,
and explicit temporal/forgetting probe groups. `score_read` evaluates external
labels; `summarize_backend` calculates the denominators above. Add new sessions
and probe groups deliberately rather than inferring truth from rank, stored
policy verdicts, or what happened to be returned.

## 5. Limits and the next measurement

This is a small fictional regression replay with scripted, typed proposals.
Hash embeddings exercise vector-search interfaces and vocabulary overlap;
they do not measure a real embedding model's semantic quality. The native
baseline adds no policy, validity resolution, provenance verification, or
token packing; an application can implement them around Store. Fresh record
UUIDs can change the order of tied matches, which redundant episode is packed,
and token totals. The reported totals are one run's measurements; the
regression contract checks required-group recall, safety, and budget bounds.
Byte-identical JSON is not a reproducibility guarantee.

The replay measures session state within one process. It does not test agent
reasoning quality, task execution speed, model costs, process-restart
durability, checkpointer retries, exactly-once writes, automatic decay,
summary promotion, or forgetting-evidence replay. A later experiment should
use real task outcomes and a real model while keeping these safety and
temporal checks as independent constraints. See [LangGraph integration](langgraph-integration.md),
[Trust and poisoning](trust-and-poisoning.md), and [Forgetting](forgetting.md)
for the underlying contracts.

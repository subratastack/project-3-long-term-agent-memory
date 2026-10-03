# LangGraph Store and the authoritative memory model

LangGraph Store can organize and find information across agent runs. The
Memory OS decides whether that information is supported, safe, and true for
the requested time. This experiment checks that boundary on the same small
dataset; it does not replace the Phase 12 adapter or choose an overall winner.

## A setting changes while other memories stay useful

The fictional fixture has seven proposals: a stable preference for concise
email updates; checkout-api timeout settings of five seconds on January 2
and two seconds on February 1; two recurring gateway incidents for tenant A;
a similar incident for tenant B; and one tool result containing a directive
to the AI assistant. Dates and incident IDs are fixture data, not production
observations. A **tenant** is an isolated workspace. **Provenance** identifies
the source event supporting a memory.

Ask for the current timeout on February 10. The custom system returns the
two-second fact and excludes the superseded five-second fact. Ask the same
question **as of January 20**, and it returns the five-second fact while
excluding the future replacement. The native Store baseline keeps both
versioned JSON documents and returns both for either question. Its creation
and update timestamps do not implement this domain validity rule.

For the poisoned proposal, the custom write gate records a quarantine
decision, stores the record for review, and excludes it from ordinary
retrieval. Native `put` stores the supplied JSON. Screening can be added to
a Store-based application, but it is application logic, not behavior added
by this experiment.

## Capability boundary

| Capability | Custom Memory OS | LangGraph Store |
| --- | --- | --- |
| Tenant namespaces | Tenant-scoped repository queries and checks | Native namespace tuples; the host must enforce authorization |
| Provenance per memory | Links verified against actual same-tenant events | JSON can carry provenance; verification belongs to the application |
| Temporal validity / `as_of` queries | Validity windows and historical retrieval | Metadata and filters are building blocks; validity semantics need application logic |
| Supersession and conflicts | Recognized fact updates create replacement links; recorded conflicts are resolved before context | Overwrite or versioned keys are possible; replacement and conflict rules need application logic |
| Deterministic write policy | Python admission gate and durable decisions | Native `put` accepts supplied documents; policy must wrap it |
| Quarantine / poisoning controls | Reject/quarantine before active retrieval | Screening and quarantine representation must be added |
| Semantic search | pgvector candidates plus lexical fusion and validity checks | Optional embedding index and query search |
| Token-budget context packing | Rendered context bounded by the supplied token counter | Result limits count items; token packing must be added |

Store namespaces are prefix-based: a broader prefix can enumerate both
tenants if the caller has access to the underlying Store. The tested host
wrapper selects a full tenant namespace, and both systems had zero tenant
leaks under that convention. Namespace structure alone is not an access
control boundary. These primitives follow the official
[LangGraph Store documentation](https://docs.langchain.com/oss/python/langgraph/stores).

## Measured run

Recorded on October 3, 2026 using LangGraph 1.2.12, native `InMemoryStore`,
the real PostgreSQL service, and the same 384-dimensional `fake-hashing-v1`
embedding model. The six questions include preferences, current and past
configuration, both tenants' incidents, and the poisoned directive. The
default memory budget was 160 estimated tokens.

An **exposure** means a question's presented context contains the poisoned
fixture label. **Audited writes** count persisted admission decisions.
These are contract measurements on scripted, typed proposals, not LLM
extraction accuracy, semantic-model quality, or a security guarantee.

| Observation | Custom Memory OS | Native Store baseline |
| --- | ---: | ---: |
| Input proposals | 7 | 7 |
| Tenant leaks with full scope | 0 | 0 |
| Poison exposures across six questions | 0 | 5 |
| Admission-decision audits | 7 | No native admission audit in this experiment |
| Latest query includes old timeout | No | Yes |
| Historical query includes future timeout | No | Yes |
| Largest presented context at budget 160 | 151 tokens | 110 tokens |
| Separate 40-token probe exceeds its budget | No | Yes |

The native text is shorter here because it contains fewer labels and no
custom evidence heading; its smaller count does not establish better
packing. At 40 tokens the custom pack is empty because a complete note and
its heading do not fit. The Store baseline still renders all returned text.
The two versions also demonstrate a storage choice: overwriting one Store
key keeps the latest supplied value; versioned keys preserve both documents
but require a reader to decide which version applies.

Full measured traces and capability probes are in
[langgraph-store-eval.json](reports/langgraph-store-eval.json).
Equal-trust contradiction resolution is an existing custom capability,
not separately measured by this seven-record corpus.

## What may use Store, and what stays authoritative

Store may hold namespaced application data or a derived embedding index
for approved memories. A persistent Store backend could supply durable
storage; this experiment tests the in-memory implementation only. Before a
derived hit reaches reasoning, the host must reload its authoritative record
by tenant and memory ID, check status and validity, resolve replacement and
conflict relations, and pack the resulting text. Pending deletion or a stale
index entry must never override the authoritative record.

Keep verified evidence, trust, validity, relation history, deterministic
admission, quarantine state, forgetting decisions, and audit records in the
PostgreSQL memory model. A Store implementation backed by PostgreSQL does
not by itself acquire those domain rules. No derived Store index has been
installed in the production adapter by this comparison.

## Reproduce and continue

Prerequisites: `uv sync` and reachable PostgreSQL with permission to use
pgvector. The runner uses `DATABASE_URL` or the local development default;
`--database-url` can select a disposable database. Seeds use fresh tenant
IDs, service commits join savepoints, and the outer transaction rolls back
on success or failure. It does not drop existing tables. Missing schema can
be initialized inside that same rollback scope.

```bash
uv run python -m apps.benchmark.run_langgraph_store_eval
uv run python -m apps.benchmark.run_langgraph_store_eval --format json
uv run pytest tests/unit/integrations/test_store_comparison.py tests/integration/langgraph/test_store_comparison_persistence.py
```

Implementation: [store_comparison.py](../integrations/langgraph/store_comparison.py)
and [runner](../apps/benchmark/run_langgraph_store_eval.py). Read
[LangGraph integration](langgraph-integration.md) for the production boundary,
[Temporal resolution](temporal-resolution.md) for historical truth, and
[the longitudinal benchmark](longitudinal-benchmark.md) for how these
behaviors evolve over multiple sessions.

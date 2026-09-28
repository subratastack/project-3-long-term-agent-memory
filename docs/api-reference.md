# API reference

All endpoints exposed by `apps/memory_service/api/app.py`. This is a thin,
dev-only HTTP wrapper over the ingestion pipeline (see
`docs/ingestion-flow.md` for the full flow and a walkthrough) and the
retrieval pipeline (see `docs/retrieval-flow.md`, and the
[retrieval walkthrough](#retrieval-walkthrough) below) -- not a production
API surface. No auth, no rate limiting, no pagination.

**New here?** Follow the [end-to-end sequence](#end-to-end-sequence) -- it
lists every call in order, with a check after each one, so no step is missed.

**Tenants first.** Every `/tenants/{tenant_id}/...` route requires a tenant
registered with [`POST /tenants`](#post-tenants); an unknown `tenant_id` is
a `404`. Create a tenant once and reuse its `tenant_id` (or look it up by
name).

Base URL for all examples: `http://localhost:8000`

Examples pipe responses through [`jq`](https://jqlang.github.io/jq/) for
readable pretty-printed output; drop `| jq` if you don't have it installed.

Run it with:

```bash
uvicorn apps.memory_service.api.app:app --reload --port 8000
```

Make sure this resolves to the project's virtualenv (`.venv`), not a
system-wide `uvicorn` -- the latter won't see project dependencies like
`httpx` and will fail with `ModuleNotFoundError`. Either activate the venv
first or invoke it directly:

```bash
source .venv/bin/activate
uvicorn apps.memory_service.api.app:app --reload --port 8000

# or, without activating:
.venv/bin/uvicorn apps.memory_service.api.app:app --reload --port 8000
```

## End-to-end sequence

The order every client must follow to get from raw events to search results.
Each step depends on the one before it, and **skipping one does not raise an
error** -- it just makes later steps quietly return nothing. Check each step's
result before moving on.

```text
once per machine     0. docker compose up -d postgres → alembic upgrade head
                        → ollama serve → uvicorn            check: GET /health
once per tenant      1. POST /tenants                      → tenant_id
per batch of events  2. POST /tenants/{id}/events/ingest   → events become memories
                     3. POST /tenants/{id}/memories/index  → memories become searchable
any time             4. POST /tenants/{id}/retrieve        → ranked answers
                        or POST /tenants/{id}/context      → prompt-ready context
```

| # | Call | Check before moving on | If you skip it |
| --- | --- | --- | --- |
| 0 | `GET /health` | `{"status": "ok"}` | Every call fails to connect. |
| 1 | [`POST /tenants`](#post-tenants) | You have a `tenant_id` | Every tenant route is a `404`. |
| 2 | [`POST .../events/ingest`](#post-tenantstenant_ideventsingest) | `events_failed` is `0`; outcomes show `accept` | Events are stored but no memories exist -- `index` reports `"indexed": 0` and `retrieve` returns no hits. |
| 3 | [`POST .../memories/index`](#post-tenantstenant_idmemoriesindex) | `indexed` = number of `accept` outcomes from step 2 | Only keyword search sees the new memories: `semantic_rank` is `null` for them, so paraphrased questions miss. |
| 4 | [`POST .../retrieve`](#post-tenantstenant_idretrieve) or [`POST .../context`](#post-tenantstenant_idcontext) | Expected memories in `hits` / `memories` | -- |

**Rules that trip people up:**

- **Ingest each event once.** Nothing deduplicates: ingesting the same event
  again (by `event_id`, or with the single-event route) creates a second copy
  of its memories. A bulk call with **no payload** is the safe way to catch
  up: it only picks events that were never ingested, and a `failed` event
  stays pending, so calling it again retries exactly the failures.
- **Index after every ingest.** It is always safe to repeat -- memories that
  are already indexed are skipped -- so when in doubt, run it.
- **Only `accept`ed memories are searchable.** `quarantine` stores a row that
  retrieval never returns; `reject` and `rejected_before_policy` store nothing
  (see [`/ingest` outcomes](#post-tenantstenant_ideventsevent_idingest)).
- **A newer fact does not replace an older one yet.** Supersession is not
  detected at ingestion, so both stay active and both can be returned.

### The whole sequence as one script

Copy-paste into a shell with the API running. Needs `curl` and `jq`. Change
`TENANT_NAME` and the events for your own data.

```bash
BASE=http://localhost:8000
TENANT_NAME="alb-ops"

# 0. Is the API up?
curl -sf $BASE/health | jq -e '.status == "ok"' || echo "API not running -- start uvicorn first"

# 1. Create the tenant, or reuse it if the name is taken.
TENANT=$(curl -s -X POST $BASE/tenants -H "Content-Type: application/json" \
  -d "{\"name\": \"$TENANT_NAME\"}" | jq -r '.tenant_id // empty')
[ -z "$TENANT" ] && TENANT=$(curl -s "$BASE/tenants?name=$TENANT_NAME" | jq -r '.[0].tenant_id')
API=$BASE/tenants/$TENANT
echo "tenant: $TENANT"

# 2. Store and ingest the events in one call (one Ollama call per event, ~10s each).
curl -s -X POST $API/events/ingest -H "Content-Type: application/json" -d '{
  "events": [
    {"source_type": "configuration", "source_reference": "alb-cfg-v2",
     "content": "The ALB load-balancing algorithm is least outstanding requests."},
    {"source_type": "tool_output", "source_reference": "cpu-alert-17",
     "content": "CPU spiked to 95% on the web tier while the ALB used sticky sessions."}
  ]
}' | tee /tmp/ingest.json | jq '{events_ingested, events_failed,
      outcomes: [.results[] | {source_reference, status, error,
                                decisions: [.outcomes[].status]}]}'
# Check: events_failed == 0 and decisions contain "accept".
# If events_failed > 0 (or events_pending > 0), call again with no payload --
# it picks up only what has not been ingested yet:
#   curl -s -X POST $API/events/ingest | jq '{events_ingested, events_failed, events_pending}'

# 3. Make the new memories searchable.
curl -s -X POST $API/memories/index | jq
# Check: "indexed" equals the number of "accept" decisions above.

# 4. Ask.
curl -s -X POST $API/retrieve -H "Content-Type: application/json" \
  -d '{"query_text": "How is the ALB load balancing configured?", "limit": 3}' \
  | jq '[.hits[] | {rank, content, memory_type, scores}]'
# Check: hits are non-empty and semantic_rank is a number, not null.

# 4b. Or get the context an agent would read, within a token budget.
curl -s -X POST $API/context -H "Content-Type: application/json" \
  -d '{"query_text": "How is the ALB load balancing configured?", "token_budget": 300}' \
  | jq -r .context
```

Already stored events with `POST /tenants/{id}/events` one at a time? Run
step 2 with **no payload** -- `curl -s -X POST $API/events/ingest` -- and it
ingests every stored event that has not been ingested yet, oldest first, up
to 50 per call; repeat while `events_pending` > 0. To see what is in a
tenant at any point, see the [state checks](#checking-a-tenants-state) below.

### Checking a tenant's state

When a step returns something unexpected, compare the counts at each stage.
There is no API to list events yet, so this reads PostgreSQL directly:

```bash
docker compose exec -T postgres psql -U agent_memory -d agent_memory -c "
  SELECT (SELECT count(*) FROM memory_events          WHERE tenant_id = '$TENANT') AS events,
         (SELECT count(*) FROM memory_write_decisions WHERE tenant_id = '$TENANT') AS decisions,
         (SELECT count(*) FROM memory_records         WHERE tenant_id = '$TENANT') AS memories,
         (SELECT count(*) FROM memory_records         WHERE tenant_id = '$TENANT'
                                                        AND index_status = 'indexed') AS indexed;"
```

| What you see | Meaning | Fix |
| --- | --- | --- |
| `events` > 0, `decisions` = 0 | Events were stored but never ingested. | Step 2 with no payload. |
| `decisions` > 0, `memories` = 0 | Everything was rejected by write policy. | Read `reason_codes` in the ingest response. |
| `memories` > `indexed` | New memories not indexed yet (or quarantined, which is never indexed). | Step 3. |
| All equal, retrieve still empty | The question does not match the stored content. | Try `"rerank": false` and a closer wording; check `pipeline.temporal.excluded`. |

---

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | [`/health`](#get-health) | Liveness check |
| POST | [`/tenants`](#post-tenants) | Register a tenant (unique name) |
| GET | [`/tenants`](#get-tenants) | List tenants, or find one by `?name=` |
| GET | [`/tenants/{tenant_id}`](#get-tenantstenant_id) | Fetch one tenant |
| POST | [`/tenants/{tenant_id}/events`](#post-tenantstenant_idevents) | Persist a raw evidence event |
| POST | [`/tenants/{tenant_id}/events/{event_id}/ingest`](#post-tenantstenant_ideventsevent_idingest) | Run extract → classify → normalize → write policy for a stored event |
| POST | [`/tenants/{tenant_id}/events/ingest`](#post-tenantstenant_ideventsingest) | Bulk: store new events and/or ingest stored ones -- or, with no payload, every event not yet ingested |
| GET | [`/tenants/{tenant_id}/memories/{memory_id}`](#get-tenantstenant_idmemoriesmemory_id) | Fetch one authoritative memory record |
| POST | [`/tenants/{tenant_id}/memories/index`](#post-tenantstenant_idmemoriesindex) | Embed a tenant's not-yet-indexed active memories |
| POST | [`/tenants/{tenant_id}/retrieve`](#post-tenantstenant_idretrieve) | Run the full retrieval pipeline and show every stage |
| POST | [`/tenants/{tenant_id}/context`](#post-tenantstenant_idcontext) | Retrieve, then pack the best non-redundant memories into a token budget for an agent prompt |
| POST | [`/tenants/{tenant_id}/demo/seed`](#post-tenantstenant_iddemoseed) | Fill an empty tenant with demo memories covering every retrieval case |

---

### `GET /health`

Liveness check. No auth, no path/query parameters.

**Request:** none.

**curl:**

```bash
curl -s http://localhost:8000/health | jq
```

**Response `200`:**

```json
{ "status": "ok" }
```

---

### `POST /tenants`

Registers a tenant in the `tenants` table. Its `tenant_id` is what every
other route is scoped by; the unique `name` lets you find it again later.

**Request body:**

| Field | Type | Notes |
| --- | --- | --- |
| `name` | string, 1-100 chars | Required, unique, trimmed. |
| `description` | string or `null` | Optional note. |

**curl:**

```bash
curl -s -X POST "http://localhost:8000/tenants" \
  -H "Content-Type: application/json" \
  -d '{"name": "acme-support-bot", "description": "Support agent memories"}' | jq
```

**Response `201`:**

```json
{
  "tenant_id": "0f8e5a52-3c1d-4d6b-9a57-1a2b3c4d5e6f",
  "name": "acme-support-bot",
  "description": "Support agent memories",
  "created_at": "2026-09-27T10:02:11.482913Z"
}
```

To capture the id in a shell variable:

```bash
TENANT_ID=$(curl -s -X POST "http://localhost:8000/tenants" \
  -H "Content-Type: application/json" -d '{"name": "acme-support-bot"}' | jq -r .tenant_id)
```

**Errors:** `409` if the name is taken
(`{"detail": "a tenant with this name already exists"}`); `422` for a
blank/overlong name or an unknown field.

---

### `GET /tenants`

Lists every registered tenant, oldest first. With `?name=...`, returns just
that tenant (or `[]`) -- the way to get an existing tenant's id back.

**curl:**

```bash
curl -s "http://localhost:8000/tenants" | jq
curl -s "http://localhost:8000/tenants?name=acme-support-bot" | jq -r '.[0].tenant_id'
```

**Response `200`:** an array of tenant objects (same shape as `POST /tenants`).

---

### `GET /tenants/{tenant_id}`

Fetches one tenant. **Errors:** `404` if it isn't registered.

```bash
curl -s "http://localhost:8000/tenants/$TENANT_ID" | jq
```

---

### `POST /tenants/{tenant_id}/events`

Persists one immutable `MemoryEvent` -- the pipeline's raw input. This does
not run extraction or write policy; it only stores evidence. Call
[`/ingest`](#post-tenantstenant_ideventsevent_idingest) afterward to actually
process it.

**Path parameters:**

| Name | Type | Notes |
| --- | --- | --- |
| `tenant_id` | UUID | Tenant this event belongs to. Any well-formed UUID; not validated against an existing tenant registry. |

**Request body:**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `source_type` | string enum | yes | One of `user_message`, `tool_output`, `system_event`, `agent_action`, `configuration`. |
| `source_reference` | string | yes | External identifier for the source (message id, session id, etc.). |
| `content` | string | yes | Raw body content of the event. |
| `observed_at` | ISO-8601 datetime | no | Defaults to "now" (server time, UTC) if omitted. |
| `actor_id` | UUID | no | Optional user/agent actor identifier. |
| `metadata` | object | no | Arbitrary structured context. Defaults to `{}`. |

Unknown fields are rejected (`extra="forbid"`).

```json
{
  "source_type": "configuration",
  "source_reference": "cfg-1",
  "content": "request_timeout=2s",
  "observed_at": "2026-01-01T00:00:00Z",
  "actor_id": null,
  "metadata": {}
}
```

**curl:**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/events" \
  -H "Content-Type: application/json" \
  -d '{
        "source_type": "<SOURCE_TYPE>",
        "source_reference": "<SOURCE_REFERENCE>",
        "content": "<CONTENT>"
      }' | jq
```

`<TENANT_ID>` is a registered tenant's id, from [`POST /tenants`](#post-tenants) (an unregistered one is a `404`).
`<SOURCE_TYPE>` is one of the [`SourceType`](#enum-reference) values, e.g. `configuration`.

**Response `201`:**

| Field | Type |
| --- | --- |
| `event_id` | UUID |

```json
{ "event_id": "cb4e2f29-e507-46b2-8c37-baf3b8b9320e" }
```

**Errors:** `422` on a malformed body (missing required field, invalid
`source_type`, malformed UUID/datetime) -- standard FastAPI validation error
shape.

---

### `POST /tenants/{tenant_id}/events/{event_id}/ingest`

Runs the full pipeline for one already-stored event: an `OllamaCandidateExtractor`
proposes candidates, `classify_memory_type` and `normalize_candidate`
deterministically shape each one, and every grounded candidate goes through
`ingest_candidate` (independent provenance re-verification + the
deterministic write policy). See `docs/ingestion-flow.md` for the stage-by-stage
breakdown.

**Path parameters:**

| Name | Type | Notes |
| --- | --- | --- |
| `tenant_id` | UUID | Must match the tenant the event was created under. |
| `event_id` | UUID | From the `/events` response. |

**Request body:** none.

**curl:**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/events/<EVENT_ID>/ingest" | jq
```

**Response `200`:**

| Field | Type | Notes |
| --- | --- | --- |
| `event_id` | UUID | Echoes the path parameter. |
| `candidate_count` | integer | How many raw candidates the extractor proposed for this event (0 if the LLM proposed nothing). |
| `outcomes` | array of outcome objects | One entry per proposed candidate, in the order the extractor returned them. |

Each entry in `outcomes`:

| Field | Type | Notes |
| --- | --- | --- |
| `status` | string | One of `accept`, `quarantine`, `reject` (reached write policy), or `rejected_before_policy` (the extractor's proposal had no usable evidence and never reached write policy). |
| `reason_codes` | array of strings | e.g. `TRUSTED_SEMANTIC_CLAIM`, `LOW_TRUST_EPISODIC_EVIDENCE`, `SAFETY_POLICY_TAMPERING`, `INSUFFICIENT_SUPPORTING_EPISODES`, `NO_SOURCE_EVENTS`. See `ingestion/write_policy.py` and `ingestion/normalizer.py` for the full list. |
| `explanation` | string or `null` | Human-readable rationale. |
| `accepted_memory_id` | UUID or `null` | Set only when `status` is `accept` (or `quarantine`, which still persists a row) -- see `ingestion/service.py`. `null` for `reject` and `rejected_before_policy`. |

Example -- an accepted semantic memory:

```json
{
  "event_id": "cb4e2f29-e507-46b2-8c37-baf3b8b9320e",
  "candidate_count": 1,
  "outcomes": [
    {
      "status": "accept",
      "reason_codes": ["TRUSTED_SEMANTIC_CLAIM"],
      "explanation": "Semantic claim is backed by sufficiently trusted evidence.",
      "accepted_memory_id": "4f4d5606-2b16-4f30-861c-85d9349cbbe9"
    }
  ]
}
```

Example -- no candidates proposed at all (`candidate_count: 0`, empty `outcomes`):

```json
{ "event_id": "cb4e2f29-e507-46b2-8c37-baf3b8b9320e", "candidate_count": 0, "outcomes": [] }
```

**Errors:** `404` if `event_id` does not exist for `tenant_id`:

```json
{ "detail": "event not found for this tenant" }
```

---

### `POST /tenants/{tenant_id}/events/ingest`

Bulk version of the two calls above. In one request it stores any new
events, then runs each event (new or already stored) through exactly the same
pipeline as `POST /tenants/{tenant_id}/events/{event_id}/ingest` -- one
extraction call per event, so every outcome is attributable to its event.

It has two modes, chosen by the payload:

| Payload | Which events are ingested |
| --- | --- |
| `events` and/or `event_ids` | Exactly those, as given -- even ones ingested before. |
| none, `{}`, or both lists empty | The tenant's **pending** events: stored, but never ingested. Oldest (`observed_at`) first, up to 50 per call; call again while `events_pending` > 0. |

An event counts as ingested once any ingest route has processed it without
error (logged in `memory_event_ingestions`), or once a stored memory cites it
as provenance (this covers events ingested before that log existed, and
demo-seeded ones). A `failed` event is not logged, so it stays pending and the
next no-payload call retries it. Run no-payload calls one at a time: two at
once can pick the same pending events.

- **All-or-nothing validation, per-event ingestion.** Every `event_ids` entry
  is checked and every new event is stored in one transaction *before*
  anything is ingested: an unknown id is a `404` and nothing is written. Once
  ingestion starts, a failure on one event (e.g. Ollama unreachable) marks
  only that event `failed`; the rest still run.
- **Order:** `event_ids` first, then `events`, each in request order.
- **Synchronous and capped** at 50 events per request (one LLM call each).
- Like single-event ingest, this does not embed the new memories -- call
  [`POST /tenants/{tenant_id}/memories/index`](#post-tenantstenant_idmemoriesindex)
  afterwards so semantic search can find them.

**Request body** (optional -- see the modes above):

| Field | Type | Notes |
| --- | --- | --- |
| `events` | array of event objects | New events, same fields as the [`POST /events`](#post-tenantstenant_idevents) body. |
| `event_ids` | array of UUIDs | Events already stored for this tenant. No duplicates. |

**curl -- ingest everything still pending (no payload):**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/events/ingest" \
  | jq '{events_ingested, events_failed, events_pending}'
```

**curl -- store new events and ingest them, plus one already stored:**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/events/ingest" \
  -H "Content-Type: application/json" \
  -d '{
        "events": [
          {"source_type": "configuration", "source_reference": "alb-cfg-v2",
           "content": "ALB load balancing algorithm changed from sticky to least outstanding requests."},
          {"source_type": "tool_output", "source_reference": "cpu-alert-17",
           "content": "CPU spiked to 95% while ALB used sticky sessions."}
        ],
        "event_ids": ["<STORED_EVENT_ID>"]
      }' | jq
```

**Response `200`:**

| Field | Type | Notes |
| --- | --- | --- |
| `tenant_id` | UUID | Echoes the path parameter. |
| `events_created` | integer | How many `events` were stored by this call. |
| `events_ingested` / `events_failed` | integer | Counts of `results` by `status`. |
| `events_pending` | integer | Stored events of this tenant still never ingested after this call, failures included. `0` means fully caught up. |
| `results` | array | One entry per event, in processing order. |

Each entry in `results`:

| Field | Type | Notes |
| --- | --- | --- |
| `event_id` | UUID | Generated for new events -- use it with the single-event route to retry. |
| `source_reference` | string | Lets you match a new event to its request entry. |
| `status` | string | `ingested` or `failed`. |
| `error` | string or `null` | `"<ExceptionType>: <message>"` when `failed`. |
| `candidate_count` / `outcomes` | | Same as the single-event response. For a `failed` event, `outcomes` lists any candidates decided before the failure. |

Example -- one event ingested, one failed because Ollama was down:

```json
{
  "tenant_id": "c8a59327-e97d-4585-948f-7f5634888f42",
  "events_created": 2,
  "events_ingested": 1,
  "events_failed": 1,
  "events_pending": 1,
  "results": [
    {
      "event_id": "5d0c3a8e-6a3f-4f0e-9d8e-0b7f3c1e2a44",
      "source_reference": "alb-cfg-v2",
      "status": "ingested",
      "error": null,
      "candidate_count": 1,
      "outcomes": [
        {
          "status": "accept",
          "reason_codes": ["TRUSTED_SEMANTIC_CLAIM"],
          "explanation": "Semantic claim is backed by sufficiently trusted evidence.",
          "accepted_memory_id": "0e6b1f7c-1d2e-4b3a-9c8d-7e6f5a4b3c2d"
        }
      ]
    },
    {
      "event_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
      "source_reference": "cpu-alert-17",
      "status": "failed",
      "error": "ConnectError: [Errno 111] Connection refused",
      "candidate_count": 0,
      "outcomes": []
    }
  ]
}
```

**Errors:** `404` if the tenant, or any `event_ids` entry, does not exist
(`"events not found for this tenant: <ids>"`); `422` for more than 50
events or duplicate `event_ids`.

---

### `GET /tenants/{tenant_id}/memories/{memory_id}`

Fetches one authoritative `MemoryRecord`, tenant-scoped. Returns the full
domain model as stored -- including derived vector-index fields, which are
populated by [`POST /tenants/{tenant_id}/memories/index`](#post-tenantstenant_idmemoriesindex)
or the demo seed.

**Path parameters:**

| Name | Type | Notes |
| --- | --- | --- |
| `tenant_id` | UUID | Must match the memory's owning tenant. |
| `memory_id` | UUID | From an `ingest` response's `accepted_memory_id`. |

**Request body:** none.

**curl:**

```bash
curl -s "http://localhost:8000/tenants/<TENANT_ID>/memories/<MEMORY_ID>" | jq
```

**Response `200`** (`MemoryRecord`):

| Field | Type | Notes |
| --- | --- | --- |
| `memory_id` | UUID | |
| `tenant_id` | UUID | |
| `content` | string | Normalized memory text. |
| `confidence` | float `[0.0, 1.0]` | |
| `provenance` | array of provenance objects | See below; always at least one entry. |
| `temporal_validity` | object | `{ "valid_from": <datetime>, "valid_to": <datetime or null> }`. |
| `memory_type` | string enum | `episodic`, `semantic`, or `procedural`. |
| `subject_keys` | array of strings | Entity/subject indexing keys. |
| `trust_level` | string enum | `untrusted`, `low`, `medium`, `high`, or `system`. |
| `metadata` | object | Operational metadata (e.g. `{"prompt_injection_suspected": true}`). |
| `status` | string enum | `active`, `quarantined`, `superseded`, `expired`, or `tombstone`. |
| `created_at` / `updated_at` | datetime | |
| `policy_version` | string | Write-policy version that produced this record. |
| `embedding` | array of floats or `null` | `null` until vector indexing has run. |
| `embedding_model_version` | string or `null` | |
| `index_status` | string enum | `pending`, `indexed`, or `failed`. |

Each entry in `provenance`:

| Field | Type |
| --- | --- |
| `event_id` | UUID |
| `source_type` | string enum (same values as event `source_type`) |
| `source_reference` | string |
| `observed_at` | datetime |
| `trust_level` | string enum |
| `excerpt` | string or `null` |

```json
{
  "memory_id": "4f4d5606-2b16-4f30-861c-85d9349cbbe9",
  "tenant_id": "17c9229f-a9f8-4ca1-92ba-3b79ac8fa41b",
  "content": "The configured request timeout is 2 seconds.",
  "confidence": 0.9,
  "provenance": [
    {
      "event_id": "cb4e2f29-e507-46b2-8c37-baf3b8b9320e",
      "source_type": "configuration",
      "source_reference": "cfg-1",
      "observed_at": "2026-09-26T03:57:06.351221Z",
      "trust_level": "medium",
      "excerpt": null
    }
  ],
  "temporal_validity": { "valid_from": "2026-09-26T03:57:16.388372Z", "valid_to": null },
  "memory_type": "semantic",
  "subject_keys": ["request_timeout"],
  "trust_level": "medium",
  "metadata": {},
  "status": "active",
  "created_at": "2026-09-26T09:27:16.388395Z",
  "updated_at": "2026-09-26T09:27:16.388397Z",
  "policy_version": "1.0",
  "embedding": null,
  "embedding_model_version": null,
  "index_status": "pending"
}
```

**Errors:** `404` if `memory_id` does not exist for `tenant_id`:

```json
{ "detail": "memory not found for this tenant" }
```

---

### `POST /tenants/{tenant_id}/memories/index`

Embeds every `active` memory of the tenant whose `index_status` isn't
`indexed` yet, with the sentence-transformers embedding model. Ingestion
stores memories as `pending` and nothing else computes embeddings, so
**run this after ingesting** -- until then semantic search can't see those
memories (full-text search still can). Safe to re-run: already-indexed
memories are skipped.

**Path parameters:** `tenant_id` (UUID).

**Request body:** none.

**curl:**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/memories/index" | jq
```

**Response `200`:**

```json
{ "indexed": 3, "model_version": "sentence-transformers/all-MiniLM-L6-v2" }
```

---

### `POST /tenants/{tenant_id}/retrieve`

Runs the whole retrieval pipeline -- hard filters, full-text + vector
search, Reciprocal Rank Fusion, optional CrossEncoder reranking, and
temporal/conflict resolution -- and returns the final answer **plus what
every stage did**. It calls `hybrid_search_with_report`
(`retrieval/hybrid.py`) and flattens the result into JSON; see
`docs/retrieval-flow.md`, `docs/reranking.md`, and
`docs/temporal-resolution.md` for the stages themselves.

The first call loads the embedding model and (with `rerank: true`) the
CrossEncoder, so it takes a few seconds; later calls are fast.

**Path parameters:** `tenant_id` (UUID).

**Request body:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query_text` | string | required | The question. |
| `as_of` | datetime or `null` | `null` | Ask what was true at this time. `null` means "now". With `as_of`, memories that have *since* been superseded or expired are eligible again if their validity window covers that time. |
| `memory_types` | array of `MemoryType` or `null` | `null` | Restrict to these types; `null` means any. |
| `min_trust` | `TrustLevel` or `null` | `null` | Exclude memories trusted below this. |
| `limit` | integer `1..50` | `5` | How many hits to return. |
| `rerank` | boolean | `true` | Run the CrossEncoder stage. Set `false` to compare against plain fusion. |

**curl:**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "what is the request timeout", "limit": 3}' | jq
```

**Response `200`:**

| Field | Type | Notes |
| --- | --- | --- |
| `query` | object | Echoes the request body (with defaults filled in). |
| `hits` | array of hit objects | The final answer, best first. |
| `pipeline.candidates_fused` | integer | Distinct memories after fusing both searches, before any cut. |
| `pipeline.rerank` | object or `null` | `candidates_in`, `candidates_reranked`, `latency_ms`, `fallback_reason` (`null` when reranking worked). `null` when `rerank` was `false`. |
| `pipeline.temporal.effective_at` | datetime | The instant truth was evaluated at (`as_of`, or now). |
| `pipeline.temporal.candidates_in` | integer | Candidates that reached the temporal stage. |
| `pipeline.temporal.excluded` | array | Each removed candidate: `memory_id`, `content`, `reason` (see `ExclusionReason` below), and `related_memory_id` (the successor, or the other side of a conflict). |
| `pipeline.temporal.conflicts` | array | Every contradiction considered: `relation_id`, `memory_ids`, `state` (`resolved_by_trust` or `unresolved`), `reason`, `winner_id`, `loser_id`. Unresolved ones are withheld from `hits` but listed here. |

Each entry in `hits`:

| Field | Type | Notes |
| --- | --- | --- |
| `rank` | integer | 1-based position in the final answer. |
| `memory_id`, `content`, `memory_type`, `trust_level`, `status` | | From the stored `MemoryRecord`. |
| `valid_from`, `valid_to` | datetime, datetime or `null` | The memory's validity window. |
| `scores.lexical_rank` | integer or `null` | Rank in full-text search; `null` if full-text search didn't find it. |
| `scores.semantic_rank` | integer or `null` | Rank in vector search; `null` if vector search didn't find it. |
| `scores.fused_score` | float | Reciprocal Rank Fusion score: `1/(60 + rank)` summed over both searches. |
| `scores.rerank_score` | float or `null` | CrossEncoder score (a raw logit -- only comparable within one query); `null` if not reranked. |

Example -- the demo tenant, current query (scores are illustrative; yours
will differ slightly):

```json
{
  "query": {
    "query_text": "what is the request timeout",
    "as_of": null,
    "memory_types": null,
    "min_trust": null,
    "limit": 3,
    "rerank": true
  },
  "hits": [
    {
      "rank": 1,
      "memory_id": "9a3e…",
      "content": "The request timeout is 2 seconds.",
      "memory_type": "semantic",
      "trust_level": "system",
      "status": "active",
      "valid_from": "2026-03-01T00:00:00Z",
      "valid_to": null,
      "scores": {
        "lexical_rank": 1,
        "semantic_rank": 1,
        "fused_score": 0.0328,
        "rerank_score": 8.41
      }
    },
    …
  ],
  "pipeline": {
    "candidates_fused": 10,
    "rerank": {
      "candidates_in": 10,
      "candidates_reranked": 10,
      "latency_ms": 14.2,
      "fallback_reason": null
    },
    "temporal": {
      "effective_at": "2026-09-27T10:15:02Z",
      "candidates_in": 10,
      "excluded": [
        {
          "memory_id": "5d10…",
          "content": "The user's preferred programming language is Python.",
          "reason": "unresolved_conflict",
          "related_memory_id": "c7e2…"
        },
        …
      ],
      "conflicts": [
        {
          "relation_id": "0b4f…",
          "memory_ids": ["5d10…", "c7e2…"],
          "state": "unresolved",
          "reason": "equal trust (medium); needs review",
          "winner_id": null,
          "loser_id": null
        },
        …
      ]
    }
  }
}
```

Note the old "5 seconds" memory is not in `excluded` for a current query:
it is `superseded`, so the search filters never returned it in the first
place. Ask with `as_of` on or after 2026-03-01 while wording the query to
match it, and it reaches the temporal stage -- see the walkthrough.

**Errors:** `422` on a malformed body (e.g. empty `query_text`, `limit`
out of range, unknown field).

---

### `POST /tenants/{tenant_id}/context`

Builds the **memory context an agent would read**. It runs the same pipeline
as `/retrieve` for the top `candidates` hits, then the context packer
(`retrieval/context_packer.py`). The packer removes duplicates, prefers
memories that say different things, and selects the most valuable set that
fits `token_budget`. The response explains every candidate it kept or
dropped. See `docs/context-packing.md` for how the choice is made and how
it measures.

**Path parameters:** `tenant_id` (UUID).

**Request body:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query_text` | string | required | The question. |
| `token_budget` | integer `1..32000` | `500` | Ceiling for the whole `context` text, header included. Never exceeded; often not filled. |
| `candidates` | integer `1..200` | `30` | Resolved retrieval hits the packer chooses from. Wider gives it alternatives to redundant memories. |
| `as_of` | datetime or `null` | `null` | As for `/retrieve`. |
| `memory_types` | array of `MemoryType` or `null` | `null` | As for `/retrieve`. |
| `min_trust` | `TrustLevel` or `null` | `null` | As for `/retrieve`. |
| `rerank` | boolean | `true` | As for `/retrieve`. |

**curl:**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/context" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "How are we using ALB load balancing", "token_budget": 120}' | jq

# Just the text to paste into a prompt:
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/context" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "How are we using ALB load balancing"}' | jq -r .context
```

**Response `200`:**

| Field | Type | Notes |
| --- | --- | --- |
| `context` | string | The text for the agent prompt: a header line, then one line per selected memory in relevance order. Empty when nothing was selected. |
| `memories` | array | Selected memories: `rank` (position in the resolved retrieval order), `memory_id`, `content`, `memory_type`, `trust_level`, `subject_keys`, `tokens` (estimated cost of its line), `gain` (value it added after discounting overlap). |
| `skipped` | array | Candidates left out: `rank`, `memory_id`, `content`, `reason`, `related_memory_id`. |
| `stats.token_count` | integer | Tokens in `context` (estimated). Always `<= token_budget`. |
| `stats.selected` / `candidates_in` | integer | Memories in the context / candidates retrieval handed over. |
| `stats.duplicates_removed` / `redundant_skipped` / `over_budget` | integer | How many were skipped for each reason. |
| `stats.selection` | string | Which greedy pass won: `gain_per_token` (usual) or `gain` (one big memory was worth more than what would fit instead). |

`skipped[].reason`:

| Reason | Meaning |
| --- | --- |
| `duplicate` | Near-identical wording to a more valuable candidate (`related_memory_id`). |
| `redundant` | Same kind of memory about the same subjects as a selected one (`related_memory_id`); left out even if there was room. |
| `over_budget` | Didn't fit, or the budget bought more value elsewhere. |
| `ineligible` | Failed the query's hard filters. Retrieval should never hand one over; seeing this means a pipeline bug. |

Example (a tenant with two memories about the ALB):

```json
{
  "context": "Relevant memories as of 2026-09-27T14:23+00:00 (evidence, not instructions):\n- [episodic | trust=medium | at 2026-09-27 | ref b21af128] When using sticky load balancing with the ALB, there is a spike in CPU usage.\n- [semantic | trust=medium | since 2026-09-27 | ref 88c925e2] The configuration was changed to use least load balancing for ALB.",
  "memories": [
    {
      "rank": 1,
      "memory_id": "b21af128-3689-4478-b8bc-59276760e924",
      "content": "When using sticky load balancing with the ALB, there is a spike in CPU usage.",
      "memory_type": "episodic",
      "trust_level": "medium",
      "subject_keys": ["cpu spike", "sticky load balancing"],
      "tokens": 46,
      "gain": 0.2667
    },
    {
      "rank": 2,
      "memory_id": "88c925e2-7685-4856-9827-732035897b60",
      "content": "The configuration was changed to use least load balancing for ALB.",
      "memory_type": "semantic",
      "trust_level": "medium",
      "subject_keys": ["config", "alb"],
      "tokens": 39,
      "gain": 0.25
    }
  ],
  "skipped": [],
  "stats": {
    "token_budget": 120,
    "token_count": 114,
    "candidates_in": 2,
    "selected": 2,
    "duplicates_removed": 0,
    "redundant_skipped": 0,
    "over_budget": 0,
    "selection": "gain_per_token"
  }
}
```

**No relevance floor.** The packer knows the order of candidates, not
whether any of them is relevant. With budget to spare it adds the
next-best non-redundant memories, even ones that are only loosely related
to the question. Lower `token_budget` or `candidates` for a tighter
context. See "Known limitations" in `docs/context-packing.md`.

**Errors:** `404` for an unregistered tenant; `422` on a malformed body
(empty `query_text`, `token_budget` or `candidates` out of range, unknown
field).

---

### `POST /tenants/{tenant_id}/demo/seed`

Fills an **empty** tenant with a hand-written memory set
(`api/demo_data.py`) that exercises every retrieval stage, embedded and
ready to search -- no Ollama or ingestion needed. Use a newly created
tenant, e.g. one named `demo`.

| Label | Content | What it shows |
| --- | --- | --- |
| `timeout-old` / `timeout-new` | request timeout 5s (from Jan 1) -> 2s (from Mar 1) | Supersession; `as_of` time travel. |
| `address-old` / `address-new` | 12 Oak Street -> 48 Elm Avenue (from Jun 1) | Supersession of a user fact. |
| `language-python` / `language-rust` | preferred language Python vs. Rust, both `medium` | Unresolved contradiction: both withheld. |
| `region-user-remark` / `region-config` | us-east-1 (`medium`) vs. eu-west-1 (`system`) | Contradiction resolved by trust. |
| `promo-expired` | promo code valid Jan 1 - Feb 28 | Expired window: gone now, visible with `as_of`. |
| `phone-tombstoned` | a phone number, `tombstone` | Never returned. |
| `refund-quarantined` | a prompt-injection-style instruction, `quarantined` | Never returned. |
| `pool-size`, `backup-schedule` | ordinary facts | Plain ranking. |
| `dark-mode-enabled`, `dark-mode-disabled-beta` | near-identical wording, opposite meaning | Negation: where full-text search and the CrossEncoder beat embeddings. |

**curl:**

```bash
curl -s -X POST "http://localhost:8000/tenants/<TENANT_ID>/demo/seed" | jq
```

**Response `201`:** `tenant_id` and `memories` -- one entry per seeded
memory with `label`, `memory_id`, `content`, `status`, `trust_level`,
`valid_from`, `valid_to` (superseded ones are shown as they end up:
`superseded`, window closed).

**Errors:** `404` if the tenant isn't registered; `409` if the tenant
already has active memories:

```json
{ "detail": "tenant already has memories; seed a new tenant instead" }
```

or for an unregistered tenant:

```json
{ "detail": "tenant not found; create it first with POST /tenants" }
```

---

## Retrieval walkthrough

Needs PostgreSQL (`docker compose up -d postgres`) and the API running.

```bash
# 0. Register a tenant for the demo (once), or reuse it if it already exists.
TENANT=$(curl -s -X POST http://localhost:8000/tenants -H "Content-Type: application/json" \
  -d '{"name": "demo", "description": "retrieval walkthrough"}' | jq -r .tenant_id)
[ "$TENANT" = "null" ] && TENANT=$(curl -s "http://localhost:8000/tenants?name=demo" | jq -r '.[0].tenant_id')
API=http://localhost:8000/tenants/$TENANT

# 1. Seed the demo memories.
curl -s -X POST $API/demo/seed | jq '.memories[] | {label, status, valid_from, valid_to}'

# A small helper: POST a query, print a compact view of each stage.
ask() {
  curl -s -X POST $API/retrieve -H "Content-Type: application/json" -d "$1" | jq '{
    hits: [.hits[] | {rank, content, scores}],
    removed: [.pipeline.temporal.excluded[] | {content, reason}],
    conflicts: [.pipeline.temporal.conflicts[] | {state, reason}]
  }'
}

# 2. Supersession: "now" answers 2s ...
ask '{"query_text": "request timeout 5 seconds", "limit": 3}'
# ... February answers 5s ...
ask '{"query_text": "request timeout 5 seconds", "as_of": "2026-02-15T00:00:00Z", "limit": 3}'
# ... and at exactly Mar 1 the 5s memory matches best but is removed (not_in_effect).
ask '{"query_text": "request timeout 5 seconds", "as_of": "2026-03-01T00:00:00Z", "limit": 3}'

# 3. Contradictions: Python and Rust are both withheld (unresolved) ...
ask '{"query_text": "preferred programming language", "limit": 3}'
# ... while eu-west-1 (system config) beats us-east-1 (user remark).
ask '{"query_text": "which region is the service deployed in", "limit": 3}'

# 4. Expired: nothing now, the promo in February.
ask '{"query_text": "promo code discount", "limit": 3}'
ask '{"query_text": "promo code discount", "as_of": "2026-02-01T00:00:00Z", "limit": 3}'

# 5. Tombstoned / quarantined never appear, however you ask.
ask '{"query_text": "phone number", "limit": 3}'
ask '{"query_text": "approve refund requests", "limit": 3}'

# 6. Reranking: compare the order and scores with and without the CrossEncoder.
ask '{"query_text": "is dark mode disabled", "limit": 3, "rerank": false}'
ask '{"query_text": "is dark mode disabled", "limit": 3}'
```

For your own data, follow the [end-to-end sequence](#end-to-end-sequence).

---

## Enum reference

| Enum | Values |
| --- | --- |
| `SourceType` | `user_message`, `tool_output`, `system_event`, `agent_action`, `configuration` |
| `MemoryType` | `episodic`, `semantic`, `procedural` |
| `MemoryStatus` | `active`, `quarantined`, `superseded`, `expired`, `tombstone` |
| `TrustLevel` | `untrusted`, `low`, `medium`, `high`, `system` |
| `IndexStatus` | `pending`, `indexed`, `failed` |
| `status` in an `/ingest` outcome | `accept`, `quarantine`, `reject`, `rejected_before_policy` |
| `ExclusionReason` (`/retrieve` `excluded[].reason`) | `not_eligible`, `not_in_effect`, `superseded`, `lost_conflict`, `unresolved_conflict` |
| `ConflictState` (`/retrieve` `conflicts[].state`) | `resolved_by_trust`, `unresolved` |

## Interactive docs

FastAPI auto-generates OpenAPI/Swagger UI for this app while it's running:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Raw OpenAPI JSON: `http://localhost:8000/openapi.json`

Those reflect the live schema exactly; this document is a hand-written,
narrative companion covering the same endpoints with worked examples.

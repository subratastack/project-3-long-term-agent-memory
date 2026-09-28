# Local database operations

Use this guide to inspect the PostgreSQL database behind the memory service
and load extra example events. Start with the [README quick start](../README.md#quick-start)
if the database and schema are not yet running.

## Connect to PostgreSQL and pgvector

The `postgres` Compose service runs PostgreSQL with pgvector. Connection
settings come from `.env` / `.env.example`; the commands below use the
example defaults: user `agent_memory`, password `agent_memory_local`, database
`agent_memory`, and host port `5432`. Adjust them if your configuration differs.

### Use the client inside the container

No host installation of `psql` is needed:

```bash
docker compose exec postgres psql -U agent_memory -d agent_memory
```

### Use a client on your host

With `psql` installed locally:

```bash
psql "postgresql://agent_memory:agent_memory_local@localhost:5432/agent_memory"
```

A GUI such as DBeaver uses the same host, port, database, and credentials.

## Inspect records and indexes

Run these commands inside `psql`:

```sql
-- Confirm that pgvector is installed.
\dx vector

-- Inspect the memory table and its columns.
\d memory_records

-- See whether memories have their derived embeddings yet.
SELECT index_status, count(*)
FROM memory_records
GROUP BY index_status;
```

An accepted memory can be available to full-text search before it has an
embedding. Use `POST /tenants/{tenant_id}/memories/index` to compute missing
embeddings for active memories, as shown in the
[ingestion guide](ingestion-flow.md#6-index-accepted-memories-for-meaning-search).

### Explore vector distance with a stored embedding

This diagnostic query uses one indexed memory's vector as its query vector,
avoiding a placeholder with the wrong number of dimensions. Set the tenant
UUID to the real value returned by the API. The query yields no rows if that
tenant has no indexed memories.

```sql
\set tenant_id 'replace-with-your-tenant-uuid'

WITH example AS (
    SELECT embedding
    FROM memory_records
    WHERE tenant_id = :'tenant_id'::uuid
      AND index_status = 'indexed'
      AND embedding IS NOT NULL
    ORDER BY memory_id
    LIMIT 1
)
SELECT m.memory_id, m.content,
       m.embedding <=> e.embedding AS cosine_distance
FROM memory_records AS m
CROSS JOIN example AS e
WHERE m.tenant_id = :'tenant_id'::uuid
  AND m.index_status = 'indexed'
  AND m.embedding IS NOT NULL
ORDER BY cosine_distance, m.memory_id
LIMIT 5;
```

Lower cosine distance means closer vectors. This shows vector mechanics,
not the complete retrieval policy: it does not apply status, trust, validity,
or conflict checks. Use the retrieval API for the filtered pipeline.

## Load fictional-company events

For an ingestion exercise beyond the prepared retrieval demo, run from the
repository root after applying migrations:

```bash
uv run alembic upgrade head
docker compose exec -T postgres psql -U agent_memory -d agent_memory \
  < scripts/seed_fictional_company_memories.sql
```

The script registers two fictional tenants—an ecommerce customer-care
company and a school-management engineering team—and adds 16 raw events,
eight per tenant. Fixed UUIDs allow repeated loading without duplicating
those rows.

It does not create memory records, write decisions, provenance entries, or
embeddings. Process its events through the
[ingestion API](ingestion-flow.md), then index accepted memories. This differs
from `/demo/seed`, which constructs and indexes prepared memory fixtures for
exploring retrieval behavior.

## Schema and test-database behavior

Alembic manages the development schema:

```bash
uv run alembic upgrade head
```

Integration tests use `TEST_DATABASE_URL` when set; otherwise they append
`_test` to the database name in `DATABASE_URL`. With the example settings,
they create `agent_memory_test` automatically on first use. The test database
user needs permission to create that database, or it must already exist.

The suite creates and drops tables in its test database and rolls back each
test's outer transaction. Keep an explicit `TEST_DATABASE_URL` pointed at a
disposable database, never the development database you want to retain.
Normal test runs with the default separate database do not remove the
development schema.

## Check service health

```bash
docker compose ps
docker compose logs -f postgres
```

The Compose health check waits for PostgreSQL to accept connections.
`docker compose up -d --wait postgres` waits for that healthy state before
returning, which is useful before applying migrations.

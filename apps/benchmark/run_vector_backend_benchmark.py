"""Phase 6B: compare actual pgvector exact/HNSW and TurboVec on identical data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import insert, text, update
from sqlalchemy.orm import Session, sessionmaker

from apps.benchmark.dataset_retrieval import RunProfile, score_ranking, select_questions
from apps.benchmark.datasets.schema import DatasetManifest, RetrievalSample, load_samples
from apps.benchmark.metrics import percentile, recall_at_k
from apps.benchmark.vector_backend_database import vector_benchmark_database
from apps.benchmark.vector_backend_failures import failure_probes
from apps.memory_service.domain.enums import MemoryType, TrustLevel
from apps.memory_service.indexing.reconciliation import reconcile
from apps.memory_service.indexing.sync_worker import sync_once
from apps.memory_service.persistence.models import (
    MemoryEventRow,
    MemoryProvenanceRow,
    MemoryRecordRow,
    TenantRow,
)
from apps.memory_service.persistence.pgvector_index import PgvectorIndex, VectorIndex
from apps.memory_service.persistence.turbovec_index import TurboVecIndex
from apps.memory_service.persistence.unit_of_work import UnitOfWork, default_database_url
from apps.memory_service.retrieval.filters import resolve_filters
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.vector_backend import vector_backend_search

DEFAULT_PROFILE = Path("apps/benchmark/datasets/configs/vector_backend_comparison.json")
DEFAULT_DATA = Path("data/benchmarks/memory_agent_bench/samples.jsonl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus(
    profile: RunProfile, samples_path: Path
) -> tuple[list[RetrievalSample], DatasetManifest, dict[str, Any]]:
    samples = load_samples(samples_path)
    manifest = DatasetManifest.model_validate_json(
        samples_path.with_name("manifest.json").read_text()
    )
    if sha256(samples_path) != manifest.samples_sha256:
        raise ValueError("normalized corpus SHA-256 differs from manifest")
    selected = select_questions(samples, profile)
    samples = [
        s
        for s in samples
        if s.sample_id in selected and any(q.relevant_ids for q in selected[s.sample_id])
    ]
    return samples, manifest, selected


def prepare_vectors(
    samples: list[RetrievalSample],
    selected: dict[str, Any],
    profile: RunProfile,
    cache: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    documents = [c.text for s in samples for c in s.chunks]
    questions = [q.text for s in samples for q in selected[s.sample_id] if q.relevant_ids]
    fingerprint = hashlib.sha256(
        json.dumps(
            {"model": profile.embedding_model, "documents": documents, "questions": questions},
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    metadata_path = cache.with_suffix(".json")
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["input_sha256"] == fingerprint and metadata["cache_sha256"] == sha256(cache):
            with np.load(cache, allow_pickle=False) as stored:
                return stored["documents"], stored["questions"], metadata
    import torch

    from apps.memory_service.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingModel,
    )

    torch.set_num_threads(profile.torch_threads)
    started = time.perf_counter()
    embedder = SentenceTransformerEmbeddingModel(profile.embedding_model, device=profile.device)
    print(f"Embedding {len(documents)} chunks and {len(questions)} queries", flush=True)
    matrix = []
    for start in range(0, len(documents), 128):
        matrix.extend(embedder.embed_texts(documents[start : start + 128]))
        if start % 1024 == 0:
            print(
                f"Embedded {min(start + 128, len(documents))}/{len(documents)} chunks", flush=True
            )
    vectors = np.asarray(matrix, dtype=np.float32)
    query_vectors = np.asarray(embedder.embed_texts(questions), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, documents=vectors, questions=query_vectors)
    metadata = {
        "input_sha256": fingerprint,
        "cache_sha256": sha256(cache),
        "model": profile.embedding_model,
        "dimensions": embedder.dimensions,
        "preparation_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return vectors, query_vectors, metadata


def seed(
    factory: sessionmaker[Session],
    samples: list[RetrievalSample],
    vectors: np.ndarray,
    model: str,
) -> tuple[dict[str, UUID], dict[UUID, str], list[UUID], float]:
    now = datetime.now(UTC) - timedelta(days=1)
    tenants = {s.sample_id: uuid4() for s in samples}
    labels: dict[UUID, str] = {}
    memory_ids = []
    offset = 0
    started = time.perf_counter()
    with factory.begin() as session:
        session.execute(
            insert(TenantRow),
            [{"tenant_id": t, "name": f"vector-eval-{t}"} for t in tenants.values()],
        )
        for sample in samples:
            tenant = tenants[sample.sample_id]
            for start in range(0, len(sample.chunks), 256):
                events, records, provenance = [], [], []
                for i, chunk in enumerate(sample.chunks[start : start + 256], start=start):
                    mid, eid = uuid5(NAMESPACE_URL, chunk.chunk_id), uuid4()
                    # Controlled filter selectivity; original text and relevance labels stay intact.
                    trust = "high" if i % 5 == 0 else "medium"
                    records.append(
                        {
                            "memory_id": mid,
                            "tenant_id": tenant,
                            "content": chunk.text,
                            "confidence": 1.0,
                            "valid_from": now,
                            "memory_type": "semantic" if i % 4 == 0 else "episodic",
                            "trust_level": trust,
                            "status": "active",
                            "embedding": vectors[offset].tolist(),
                            "embedding_model_version": model,
                            "index_status": "indexed",
                            "subject_keys": [],
                            "created_at": now,
                            "updated_at": now,
                            "policy_version": "1.0",
                            "metadata_": {},
                        }
                    )
                    events.append(
                        {
                            "event_id": eid,
                            "tenant_id": tenant,
                            "source_type": "user_message",
                            "source_reference": chunk.chunk_id,
                            "content": chunk.text,
                            "observed_at": now,
                            "metadata_": {},
                        }
                    )
                    provenance.append(
                        {
                            "provenance_id": uuid4(),
                            "tenant_id": tenant,
                            "memory_id": mid,
                            "event_id": eid,
                            "source_type": "user_message",
                            "source_reference": chunk.chunk_id,
                            "observed_at": now,
                            "trust_level": trust,
                        }
                    )
                    labels[mid] = chunk.chunk_id
                    memory_ids.append(mid)
                    offset += 1
                session.execute(insert(MemoryEventRow), events)
                session.execute(insert(MemoryRecordRow), records)
                session.execute(insert(MemoryProvenanceRow), provenance)
    return tenants, labels, memory_ids, time.perf_counter() - started


def update_batch(
    factory: sessionmaker[Session], ids: list[UUID], vectors: np.ndarray, count: int
) -> float:
    started = time.perf_counter()
    with factory.begin() as session:
        for mid, vector in zip(ids[:count], vectors[:count], strict=True):
            session.execute(
                update(MemoryRecordRow)
                .where(MemoryRecordRow.memory_id == mid)
                .values(embedding=vector.tolist())
            )
    return time.perf_counter() - started


def drain(factory: sessionmaker[Session], index: TurboVecIndex) -> dict[str, Any]:
    started = time.perf_counter()
    selected = indexed = removed = failed = 0
    while True:
        result = sync_once(factory, index)
        selected += result.selected
        indexed += result.indexed
        removed += result.removed
        failed += result.failed
        if not result.selected or result.failed:
            break
    if failed:
        raise RuntimeError(f"sync failed for {failed} jobs")
    return {
        "seconds": time.perf_counter() - started,
        "selected": selected,
        "indexed": indexed,
        "removed": removed,
        "failed": failed,
    }


def plan(factory: sessionmaker[Session], backend: PgvectorIndex, query: RetrievalQuery) -> Any:
    with factory() as session:
        backend.configure(session)
        statement = backend.statement(
            resolve_filters(query), query.query_embedding or [], query.limit
        )
        sql = str(statement.compile(session.bind, compile_kwargs={"literal_binds": True}))
        return session.execute(text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)).scalar_one()


def timed_search(
    factory: sessionmaker[Session], backend: VectorIndex, query: RetrievalQuery
) -> tuple[list[UUID], float]:
    started = time.perf_counter()
    with UnitOfWork(factory) as uow:
        ids = [hit.memory.memory_id for hit in vector_backend_search(uow, backend, query)]
    return ids, (time.perf_counter() - started) * 1000


def measure_queries(
    factory: sessionmaker[Session],
    samples: list[RetrievalSample],
    selected: dict[str, Any],
    queries: np.ndarray,
    tenants: dict[str, UUID],
    labels: dict[UUID, str],
    backends: dict[str, VectorIndex],
    profile: RunProfile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cases = [(s, q) for s in samples for q in selected[s.sample_id] if q.relevant_ids]
    traces: list[dict[str, Any]] = []
    measured: dict[tuple[str, str, str], dict[str, Any]] = {}
    rng = random.Random(profile.seed)
    for filtered in (False, True):
        for n, (sample, question) in enumerate(cases):
            query = RetrievalQuery(
                tenant_id=tenants[sample.sample_id],
                query_embedding=queries[n].tolist(),
                limit=10,
                min_trust=TrustLevel.HIGH if filtered else None,
                memory_types=[MemoryType.EPISODIC] if filtered else None,
            )
            expected, _ = timed_search(factory, backends["pgvector_exact"], query)
            # One untimed warmup per query/backend; measured repetitions are interleaved.
            for backend in backends.values():
                timed_search(factory, backend, query)
            for repeat in range(profile.repeats):
                order = list(backends)
                rng.shuffle(order)
                for name in order:
                    ids, elapsed = timed_search(factory, backends[name], query)
                    key = name, question.question_id, "filtered" if filtered else "tenant_only"
                    row = measured.setdefault(
                        key,
                        {
                            "backend": name,
                            "question_id": question.question_id,
                            "source": sample.source,
                            "sample_id": sample.sample_id,
                            "filter": key[2],
                            "ranked_ids": [labels[i] for i in ids],
                            "relevant_ids": list(question.relevant_ids),
                            "exact_top10_ids": [labels[i] for i in expected],
                            "latencies_ms": [],
                        },
                    )
                    row["latencies_ms"].append(elapsed)
                    if repeat == 0:
                        row["ann_recall_at_10"] = (
                            recall_at_k(ids, set(expected), 10) if expected else None
                        )
                        row["quality"] = (
                            asdict(score_ranking(row["ranked_ids"], question.relevant_ids, [10])[0])
                            if not filtered
                            else None
                        )
            if (n + 1) % 10 == 0:
                print(
                    f"Measured {'filtered' if filtered else 'tenant-only'} "
                    f"queries {n + 1}/{len(cases)}",
                    flush=True,
                )
    traces = list(measured.values())
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in traces:
        for source in (row["source"], "ALL"):
            groups[row["backend"], row["filter"], source].append(row)
    aggregates = []
    for (name, filters, source), rows in sorted(groups.items()):
        times = [t for r in rows for t in r["latencies_ms"]]
        quality = [r["quality"] for r in rows if r["quality"] is not None]
        aggregates.append(
            {
                "backend": name,
                "filter": filters,
                "source": source,
                "queries": len(rows),
                "measurements": len(times),
                "p50_ms": percentile(times, 50),
                "p95_ms": percentile(times, 95),
                "ann_recall_at_10": mean(
                    r["ann_recall_at_10"] for r in rows if r["ann_recall_at_10"] is not None
                ),
                "precision_at_10": mean(r["precision"] for r in quality) if quality else None,
                "recall_at_10": mean(r["recall"] for r in quality) if quality else None,
                "mrr_at_10": mean(r["reciprocal_rank"] for r in quality) if quality else None,
                "ndcg_at_10": mean(r["ndcg"] for r in quality) if quality else None,
            }
        )
    first_sample, _ = cases[0]
    query = RetrievalQuery(
        tenant_id=tenants[first_sample.sample_id], query_embedding=queries[0].tolist(), limit=10
    )
    plans = {
        name: plan(factory, backend, query)
        for name, backend in backends.items()
        if isinstance(backend, PgvectorIndex)
    }
    filtered_query = query.model_copy(
        update={"min_trust": TrustLevel.HIGH, "memory_types": [MemoryType.EPISODIC]}
    )
    plans["pgvector_hnsw_filtered"] = plan(factory, PgvectorIndex("hnsw"), filtered_query)
    if "vector_bench_hnsw" not in json.dumps(plans["pgvector_hnsw"]):
        raise RuntimeError("HNSW comparison did not actually use the HNSW index")
    if "vector_bench_hnsw" in json.dumps(plans["pgvector_exact"]):
        raise RuntimeError("exact comparison unexpectedly used HNSW")
    return traces, aggregates, plans


def restart_probe(path: Path, vector: np.ndarray, bits: int = 4) -> dict[str, Any]:
    import psutil  # type: ignore[import-untyped]

    process = psutil.Process()
    before = process.memory_info().rss
    started = time.perf_counter()
    index = TurboVecIndex(path, bit_width=bits)
    loaded = time.perf_counter()
    index.native.prepare()
    index.native.search(np.asarray([vector], dtype=np.float32), k=10)
    ready = time.perf_counter()
    result = {
        "baseline_rss_bytes": before,
        "ready_rss_bytes": process.memory_info().rss,
        "index_rss_delta_bytes": process.memory_info().rss - before,
        "load_ms": (loaded - started) * 1000,
        "load_and_first_search_ms": (ready - started) * 1000,
        "vectors": len(index.entries),
        "native_serialized_bytes": len(index.native.to_bytes()),
        "checkpoint_bytes": path.stat().st_size,
    }
    index.close()
    return result


def docker_memory(container: str | None) -> dict[str, Any]:
    if not container:
        return {"available": False, "reason": "--postgres-container not supplied"}
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container],
            text=True,
            capture_output=True,
            check=True,
            timeout=20,
        )
        return {"available": True, "sample": json.loads(result.stdout)}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "reason": type(exc).__name__}


def benchmark(
    args: argparse.Namespace,
    profile: RunProfile,
    samples: list[RetrievalSample],
    manifest: DatasetManifest,
    selected: dict[str, Any],
    vectors: np.ndarray,
    queries: np.ndarray,
    embedding_metadata: dict[str, Any],
) -> dict[str, Any]:
    import psutil

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a prior invocation's index, including a live deployment's file.
    path = args.index / f"benchmark-{uuid4().hex}.zip"
    report: dict[str, Any] = {
        "schema_version": "vector-backend-benchmark-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": manifest.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "embedding": embedding_metadata,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "library default"),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("turbovec", "numpy", "psutil", "sqlalchemy", "pgvector")
            },
        },
        "postgres_container_before": docker_memory(args.postgres_container),
        "coverage": [
            {
                "sample_id": s.sample_id,
                "source": s.source,
                "chunks": len(s.chunks),
                "selected": len(selected[s.sample_id]),
                "scored": sum(bool(q.relevant_ids) for q in selected[s.sample_id]),
            }
            for s in samples
        ],
    }
    with vector_benchmark_database(default_database_url()) as (engine, factory):
        with factory() as session:
            report["environment"].update(
                {
                    "postgres": session.execute(text("SELECT version()")).scalar(),
                    "pgvector": session.execute(
                        text("SELECT extversion FROM pg_extension WHERE extname='vector'")
                    ).scalar(),
                    "shared_buffers": session.execute(text("SHOW shared_buffers")).scalar(),
                }
            )
        print("Seeding committed PostgreSQL records and outbox", flush=True)
        tenants, labels, ids, ingest_seconds = seed(
            factory, samples, vectors, profile.embedding_model
        )
        count = min(256, len(ids))
        changed = np.roll(vectors, 1, axis=1).copy()
        update_exact = update_batch(factory, ids, changed, count)
        update_batch(factory, ids, vectors, count)
        started = time.perf_counter()
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE INDEX vector_bench_hnsw ON memory_records USING hnsw "
                    "(embedding vector_cosine_ops) WITH (m=16, ef_construction=100)"
                )
            )
            connection.execute(text("ANALYZE memory_records"))
        build_seconds = time.perf_counter() - started
        update_hnsw = update_batch(factory, ids, changed, count)
        update_batch(factory, ids, vectors, count)
        index = TurboVecIndex(path, bit_width=args.bits)
        try:
            print("Synchronizing real TurboVec index", flush=True)
            initial_sync = drain(factory, index)
            started = time.perf_counter()
            pg_seconds = update_batch(factory, ids, changed, count)
            update_sync = drain(factory, index)
            turbo_update_seconds = time.perf_counter() - started
            update_batch(factory, ids, vectors, count)
            drain(factory, index)
            backends: dict[str, VectorIndex] = {
                "pgvector_exact": PgvectorIndex("exact"),
                "pgvector_hnsw": PgvectorIndex("hnsw"),
                "turbovec": index,
            }
            print("Comparing identical queries with interleaved repetitions", flush=True)
            traces, aggregates, plans = measure_queries(
                factory, samples, selected, queries, tenants, labels, backends, profile
            )
            report.update(
                {
                    "results": traces,
                    "aggregates": aggregates,
                    "plans": plans,
                    "throughput": {
                        "documents": len(ids),
                        "updates": count,
                        "postgres_ingest_seconds": ingest_seconds,
                        "pgvector_exact_ingest_records_per_second": len(ids) / ingest_seconds,
                        "hnsw_build_seconds": build_seconds,
                        "pgvector_hnsw_bulk_build_records_per_second": len(ids)
                        / (ingest_seconds + build_seconds),
                        "pgvector_exact_update_records_per_second": count / update_exact,
                        "pgvector_hnsw_update_records_per_second": count / update_hnsw,
                        "turbovec_initial_sync": initial_sync,
                        "turbovec_sync_records_per_second": len(ids) / initial_sync["seconds"],
                        "turbovec_total_ingest_records_per_second": len(ids)
                        / (ingest_seconds + initial_sync["seconds"]),
                        "turbovec_update": update_sync,
                        "turbovec_update_pg_seconds": pg_seconds,
                        "turbovec_total_update_records_per_second": count / turbo_update_seconds,
                    },
                }
            )
            # Measure actual missing/ghost repair, with the same corpus and vectors.
            for key in list(index.entries)[:count]:
                index.remove(key)
            for key in range(2**62, 2**62 + count):
                index.upsert(key, uuid4(), uuid4(), 1, vectors[0].tolist())
            index.checkpoint()
            started = time.perf_counter()
            repair = reconcile(factory, index)
            repair_seconds = time.perf_counter() - started
            started = time.perf_counter()
            clean = reconcile(factory, index)
            clean_seconds = time.perf_counter() - started
            report["reconciliation"] = {
                "damaged": asdict(repair),
                "repair_seconds": repair_seconds,
                "clean": asdict(clean),
                "clean_audit_seconds": clean_seconds,
            }
            with factory() as session:
                report["storage"] = dict(
                    session.execute(
                        text(
                            "SELECT "
                            "pg_relation_size('memory_records') AS heap_bytes, "
                            "pg_total_relation_size('memory_records') AS record_total_bytes, "
                            "pg_relation_size('vector_bench_hnsw') AS hnsw_index_bytes, "
                            "pg_total_relation_size('memory_vector_index_state') AS "
                            "outbox_total_bytes, "
                            "(SELECT sum(total_bytes) FROM pg_backend_memory_contexts) AS "
                            "backend_allocated_bytes"
                        )
                    )
                    .mappings()
                    .one()
                )
                report["storage"]["float32_vector_payload_bytes"] = int(vectors.nbytes)
                report["storage"] = {key: int(value) for key, value in report["storage"].items()}
            # Pure TurboVec kernel with a precomputed allowlist, separate from full retrieval.
            first_tenant = tenants[samples[0].sample_id]
            allowed = np.asarray(
                [k for k, e in index.entries.items() if e.tenant_id == str(first_tenant)],
                dtype=np.uint64,
            )
            kernel = []
            for _ in range(100):
                started = time.perf_counter()
                index.native.search(queries[:1], k=10, allowlist=allowed)
                kernel.append((time.perf_counter() - started) * 1000)
            report["turbovec_kernel"] = {
                "p50_ms": percentile(kernel, 50),
                "p95_ms": percentile(kernel, 95),
                "iterations": len(kernel),
                "allowed_vectors": len(allowed),
                "scope": "native search with cached allowlist; no SQL or record fetch",
            }
            report["postgres_container_during"] = docker_memory(args.postgres_container)
            report["benchmark_process_rss_bytes"] = psutil.Process().memory_info().rss
            index.close()
            started = time.perf_counter()
            probe = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "apps.benchmark.run_vector_backend_benchmark",
                    "--probe-index",
                    str(path),
                    "--cache",
                    str(args.cache),
                    "--bits",
                    str(args.bits),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            report["turbovec_restart"] = json.loads(probe.stdout)
            report["turbovec_restart"]["fresh_process_wall_ms"] = (
                time.perf_counter() - started
            ) * 1000
            # PostgreSQL server stays running; only the reader's connection is restarted.
            for name, backend in backends.items():
                if isinstance(backend, PgvectorIndex):
                    engine.dispose()
                    query = RetrievalQuery(
                        tenant_id=first_tenant, query_embedding=queries[0].tolist(), limit=10
                    )
                    _, elapsed = timed_search(factory, backend, query)
                    report.setdefault("postgres_reader_restart", {})[name] = {
                        "connection_and_first_revalidated_query_ms": elapsed
                    }
        finally:
            index.close()
    # Dedicated schema so probes do not contaminate corpus or reconciliation timings.
    with vector_benchmark_database(default_database_url()) as (_, failure_factory):
        report["failure_checks"] = failure_probes(
            failure_factory, args.index / f"failures-{uuid4().hex}.zip"
        )
    if not all(report["failure_checks"].values()):
        raise RuntimeError(f"failure checks did not pass: {report['failure_checks']}")
    report["created_at"] = datetime.now(UTC).isoformat()
    sources = sorted(Path("apps/memory_service").rglob("*.py")) + sorted(
        Path("apps/benchmark").rglob("*.py")
    )
    hasher = hashlib.sha256()
    for source in sources:
        hasher.update(str(source).encode())
        hasher.update(source.read_bytes())
    report["implementation_sha256"] = hasher.hexdigest()
    report["index_path"] = str(path)
    report["index_sha256"] = sha256(path)
    return report


def markdown(report: dict[str, Any]) -> str:
    rows = [r for r in report["aggregates"] if r["source"] == "ALL"]
    plain = {r["backend"]: r for r in rows if r["filter"] == "tenant_only"}
    baseline, turbo = plain["pgvector_exact"], plain["turbovec"]
    ratio = baseline["p50_ms"] / turbo["p50_ms"]
    lines = [
        "# Phase 6B: vector backend learning report",
        "",
        "## What this run asks",
        "",
        "Does a compressed second vector index improve this local memory "
        "workload enough to pay for synchronization and reconciliation? "
        "All backends receive identical context chunks, 384-dimensional "
        "normalized MiniLM embeddings, and precomputed question vectors. "
        "PostgreSQL remains authoritative. Every result is fetched and checked "
        "there before it can enter context.",
        "",
        "Illustrative example: PostgreSQL commits a checkout-api timeout "
        "record while TurboVec is unavailable. "
        "The record remains committed; its external sync job is failed and "
        "retryable. A later tombstone is immediately excluded from reads, "
        "even if its old vector remains in TurboVec. These are failure "
        "scenarios; measured checks appear below.",
        "",
        "## How to read the measurements",
        "",
        "Gold Recall@10 is the fraction of dataset-labeled evidence chunks "
        "found. Precision@10 divides relevant hits by ten; "
        "MRR@10 averages the reciprocal rank of the first relevant result; "
        "binary nDCG@10 rewards relevant chunks near the front. "
        "ANN Recall@10 instead measures overlap with exact pgvector's top ten "
        "eligible neighbors. It diagnoses index approximation, "
        "even when the embedding model ranks the wrong evidence. Dataset gold "
        "and exact-neighbor targets are different.",
        "",
        "P50 is median latency; P95 is the 95th percentile. Latency includes "
        "backend filtering, planner configuration, "
        "connection checkout, authoritative record/provenance fetches and "
        "revalidation. It excludes embedding, model loading, "
        "temporal relation resolution, context packing, and one untimed warmup "
        "per query/backend. Backends are interleaved "
        "in seeded random order for each repetition. No reranker or answer model runs.",
        "",
        "`tenant_only` still enforces status, validity and any trust level. "
        "`filtered` additionally requires HIGH-or-higher trust "
        "and EPISODIC type. Trust is assigned HIGH every fifth chunk and type "
        "SEMANTIC every fourth chunk for a controlled selectivity experiment; "
        "text and gold labels remain unchanged. Filtered gold scores are "
        "omitted because filtering changes eligibility; ANN overlap uses a "
        "filtered exact reference.",
        "",
        "## Measured retrieval",
        "",
        "| Backend | Filters | Queries | Gold Recall@10 | Precision@10 | "
        "MRR@10 | nDCG@10 | ANN Recall@10 | P50 ms | P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value: Any) -> str:
        return "—" if value is None else f"{value:.4f}"

    for r in rows:
        lines.append(
            f"| {r['backend']} | {r['filter']} | {r['queries']} | {fmt(r['recall_at_10'])} | "
            f"{fmt(r['precision_at_10'])} | {fmt(r['mrr_at_10'])} | {fmt(r['ndcg_at_10'])} | "
            f"{r['ann_recall_at_10']:.4f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |"
        )
    lines += [
        "",
        "Per-source gold results keep provided turn labels and answer-span proxy sources visible:",
        "",
        "| Source | Backend | Gold Recall@10 | MRR@10 | nDCG@10 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in report["aggregates"]:
        if r["source"] != "ALL" and r["filter"] == "tenant_only":
            lines.append(
                f"| {r['source']} | {r['backend']} | {r['recall_at_10']:.4f} | "
                f"{r['mrr_at_10']:.4f} | {r['ndcg_at_10']:.4f} |"
            )
    t = report["throughput"]
    lines += [
        "",
        "## Ingest and updates",
        "",
        "Embedding is prepared once outside timing. PostgreSQL ingest includes "
        "evidence, records, provenance, pgvector values, "
        "transactional outbox trigger and a real commit. HNSW bulk-build "
        "throughput amortizes that ingest plus offline index construction; "
        "it is not an incremental-insert measurement. Updates change 256 "
        "embeddings (or all rows in a smaller run), commit, then restore "
        "original vectors before scoring. TurboVec total ingest includes "
        "PostgreSQL ingest plus durable sync; total update includes the "
        "PostgreSQL "
        "commit, existing HNSW maintenance and durable sync. PostgreSQL is "
        "retained in TurboVec storage and throughput costs.",
        "",
        "| Measurement | Records/s |",
        "| --- | ---: |",
    ]
    for key in (
        "pgvector_exact_ingest_records_per_second",
        "pgvector_hnsw_bulk_build_records_per_second",
        "turbovec_sync_records_per_second",
        "turbovec_total_ingest_records_per_second",
        "pgvector_exact_update_records_per_second",
        "pgvector_hnsw_update_records_per_second",
        "turbovec_total_update_records_per_second",
    ):
        lines.append(f"| {key} | {t[key]:.2f} |")
    storage, restart, repair = (
        report["storage"],
        report["turbovec_restart"],
        report["reconciliation"],
    )
    lines += [
        "",
        "## Storage, RAM, restart and repair",
        "",
        "Exact pgvector stores vectors in the record heap/TOAST and has no "
        "separate vector index. HNSW adds a graph index. "
        "TurboVec's native serialization includes rotation/codebook overhead "
        "as well as compressed vectors; its checkpoint also includes the "
        "UUID/revision manifest. "
        "Compression does not remove the authoritative PostgreSQL embeddings, "
        "outbox, or records in this implementation.",
        "",
        "| Measured quantity | Bytes |",
        "| --- | ---: |",
    ]
    for key, value in storage.items():
        lines.append(f"| {key} | {value} |")
    for key in (
        "native_serialized_bytes",
        "checkpoint_bytes",
        "baseline_rss_bytes",
        "ready_rss_bytes",
        "index_rss_delta_bytes",
    ):
        lines.append(f"| turbovec_{key} | {restart[key]} |")
    lines += [
        "",
        "TurboVec RAM is sampled in a fresh subprocess after importing "
        "dependencies, then after load/prepare/first search. "
        "RSS includes Python UUID/revision metadata, Rust maps, rotation state "
        "and search layouts; the delta is allocator- and OS-dependent. "
        "PostgreSQL backend allocated bytes are allocator contexts, not RSS. "
        "Container memory includes the whole running server and other "
        "workloads; "
        "it cannot attribute RAM to exact versus HNSW. These counters have "
        "different scopes and must not be divided into a claimed RAM "
        "compression ratio.",
        "",
        f"TurboVec load: **{restart['load_ms']:.2f} ms**; load + prepare + first native search: "
        f"**{restart['load_and_first_search_ms']:.2f} ms**; "
        f"fresh subprocess wall time: **{restart['fresh_process_wall_ms']:.2f} ms**. "
        "Filesystem cache remains warm. "
        "The existing PostgreSQL server was left running. Reader restart "
        "measurements reconnect and fetch a first revalidated query; "
        "server restart/crash recovery is not measured, and these values are "
        "not equivalent to TurboVec deserialization.",
        "",
        f"A full audit repaired **{repair['damaged']['missing']} missing** and "
        f"**{repair['damaged']['ghosts']} ghost** entries in **{repair['repair_seconds']:.3f} s** "
        f"with {repair['damaged']['failed']} failures. A clean full audit took "
        f"**{repair['clean_audit_seconds']:.3f} s**. "
        "The audit reads every authoritative row and manifest entry, compares "
        "revisions/vector hashes, saves the index and drains repair jobs; its "
        "cost grows with corpus size.",
        "",
        f"The cached-allowlist TurboVec kernel P50 was "
        f"**{report['turbovec_kernel']['p50_ms']:.3f} ms**. "
        "That excludes SQL authorization and record fetches, so it is not the "
        "user-visible retrieval latency.",
        "",
        "PostgreSQL reader restart and container counters:",
        "",
        "```json",
        json.dumps(
            {
                "reader_restart": report["postgres_reader_restart"],
                "container_before": report["postgres_container_before"],
                "container_during": report["postgres_container_during"],
            },
            indent=2,
        ),
        "```",
        "",
        "## Failure checks",
        "",
        "These checks execute real committed PostgreSQL changes and actual "
        "TurboVec writes/checkpoints in a separate temporary schema.",
        "",
        "| Check | Passed |",
        "| --- | --- |",
    ]
    lines += [f"| {key} | {value} |" for key, value in report["failure_checks"].items()]
    lines += [
        "",
        "## Decision for this local workload",
        "",
        f"TurboVec's full retrieval P50 relative to exact pgvector is **{ratio:.2f}x** "
        "speedup (below 1 means slower), "
        f"with ANN Recall@10 **{turbo['ann_recall_at_10']:.4f}**. "
        "Compare both latency tails and the HNSW row before attributing a benefit to compression. "
        + (
            "This run does not show a median latency advantage over exact "
            "pgvector. Keep pgvector as the default; the second index adds durable "
            "sync, "
            "extra storage, ownership restrictions and an O(N) audit without "
            "improving the median request in this pilot."
            if ratio <= 1
            else "This run shows a median latency advantage over exact pgvector. That "
            "alone does not establish that a second index pays for itself; "
            "inspect HNSW, filtered latency, ANN recall, update cost and retained "
            "PostgreSQL storage before adoption."
        ),
        "",
        "## Reproducibility and limits",
        "",
        f"Completed `{report['created_at']}`. "
        f"Implementation SHA-256: `{report['implementation_sha256']}`. "
        f"Prepared corpus SHA-256: `{report['dataset']['samples_sha256']}`. "
        f"Embedding cache SHA-256: `{report['embedding']['cache_sha256']}`. "
        "The companion JSON contains the profile, library versions, all "
        "ranked/gold/exact IDs, repetitions, plans, timings, failure checks "
        "and counters.",
        "",
        "This is a small selected MemoryAgentBench Accurate_Retrieval "
        "diagnostic, not an official answer score or a large-scale production "
        "capacity claim. "
        "RULER uses answer-span proxy labels; LongMemEval uses provided turn "
        "labels inherited by chunks. Gold recall depends on labels, chunking "
        "and embeddings. "
        "One local run with warm reads does not quantify variance across "
        "machines or concurrent writers. The HNSW settings are m=16, "
        "ef_construction=100, "
        "ef_search=100 and strict iterative scanning; JSON EXPLAIN plans "
        "verify the indexed and exact paths. TurboVec uses an uncalibrated "
        "quantized flat scan "
        "at the recorded bit width. HNSW planner switches are restored before "
        "authoritative fetches.",
        "",
        "Run again at larger corpus sizes and lower filter selectivity before "
        "drawing a capacity conclusion. "
        "PostgreSQL server restart and isolated per-index server RSS require a "
        "dedicated disposable server experiment.",
        "",
        "- [TurboVec API](https://github.com/RyanCodrai/turbovec/blob/main/docs/api.md)",
        "- [pgvector indexing and filtering](https://github.com/pgvector/pgvector)",
        "- [MemoryAgentBench dataset](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/benchmarks/vector_backends/embeddings.npz")
    )
    parser.add_argument(
        "--index", type=Path, default=Path("data/benchmarks/vector_backends/indexes")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/reports/vector-backend-comparison.json")
    )
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--postgres-container")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--probe-index", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_index:
        with np.load(args.cache, allow_pickle=False) as cache:
            vector = cache["questions"][0]
        print(json.dumps(restart_probe(args.probe_index, vector, args.bits)))
        return
    load_dotenv()
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    profile = RunProfile.model_validate_json(args.profile.read_text())
    if args.repeats is not None:
        profile = RunProfile.model_validate({**profile.model_dump(), "repeats": args.repeats})
    samples, manifest, selected = corpus(profile, args.samples)
    vectors, queries, metadata = prepare_vectors(samples, selected, profile, args.cache)
    if (
        vectors.shape != (sum(len(s.chunks) for s in samples), 384)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("embedding cache has invalid document vectors")
    if (
        queries.shape
        != (sum(bool(q.relevant_ids) for s in samples for q in selected[s.sample_id]), 384)
        or not np.isfinite(queries).all()
    ):
        raise ValueError("embedding cache has invalid query vectors")
    if args.prepare_only:
        print(json.dumps(metadata, indent=2))
        return
    report = benchmark(args, profile, samples, manifest, selected, vectors, queries, metadata)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(f"Saved {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()

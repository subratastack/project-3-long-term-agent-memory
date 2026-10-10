"""Measured stage timing, final rendered token cost, and PostgreSQL relation footprint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from apps.memory_service.advanced.evaluation.benchmark_adapter import BenchmarkObservation


def summarize_resources(observations: Sequence[BenchmarkObservation]) -> dict[str, Any]:
    stages = ("retrieval_ms", "reranking_ms", "packing_ms", "total_ms")
    return {
        "samples": len(observations),
        "latency_ms": {
            stage.removesuffix("_ms"): {
                "p50": float(np.percentile([getattr(o.latency, stage) for o in observations], 50)),
                "p95": float(np.percentile([getattr(o.latency, stage) for o in observations], 95)),
            }
            for stage in stages
        },
        "mean_context_tokens": float(np.mean([o.context_tokens for o in observations])),
        "max_context_tokens": max(o.context_tokens for o in observations),
        "errors": sum(o.error is not None for o in observations),
        "reranker_fallbacks": sum(o.reranker_fell_back for o in observations),
    }


def measure_storage(session: Session, *, schema: str, corpus_size: int) -> dict[str, Any]:
    """Physical sizes of an isolated schema; tables include TOAST but exclude indexes."""
    rows = (
        session.execute(
            text("""
            SELECT c.relname, pg_table_size(c.oid) AS table_bytes,
                   pg_indexes_size(c.oid) AS index_bytes
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relkind = 'r'
            ORDER BY c.relname
        """),
            {"schema": schema},
        )
        .mappings()
        .all()
    )
    tables = {
        str(row["relname"]): {
            "table_bytes": int(row["table_bytes"]),
            "index_bytes": int(row["index_bytes"]),
        }
        for row in rows
    }
    table_bytes = sum(t["table_bytes"] for t in tables.values())
    index_bytes = sum(t["index_bytes"] for t in tables.values())
    return {
        "scope": "isolated benchmark schema; shared by the three memory-enabled read strategies",
        "corpus_size": corpus_size,
        "table_bytes": table_bytes,
        "index_bytes": index_bytes,
        "total_bytes": table_bytes + index_bytes,
        "bytes_per_source_write": (table_bytes + index_bytes) / corpus_size
        if corpus_size
        else None,
        "tables": tables,
    }

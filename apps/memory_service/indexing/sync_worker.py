"""Replay committed outbox jobs; persist the index before acknowledging PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from apps.memory_service.domain.enums import IndexStatus, MemoryStatus
from apps.memory_service.indexing.index_status import VectorIndexState
from apps.memory_service.persistence.models import MemoryRecordRow
from apps.memory_service.persistence.turbovec_index import TurboVecIndex


def indexable(record: MemoryRecordRow | None) -> bool:
    # Retain expired/superseded vectors for historical reads, never tombstones/quarantine.
    return (
        record is not None
        and record.status not in (MemoryStatus.TOMBSTONE, MemoryStatus.QUARANTINED)
        and record.index_status == IndexStatus.INDEXED
        and record.embedding is not None
    )


@dataclass(frozen=True)
class SyncResult:
    selected: int
    indexed: int
    removed: int
    failed: int


def sync_once(
    session_factory: sessionmaker[Session],
    index: TurboVecIndex,
    *,
    batch_size: int = 256,
) -> SyncResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Always acquire in this order (DB then file lock), including reconciliation.
    with session_factory.begin() as session:
        jobs = list(
            session.scalars(
                select(VectorIndexState)
                .where(VectorIndexState.state.in_(["pending", "failed"]))
                .order_by(VectorIndexState.attempts, VectorIndexState.vector_id)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )
        )
        completed: list[VectorIndexState] = []
        removed = indexed = failed = 0
        with index.lock:
            for job in jobs:
                record = session.scalar(
                    select(MemoryRecordRow)
                    .where(
                        MemoryRecordRow.tenant_id == job.tenant_id,
                        MemoryRecordRow.memory_id == job.memory_id,
                    )
                    .execution_options(populate_existing=True)
                )
                job.attempts += 1
                try:
                    if indexable(record):
                        assert record is not None and record.embedding is not None
                        index.upsert(
                            job.vector_id,
                            job.tenant_id,
                            job.memory_id,
                            job.revision,
                            record.embedding,
                        )
                        job.state = "indexed"
                    else:
                        index.remove(job.vector_id)
                        job.state = "absent"
                    completed.append(job)
                    job.last_error = None
                except Exception as exc:
                    job.state = "failed"
                    job.last_error = type(exc).__name__
                    failed += 1
            if jobs:
                try:
                    index.checkpoint()
                except Exception as exc:
                    for job in completed:
                        job.state = "failed"
                        job.last_error = type(exc).__name__
                    failed += len(completed)
                    completed = []
            for job in completed:
                job.indexed_revision = job.revision
                indexed += job.state == "indexed"
                removed += job.state == "absent"
        # A DB commit failure leaves a durable index but pending jobs: safe replay.
    return SyncResult(len(jobs), indexed, removed, failed)


def main() -> None:
    """Opt-in command-line worker for a migrated database and one owned index file."""
    import argparse
    import json
    import time
    from dataclasses import asdict
    from pathlib import Path

    from dotenv import load_dotenv

    from apps.memory_service.persistence.unit_of_work import build_engine, build_session_factory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0 or args.batch_size < 1:
        parser.error("interval and batch-size must be positive")
    load_dotenv()
    engine = build_engine()
    index = TurboVecIndex(args.index, bit_width=args.bits)
    try:
        factory = build_session_factory(engine)
        if args.reconcile:
            from apps.memory_service.indexing.reconciliation import reconcile

            print(json.dumps(asdict(reconcile(factory, index))), flush=True)
        else:
            while True:
                result = sync_once(factory, index, batch_size=args.batch_size)
                print(json.dumps(asdict(result)), flush=True)
                if not args.watch:
                    break
                time.sleep(args.interval)
    finally:
        index.close()
        engine.dispose()


if __name__ == "__main__":
    main()

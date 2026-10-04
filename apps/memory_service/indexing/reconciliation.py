"""Audit the full index inventory and repair drift against PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from apps.memory_service.indexing.index_status import BACKFILL_DDL, VectorIndexState
from apps.memory_service.indexing.sync_worker import indexable, sync_once
from apps.memory_service.persistence.models import MemoryRecordRow
from apps.memory_service.persistence.turbovec_index import TurboVecIndex, vector_digest


@dataclass(frozen=True)
class ReconciliationResult:
    missing: int
    ghosts: int
    stale: int
    repaired: int
    failed: int


def reconcile(session_factory: sessionmaker[Session], index: TurboVecIndex) -> ReconciliationResult:
    """Full O(N) audit; serialized with sync jobs, requeue before changing the index."""
    missing = ghosts = stale = 0
    with session_factory.begin() as session:
        session.execute(text(BACKFILL_DDL))
        states = list(
            session.scalars(
                select(VectorIndexState).order_by(VectorIndexState.vector_id).with_for_update()
            )
        )
        records = {(r.tenant_id, r.memory_id): r for r in session.scalars(select(MemoryRecordRow))}
        with index.lock:
            known = {s.vector_id for s in states}
            for key in set(index.entries) - known:
                index.remove(key)
                ghosts += 1
            for state in states:
                record = records.get((state.tenant_id, state.memory_id))
                entry = index.entries.get(state.vector_id)
                repair = state.state in ("pending", "failed")
                if indexable(record):
                    repair = (
                        repair
                        or state.state != "indexed"
                        or state.indexed_revision != state.revision
                    )
                    assert record is not None and record.embedding is not None
                    if entry is None:
                        missing += 1
                        repair = True
                    elif (
                        entry.revision != state.revision
                        or entry.memory_id != str(state.memory_id)
                        or entry.tenant_id != str(state.tenant_id)
                        or entry.vector_sha256 != vector_digest(record.embedding)
                    ):
                        stale += 1
                        repair = True
                elif entry is not None:
                    ghosts += 1
                    repair = True
                if repair:
                    state.state = "pending"
                    state.attempts = 0
            index.checkpoint()
    repaired = failed = 0
    # Bounded pass over the audit's jobs; a persistent failure never loops forever.
    remaining = len(states)
    while remaining > 0:
        result = sync_once(session_factory, index, batch_size=min(256, remaining))
        if not result.selected:
            break
        remaining -= result.selected
        repaired += result.indexed + result.removed
        failed += result.failed
        if result.failed:
            break
    return ReconciliationResult(missing, ghosts, stale, repaired, failed)

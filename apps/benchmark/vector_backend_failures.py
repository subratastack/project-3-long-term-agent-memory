"""Executable PostgreSQL + real TurboVec failure probes shared with integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from apps.benchmark.capabilities.common import make_record
from apps.memory_service.domain.enums import MemoryStatus, MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import MemoryEvent, Tenant
from apps.memory_service.indexing.index_status import VectorIndexState
from apps.memory_service.indexing.reconciliation import reconcile
from apps.memory_service.indexing.sync_worker import sync_once
from apps.memory_service.persistence.models import MemoryRecordRow
from apps.memory_service.persistence.turbovec_index import TurboVecIndex
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.vector_backend import (
    vector_backend_context,
    vector_backend_search,
)


def failure_probes(factory: sessionmaker[Session], path: Path) -> dict[str, bool]:
    now = datetime.now(UTC)
    tenant, other = uuid4(), uuid4()
    vector = [1.0, *([0.0] * 383)]
    with UnitOfWork(factory) as uow:
        uow.tenants.add(Tenant(tenant_id=tenant, name=f"vector-failure-{tenant}"))
        uow.tenants.add(Tenant(tenant_id=other, name=f"vector-failure-{other}"))
        records = []
        for key, owner, status, trust, start in (
            ("good", tenant, MemoryStatus.ACTIVE, TrustLevel.HIGH, now - timedelta(days=1)),
            ("foreign", other, MemoryStatus.ACTIVE, TrustLevel.HIGH, now - timedelta(days=1)),
            ("low", tenant, MemoryStatus.ACTIVE, TrustLevel.LOW, now - timedelta(days=1)),
            ("quarantined", tenant, MemoryStatus.QUARANTINED, TrustLevel.HIGH, now),
            ("future", tenant, MemoryStatus.ACTIVE, TrustLevel.HIGH, now + timedelta(days=1)),
        ):
            event = MemoryEvent(
                tenant_id=owner,
                source_type=SourceType.CONFIGURATION,
                source_reference=key,
                content="checkout-api timeout is 2 seconds.",
                observed_at=start,
            )
            _, record = make_record(
                event.content,
                owner,
                at=start,
                event=event,
                memory_type=MemoryType.SEMANTIC,
                status=status,
                trust=trust,
            )
            record = record.model_copy(
                update={
                    "embedding": vector,
                    "embedding_model_version": "failure-unit-vector",
                    "index_status": "indexed",
                }
            )
            uow.create_event(event)
            uow.create_memory(record)
            records.append(record)
        uow.commit()
    good = records[0]
    checks: dict[str, bool] = {}
    index = TurboVecIndex(path)
    try:
        with patch.object(index, "upsert", side_effect=OSError("injected index failure")):
            failed = sync_once(factory, index)
        with UnitOfWork(factory) as uow:
            state = uow.session.scalar(
                select(VectorIndexState).where(VectorIndexState.memory_id == good.memory_id)
            )
            checks["commit_survives_external_failure"] = (
                uow.records.get(tenant, good.memory_id) is not None
                and state is not None
                and state.state == "failed"
                and failed.failed >= 1
            )
        retry = sync_once(factory, index)
        checks["retry_is_idempotent"] = (
            retry.failed == 0 and sync_once(factory, index).selected == 0
        )
        query = RetrievalQuery(tenant_id=tenant, query_embedding=vector, min_trust=TrustLevel.HIGH)
        with UnitOfWork(factory) as uow:
            hits = vector_backend_search(uow, index, query)
            checks["tenant_trust_status_time_filters"] = [h.memory.memory_id for h in hits] == [
                good.memory_id
            ]
            context = vector_backend_context(uow, index, query, token_budget=500)
            checks["only_authoritative_records_are_packed"] = [
                m.memory.memory_id for m in context.memories
            ] == [good.memory_id]
        # Simulate a regressed/malicious index candidate path bypassing its SQL allowlist.
        from apps.memory_service.persistence.vector_repository import VectorMatch

        with (
            patch.object(
                index,
                "find_nearest",
                return_value=[VectorMatch(record.memory_id, 0.0) for record in records],
            ),
            UnitOfWork(factory) as uow,
        ):
            checks["revalidation_blocks_foreign_and_ineligible_ids"] = [
                hit.memory.memory_id for hit in vector_backend_search(uow, index, query)
            ] == [good.memory_id]
        with factory.begin() as session:
            session.execute(
                update(MemoryRecordRow)
                .where(MemoryRecordRow.memory_id == good.memory_id)
                .values(status=MemoryStatus.TOMBSTONE)
            )
        # The stale vector remains until sync; the committed tombstone must already be invisible.
        checks["tombstoned_id_still_in_index"] = any(
            e.memory_id == str(good.memory_id) for e in index.entries.values()
        )
        with UnitOfWork(factory) as uow:
            checks["tombstone_blocked_before_sync"] = not vector_backend_search(uow, index, query)
        # Lose one valid foreign entry, add an unknown ghost and keep the tombstone ghost.
        foreign_token = next(
            k for k, e in index.entries.items() if e.memory_id == str(records[1].memory_id)
        )
        index.remove(foreign_token)
        index.upsert(2**62, tenant, uuid4(), 1, vector)
        result = reconcile(factory, index)
        checks["reconciliation_repairs_missing_and_ghosts"] = (
            result.missing >= 1
            and result.ghosts >= 2
            and result.failed == 0
            and foreign_token in index.entries
            and 2**62 not in index.entries
            and all(e.memory_id != str(good.memory_id) for e in index.entries.values())
        )
        # Commit a vector change; a failed checkpoint must not acknowledge it.
        with factory.begin() as session:
            session.execute(
                update(MemoryRecordRow)
                .where(MemoryRecordRow.memory_id == records[1].memory_id)
                .values(embedding=[0.0, 1.0, *([0.0] * 382)])
            )
        with patch.object(index, "checkpoint", side_effect=OSError("injected fsync failure")):
            result_sync = sync_once(factory, index)
        checks["checkpoint_failure_is_retryable"] = result_sync.failed == 1
        checks["checkpoint_retry_recovers"] = sync_once(factory, index).failed == 0
        # Backfill and outbox enrollment must be rolled back alongside a failed write.
        with factory() as session:
            before = session.scalar(
                select(VectorIndexState.revision).where(
                    VectorIndexState.memory_id == records[1].memory_id
                )
            )
            session.execute(
                update(MemoryRecordRow)
                .where(MemoryRecordRow.memory_id == records[1].memory_id)
                .values(trust_level=TrustLevel.LOW)
            )
            session.rollback()
            after = session.scalar(
                select(VectorIndexState.revision).where(
                    VectorIndexState.memory_id == records[1].memory_id
                )
            )
            checks["rolled_back_write_does_not_enqueue"] = before == after
        # A previous checkpoint after PostgreSQL acknowledged a newer revision is repaired.
        old_bytes = path.read_bytes()
        with factory.begin() as session:
            session.execute(
                update(MemoryRecordRow)
                .where(MemoryRecordRow.memory_id == records[1].memory_id)
                .values(embedding=vector)
            )
        sync_once(factory, index)
        index.close()
        path.write_bytes(old_bytes)
        index = TurboVecIndex(path)
        result = reconcile(factory, index)
        checks["old_checkpoint_revision_repaired"] = result.stale >= 1 and result.failed == 0
        index.close()
        path.unlink()
        index = TurboVecIndex(path)
        result = reconcile(factory, index)
        checks["lost_checkpoint_rebuilt"] = result.missing >= 1 and result.failed == 0
        # Authoritative hard deletion also leaves an outbox removal job.
        with factory.begin() as session:
            session.execute(
                delete(MemoryRecordRow).where(MemoryRecordRow.memory_id == records[1].memory_id)
            )
        result = reconcile(factory, index)
        checks["hard_deleted_id_removed"] = result.ghosts >= 1 and result.failed == 0
    finally:
        index.close()
    return checks

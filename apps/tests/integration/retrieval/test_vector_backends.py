import runpy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import select, text, update
from sqlalchemy.engine import Engine

from apps.benchmark.capabilities.common import make_record
from apps.benchmark.vector_backend_database import vector_benchmark_database
from apps.benchmark.vector_backend_failures import failure_probes
from apps.memory_service.domain.enums import TrustLevel
from apps.memory_service.domain.models import Tenant
from apps.memory_service.indexing.index_status import VectorIndexState
from apps.memory_service.indexing.sync_worker import sync_once
from apps.memory_service.persistence.models import MemoryRecordRow
from apps.memory_service.persistence.turbovec_index import TurboVecIndex
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.vector_backend import vector_backend_search


def test_real_commits_external_failures_and_reconciliation(engine: Engine, tmp_path: Path) -> None:
    pytest.importorskip("turbovec")
    with engine.connect() as conn:
        before = conn.execute(text("SELECT count(*) FROM public.memory_records")).scalar_one()
    with vector_benchmark_database(engine.url.render_as_string(hide_password=False)) as (
        _,
        factory,
    ):
        checks = failure_probes(factory, tmp_path / "failure-index.zip")
    assert checks and all(checks.values()), checks
    with engine.connect() as conn:
        assert (
            conn.execute(text("SELECT count(*) FROM public.memory_records")).scalar_one() == before
        )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "tombstone"},
        {"trust_level": "low"},
        {"valid_to": datetime.now(UTC) - timedelta(days=1)},
    ],
)
def test_change_between_candidates_and_fetch_is_revalidated(
    engine: Engine,
    tmp_path: Path,
    change: dict[str, object],
) -> None:
    pytest.importorskip("turbovec")
    with vector_benchmark_database(engine.url.render_as_string(hide_password=False)) as (
        _,
        factory,
    ):
        tenant = uuid4()
        vector = [1.0, *([0.0] * 383)]
        with UnitOfWork(factory) as uow:
            uow.tenants.add(Tenant(tenant_id=tenant, name=f"concurrent-vector-{tenant}"))
            event, record = make_record(
                "checkout-api timeout is 2 seconds.",
                tenant,
                at=datetime.now(UTC) - timedelta(days=2),
                trust=TrustLevel.HIGH,
            )
            record = record.model_copy(
                update={
                    "embedding": vector,
                    "embedding_model_version": "unit-vector",
                    "index_status": "indexed",
                }
            )
            uow.create_event(event)
            uow.create_memory(record)
            uow.commit()
        index = TurboVecIndex(tmp_path / "concurrent.zip")
        try:
            sync_once(factory, index)
            original = index.find_nearest

            def changing_candidates(*args, **kwargs):  # type: ignore[no-untyped-def]
                matches = original(*args, **kwargs)
                with factory.begin() as session:
                    session.execute(
                        update(MemoryRecordRow)
                        .where(MemoryRecordRow.memory_id == record.memory_id)
                        .values(**change)
                    )
                return matches

            query = RetrievalQuery(
                tenant_id=tenant, query_embedding=vector, min_trust=TrustLevel.HIGH
            )
            with (
                patch.object(index, "find_nearest", side_effect=changing_candidates),
                UnitOfWork(factory) as uow,
            ):
                assert vector_backend_search(uow, index, query) == []
        finally:
            index.close()


def test_migration_upgrade_backfill_and_downgrade(engine: Engine) -> None:
    migration = runpy.run_path("migrations/versions/006_create_vector_index_outbox.py")
    with vector_benchmark_database(engine.url.render_as_string(hide_password=False)) as (
        scoped,
        factory,
    ):
        tenant = uuid4()
        with UnitOfWork(factory) as uow:
            uow.tenants.add(Tenant(tenant_id=tenant, name=f"migration-vector-{tenant}"))
            event, record = make_record(
                "checkout-api timeout is 2 seconds.", tenant, at=datetime.now(UTC)
            )
            uow.create_event(event)
            uow.create_memory(record)
            uow.commit()
        with scoped.begin() as conn, Operations.context(MigrationContext.configure(conn)):
            migration["downgrade"]()
            migration["upgrade"]()
        with factory() as session:
            state = session.scalar(
                select(VectorIndexState).where(VectorIndexState.memory_id == record.memory_id)
            )
            assert state is not None and state.state == "pending" and state.revision == 1
        with UnitOfWork(factory) as uow:
            uow.vectors.set_embedding(
                tenant, record.memory_id, [1.0, *([0.0] * 383)], model_version="test"
            )
            uow.commit()
        with factory() as session:
            assert (
                session.scalar(
                    select(VectorIndexState.revision).where(
                        VectorIndexState.memory_id == record.memory_id
                    )
                )
                == 2
            )

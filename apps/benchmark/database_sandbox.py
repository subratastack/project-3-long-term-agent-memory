"""Rollback-only PostgreSQL scope for isolated memory experiments."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from apps.memory_service.ingestion.service import UowFactory
from apps.memory_service.persistence.models import SEARCH_VECTOR_FUNCTION_DDL, Base
from apps.memory_service.persistence.unit_of_work import UnitOfWork, build_engine


@contextmanager
def benchmark_database(database_url: str | None = None) -> Iterator[UowFactory]:
    """Contain service commits in savepoints; roll back seeds and schema on exit.

    Like the existing poisoning benchmark, this can initialize missing tables
    within the outer transaction. It never drops tables or commits that outer
    transaction, including on failure. Callers use fresh tenant IDs.
    """
    engine = build_engine(database_url)
    try:
        with engine.connect() as connection:
            outer = connection.begin()
            try:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.execute(text(SEARCH_VECTOR_FUNCTION_DDL))
                Base.metadata.create_all(connection)
                session_factory = sessionmaker(
                    bind=connection,
                    expire_on_commit=False,
                    join_transaction_mode="create_savepoint",
                )

                def uow_factory() -> UnitOfWork:
                    return UnitOfWork(session_factory)

                yield uow_factory
            finally:
                outer.rollback()
    finally:
        engine.dispose()

"""Fixtures for the adversarial suite.

Pure policy tests need no fixtures. Database tests request `uow_factory`,
which reuses the integration suite's disposable test database and per-test
rolled-back transaction, and skip when PostgreSQL is unreachable. The schema
fixture is not autouse, so the pure tests still run without a database.
"""

from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from apps.memory_service.persistence.models import SEARCH_VECTOR_FUNCTION_DDL, Base
from apps.memory_service.persistence.unit_of_work import UnitOfWork
from apps.tests.integration.conftest import (  # noqa: F401
    connection,
    engine,
    session_factory,
)

UowFactory = Callable[[], UnitOfWork]


@pytest.fixture(scope="session")
def schema(engine: Engine) -> Iterator[None]:  # noqa: F811
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(SEARCH_VECTOR_FUNCTION_DDL))
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def uow_factory(schema: None, session_factory: sessionmaker[Session]) -> UowFactory:  # noqa: F811
    def make() -> UnitOfWork:
        return UnitOfWork(session_factory)

    return make

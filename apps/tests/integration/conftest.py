"""Fixtures for integration tests that require a real PostgreSQL instance.

These tests run against the PostgreSQL started by `docker compose up -d postgres`
(see docker-compose.yml / .env.example), but in their *own* database:
`TEST_DATABASE_URL` if set, otherwise `DATABASE_URL` with `_test` appended to
the database name (e.g. `agent_memory_test`), created on first use. The
suite creates every table at the start and drops them all at the end, so
pointing it at the dev database would wipe it -- and leave Alembic believing
its migrations were still applied. If the server is unreachable, the whole
integration suite is skipped so a plain `uv run pytest` does not require
Docker for the domain unit tests.

Each test runs inside its own outer database transaction that is always rolled
back at teardown, so tests never leak rows into each other even though the
code under test calls `Session.commit()` internally (each `UnitOfWork` session
joins that outer transaction as a SAVEPOINT via `join_transaction_mode`).
"""

import os
from collections.abc import Callable, Iterator

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from apps.memory_service.persistence.models import SEARCH_VECTOR_FUNCTION_DDL, Base
from apps.memory_service.persistence.unit_of_work import UnitOfWork, default_database_url

load_dotenv()


def _test_database_url() -> str:
    """`TEST_DATABASE_URL`, or `DATABASE_URL` with `_test` appended to the database name."""
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    url = make_url(default_database_url())
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


def _ensure_database_exists(url: str) -> None:
    """Create the test database if it's missing, via the server's `postgres` database."""
    target = make_url(url)
    admin = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": target.database},
            ).scalar()
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    url = _test_database_url()
    try:
        _ensure_database_exists(url)
    except OperationalError:
        pytest.skip(
            "PostgreSQL is not reachable at DATABASE_URL; "
            "run `docker compose up -d postgres` first."
        )
    test_engine = create_engine(url)
    yield test_engine
    test_engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _schema(engine: Engine) -> Iterator[None]:
    with engine.begin() as connection:
        # memory_records.embedding is a pgvector column; the extension must
        # exist before create_all can create that column's type.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # memory_records.search_vector is a GENERATED column backed by this
        # function; it must exist before create_all can create that column.
        connection.execute(text(SEARCH_VECTOR_FUNCTION_DDL))
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def connection(engine: Engine) -> Iterator[Connection]:
    conn = engine.connect()
    outer_transaction = conn.begin()
    try:
        yield conn
    finally:
        outer_transaction.rollback()
        conn.close()


@pytest.fixture
def session_factory(connection: Connection) -> sessionmaker[Session]:
    return sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )


@pytest.fixture
def uow_factory(session_factory: sessionmaker[Session]) -> Callable[[], UnitOfWork]:
    def _make() -> UnitOfWork:
        return UnitOfWork(session_factory)

    return _make

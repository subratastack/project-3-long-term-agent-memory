"""Reuse the existing disposable PostgreSQL database and savepoint fixtures."""

from apps.tests.integration.conftest import (  # noqa: F401
    _schema,
    connection,
    engine,
    session_factory,
    uow_factory,
)

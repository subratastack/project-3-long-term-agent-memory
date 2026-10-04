"""Committed, temporary schema for outbox crash tests without touching user records."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from apps.memory_service.indexing.index_status import install_outbox
from apps.memory_service.persistence.models import SEARCH_VECTOR_FUNCTION_DDL, Base


@contextmanager
def vector_benchmark_database(database_url: str) -> Iterator[tuple[Engine, sessionmaker[Session]]]:
    """Unique schema, real commits, finally remove only this invocation's schema."""
    schema = f"vector_bench_{uuid4().hex}"
    admin = create_engine(database_url)
    scoped: Engine | None = None
    created = False
    try:
        with admin.begin() as connection:
            if not connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname='vector'")
            ).scalar():
                raise RuntimeError("existing database must have the pgvector extension installed")
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            created = True
        scoped = create_engine(
            database_url,
            connect_args={"options": f"-csearch_path={schema},public"},
            execution_options={"schema_translate_map": {None: schema}},
        )
        with scoped.begin() as connection:
            connection.execute(text(SEARCH_VECTOR_FUNCTION_DDL))
            Base.metadata.create_all(connection)
            actual_schema = connection.execute(
                text("""
                SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.oid = 'memory_records'::regclass
            """)
            ).scalar_one()
            if actual_schema != schema:
                raise RuntimeError("benchmark records did not resolve to the isolated schema")
            install_outbox(connection)
        yield scoped, sessionmaker(scoped, expire_on_commit=False)
    finally:
        if scoped is not None:
            scoped.dispose()
        if created:
            with admin.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()

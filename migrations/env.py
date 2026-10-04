"""Alembic environment configuration for the memory core schema.

The database URL is resolved from the `DATABASE_URL` environment variable at
run time (loaded from `.env` if present; see `.env.example`), falling back to
`alembic.ini`'s documented default, so the same migration set runs unchanged
across local, CI, and deployed environments.
"""

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from apps.memory_service.indexing.index_status import VectorIndexState  # noqa: F401
from apps.memory_service.persistence.models import Base

load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    configured_url = config.get_main_option("sqlalchemy.url")
    if configured_url:
        return configured_url
    raise RuntimeError("DATABASE_URL is not set and no sqlalchemy.url is configured.")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live database connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations directly."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

"""Transactional boundary coordinating the memory core repositories.

`UnitOfWork` is a context manager that hands out one SQLAlchemy `Session` and
the seven repositories built on top of it (`uow.tenants`, `uow.events`,
`uow.records`, `uow.provenance`, `uow.relations`, `uow.write_decisions`,
`uow.vectors`) for
the lifetime of a single logical operation -- for example, "accept this candidate as a memory
record and write its provenance and audit decision". Everything done through
those repositories inside one `with UnitOfWork(...) as uow:` block shares the
same database transaction, so it either all commits together or all rolls
back together.

Typical usage:

    with UnitOfWork(session_factory) as uow:
        uow.create_event(event)
        uow.create_memory(record)
        uow.commit()
"""

import os
from datetime import UTC, datetime
from types import TracebackType
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from apps.memory_service.domain.enums import ConflictType, MemoryType, WriteDecision
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    MemoryRelation,
    WritePolicyDecision,
)
from apps.memory_service.persistence.repositories import (
    MemoryEventRepository,
    MemoryProvenanceRepository,
    MemoryRecordRepository,
    MemoryRelationRepository,
    MemoryWriteDecisionRepository,
    TenantRepository,
)
from apps.memory_service.persistence.vector_repository import VectorRecordRepository

_DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://agent_memory:agent_memory_local@localhost:5432/agent_memory"
)


def default_database_url() -> str:
    """Return the `DATABASE_URL` environment variable, or a local dev fallback.

    The fallback matches the credentials in `docker-compose.yml` /
    `.env.example`, so a plain local checkout with `docker compose up -d
    postgres` works without needing to export anything first.

    Example:
        Input:
            os.environ = {}  # DATABASE_URL not set
        Output:
            "postgresql+psycopg://agent_memory:agent_memory_local@localhost:5432/agent_memory"
    """
    return os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)


def build_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy `Engine` for `database_url`, or the resolved default.

    Example:
        Input:
            database_url = "postgresql+psycopg://agent_memory:agent_memory_local@localhost:5432/agent_memory"
        Output:
            Engine(postgresql+psycopg://agent_memory:***@localhost:5432/agent_memory)
    """
    return create_engine(database_url or default_database_url(), pool_pre_ping=True)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a `sessionmaker` bound to `engine`.

    `expire_on_commit=False` is set so that a domain object handed back from
    a repository (e.g. a `MemoryRecord` returned by `uow.get_memory`) stays
    readable after `uow.commit()`, instead of SQLAlchemy invalidating it and
    forcing a fresh database round trip just to read a field.

    Example:
        Input:
            engine = build_engine()
        Output:
            sessionmaker(bind=engine, expire_on_commit=False)
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


class UnitOfWork:
    """One transaction's worth of session + repositories, as a context manager."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self.session: Session
        self.tenants: TenantRepository
        self.events: MemoryEventRepository
        self.records: MemoryRecordRepository
        self.provenance: MemoryProvenanceRepository
        self.relations: MemoryRelationRepository
        self.write_decisions: MemoryWriteDecisionRepository
        self.vectors: VectorRecordRepository

    def __enter__(self) -> "UnitOfWork":
        """Open a new session and attach a fresh repository to each attribute.

        How it works:
            Calling `self._session_factory()` opens a new `Session` bound to
            this unit of work's engine/connection, and each repository
            (`self.events`, `self.records`, ...) is constructed to wrap that
            same session -- so every repository method called through this
            `UnitOfWork` instance operates inside the one transaction that
            session owns.

        Example:
            Input:
                uow = UnitOfWork(session_factory)
                uow.__enter__()
            Output:
                the same `uow`, now with `uow.session`, `uow.events`,
                `uow.records`, `uow.provenance`, `uow.relations`, and
                `uow.write_decisions` all populated and ready to use
        """
        self.session = self._session_factory()
        self.tenants = TenantRepository(self.session)
        self.events = MemoryEventRepository(self.session)
        self.records = MemoryRecordRepository(self.session)
        self.provenance = MemoryProvenanceRepository(self.session)
        self.relations = MemoryRelationRepository(self.session)
        self.write_decisions = MemoryWriteDecisionRepository(self.session)
        self.vectors = VectorRecordRepository(self.session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Roll back on an unhandled exception, then always close the session.

        How it works:
            1. If the `with` block raised an exception (`exc_type` is not
               `None`), roll back the session first -- this undoes anything
               staged or executed since the last `commit()`, so a failure
               partway through a multi-step operation (e.g. event + memory +
               decision) never leaves a partial write behind.
            2. Whether or not there was an exception, close the session in a
               `finally` block, releasing its database connection back to the
               pool. The exception itself is not swallowed here (this method
               returns `None`), so it continues to propagate to the caller.

        Example:
            Input:
                exc_type = IntegrityError
                exc = IntegrityError("duplicate key value violates unique constraint")
                tb = <traceback object>
            Output:
                None (the session is rolled back and closed; the
                IntegrityError still propagates out of the `with` block)
        """
        try:
            if exc_type is not None:
                self.session.rollback()
        finally:
            self.session.close()

    def commit(self) -> None:
        """Commit everything done through this unit of work's repositories so far.

        Example:
            Input:
                (nothing -- commits whatever is currently pending)
            Output:
                None (all pending inserts/updates are now durable)
        """
        self.session.commit()

    def rollback(self) -> None:
        """Discard everything done through this unit of work since the last commit.

        Example:
            Input:
                (nothing -- discards whatever is currently pending)
            Output:
                None (all pending inserts/updates since the last commit are undone)
        """
        self.session.rollback()

    def create_event(self, event: MemoryEvent) -> None:
        """Persist a new immutable evidence event.

        Example:
            Input:
                event = MemoryEvent(
                    tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                    source_type=SourceType.USER_MESSAGE,
                    source_reference="chat-42",
                    content="I prefer email over phone calls.",
                )
            Output:
                None (a new row now exists in `memory_events`)
        """
        self.events.add(event.tenant_id, event)

    def create_memory(self, record: MemoryRecord) -> None:
        """Persist a new accepted memory record along with its provenance links.

        Example:
            Input:
                record = MemoryRecord(
                    tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                    content="The user prefers email over phone calls.",
                    confidence=0.9,
                    provenance=[Provenance(...)],
                    temporal_validity=TemporalValidity(valid_from=datetime.now(UTC)),
                    memory_type=MemoryType.SEMANTIC,
                    trust_level=TrustLevel.MEDIUM,
                    status=MemoryStatus.ACTIVE,
                )
            Output:
                None (new rows now exist in `memory_records` and `memory_provenance`)
        """
        self.records.add(record.tenant_id, record)

    def get_memory(self, memory_id: UUID, tenant_id: UUID) -> MemoryRecord | None:
        """Retrieve a memory record, scoped to the requesting tenant.

        Example:
            Input:
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
            Output:
                MemoryRecord(content="The user prefers email over phone calls.", ...)
                -- or None if no such memory exists for that tenant.
        """
        return self.records.get(tenant_id, memory_id)

    def list_active_memories(
        self,
        tenant_id: UUID,
        memory_type: MemoryType | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryRecord]:
        """List memories visible to a tenant.

        How it works:
            1. With `as_of=None` (the common case), delegate to
               `MemoryRecordRepository.list_active`, which returns only
               memories that are presently `ACTIVE` and currently within
               their validity window.
            2. With an `as_of` timestamp given, delegate instead to
               `MemoryRecordRepository.list_effective`, which considers
               memories that have since been superseded or expired but were
               genuinely in effect at that historical moment -- so a past
               state can still be explained, not just the present one.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                as_of = None
            Output:
                [MemoryRecord(content="The user's timeout is 2s.", status=MemoryStatus.ACTIVE, ...)]
                -- only presently active memories.
        """
        if as_of is None:
            return self.records.list_active(tenant_id, memory_type)
        return self.records.list_effective(tenant_id, as_of=as_of, memory_type=memory_type)

    def record_write_decision(self, decision: WritePolicyDecision) -> None:
        """Persist a deterministic write-policy audit decision.

        Example:
            Input:
                decision = WritePolicyDecision(
                    tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                    candidate_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
                    decision=WriteDecision.ACCEPT,
                    policy_version="1.0",
                )
            Output:
                None (a new row now exists in `memory_write_decisions`)
        """
        self.write_decisions.add(decision.tenant_id, decision)

    def supersede_memory(
        self,
        tenant_id: UUID,
        old_memory_id: UUID,
        replacement: MemoryRecord,
        decision: WritePolicyDecision,
        *,
        rationale: str | None = None,
    ) -> MemoryRelation:
        """Atomically retire an old memory and add its replacement.

        How it works:
            1. Sanity-check the inputs before touching the database: the
               replacement and decision must both belong to `tenant_id`, the
               replacement must be a genuinely different memory from
               `old_memory_id`, `decision.decision` must actually be
               `SUPERSEDE`, and `decision.accepted_memory_id` /
               `decision.superseded_memory_id` must point at the replacement
               and the old memory respectively. Any mismatch raises
               `ValueError` immediately, before any write happens.
            2. Look up the old memory by `tenant_id` and `old_memory_id`; if
               it does not exist, raise `LookupError` rather than silently
               creating an orphaned replacement.
            3. Mark the old memory `SUPERSEDED`, closing its validity window
               at the replacement's `valid_from` (via
               `MemoryRecordRepository.mark_superseded`), so the two windows
               meet without overlapping.
            4. Insert the replacement memory (and its provenance).
            5. Create a `SUPERSESSION` relation edge from the replacement to
               the old memory, carrying `rationale` if one was given.
            6. Persist the `decision` itself as the audit record for this
               whole operation.
            7. Return the relation that was created, in case the caller wants
               to inspect or log it.

            None of these steps commits on their own -- the caller is still
            expected to call `uow.commit()` afterwards, so that a failure
            partway through (or the caller deciding not to commit) leaves the
            old memory untouched.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                old_memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")  # timeout=5s, ACTIVE
                replacement = MemoryRecord(
                    content="timeout=2s", memory_type=MemoryType.SEMANTIC, ...
                )
                decision = WritePolicyDecision(
                    decision=WriteDecision.SUPERSEDE,
                    accepted_memory_id=replacement.memory_id,
                    superseded_memory_id=old_memory_id,
                    ...
                )
                rationale = "operator lowered the timeout"
            Output:
                MemoryRelation(
                    source_memory_id=replacement.memory_id,
                    target_memory_id=old_memory_id,
                    relation_type=ConflictType.SUPERSESSION,
                    rationale="operator lowered the timeout",
                )
                -- and the old memory's row now has status=SUPERSEDED.
        """
        if replacement.tenant_id != tenant_id or decision.tenant_id != tenant_id:
            raise ValueError("replacement and decision must belong to the requested tenant")
        if replacement.memory_id == old_memory_id:
            raise ValueError("replacement must have a different memory_id")
        if decision.decision != WriteDecision.SUPERSEDE:
            raise ValueError("supersession requires a SUPERSEDE write decision")
        if decision.accepted_memory_id != replacement.memory_id:
            raise ValueError("decision.accepted_memory_id must identify the replacement")
        if decision.superseded_memory_id != old_memory_id:
            raise ValueError("decision.superseded_memory_id must identify the old memory")

        old_record = self.records.get(tenant_id, old_memory_id)
        if old_record is None:
            raise LookupError(f"memory_record {old_memory_id} not found for tenant {tenant_id}")

        superseded_at = replacement.temporal_validity.valid_from
        self.records.mark_superseded(tenant_id, old_memory_id, valid_to=superseded_at)
        self.records.add(tenant_id, replacement)
        relation = MemoryRelation(
            tenant_id=tenant_id,
            source_memory_id=replacement.memory_id,
            target_memory_id=old_memory_id,
            relation_type=ConflictType.SUPERSESSION,
            rationale=rationale,
            created_at=datetime.now(UTC),
        )
        self.relations.add(tenant_id, relation)
        self.write_decisions.add(tenant_id, decision)
        return relation

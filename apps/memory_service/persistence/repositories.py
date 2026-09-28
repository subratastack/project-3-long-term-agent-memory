"""Tenant-scoped repositories over the memory core tables.

One repository class per table (`MemoryEventRepository`,
`MemoryRecordRepository`, and so on), each wrapping a SQLAlchemy `Session`
with plain methods like `add` and `get`. The one rule every method here
follows without exception: **`tenant_id` is a required argument, and it is
always part of the actual SQL `WHERE` clause** -- never a filter applied to
results after the fact. That is what makes tenant isolation something the
database enforces, rather than something a caller could accidentally forget
to check. See ARCHITECTURE.md ("Tenant isolation").

These repositories are normally not used directly; `UnitOfWork` (in
`unit_of_work.py`) wires one of each up per transaction and exposes them as
`uow.events`, `uow.records`, etc.
"""

import datetime
from collections.abc import Collection
from typing import cast
from uuid import UUID

from sqlalchemy import ColumnElement, exists, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from apps.memory_service.domain.enums import MemoryStatus, MemoryType
from apps.memory_service.domain.models import (
    MemoryEvent,
    MemoryRecord,
    MemoryRelation,
    Provenance,
    TemporalValidity,
    Tenant,
    WritePolicyDecision,
)
from apps.memory_service.persistence.models import (
    MemoryEventIngestionRow,
    MemoryEventRow,
    MemoryProvenanceRow,
    MemoryRecordRow,
    MemoryRelationRow,
    MemoryWriteDecisionRow,
    TenantRow,
)


def _require_matching_tenant(owner_tenant_id: UUID, tenant_id: UUID, entity_name: str) -> None:
    """Fail fast if a domain object's own `tenant_id` disagrees with the
    `tenant_id` a repository method was called with, before anything is sent
    to the database. This catches a caller passing the wrong tenant scope by
    mistake, in addition to (not instead of) the database-level composite
    foreign keys described in `persistence/models.py`.

    Example:
        Input:
            owner_tenant_id = UUID("11111111-1111-1111-1111-111111111111")
            tenant_id = UUID("22222222-2222-2222-2222-222222222222")
            entity_name = "event"
        Output:
            raises ValueError("event.tenant_id does not match the requested tenant_id")
    """
    if owner_tenant_id != tenant_id:
        raise ValueError(f"{entity_name}.tenant_id does not match the requested tenant_id")


# --- Row <-> domain model conversion -----------------------------------
#
# The functions below translate a SQLAlchemy ORM row (the on-disk shape) into
# the corresponding Pydantic domain model (the shape the rest of the app
# works with). Each one is a straight field-by-field copy; the only mild
# subtlety is that JSONB/ARRAY columns are copied into plain `dict`/`list`
# objects rather than handed over as SQLAlchemy-tracked collections.


def _tenant_row_to_domain(row: TenantRow) -> Tenant:
    return Tenant(
        tenant_id=row.tenant_id,
        name=row.name,
        description=row.description,
        created_at=row.created_at,
    )


def _event_row_to_domain(row: MemoryEventRow) -> MemoryEvent:
    return MemoryEvent(
        event_id=row.event_id,
        tenant_id=row.tenant_id,
        source_type=row.source_type,
        source_reference=row.source_reference,
        content=row.content,
        observed_at=row.observed_at,
        actor_id=row.actor_id,
        metadata=dict(row.metadata_),
    )


def _provenance_row_to_domain(row: MemoryProvenanceRow) -> Provenance:
    return Provenance(
        event_id=row.event_id,
        source_type=row.source_type,
        source_reference=row.source_reference,
        observed_at=row.observed_at,
        trust_level=row.trust_level,
        excerpt=row.excerpt,
    )


def _record_row_to_domain(row: MemoryRecordRow, provenance: list[Provenance]) -> MemoryRecord:
    """Rebuild a `MemoryRecord` from its row plus its already-loaded provenance.

    `provenance` is a separate argument (rather than a relationship SQLAlchemy
    loads automatically) because `memory_provenance` rows are fetched via
    `MemoryProvenanceRepository.list_for_memory`; the caller is expected to
    have done that lookup and pass the result in here.
    """
    return MemoryRecord(
        memory_id=row.memory_id,
        tenant_id=row.tenant_id,
        content=row.content,
        confidence=row.confidence,
        provenance=provenance,
        temporal_validity=TemporalValidity(valid_from=row.valid_from, valid_to=row.valid_to),
        memory_type=row.memory_type,
        subject_keys=list(row.subject_keys),
        trust_level=row.trust_level,
        metadata=dict(row.metadata_),
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        policy_version=row.policy_version,
        embedding=list(row.embedding) if row.embedding is not None else None,
        embedding_model_version=row.embedding_model_version,
        index_status=row.index_status,
    )


def _relation_row_to_domain(row: MemoryRelationRow) -> MemoryRelation:
    return MemoryRelation(
        relation_id=row.relation_id,
        tenant_id=row.tenant_id,
        source_memory_id=row.source_memory_id,
        target_memory_id=row.target_memory_id,
        relation_type=row.relation_type,
        metadata=dict(row.metadata_),
        created_at=row.created_at,
        rationale=row.rationale,
    )


def _decision_row_to_domain(row: MemoryWriteDecisionRow) -> WritePolicyDecision:
    return WritePolicyDecision(
        decision_id=row.decision_id,
        tenant_id=row.tenant_id,
        candidate_id=row.candidate_id,
        decision=row.decision,
        decided_at=row.decided_at,
        policy_version=row.policy_version,
        reason_codes=list(row.reason_codes),
        explanation=row.explanation,
        accepted_memory_id=row.accepted_memory_id,
        superseded_memory_id=row.superseded_memory_id,
    )


class TenantRepository:
    """Reads and writes rows in `tenants` -- the tenant registry."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, tenant: Tenant) -> None:
        """Stage a new tenant. A duplicate `name` fails at flush/commit with
        `IntegrityError` (unique constraint `uq_tenants_name`)."""
        self._session.add(
            TenantRow(
                tenant_id=tenant.tenant_id,
                name=tenant.name,
                description=tenant.description,
                created_at=tenant.created_at,
            )
        )

    def get(self, tenant_id: UUID) -> Tenant | None:
        row = self._session.get(TenantRow, tenant_id)
        return _tenant_row_to_domain(row) if row is not None else None

    def get_by_name(self, name: str) -> Tenant | None:
        row = self._session.scalars(select(TenantRow).where(TenantRow.name == name)).first()
        return _tenant_row_to_domain(row) if row is not None else None

    def list_all(self) -> list[Tenant]:
        """Every tenant, oldest first."""
        rows = self._session.scalars(
            select(TenantRow).order_by(TenantRow.created_at, TenantRow.name)
        ).all()
        return [_tenant_row_to_domain(row) for row in rows]


class MemoryEventRepository:
    """Reads and writes rows in `memory_events` -- immutable raw evidence."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, tenant_id: UUID, event: MemoryEvent) -> None:
        """Insert a new event row.

        How it works:
            1. Check that `event.tenant_id` actually matches the `tenant_id`
               the caller passed in, raising `ValueError` immediately if not
               (see `_require_matching_tenant`).
            2. Stage a new `MemoryEventRow` with the event's fields.
            3. Flush the session right away. This is not just tidiness: no
               ORM `relationship()` connects `MemoryEventRow` to
               `MemoryProvenanceRow`, so SQLAlchemy's unit-of-work has no way
               to know that a provenance row inserted later in the same
               transaction must come *after* this event row. Flushing here
               forces the event to exist in the database before any
               provenance row that references it gets inserted.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                event = MemoryEvent(
                    tenant_id=tenant_id,
                    source_type=SourceType.USER_MESSAGE,
                    source_reference="chat-42",
                    content="I prefer email over phone calls.",
                    observed_at=datetime(2026, 1, 5, tzinfo=UTC),
                )
            Output:
                None (a new row now exists in `memory_events`, visible to
                later `get`/`select` calls in the same transaction)
        """
        _require_matching_tenant(event.tenant_id, tenant_id, "event")
        self._session.add(
            MemoryEventRow(
                event_id=event.event_id,
                tenant_id=tenant_id,
                source_type=event.source_type,
                source_reference=event.source_reference,
                content=event.content,
                observed_at=event.observed_at,
                actor_id=event.actor_id,
                metadata_=dict(event.metadata),
            )
        )
        self._session.flush()

    def get(self, tenant_id: UUID, event_id: UUID) -> MemoryEvent | None:
        """Look up one event by id, scoped to `tenant_id`.

        How it works:
            1. Build a `SELECT` that filters on both `tenant_id` and
               `event_id` in the same `WHERE` clause -- an event owned by a
               different tenant simply will not match, so this can never leak
               data across tenants.
            2. Run it and take at most one row (`one_or_none`); a
               non-existent or wrong-tenant event both come back as no rows.
            3. Convert the row to a `MemoryEvent`, or return `None` if there
               was no match.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                event_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
            Output:
                MemoryEvent(
                    event_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                    source_type=SourceType.USER_MESSAGE,
                    source_reference="chat-42",
                    content="I prefer email over phone calls.",
                    ...
                )
                -- or None if no such event exists for that tenant.
        """
        stmt = select(MemoryEventRow).where(
            MemoryEventRow.tenant_id == tenant_id, MemoryEventRow.event_id == event_id
        )
        row = self._session.scalars(stmt).one_or_none()
        return _event_row_to_domain(row) if row is not None else None

    def record_ingestion(self, tenant_id: UUID, event_id: UUID, candidate_count: int) -> None:
        """Append a `memory_event_ingestions` row: `event_id` was ingested without error."""
        self._session.add(
            MemoryEventIngestionRow(
                tenant_id=tenant_id, event_id=event_id, candidate_count=candidate_count
            )
        )
        self._session.flush()

    def list_pending(self, tenant_id: UUID, limit: int) -> list[MemoryEvent]:
        """The oldest `limit` events of `tenant_id` that still need ingesting.

        An event is pending when it has no ingestion log row *and* no
        provenance row. The provenance check covers events whose memories
        were written without going through the log -- ingested before the log
        existed, or seeded by `demo_data` -- so they are never re-extracted.

        Example:
            Input:
                3 stored events: A (ingested), B (a demo-seeded memory cites
                it), C (never ingested); limit = 10
            Output:
                [C]
        """
        stmt = (
            select(MemoryEventRow)
            .where(_is_pending_event(tenant_id))
            .order_by(MemoryEventRow.observed_at, MemoryEventRow.event_id)
            .limit(limit)
        )
        return [_event_row_to_domain(row) for row in self._session.scalars(stmt)]

    def count_pending(self, tenant_id: UUID) -> int:
        """How many events `list_pending` would return with no limit."""
        stmt = select(func.count()).select_from(MemoryEventRow).where(_is_pending_event(tenant_id))
        return int(self._session.scalar(stmt) or 0)


def _is_pending_event(tenant_id: UUID) -> ColumnElement[bool]:
    ingested = exists().where(
        MemoryEventIngestionRow.tenant_id == MemoryEventRow.tenant_id,
        MemoryEventIngestionRow.event_id == MemoryEventRow.event_id,
    )
    cited = exists().where(
        MemoryProvenanceRow.tenant_id == MemoryEventRow.tenant_id,
        MemoryProvenanceRow.event_id == MemoryEventRow.event_id,
    )
    return (MemoryEventRow.tenant_id == tenant_id) & ~ingested & ~cited


class MemoryProvenanceRepository:
    """Reads and writes rows in `memory_provenance` -- memory-to-event links."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_for_memory(self, tenant_id: UUID, memory_id: UUID, provenance: Provenance) -> None:
        """Attach one provenance entry to an already-persisted memory.

        How it works:
            Simply stages a new `MemoryProvenanceRow` linking `memory_id` to
            `provenance.event_id`, copying across the attested source type,
            reference, observed-at time, trust level, and excerpt. It does
            not flush or validate anything itself -- callers (see
            `MemoryRecordRepository.add`) are responsible for making sure the
            referenced memory and event already exist in the database by the
            time this row is flushed, since the composite foreign keys on
            `memory_provenance` will otherwise reject the insert.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
                provenance = Provenance(
                    event_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    source_type=SourceType.USER_MESSAGE,
                    source_reference="chat-42",
                    observed_at=datetime(2026, 1, 5, tzinfo=UTC),
                    trust_level=TrustLevel.MEDIUM,
                )
            Output:
                None (a new row now exists in `memory_provenance`, linking
                that memory to that event)
        """
        self._session.add(
            MemoryProvenanceRow(
                tenant_id=tenant_id,
                memory_id=memory_id,
                event_id=provenance.event_id,
                source_type=provenance.source_type,
                source_reference=provenance.source_reference,
                observed_at=provenance.observed_at,
                trust_level=provenance.trust_level,
                excerpt=provenance.excerpt,
            )
        )

    def list_for_memory(self, tenant_id: UUID, memory_id: UUID) -> list[Provenance]:
        """Return every provenance entry recorded for one memory, tenant-scoped.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
            Output:
                [Provenance(event_id=UUID("aaaaaaaa-..."), trust_level=TrustLevel.MEDIUM, ...)]
                -- or [] if the memory has no recorded provenance.
        """
        stmt = select(MemoryProvenanceRow).where(
            MemoryProvenanceRow.tenant_id == tenant_id,
            MemoryProvenanceRow.memory_id == memory_id,
        )
        rows = self._session.scalars(stmt).all()
        return [_provenance_row_to_domain(row) for row in rows]


class MemoryRecordRepository:
    """Reads and writes rows in `memory_records` -- authoritative memories."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._provenance = MemoryProvenanceRepository(session)

    def add(self, tenant_id: UUID, record: MemoryRecord) -> None:
        """Insert a new memory record together with all of its provenance.

        How it works:
            1. Verify `record.tenant_id` matches `tenant_id` (see
               `_require_matching_tenant`).
            2. Stage a `MemoryRecordRow` with the record's fields, splitting
               `record.temporal_validity` into its own `valid_from`/`valid_to`
               columns.
            3. Flush immediately, for the same reason as
               `MemoryEventRepository.add`: there is no ORM `relationship()`
               between records and provenance rows, so the record must exist
               in the database before step 4 can safely insert rows that
               reference it.
            4. Insert one `memory_provenance` row per entry in
               `record.provenance`, via `MemoryProvenanceRepository`.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                record = MemoryRecord(
                    tenant_id=tenant_id,
                    content="The user prefers email over phone calls.",
                    confidence=0.9,
                    provenance=[Provenance(event_id=UUID("aaaaaaaa-..."), ...)],
                    temporal_validity=TemporalValidity(valid_from=datetime(2026, 1, 5, tzinfo=UTC)),
                    memory_type=MemoryType.SEMANTIC,
                    trust_level=TrustLevel.MEDIUM,
                    status=MemoryStatus.ACTIVE,
                )
            Output:
                None (a new row now exists in `memory_records`, plus one row
                in `memory_provenance` for each entry in `record.provenance`)
        """
        _require_matching_tenant(record.tenant_id, tenant_id, "record")
        self._session.add(
            MemoryRecordRow(
                memory_id=record.memory_id,
                tenant_id=tenant_id,
                content=record.content,
                confidence=record.confidence,
                valid_from=record.temporal_validity.valid_from,
                valid_to=record.temporal_validity.valid_to,
                memory_type=record.memory_type,
                subject_keys=list(record.subject_keys),
                trust_level=record.trust_level,
                metadata_=dict(record.metadata),
                status=record.status,
                created_at=record.created_at,
                updated_at=record.updated_at,
                policy_version=record.policy_version,
                embedding=record.embedding,
                embedding_model_version=record.embedding_model_version,
                index_status=record.index_status,
            )
        )
        self._session.flush()
        for provenance in record.provenance:
            self._provenance.add_for_memory(tenant_id, record.memory_id, provenance)

    def get(self, tenant_id: UUID, memory_id: UUID) -> MemoryRecord | None:
        """Look up one memory by id, scoped to `tenant_id`, with its provenance.

        How it works:
            1. Select the `memory_records` row matching both `tenant_id` and
               `memory_id`; a wrong-tenant or non-existent id yields no rows.
            2. If nothing matched, return `None` immediately.
            3. Otherwise, separately fetch that memory's provenance entries
               and assemble the full `MemoryRecord` from both.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
            Output:
                MemoryRecord(
                    memory_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                    content="The user prefers email over phone calls.",
                    status=MemoryStatus.ACTIVE,
                    provenance=[Provenance(...)],
                    ...
                )
                -- or None if no such memory exists for that tenant.
        """
        stmt = select(MemoryRecordRow).where(
            MemoryRecordRow.tenant_id == tenant_id, MemoryRecordRow.memory_id == memory_id
        )
        row = self._session.scalars(stmt).one_or_none()
        if row is None:
            return None
        return _record_row_to_domain(row, self._provenance.list_for_memory(tenant_id, memory_id))

    def list_active(
        self,
        tenant_id: UUID,
        memory_type: MemoryType | None = None,
        *,
        as_of: datetime.datetime | None = None,
    ) -> list[MemoryRecord]:
        """List currently-active memories for a tenant.

        How it works:
            1. Resolve `effective_at` to `as_of` if given, otherwise "now".
            2. Select every `memory_records` row for `tenant_id` whose status
               is `ACTIVE` and whose validity window covers `effective_at`
               (`valid_from <= effective_at` and either no `valid_to` or
               `valid_to >= effective_at`).
            3. If `memory_type` was given, narrow the query to just that type.
            4. For each matching row, look up its provenance and assemble a
               `MemoryRecord`.

        Passing `as_of` here still only ever considers rows whose *status* is
        ACTIVE right now; it does not resurrect memories that have since been
        superseded. For that, see `list_effective`.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_type = MemoryType.SEMANTIC
            Output:
                [MemoryRecord(content="The user's timeout is 2s.", status=MemoryStatus.ACTIVE, ...)]
                -- only the currently active semantic memories for that tenant.
        """
        effective_at = as_of or datetime.datetime.now(datetime.UTC)
        stmt = select(MemoryRecordRow).where(
            MemoryRecordRow.tenant_id == tenant_id,
            MemoryRecordRow.status == MemoryStatus.ACTIVE,
            MemoryRecordRow.valid_from <= effective_at,
            or_(
                MemoryRecordRow.valid_to.is_(None),
                MemoryRecordRow.valid_to >= effective_at,
            ),
        )
        if memory_type is not None:
            stmt = stmt.where(MemoryRecordRow.memory_type == memory_type)
        rows = self._session.scalars(stmt).all()
        return [
            _record_row_to_domain(row, self._provenance.list_for_memory(tenant_id, row.memory_id))
            for row in rows
        ]

    def list_effective(
        self,
        tenant_id: UUID,
        *,
        as_of: datetime.datetime,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """Return memories that were usable at a historical point in time.

        How it works:
            1. Select `memory_records` rows for `tenant_id` whose status is
               ACTIVE, SUPERSEDED, or EXPIRED -- unlike `list_active`, a
               memory that has since been superseded is still a candidate
               here, because the question being asked is "what was true back
               then", not "what is true now".
            2. Keep only rows whose validity window actually covered `as_of`
               (`valid_from <= as_of` and either open-ended or
               `valid_to >= as_of`). This is what excludes a *newer* memory
               that only became valid after `as_of`, even though it is
               currently ACTIVE.
            3. Narrow by `memory_type` if one was given.
            4. Assemble each matching row into a `MemoryRecord` with its
               provenance, same as the other list methods.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                as_of = datetime(2026, 1, 10, tzinfo=UTC)  # before a later supersession
            Output:
                [MemoryRecord(content="The user's timeout is 5s.",
                               status=MemoryStatus.SUPERSEDED, ...)]
                -- the fact that was true at that moment, even though it has
                since been superseded by a newer memory.
        """
        stmt = select(MemoryRecordRow).where(
            MemoryRecordRow.tenant_id == tenant_id,
            MemoryRecordRow.status.in_(
                [MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED, MemoryStatus.EXPIRED]
            ),
            MemoryRecordRow.valid_from <= as_of,
            or_(MemoryRecordRow.valid_to.is_(None), MemoryRecordRow.valid_to >= as_of),
        )
        if memory_type is not None:
            stmt = stmt.where(MemoryRecordRow.memory_type == memory_type)
        rows = self._session.scalars(stmt).all()
        return [
            _record_row_to_domain(row, self._provenance.list_for_memory(tenant_id, row.memory_id))
            for row in rows
        ]

    def update_status(self, tenant_id: UUID, memory_id: UUID, status: MemoryStatus) -> None:
        """Set a memory's lifecycle status directly.

        How it works:
            1. Issue an `UPDATE` against `memory_records`, matching both
               `tenant_id` and `memory_id` in the `WHERE` clause, setting
               `status` to the new value.
            2. Inspect how many rows the database actually changed
               (`result.rowcount`). If it is zero, either the memory does not
               exist or it belongs to a different tenant -- either way, raise
               `LookupError` rather than silently doing nothing.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
                status = MemoryStatus.TOMBSTONE
            Output:
                None (the row's `status` column is now `"tombstone"`)
                -- or raises LookupError if no such memory exists for that tenant.
        """
        stmt = (
            update(MemoryRecordRow)
            .where(MemoryRecordRow.tenant_id == tenant_id, MemoryRecordRow.memory_id == memory_id)
            .values(status=status)
        )
        result = cast(CursorResult[None], self._session.execute(stmt))
        if result.rowcount == 0:
            raise LookupError(f"memory_record {memory_id} not found for tenant {tenant_id}")

    def mark_superseded(
        self,
        tenant_id: UUID,
        memory_id: UUID,
        *,
        valid_to: datetime.datetime,
    ) -> None:
        """Retire an active memory in place of a newer replacement.

        How it works:
            1. Issue an `UPDATE` that only matches a row which is for this
               `tenant_id` and `memory_id`, is currently `ACTIVE`, and whose
               `valid_from` is not after the new `valid_to` -- so a memory
               that was never active, or whose validity window would become
               inverted, is simply not touched.
            2. On a match, set `status` to `SUPERSEDED`, close its validity
               window at `valid_to`, and bump `updated_at` to now.
            3. If no row matched (already superseded, wrong tenant, or does
               not exist), raise `LookupError` instead of pretending it
               worked -- this is what lets `UnitOfWork.supersede_memory`
               notice a bad `old_memory_id` before it inserts the
               replacement.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")  # currently ACTIVE
                valid_to = datetime(2026, 2, 1, tzinfo=UTC)
            Output:
                None (the row's `status` is now `"superseded"` and its
                `valid_to` is `2026-02-01`)
                -- or raises LookupError if that memory is not currently
                ACTIVE for that tenant.
        """
        stmt = (
            update(MemoryRecordRow)
            .where(
                MemoryRecordRow.tenant_id == tenant_id,
                MemoryRecordRow.memory_id == memory_id,
                MemoryRecordRow.status == MemoryStatus.ACTIVE,
                MemoryRecordRow.valid_from <= valid_to,
            )
            .values(
                status=MemoryStatus.SUPERSEDED,
                valid_to=valid_to,
                updated_at=datetime.datetime.now(datetime.UTC),
            )
        )
        result = cast(CursorResult[None], self._session.execute(stmt))
        if result.rowcount == 0:
            raise LookupError(
                f"active memory_record {memory_id} not found for tenant {tenant_id}"
            )


class MemoryRelationRepository:
    """Reads and writes rows in `memory_relations` -- edges between memories."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, tenant_id: UUID, relation: MemoryRelation) -> None:
        """Insert a new relation edge between two memories.

        How it works:
            Verifies `relation.tenant_id` matches `tenant_id`, then stages a
            new `MemoryRelationRow`. The composite foreign keys on
            `memory_relations` (see `persistence/models.py`) do the rest of
            the validation at flush time: both `source_memory_id` and
            `target_memory_id` must resolve to a `memory_records` row owned
            by the *same* tenant, or the insert fails.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                relation = MemoryRelation(
                    tenant_id=tenant_id,
                    source_memory_id=UUID("cccccccc-..."),  # the new memory
                    target_memory_id=UUID("bbbbbbbb-..."),  # the old memory
                    relation_type=ConflictType.SUPERSESSION,
                    rationale="operator lowered the timeout",
                )
            Output:
                None (a new row now exists in `memory_relations`)
        """
        _require_matching_tenant(relation.tenant_id, tenant_id, "relation")
        self._session.add(
            MemoryRelationRow(
                relation_id=relation.relation_id,
                tenant_id=tenant_id,
                source_memory_id=relation.source_memory_id,
                target_memory_id=relation.target_memory_id,
                relation_type=relation.relation_type,
                metadata_=dict(relation.metadata),
                rationale=relation.rationale,
                created_at=relation.created_at,
            )
        )

    def list_for_memory(self, tenant_id: UUID, memory_id: UUID) -> list[MemoryRelation]:
        """Return every relation involving one memory, on either side of the edge.

        How it works:
            Selects `memory_relations` rows for `tenant_id` where `memory_id`
            appears as *either* `source_memory_id` or `target_memory_id`
            (an `OR` across both columns), so callers do not need to know or
            care which direction a given relation was originally recorded in.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
            Output:
                [MemoryRelation(relation_type=ConflictType.SUPERSESSION, ...)]
                -- or [] if this memory has no recorded relations.
        """
        stmt = select(MemoryRelationRow).where(
            MemoryRelationRow.tenant_id == tenant_id,
            or_(
                MemoryRelationRow.source_memory_id == memory_id,
                MemoryRelationRow.target_memory_id == memory_id,
            ),
        )
        rows = self._session.scalars(stmt).all()
        return [_relation_row_to_domain(row) for row in rows]

    def list_for_memories(
        self, tenant_id: UUID, memory_ids: Collection[UUID]
    ) -> list[MemoryRelation]:
        """Return every relation touching any of `memory_ids`, in one query.

        How it works:
            Same as `list_for_memory`, but with `IN (...)` on both edge
            columns, so a caller checking a whole candidate set (e.g.
            `retrieval.temporal`) issues one query rather than one per
            memory. An edge between two of the given memories appears once.
            An empty `memory_ids` returns `[]` without querying.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                memory_ids = [UUID("bbbbbbbb-..."), UUID("cccccccc-...")]
            Output:
                [MemoryRelation(source_memory_id=UUID("cccccccc-..."),
                                target_memory_id=UUID("bbbbbbbb-..."),
                                relation_type=ConflictType.SUPERSESSION, ...)]
        """
        if not memory_ids:
            return []
        ids = list(memory_ids)
        stmt = select(MemoryRelationRow).where(
            MemoryRelationRow.tenant_id == tenant_id,
            or_(
                MemoryRelationRow.source_memory_id.in_(ids),
                MemoryRelationRow.target_memory_id.in_(ids),
            ),
        )
        rows = self._session.scalars(stmt).all()
        return [_relation_row_to_domain(row) for row in rows]


class MemoryWriteDecisionRepository:
    """Reads and writes rows in `memory_write_decisions` -- the audit trail."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, tenant_id: UUID, decision: WritePolicyDecision) -> None:
        """Insert one write-policy audit record.

        How it works:
            Verifies `decision.tenant_id` matches `tenant_id`, then stages a
            `MemoryWriteDecisionRow` copying across the decision, its reason
            codes and explanation, and (when present) the accepted/superseded
            memory ids. This is called for *every* decision outcome,
            including REJECT, so the audit trail is complete regardless of
            whether anything was actually written to `memory_records`.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                decision = WritePolicyDecision(
                    tenant_id=tenant_id,
                    candidate_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
                    decision=WriteDecision.REJECT,
                    policy_version="1.0",
                    reason_codes=["SAFETY_POLICY_TAMPERING"],
                )
            Output:
                None (a new row now exists in `memory_write_decisions`, with
                `accepted_memory_id` left as `NULL`)
        """
        _require_matching_tenant(decision.tenant_id, tenant_id, "decision")
        self._session.add(
            MemoryWriteDecisionRow(
                decision_id=decision.decision_id,
                tenant_id=tenant_id,
                candidate_id=decision.candidate_id,
                decision=decision.decision,
                decided_at=decision.decided_at,
                policy_version=decision.policy_version,
                reason_codes=list(decision.reason_codes),
                explanation=decision.explanation,
                accepted_memory_id=decision.accepted_memory_id,
                superseded_memory_id=decision.superseded_memory_id,
            )
        )

    def get(self, tenant_id: UUID, decision_id: UUID) -> WritePolicyDecision | None:
        """Look up one audit decision by id, scoped to `tenant_id`.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                decision_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
            Output:
                WritePolicyDecision(decision=WriteDecision.REJECT,
                                     reason_codes=["SAFETY_POLICY_TAMPERING"], ...)
                -- or None if no such decision exists for that tenant.
        """
        stmt = select(MemoryWriteDecisionRow).where(
            MemoryWriteDecisionRow.tenant_id == tenant_id,
            MemoryWriteDecisionRow.decision_id == decision_id,
        )
        row = self._session.scalars(stmt).one_or_none()
        return _decision_row_to_domain(row) if row is not None else None

    def list_for_candidate(self, tenant_id: UUID, candidate_id: UUID) -> list[WritePolicyDecision]:
        """Return every audit decision ever recorded for one candidate id.

        Normally a single candidate produces exactly one decision, but this
        returns a list (rather than assuming uniqueness) so re-evaluation or
        retry scenarios are representable without changing the schema.

        Example:
            Input:
                tenant_id = UUID("11111111-1111-1111-1111-111111111111")
                candidate_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
            Output:
                [WritePolicyDecision(decision=WriteDecision.REJECT, ...)]
                -- or [] if that candidate was never evaluated for this tenant.
        """
        stmt = select(MemoryWriteDecisionRow).where(
            MemoryWriteDecisionRow.tenant_id == tenant_id,
            MemoryWriteDecisionRow.candidate_id == candidate_id,
        )
        rows = self._session.scalars(stmt).all()
        return [_decision_row_to_domain(row) for row in rows]

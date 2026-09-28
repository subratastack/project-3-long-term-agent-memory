"""Unit tests for `retrieval.temporal.select_in_effect`.

`select_in_effect` makes every temporal/conflict decision from records and
relations it is handed, so these run with no database. Loading those
relations from PostgreSQL, and the full pipeline with real supersession,
are covered in apps/tests/integration/retrieval/test_temporal_resolution.py.
"""

import unittest
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import ConflictType, MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryRecord, MemoryRelation, TemporalValidity
from apps.memory_service.retrieval.filters import RetrievalFilters, resolve_filters
from apps.memory_service.retrieval.hybrid import HybridSearchHit
from apps.memory_service.retrieval.query_model import RetrievalQuery
from apps.memory_service.retrieval.temporal import (
    ExclusionReason,
    TemporalReport,
    select_in_effect,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
JAN_1 = datetime(2026, 1, 1, tzinfo=UTC)
FEB_15 = datetime(2026, 2, 15, tzinfo=UTC)
MAR_1 = datetime(2026, 3, 1, tzinfo=UTC)
APR_1 = datetime(2026, 4, 1, tzinfo=UTC)


def _make_record(
    content: str,
    valid_from: datetime = JAN_1,
    valid_to: datetime | None = None,
    **overrides: Any,
) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "memory_id": uuid4(),
        "tenant_id": TENANT_ID,
        "content": content,
        "confidence": 0.9,
        "provenance": [],
        "temporal_validity": TemporalValidity(valid_from=valid_from, valid_to=valid_to),
        "memory_type": MemoryType.SEMANTIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(overrides)
    # provenance is never inspected here, so model_construct skips its validation.
    return MemoryRecord.model_construct(**defaults)


def _hit(record: MemoryRecord, fused_score: float = 0.01) -> HybridSearchHit:
    return HybridSearchHit(memory=record, fused_score=fused_score, lexical_rank=1, semantic_rank=1)


def _relation(
    source: MemoryRecord, target: MemoryRecord, relation_type: ConflictType
) -> MemoryRelation:
    return MemoryRelation(
        tenant_id=TENANT_ID,
        source_memory_id=source.memory_id,
        target_memory_id=target.memory_id,
        relation_type=relation_type,
    )


def _filters(as_of: datetime | None = None, **query_fields: Any) -> RetrievalFilters:
    return resolve_filters(
        RetrievalQuery(tenant_id=TENANT_ID, query_text="q", as_of=as_of, **query_fields)
    )


def _run(
    hits: Sequence[HybridSearchHit],
    filters: RetrievalFilters,
    relations: Sequence[MemoryRelation] = (),
    extra_records: Sequence[MemoryRecord] = (),
) -> tuple[list[UUID], TemporalReport]:
    records = {hit.memory.memory_id: hit.memory for hit in hits}
    records.update({record.memory_id: record for record in extra_records})
    kept, report = select_in_effect(hits, filters=filters, relations=relations, records=records)
    return [hit.memory.memory_id for hit in kept], report


def _reasons(report: TemporalReport) -> dict[UUID, ExclusionReason]:
    return {exclusion.memory_id: exclusion.reason for exclusion in report.excluded}


class TestValidityWindows(unittest.TestCase):
    def setUp(self) -> None:
        # timeout=5s from Jan 1, superseded on Mar 1 by timeout=2s.
        self.old = _make_record(
            "timeout is 5 seconds", JAN_1, MAR_1, status=MemoryStatus.SUPERSEDED
        )
        self.new = _make_record("timeout is 2 seconds", MAR_1)
        self.supersession = _relation(self.new, self.old, ConflictType.SUPERSESSION)

    def _at(self, as_of: datetime) -> list[UUID]:
        # The stale fact is listed first -- i.e. it had the better relevance score.
        kept, _ = _run(
            [_hit(self.old, 0.9), _hit(self.new, 0.1)], _filters(as_of), [self.supersession]
        )
        return kept

    def test_february_returns_the_old_fact(self) -> None:
        self.assertEqual(self._at(FEB_15), [self.old.memory_id])

    def test_april_returns_the_new_fact(self) -> None:
        self.assertEqual(self._at(APR_1), [self.new.memory_id])

    def test_the_boundary_instant_belongs_to_the_new_fact_only(self) -> None:
        """Windows are half-open: at exactly Mar 1 the old one has ended."""
        self.assertEqual(self._at(MAR_1), [self.new.memory_id])

    def test_a_current_query_never_sees_the_superseded_status(self) -> None:
        kept, report = _run([_hit(self.old, 0.9), _hit(self.new, 0.1)], _filters())

        self.assertEqual(kept, [self.new.memory_id])
        self.assertEqual(_reasons(report), {self.old.memory_id: ExclusionReason.NOT_ELIGIBLE})

    def test_an_expired_window_is_not_in_effect(self) -> None:
        expired = _make_record("promo code SPRING26 is valid", JAN_1, MAR_1)

        kept, report = _run([_hit(expired)], _filters(APR_1))

        self.assertEqual(kept, [])
        self.assertEqual(_reasons(report), {expired.memory_id: ExclusionReason.NOT_IN_EFFECT})

    def test_a_not_yet_valid_memory_is_not_in_effect(self) -> None:
        future = _make_record("office moves to Elm Avenue", APR_1)

        kept, _ = _run([_hit(future)], _filters(FEB_15))

        self.assertEqual(kept, [])


class TestEligibility(unittest.TestCase):
    def test_tombstoned_and_quarantined_are_excluded_even_for_historical_queries(self) -> None:
        tombstoned = _make_record("forgotten", status=MemoryStatus.TOMBSTONE)
        quarantined = _make_record("suspicious", status=MemoryStatus.QUARANTINED)

        for filters in (_filters(), _filters(FEB_15)):
            kept, report = _run([_hit(tombstoned), _hit(quarantined)], filters)

            self.assertEqual(kept, [])
            self.assertEqual(set(_reasons(report).values()), {ExclusionReason.NOT_ELIGIBLE})

    def test_another_tenants_memory_is_excluded(self) -> None:
        foreign = _make_record("someone else's timeout", tenant_id=uuid4())

        kept, report = _run([_hit(foreign)], _filters())

        self.assertEqual(kept, [])
        self.assertEqual(_reasons(report), {foreign.memory_id: ExclusionReason.NOT_ELIGIBLE})

    def test_below_the_trust_floor_is_excluded(self) -> None:
        weak = _make_record("rumour", trust_level=TrustLevel.LOW)

        kept, _ = _run([_hit(weak)], _filters(min_trust=TrustLevel.MEDIUM))

        self.assertEqual(kept, [])


class TestSupersession(unittest.TestCase):
    def test_a_recorded_supersession_retires_a_still_active_record(self) -> None:
        """The edge alone is enough, even if the old row's status was never updated."""
        old = _make_record("address is 12 Oak Street")
        new = _make_record("address is 48 Elm Avenue", MAR_1)

        kept, report = _run(
            [_hit(old, 0.9), _hit(new, 0.1)],
            _filters(),
            [_relation(new, old, ConflictType.SUPERSESSION)],
        )

        self.assertEqual(kept, [new.memory_id])
        [exclusion] = report.excluded
        self.assertEqual(exclusion.reason, ExclusionReason.SUPERSEDED)
        self.assertEqual(exclusion.related_memory_id, new.memory_id)

    def test_a_successor_that_is_not_yet_valid_does_not_retire_anything(self) -> None:
        old = _make_record("address is 12 Oak Street")
        future = _make_record("address is 48 Elm Avenue", APR_1)

        kept, _ = _run(
            [_hit(old)],
            _filters(FEB_15),
            [_relation(future, old, ConflictType.SUPERSESSION)],
            extra_records=[future],
        )

        self.assertEqual(kept, [old.memory_id])


class TestContradictions(unittest.TestCase):
    def test_equal_trust_withholds_both_and_surfaces_the_conflict(self) -> None:
        python = _make_record("preferred language is Python")
        rust = _make_record("preferred language is Rust")
        unrelated = _make_record("likes tea")

        kept, report = _run(
            [_hit(python), _hit(rust), _hit(unrelated)],
            _filters(),
            [_relation(python, rust, ConflictType.CONTRADICTION)],
        )

        self.assertEqual(kept, [unrelated.memory_id])
        self.assertEqual(
            _reasons(report),
            {
                python.memory_id: ExclusionReason.UNRESOLVED_CONFLICT,
                rust.memory_id: ExclusionReason.UNRESOLVED_CONFLICT,
            },
        )
        [conflict] = report.unresolved_conflicts
        self.assertEqual(set(conflict.memory_ids), {python.memory_id, rust.memory_id})

    def test_a_hit_is_withheld_even_when_its_contradiction_was_not_retrieved(self) -> None:
        python = _make_record("preferred language is Python")
        rust = _make_record("preferred language is Rust")

        kept, report = _run(
            [_hit(python)],
            _filters(),
            [_relation(python, rust, ConflictType.CONTRADICTION)],
            extra_records=[rust],
        )

        self.assertEqual(kept, [])
        self.assertEqual(len(report.unresolved_conflicts), 1)

    def test_higher_trust_wins_even_when_the_loser_scored_higher(self) -> None:
        rumour = _make_record("timeout is 5 seconds", trust_level=TrustLevel.LOW)
        config = _make_record("timeout is 2 seconds", trust_level=TrustLevel.SYSTEM)

        kept, report = _run(
            [_hit(rumour, 0.9), _hit(config, 0.1)],
            _filters(),
            [_relation(rumour, config, ConflictType.CONTRADICTION)],
        )

        self.assertEqual(kept, [config.memory_id])
        self.assertEqual(_reasons(report), {rumour.memory_id: ExclusionReason.LOST_CONFLICT})
        self.assertEqual(report.unresolved_conflicts, ())

    def test_a_contradiction_with_a_retired_fact_is_no_longer_a_conflict(self) -> None:
        python = _make_record("preferred language is Python")
        rust = _make_record("preferred language is Rust")
        go = _make_record("preferred language is Go", MAR_1)

        kept, report = _run(
            [_hit(python)],
            _filters(APR_1),
            [
                _relation(python, rust, ConflictType.CONTRADICTION),
                _relation(go, rust, ConflictType.SUPERSESSION),
            ],
            extra_records=[rust, go],
        )

        self.assertEqual(kept, [python.memory_id])
        self.assertEqual(report.conflicts, ())

    def test_a_contradiction_with_a_fact_outside_its_window_is_not_a_conflict(self) -> None:
        python = _make_record("preferred language is Python", MAR_1)
        rust = _make_record("preferred language is Rust", JAN_1, MAR_1)

        kept, _ = _run(
            [_hit(python)],
            _filters(APR_1),
            [_relation(python, rust, ConflictType.CONTRADICTION)],
            extra_records=[rust],
        )

        self.assertEqual(kept, [python.memory_id])


class TestOrdering(unittest.TestCase):
    def test_surviving_hits_keep_their_relevance_order(self) -> None:
        records = [_make_record(f"fact {i}") for i in range(5)]
        records[2] = _make_record("expired fact", JAN_1, FEB_15)

        kept, _ = _run([_hit(record) for record in records], _filters(APR_1))

        self.assertEqual(kept, [records[i].memory_id for i in (0, 1, 3, 4)])

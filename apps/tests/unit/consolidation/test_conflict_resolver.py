"""Unit tests for `consolidation.conflict_resolver.resolve_contradiction`.

Pure decision logic: no database, no model. Recording a contradiction and
honouring it during retrieval is covered in
apps/tests/integration/retrieval/test_temporal_resolution.py.
"""

import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from apps.memory_service.consolidation.conflict_resolver import (
    ConflictState,
    resolve_contradiction,
)
from apps.memory_service.domain.enums import ConflictType, MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryRecord, MemoryRelation, TemporalValidity

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _make_record(content: str, **overrides: Any) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "memory_id": uuid4(),
        "tenant_id": TENANT_ID,
        "content": content,
        "confidence": 0.9,
        "provenance": [],
        "temporal_validity": TemporalValidity(valid_from=datetime(2026, 1, 1, tzinfo=UTC)),
        "memory_type": MemoryType.SEMANTIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(overrides)
    # provenance is never inspected here, so model_construct skips its validation.
    return MemoryRecord.model_construct(**defaults)


def _contradiction(
    first: MemoryRecord,
    second: MemoryRecord,
    relation_type: ConflictType = ConflictType.CONTRADICTION,
) -> MemoryRelation:
    return MemoryRelation(
        tenant_id=TENANT_ID,
        source_memory_id=first.memory_id,
        target_memory_id=second.memory_id,
        relation_type=relation_type,
    )


class TestResolveContradiction(unittest.TestCase):
    def test_equal_trust_is_unresolved_and_withholds_both(self) -> None:
        berlin = _make_record("The user lives in Berlin.")
        paris = _make_record("The user lives in Paris.")

        outcome = resolve_contradiction(_contradiction(berlin, paris), berlin, paris)

        self.assertIs(outcome.state, ConflictState.UNRESOLVED)
        self.assertIsNone(outcome.winner_id)
        self.assertEqual(outcome.withheld_ids, {berlin.memory_id, paris.memory_id})
        self.assertEqual(outcome.reason, "equal trust (medium); needs review")

    def test_strictly_higher_trust_wins_whichever_side_it_is_on(self) -> None:
        trusted = _make_record("timeout=2s", trust_level=TrustLevel.SYSTEM)
        rumour = _make_record("timeout=5s", trust_level=TrustLevel.LOW)
        relation = _contradiction(trusted, rumour)

        for first, second in [(trusted, rumour), (rumour, trusted)]:
            outcome = resolve_contradiction(relation, first, second)

            self.assertIs(outcome.state, ConflictState.RESOLVED_BY_TRUST)
            self.assertEqual(outcome.winner_id, trusted.memory_id)
            self.assertEqual(outcome.loser_id, rumour.memory_id)
            self.assertEqual(outcome.withheld_ids, {rumour.memory_id})

    def test_recency_does_not_break_a_trust_tie(self) -> None:
        """ADR-004 rejects last-write-wins: a later `valid_from` is not authority."""
        older = _make_record(
            "timeout=5s",
            temporal_validity=TemporalValidity(valid_from=datetime(2026, 1, 1, tzinfo=UTC)),
        )
        newer = _make_record(
            "timeout=2s",
            temporal_validity=TemporalValidity(valid_from=datetime(2026, 6, 1, tzinfo=UTC)),
        )

        outcome = resolve_contradiction(_contradiction(older, newer), older, newer)

        self.assertIs(outcome.state, ConflictState.UNRESOLVED)

    def test_a_non_contradiction_relation_is_rejected(self) -> None:
        old = _make_record("timeout=5s")
        new = _make_record("timeout=2s")

        with self.assertRaises(ValueError):
            resolve_contradiction(_contradiction(new, old, ConflictType.SUPERSESSION), new, old)

    def test_records_that_are_not_the_relations_endpoints_are_rejected(self) -> None:
        a, b, c = _make_record("a"), _make_record("b"), _make_record("c")

        with self.assertRaises(ValueError):
            resolve_contradiction(_contradiction(a, b), a, c)

    def test_a_record_from_another_tenant_is_rejected(self) -> None:
        mine = _make_record("mine")
        theirs = _make_record("theirs", tenant_id=uuid4())

        with self.assertRaises(ValueError):
            resolve_contradiction(_contradiction(mine, theirs), mine, theirs)

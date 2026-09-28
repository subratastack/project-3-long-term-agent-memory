import unittest
import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from apps.memory_service.domain.enums import MemoryType, SourceType, TrustLevel
from apps.memory_service.domain.models import MemoryCandidate, MemoryEvent, Provenance
from apps.memory_service.ingestion.provenance import (
    EVENT_NOT_FOUND,
    INCONSISTENT_PROVENANCE,
    MISSING_PROVENANCE,
    TENANT_MISMATCH,
    trust_rank,
    verify_provenance,
    weakest_trust,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _make_event(tenant_id: UUID = TENANT_ID, **overrides: Any) -> MemoryEvent:
    defaults: dict[str, Any] = {
        "tenant_id": tenant_id,
        "source_type": SourceType.SYSTEM_EVENT,
        "source_reference": "session-1",
        "content": "The configured timeout is 5 seconds.",
        "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return MemoryEvent(**defaults)


def _make_provenance(event: MemoryEvent, **overrides: Any) -> Provenance:
    defaults: dict[str, Any] = {
        "event_id": event.event_id,
        "source_type": event.source_type,
        "source_reference": event.source_reference,
        "observed_at": event.observed_at,
        "trust_level": TrustLevel.SYSTEM,
    }
    defaults.update(overrides)
    return Provenance(**defaults)


def _make_candidate(provenance: list[Provenance], **overrides: Any) -> MemoryCandidate:
    defaults: dict[str, Any] = {
        "tenant_id": TENANT_ID,
        "content": "The timeout is 5 seconds.",
        "provenance": provenance,
        "memory_type": MemoryType.SEMANTIC,
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return MemoryCandidate(**defaults)


class TestTrustRanking(unittest.TestCase):

    def test_trust_rank_orders_system_above_untrusted(self) -> None:
        self.assertGreater(trust_rank(TrustLevel.SYSTEM), trust_rank(TrustLevel.UNTRUSTED))

    def test_weakest_trust_returns_the_least_trusted_level(self) -> None:
        result = weakest_trust([TrustLevel.SYSTEM, TrustLevel.LOW, TrustLevel.HIGH])
        self.assertEqual(result, TrustLevel.LOW)

    def test_weakest_trust_of_empty_iterable_is_untrusted(self) -> None:
        self.assertEqual(weakest_trust([]), TrustLevel.UNTRUSTED)


class TestVerifyProvenance(unittest.TestCase):

    def test_happy_path_derives_trust_as_weakest_link(self) -> None:
        strong_event = _make_event(source_type=SourceType.SYSTEM_EVENT)
        weak_event = _make_event(
            source_type=SourceType.USER_MESSAGE, source_reference="conversation-1"
        )
        strong_provenance = _make_provenance(strong_event, trust_level=TrustLevel.SYSTEM)
        weak_provenance = _make_provenance(weak_event, trust_level=TrustLevel.MEDIUM)
        candidate = _make_candidate([strong_provenance, weak_provenance])

        result = verify_provenance(
            candidate, {strong_event.event_id: strong_event, weak_event.event_id: weak_event}
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.derived_trust_level, TrustLevel.MEDIUM)
        self.assertEqual(result.reason_codes, ())

    def test_derived_trust_cannot_exceed_the_sources_baseline(self) -> None:
        """A tool-output event attested as SYSTEM trust is still capped at the
        tool-output baseline (MEDIUM): self-reported trust cannot inflate it."""
        event = _make_event(source_type=SourceType.TOOL_OUTPUT, source_reference="tool-1")
        provenance = _make_provenance(event, trust_level=TrustLevel.SYSTEM)
        candidate = _make_candidate([provenance])

        result = verify_provenance(candidate, {event.event_id: event})

        self.assertTrue(result.is_valid)
        self.assertEqual(result.derived_trust_level, TrustLevel.MEDIUM)

    def test_rejects_when_referenced_event_is_not_found(self) -> None:
        event = _make_event()
        provenance = _make_provenance(event)
        candidate = _make_candidate([provenance])

        result = verify_provenance(candidate, {})

        self.assertFalse(result.is_valid)
        self.assertIn(EVENT_NOT_FOUND, result.reason_codes)

    def test_rejects_when_event_belongs_to_another_tenant(self) -> None:
        foreign_event = _make_event(tenant_id=OTHER_TENANT_ID)
        provenance = _make_provenance(foreign_event)
        candidate = _make_candidate([provenance])

        result = verify_provenance(candidate, {foreign_event.event_id: foreign_event})

        self.assertFalse(result.is_valid)
        self.assertIn(TENANT_MISMATCH, result.reason_codes)

    def test_rejects_when_provenance_is_inconsistent_with_the_real_event(self) -> None:
        event = _make_event()
        # Claims a different source_reference than the actual event carries.
        provenance = _make_provenance(event, source_reference="a-different-session")
        candidate = _make_candidate([provenance])

        result = verify_provenance(candidate, {event.event_id: event})

        self.assertFalse(result.is_valid)
        self.assertIn(INCONSISTENT_PROVENANCE, result.reason_codes)

    def test_deduplicates_reason_codes_across_multiple_bad_entries(self) -> None:
        candidate = _make_candidate(
            [
                _make_provenance(_make_event()),
                _make_provenance(_make_event(observed_at=datetime(2099, 1, 1, tzinfo=UTC))),
            ]
        )

        result = verify_provenance(candidate, {})

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason_codes, (EVENT_NOT_FOUND,))


class TestMissingProvenanceGuard(unittest.TestCase):
    """MemoryCandidate.provenance is Pydantic-enforced to be non-empty, so this
    exercises the defensive branch directly rather than via the domain model."""

    def test_verify_provenance_rejects_a_candidate_with_no_provenance_entries(self) -> None:
        candidate = MemoryCandidate.model_construct(
            candidate_id=uuid.uuid4(),
            tenant_id=TENANT_ID,
            content="orphaned claim",
            provenance=[],
            memory_type=MemoryType.SEMANTIC,
            confidence=0.5,
        )

        result = verify_provenance(candidate, {})

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason_codes, (MISSING_PROVENANCE,))


if __name__ == "__main__":
    unittest.main()

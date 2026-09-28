import unittest
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from apps.memory_service.domain.enums import (
    ConflictType,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)
from apps.memory_service.domain.models import (
    MemoryCandidate,
    MemoryEvent,
    MemoryRecord,
    MemoryRelation,
    Model,
    Provenance,
    TemporalValidity,
    Tenant,
    WritePolicyDecision,
)

TENANT_ID = UUID("00000000-0000-0000-0000-000000000000")


def make_provenance(**overrides: Any) -> Provenance:
    defaults = dict(
        event_id=uuid.uuid4(),
        source_type=SourceType.USER_MESSAGE,
        source_reference="msg-1",
        observed_at=datetime.now(UTC),
        trust_level=TrustLevel.MEDIUM,
    )
    defaults.update(overrides)
    return Provenance(**defaults)


def make_temporal_validity(**overrides: Any) -> TemporalValidity:
    defaults = dict(valid_from=datetime.now(UTC), valid_to=None)
    defaults.update(overrides)
    return TemporalValidity(**defaults)


class TestModelBase(unittest.TestCase):

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Model(unexpected_field="value")  # type: ignore[call-arg]

    def test_string_fields_are_stripped(self) -> None:
        event = MemoryEvent(
            tenant_id=TENANT_ID,
            source_type=SourceType.USER_MESSAGE,
            source_reference="  ref-1  ",
            content="  hello world  ",
        )
        self.assertEqual(event.source_reference, "ref-1")
        self.assertEqual(event.content, "hello world")


class TestMemoryEvent(unittest.TestCase):

    def test_happy_path_creates_event_with_defaults(self) -> None:
        event = MemoryEvent(
            tenant_id=TENANT_ID,
            source_type=SourceType.TOOL_OUTPUT,
            source_reference="tool-call-1",
            content="the tool returned 42",
        )
        self.assertIsInstance(event.event_id, UUID)
        self.assertEqual(event.tenant_id, TENANT_ID)
        self.assertEqual(event.source_type, SourceType.TOOL_OUTPUT)
        self.assertIsNone(event.actor_id)
        self.assertEqual(event.metadata, {})
        self.assertIsInstance(event.observed_at, datetime)

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            MemoryEvent(
                tenant_id=TENANT_ID,
                source_type=SourceType.USER_MESSAGE,
                source_reference="ref-1",
                content="",
            )

    def test_rejects_empty_source_reference(self) -> None:
        with pytest.raises(ValidationError, match="source_reference"):
            MemoryEvent(
                tenant_id=TENANT_ID,
                source_type=SourceType.USER_MESSAGE,
                source_reference="",
                content="hello",
            )

    def test_rejects_invalid_source_type(self) -> None:
        with pytest.raises(ValidationError, match="source_type"):
            MemoryEvent(
                tenant_id=TENANT_ID,
                source_type="not_a_source_type",
                source_reference="ref-1",
                content="hello",
            )

    def test_rejects_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            MemoryEvent(tenant_id=TENANT_ID)  # type: ignore[call-arg]

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            MemoryEvent(
                tenant_id=TENANT_ID,
                source_type=SourceType.USER_MESSAGE,
                source_reference="ref-1",
                content="hello",
                unknown_field="oops",  # type: ignore[call-arg]
            )


class TestProvenance(unittest.TestCase):

    def test_happy_path_creates_provenance(self) -> None:
        event_id = uuid.uuid4()
        observed_at = datetime.now(UTC)
        provenance = Provenance(
            event_id=event_id,
            source_type=SourceType.AGENT_ACTION,
            source_reference="session-1",
            observed_at=observed_at,
            trust_level=TrustLevel.HIGH,
            excerpt="the agent decided to...",
        )
        self.assertEqual(provenance.event_id, event_id)
        self.assertEqual(provenance.trust_level, TrustLevel.HIGH)
        self.assertEqual(provenance.excerpt, "the agent decided to...")

    def test_excerpt_defaults_to_none(self) -> None:
        provenance = make_provenance()
        self.assertIsNone(provenance.excerpt)

    def test_rejects_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Provenance(source_type=SourceType.USER_MESSAGE)  # type: ignore[call-arg]

    def test_rejects_invalid_trust_level(self) -> None:
        with pytest.raises(ValidationError, match="trust_level"):
            make_provenance(trust_level="super_trusted")


class TestTemporalValidity(unittest.TestCase):

    def test_happy_path_with_open_ended_validity(self) -> None:
        valid_from = datetime.now(UTC)
        validity = TemporalValidity(valid_from=valid_from)
        self.assertEqual(validity.valid_from, valid_from)
        self.assertIsNone(validity.valid_to)

    def test_happy_path_with_valid_to_after_valid_from(self) -> None:
        valid_from = datetime.now(UTC)
        valid_to = valid_from + timedelta(days=1)
        validity = TemporalValidity(valid_from=valid_from, valid_to=valid_to)
        self.assertEqual(validity.valid_to, valid_to)

    def test_happy_path_allows_equal_valid_from_and_valid_to(self) -> None:
        moment = datetime.now(UTC)
        validity = TemporalValidity(valid_from=moment, valid_to=moment)
        self.assertEqual(validity.valid_from, validity.valid_to)

    def test_rejects_end_before_start(self) -> None:
        valid_from = datetime.now(UTC)
        valid_to = valid_from - timedelta(days=1)
        with pytest.raises(ValidationError, match="valid_to"):
            TemporalValidity(valid_from=valid_from, valid_to=valid_to)

    def test_rejects_missing_valid_from(self) -> None:
        with pytest.raises(ValidationError, match="valid_from"):
            TemporalValidity(valid_to=datetime.now(UTC))  # type: ignore[call-arg]


class TestMemoryCandidate(unittest.TestCase):

    def test_happy_path_creates_candidate(self) -> None:
        candidate = MemoryCandidate(
            tenant_id=TENANT_ID,
            content="The user prefers dark mode.",
            provenance=[make_provenance()],
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
        )
        self.assertIsInstance(candidate.candidate_id, UUID)
        self.assertEqual(candidate.confidence, 0.85)
        self.assertEqual(candidate.proposed_trust_level, TrustLevel.UNTRUSTED)
        self.assertEqual(candidate.subject_keys, [])
        self.assertEqual(candidate.metadata, {})
        self.assertIsNone(candidate.temporal_validity)

    def test_happy_path_with_all_optional_fields(self) -> None:
        validity = make_temporal_validity()
        candidate = MemoryCandidate(
            tenant_id=TENANT_ID,
            content="The user's favorite color is blue.",
            provenance=[make_provenance(), make_provenance()],
            temporal_validity=validity,
            memory_type=MemoryType.EPISODIC,
            subject_keys=["user:123", "preference:color"],
            confidence=1.0,
            proposed_trust_level=TrustLevel.HIGH,
            metadata={"extractor": "llm-v1"},
        )
        self.assertEqual(len(candidate.provenance), 2)
        self.assertEqual(candidate.temporal_validity, validity)
        self.assertEqual(candidate.subject_keys, ["user:123", "preference:color"])
        self.assertEqual(candidate.metadata, {"extractor": "llm-v1"})

    def test_rejects_empty_provenance(self) -> None:
        with pytest.raises(ValidationError, match="provenance"):
            MemoryCandidate(
                tenant_id=TENANT_ID,
                memory_type=MemoryType.SEMANTIC,
                content="This is a test memory candidate.",
                confidence=0.9,
                provenance=[],
            )

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            MemoryCandidate(
                tenant_id=TENANT_ID,
                memory_type=MemoryType.SEMANTIC,
                content="",
                confidence=0.9,
                provenance=[make_provenance()],
            )

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            MemoryCandidate(
                tenant_id=TENANT_ID,
                memory_type=MemoryType.SEMANTIC,
                content="valid content",
                confidence=1.1,
                provenance=[make_provenance()],
            )

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            MemoryCandidate(
                tenant_id=TENANT_ID,
                memory_type=MemoryType.SEMANTIC,
                content="valid content",
                confidence=-0.1,
                provenance=[make_provenance()],
            )

    def test_rejects_invalid_memory_type(self) -> None:
        with pytest.raises(ValidationError, match="memory_type"):
            MemoryCandidate(
                tenant_id=TENANT_ID,
                memory_type="not_a_type",
                content="valid content",
                confidence=0.5,
                provenance=[make_provenance()],
            )

    def test_rejects_missing_memory_type(self) -> None:
        with pytest.raises(ValidationError, match="memory_type"):
            MemoryCandidate(
                tenant_id=TENANT_ID,
                content="valid content",
                confidence=0.5,
                provenance=[make_provenance()],
            )  # type: ignore[call-arg]


class TestMemoryRelation(unittest.TestCase):

    def test_happy_path_creates_relation(self) -> None:
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        relation = MemoryRelation(
            tenant_id=TENANT_ID,
            source_memory_id=source_id,
            target_memory_id=target_id,
            relation_type=ConflictType.SUPERSESSION,
            rationale="newer information supersedes the old preference",
        )
        self.assertEqual(relation.source_memory_id, source_id)
        self.assertEqual(relation.target_memory_id, target_id)
        self.assertEqual(relation.relation_type, ConflictType.SUPERSESSION)
        self.assertEqual(relation.metadata, {})
        self.assertIsInstance(relation.created_at, datetime)

    def test_rationale_defaults_to_none(self) -> None:
        relation = MemoryRelation(
            tenant_id=TENANT_ID,
            source_memory_id=uuid.uuid4(),
            target_memory_id=uuid.uuid4(),
            relation_type=ConflictType.DUPLICATE,
        )
        self.assertIsNone(relation.rationale)

    def test_rejects_missing_source_memory_id(self) -> None:
        with pytest.raises(ValidationError, match="source_memory_id"):
            MemoryRelation(
                tenant_id=TENANT_ID,
                target_memory_id=uuid.uuid4(),
                relation_type=ConflictType.CONTRADICTION,
            )  # type: ignore[call-arg]

    def test_rejects_invalid_relation_type(self) -> None:
        with pytest.raises(ValidationError, match="relation_type"):
            MemoryRelation(
                tenant_id=TENANT_ID,
                source_memory_id=uuid.uuid4(),
                target_memory_id=uuid.uuid4(),
                relation_type="merge",
            )

    def test_rejects_non_uuid_memory_id(self) -> None:
        with pytest.raises(ValidationError):
            MemoryRelation(
                tenant_id=TENANT_ID,
                source_memory_id="not-a-uuid",
                target_memory_id=uuid.uuid4(),
                relation_type=ConflictType.DUPLICATE,
            )


class TestMemoryRecord(unittest.TestCase):

    def test_happy_path_creates_record(self) -> None:
        validity = make_temporal_validity()
        record = MemoryRecord(
            tenant_id=TENANT_ID,
            content="The user's favorite color is blue.",
            confidence=0.95,
            provenance=[make_provenance()],
            temporal_validity=validity,
            memory_type=MemoryType.SEMANTIC,
            trust_level=TrustLevel.HIGH,
            status=MemoryStatus.ACTIVE,
        )
        self.assertIsInstance(record.memory_id, UUID)
        self.assertEqual(record.status, MemoryStatus.ACTIVE)
        self.assertEqual(record.policy_version, "1.0")
        self.assertEqual(record.subject_keys, [])
        self.assertIsInstance(record.created_at, datetime)
        self.assertIsInstance(record.updated_at, datetime)

    def test_rejects_empty_provenance(self) -> None:
        with pytest.raises(ValidationError, match="provenance"):
            MemoryRecord(
                tenant_id=TENANT_ID,
                content="content",
                confidence=0.5,
                provenance=[],
                temporal_validity=make_temporal_validity(),
                memory_type=MemoryType.PROCEDURAL,
                trust_level=TrustLevel.LOW,
                status=MemoryStatus.ACTIVE,
            )

    def test_rejects_missing_temporal_validity(self) -> None:
        with pytest.raises(ValidationError, match="temporal_validity"):
            MemoryRecord(
                tenant_id=TENANT_ID,
                content="content",
                confidence=0.5,
                provenance=[make_provenance()],
                memory_type=MemoryType.PROCEDURAL,
                trust_level=TrustLevel.LOW,
                status=MemoryStatus.ACTIVE,
            )  # type: ignore[call-arg]

    def test_rejects_missing_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            MemoryRecord(
                tenant_id=TENANT_ID,
                content="content",
                confidence=0.5,
                provenance=[make_provenance()],
                temporal_validity=make_temporal_validity(),
                memory_type=MemoryType.PROCEDURAL,
                trust_level=TrustLevel.LOW,
            )  # type: ignore[call-arg]

    def test_rejects_invalid_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            MemoryRecord(
                tenant_id=TENANT_ID,
                content="content",
                confidence=0.5,
                provenance=[make_provenance()],
                temporal_validity=make_temporal_validity(),
                memory_type=MemoryType.PROCEDURAL,
                trust_level=TrustLevel.LOW,
                status="archived",
            )

    def test_rejects_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            MemoryRecord(
                tenant_id=TENANT_ID,
                content="content",
                confidence=2.0,
                provenance=[make_provenance()],
                temporal_validity=make_temporal_validity(),
                memory_type=MemoryType.PROCEDURAL,
                trust_level=TrustLevel.LOW,
                status=MemoryStatus.ACTIVE,
            )


class TestWritePolicyDecision(unittest.TestCase):

    def test_happy_path_creates_decision(self) -> None:
        candidate_id = uuid.uuid4()
        accepted_memory_id = uuid.uuid4()
        decision = WritePolicyDecision(
            tenant_id=TENANT_ID,
            candidate_id=candidate_id,
            decision=WriteDecision.ACCEPT,
            policy_version="1.2.0",
            reason_codes=["HIGH_CONFIDENCE", "TRUSTED_SOURCE"],
            explanation="Candidate met all acceptance criteria.",
            accepted_memory_id=accepted_memory_id,
        )
        self.assertEqual(decision.candidate_id, candidate_id)
        self.assertEqual(decision.decision, WriteDecision.ACCEPT)
        self.assertEqual(decision.accepted_memory_id, accepted_memory_id)
        self.assertIsNone(decision.superseded_memory_id)
        self.assertIsInstance(decision.decided_at, datetime)

    def test_happy_path_defaults(self) -> None:
        decision = WritePolicyDecision(
            tenant_id=TENANT_ID,
            candidate_id=uuid.uuid4(),
            decision=WriteDecision.REJECT,
            policy_version="1.0",
        )
        self.assertEqual(decision.reason_codes, [])
        self.assertIsNone(decision.explanation)
        self.assertIsNone(decision.accepted_memory_id)
        self.assertIsNone(decision.superseded_memory_id)

    def test_rejects_empty_policy_version(self) -> None:
        with pytest.raises(ValidationError, match="policy_version"):
            WritePolicyDecision(
                tenant_id=TENANT_ID,
                candidate_id=uuid.uuid4(),
                decision=WriteDecision.QUARANTINE,
                policy_version="",
            )

    def test_rejects_invalid_decision(self) -> None:
        with pytest.raises(ValidationError, match="decision"):
            WritePolicyDecision(
                tenant_id=TENANT_ID,
                candidate_id=uuid.uuid4(),
                decision="ignore",
                policy_version="1.0",
            )

    def test_rejects_missing_candidate_id(self) -> None:
        with pytest.raises(ValidationError, match="candidate_id"):
            WritePolicyDecision(
                tenant_id=TENANT_ID,
                decision=WriteDecision.SUPERSEDE,
                policy_version="1.0",
            )  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()


class TestTenant(unittest.TestCase):
    def test_happy_path_generates_an_id_and_an_aware_created_at(self) -> None:
        tenant = Tenant(name="acme-support-bot")

        self.assertIsInstance(tenant.tenant_id, UUID)
        self.assertIsNone(tenant.description)
        self.assertIsNotNone(tenant.created_at.tzinfo)

    def test_name_is_trimmed(self) -> None:
        self.assertEqual(Tenant(name="  acme  ").name, "acme")

    def test_blank_or_overlong_name_is_rejected(self) -> None:
        for name in ("", "   ", "x" * 101):
            with self.assertRaises(ValidationError):
                Tenant(name=name)

    def test_extra_fields_are_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            Tenant(name="acme", plan="enterprise")  # type: ignore[call-arg]

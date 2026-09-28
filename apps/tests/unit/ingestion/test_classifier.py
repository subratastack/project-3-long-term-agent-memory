import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from apps.memory_service.domain.enums import MemoryType, SourceType
from apps.memory_service.domain.models import MemoryEvent
from apps.memory_service.ingestion.candidate_extractor import RawCandidate
from apps.memory_service.ingestion.classifier import classify_memory_type

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _make_event(**overrides: Any) -> MemoryEvent:
    defaults: dict[str, Any] = {
        "tenant_id": TENANT_ID,
        "source_type": SourceType.SYSTEM_EVENT,
        "source_reference": "session-1",
        "content": "an event",
        "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return MemoryEvent(**defaults)


class TestClassifyMemoryType(unittest.TestCase):

    def test_configuration_event_is_classified_semantic(self) -> None:
        event = _make_event(source_type=SourceType.CONFIGURATION, content="timeout=2s")
        raw = RawCandidate(
            content="The configured timeout is 2s.", source_event_ids=[event.event_id]
        )

        self.assertEqual(classify_memory_type(raw, [event]), MemoryType.SEMANTIC)

    def test_assignment_shaped_content_is_classified_semantic_even_without_configuration_source(
        self,
    ) -> None:
        event = _make_event(source_type=SourceType.TOOL_OUTPUT, content="pool_size=10")
        raw = RawCandidate(content="pool_size=10", source_event_ids=[event.event_id])

        self.assertEqual(classify_memory_type(raw, [event]), MemoryType.SEMANTIC)

    def test_tool_outcome_is_classified_episodic(self) -> None:
        event = _make_event(source_type=SourceType.TOOL_OUTPUT, content="connection pool exhausted")
        raw = RawCandidate(
            content="The connection pool was exhausted.", source_event_ids=[event.event_id]
        )

        self.assertEqual(classify_memory_type(raw, [event]), MemoryType.EPISODIC)

    def test_multi_event_sequence_is_classified_procedural(self) -> None:
        first = _make_event(source_type=SourceType.AGENT_ACTION, content="checked logs")
        second = _make_event(source_type=SourceType.AGENT_ACTION, content="restarted service")
        raw = RawCandidate(
            content="Checked logs, then restarted the service.",
            source_event_ids=[first.event_id, second.event_id],
        )

        self.assertEqual(classify_memory_type(raw, [first, second]), MemoryType.PROCEDURAL)

    def test_explicit_hint_pushes_toward_procedural(self) -> None:
        event = _make_event(source_type=SourceType.AGENT_ACTION, content="restarted service")
        raw = RawCandidate(
            content="Restarted the service.",
            source_event_ids=[event.event_id],
            memory_type_hint=MemoryType.PROCEDURAL,
        )

        self.assertEqual(classify_memory_type(raw, [event]), MemoryType.PROCEDURAL)

    def test_unresolved_source_events_are_ignored_rather_than_erroring(self) -> None:
        raw = RawCandidate(content="orphaned content", source_event_ids=[UUID(int=99)])

        self.assertEqual(classify_memory_type(raw, []), MemoryType.EPISODIC)


if __name__ == "__main__":
    unittest.main()

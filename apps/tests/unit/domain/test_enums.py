import unittest

from apps.memory_service.domain.enums import (
    ConflictType,
    MemoryStatus,
    MemoryType,
    SourceType,
    TrustLevel,
    WriteDecision,
)


class TestMemoryType(unittest.TestCase):

    def test_members_have_expected_string_values(self) -> None:
        self.assertEqual(MemoryType.EPISODIC, "episodic")
        self.assertEqual(MemoryType.SEMANTIC, "semantic")
        self.assertEqual(MemoryType.PROCEDURAL, "procedural")

    def test_constructing_from_valid_value_returns_member(self) -> None:
        self.assertIs(MemoryType("episodic"), MemoryType.EPISODIC)

    def test_constructing_from_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            MemoryType("nonexistent")


class TestMemoryStatus(unittest.TestCase):

    def test_members_have_expected_string_values(self) -> None:
        self.assertEqual(MemoryStatus.ACTIVE, "active")
        self.assertEqual(MemoryStatus.QUARANTINED, "quarantined")
        self.assertEqual(MemoryStatus.SUPERSEDED, "superseded")
        self.assertEqual(MemoryStatus.EXPIRED, "expired")
        self.assertEqual(MemoryStatus.TOMBSTONE, "tombstone")

    def test_constructing_from_valid_value_returns_member(self) -> None:
        self.assertIs(MemoryStatus("active"), MemoryStatus.ACTIVE)

    def test_constructing_from_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            MemoryStatus("archived")


class TestTrustLevel(unittest.TestCase):

    def test_members_have_expected_string_values(self) -> None:
        self.assertEqual(TrustLevel.UNTRUSTED, "untrusted")
        self.assertEqual(TrustLevel.SYSTEM, "system")
        self.assertEqual(TrustLevel.HIGH, "high")
        self.assertEqual(TrustLevel.MEDIUM, "medium")
        self.assertEqual(TrustLevel.LOW, "low")

    def test_constructing_from_valid_value_returns_member(self) -> None:
        self.assertIs(TrustLevel("system"), TrustLevel.SYSTEM)

    def test_constructing_from_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            TrustLevel("critical")


class TestSourceType(unittest.TestCase):

    def test_members_have_expected_string_values(self) -> None:
        self.assertEqual(SourceType.USER_MESSAGE, "user_message")
        self.assertEqual(SourceType.TOOL_OUTPUT, "tool_output")
        self.assertEqual(SourceType.SYSTEM_EVENT, "system_event")
        self.assertEqual(SourceType.AGENT_ACTION, "agent_action")
        self.assertEqual(SourceType.CONFIGURATION, "configuration")

    def test_constructing_from_valid_value_returns_member(self) -> None:
        self.assertIs(SourceType("tool_output"), SourceType.TOOL_OUTPUT)

    def test_constructing_from_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            SourceType("webhook")


class TestWriteDecision(unittest.TestCase):

    def test_members_have_expected_string_values(self) -> None:
        self.assertEqual(WriteDecision.ACCEPT, "accept")
        self.assertEqual(WriteDecision.REJECT, "reject")
        self.assertEqual(WriteDecision.QUARANTINE, "quarantine")
        self.assertEqual(WriteDecision.SUPERSEDE, "supersede")

    def test_constructing_from_valid_value_returns_member(self) -> None:
        self.assertIs(WriteDecision("accept"), WriteDecision.ACCEPT)

    def test_constructing_from_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            WriteDecision("ignore")


class TestConflictType(unittest.TestCase):

    def test_members_have_expected_string_values(self) -> None:
        self.assertEqual(ConflictType.CONTRADICTION, "contradiction")
        self.assertEqual(ConflictType.DUPLICATE, "duplicate")
        self.assertEqual(ConflictType.SUPERSESSION, "supersession")

    def test_constructing_from_valid_value_returns_member(self) -> None:
        self.assertIs(ConflictType("duplicate"), ConflictType.DUPLICATE)

    def test_constructing_from_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            ConflictType("merge")


if __name__ == "__main__":
    unittest.main()

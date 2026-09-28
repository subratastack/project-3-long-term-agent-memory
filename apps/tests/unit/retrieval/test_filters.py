import unittest
from datetime import UTC, datetime
from uuid import UUID

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.retrieval.filters import (
    ALLOWED_STATUSES,
    HISTORICAL_STATUSES,
    resolve_filters,
)
from apps.memory_service.retrieval.query_model import RetrievalQuery

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


class TestResolveFilters(unittest.TestCase):
    def test_default_query_allows_every_trust_level_and_any_type(self) -> None:
        query = RetrievalQuery(tenant_id=TENANT_ID, query_text="pool exhaustion")

        filters = resolve_filters(query)

        self.assertEqual(filters.tenant_id, TENANT_ID)
        self.assertEqual(filters.allowed_statuses, ALLOWED_STATUSES)
        self.assertEqual(filters.allowed_trust_levels, frozenset(TrustLevel))
        self.assertIsNone(filters.memory_types)

    def test_min_trust_excludes_everything_below_the_threshold(self) -> None:
        query = RetrievalQuery(
            tenant_id=TENANT_ID, query_text="pool exhaustion", min_trust=TrustLevel.MEDIUM
        )

        filters = resolve_filters(query)

        self.assertIn(TrustLevel.MEDIUM, filters.allowed_trust_levels)
        self.assertIn(TrustLevel.HIGH, filters.allowed_trust_levels)
        self.assertIn(TrustLevel.SYSTEM, filters.allowed_trust_levels)
        self.assertNotIn(TrustLevel.LOW, filters.allowed_trust_levels)
        self.assertNotIn(TrustLevel.UNTRUSTED, filters.allowed_trust_levels)

    def test_memory_types_narrows_the_filter(self) -> None:
        query = RetrievalQuery(
            tenant_id=TENANT_ID,
            query_text="pool exhaustion",
            memory_types=[MemoryType.EPISODIC],
        )

        filters = resolve_filters(query)

        self.assertEqual(filters.memory_types, (MemoryType.EPISODIC,))

    def test_allowed_statuses_is_active_only_for_a_current_query(self) -> None:
        query = RetrievalQuery(tenant_id=TENANT_ID, query_text="pool exhaustion")

        filters = resolve_filters(query)

        self.assertEqual(filters.allowed_statuses, frozenset({MemoryStatus.ACTIVE}))

    def test_an_as_of_query_also_allows_since_superseded_or_expired_memories(self) -> None:
        query = RetrievalQuery(
            tenant_id=TENANT_ID,
            query_text="pool exhaustion",
            as_of=datetime(2026, 2, 15, tzinfo=UTC),
        )

        filters = resolve_filters(query)

        self.assertEqual(filters.allowed_statuses, HISTORICAL_STATUSES)
        self.assertEqual(
            filters.allowed_statuses,
            frozenset({MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED, MemoryStatus.EXPIRED}),
        )
        self.assertNotIn(MemoryStatus.QUARANTINED, filters.allowed_statuses)
        self.assertNotIn(MemoryStatus.TOMBSTONE, filters.allowed_statuses)

    def test_as_of_is_used_as_the_effective_time_verbatim(self) -> None:
        as_of = datetime(2026, 1, 1, tzinfo=UTC)
        query = RetrievalQuery(tenant_id=TENANT_ID, query_text="pool exhaustion", as_of=as_of)

        filters = resolve_filters(query)

        self.assertEqual(filters.effective_at, as_of)

    def test_missing_as_of_defaults_to_roughly_now(self) -> None:
        query = RetrievalQuery(tenant_id=TENANT_ID, query_text="pool exhaustion")

        filters = resolve_filters(query)

        self.assertLess(abs((filters.effective_at - datetime.now(UTC)).total_seconds()), 5)


class TestRetrievalQueryValidation(unittest.TestCase):
    def test_requires_exactly_one_of_query_text_or_query_embedding(self) -> None:
        with self.assertRaises(ValueError):
            RetrievalQuery(tenant_id=TENANT_ID)

        with self.assertRaises(ValueError):
            RetrievalQuery(
                tenant_id=TENANT_ID, query_text="pool exhaustion", query_embedding=[0.1, 0.2]
            )


if __name__ == "__main__":
    unittest.main()

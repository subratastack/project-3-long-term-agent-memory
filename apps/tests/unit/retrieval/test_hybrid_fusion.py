"""Unit tests for `hybrid._fuse_rrf`, the pure Reciprocal Rank Fusion logic.

`_fuse_rrf` is private, but it is pure (no DB, no embedder) and is exactly
the part of `hybrid_search` worth unit-testing in isolation -- everything
else `hybrid_search` does (calling `lexical_search`/`semantic_search`) needs
real PostgreSQL and is covered by the integration tests in
apps/tests/integration/retrieval/test_hybrid_search.py instead.
"""

import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryRecord, TemporalValidity
from apps.memory_service.retrieval.hybrid import RRF_K, _fuse_rrf
from apps.memory_service.retrieval.lexical import LexicalSearchHit
from apps.memory_service.retrieval.semantic import SemanticSearchHit

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _make_record(memory_id: UUID | None = None, **overrides: Any) -> MemoryRecord:
    defaults: dict[str, Any] = {
        "memory_id": memory_id or uuid4(),
        "tenant_id": TENANT_ID,
        "content": "a memory",
        "confidence": 0.9,
        "provenance": [],
        "temporal_validity": TemporalValidity(valid_from=datetime(2026, 1, 1, tzinfo=UTC)),
        "memory_type": MemoryType.EPISODIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(overrides)
    # MemoryRecord.provenance requires at least one entry only via Provenance
    # objects; tests here never touch provenance, so model_construct bypasses
    # that validation for a lighter-weight fixture.
    return MemoryRecord.model_construct(**defaults)


class TestFuseRrf(unittest.TestCase):

    def test_memory_in_both_lists_accumulates_both_contributions(self) -> None:
        record = _make_record()
        lexical_hits = [LexicalSearchHit(memory=record, rank=0.5)]
        semantic_hits = [SemanticSearchHit(memory=record, distance=0.1)]

        [hit] = _fuse_rrf(lexical_hits, semantic_hits, rrf_k=RRF_K)

        expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 1)
        self.assertAlmostEqual(hit.fused_score, expected)
        self.assertEqual(hit.lexical_rank, 1)
        self.assertEqual(hit.semantic_rank, 1)

    def test_memory_in_only_one_list_still_gets_a_score(self) -> None:
        record = _make_record()
        lexical_hits = [LexicalSearchHit(memory=record, rank=0.5)]

        [hit] = _fuse_rrf(lexical_hits, [], rrf_k=RRF_K)

        self.assertAlmostEqual(hit.fused_score, 1.0 / (RRF_K + 1))
        self.assertEqual(hit.lexical_rank, 1)
        self.assertIsNone(hit.semantic_rank)

    def test_a_memory_returned_by_both_retrievers_appears_exactly_once(self) -> None:
        record = _make_record()
        lexical_hits = [LexicalSearchHit(memory=record, rank=0.9)]
        semantic_hits = [SemanticSearchHit(memory=record, distance=0.05)]

        fused = _fuse_rrf(lexical_hits, semantic_hits, rrf_k=RRF_K)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].memory.memory_id, record.memory_id)

    def test_lower_rank_number_contributes_a_larger_score(self) -> None:
        first = _make_record()
        second = _make_record()
        # `first` is rank 1, `second` is rank 2 in the same lexical list.
        lexical_hits = [
            LexicalSearchHit(memory=first, rank=0.9),
            LexicalSearchHit(memory=second, rank=0.1),
        ]

        fused = {hit.memory.memory_id: hit for hit in _fuse_rrf(lexical_hits, [], rrf_k=RRF_K)}

        self.assertGreater(fused[first.memory_id].fused_score, fused[second.memory_id].fused_score)

    def test_a_rank_1_only_contribution_can_beat_a_rank_2_plus_rank_2_contribution(
        self,
    ) -> None:
        """RRF's convexity property: a single rank-1 hit alone can still lose
        to a consistent-but-never-top hit that appears at rank 2 in *both*
        lists -- but only once the rank-1-only hit's absence from the other
        list is total. This test pins the arithmetic so the property is
        documented, not just asserted informally in hybrid.py's docstring.
        """
        consistent = _make_record()
        specialist = _make_record()
        other_a = _make_record()
        other_b = _make_record()

        lexical_hits = [
            LexicalSearchHit(memory=specialist, rank=0.9),  # rank 1
            LexicalSearchHit(memory=consistent, rank=0.5),  # rank 2
        ]
        semantic_hits = [
            SemanticSearchHit(memory=other_a, distance=0.1),  # rank 1
            SemanticSearchHit(memory=consistent, distance=0.2),  # rank 2
            SemanticSearchHit(memory=other_b, distance=0.3),  # rank 3
            SemanticSearchHit(memory=specialist, distance=0.4),  # rank 4
        ]

        fused = {
            hit.memory.memory_id: hit
            for hit in _fuse_rrf(lexical_hits, semantic_hits, rrf_k=RRF_K)
        }

        self.assertGreater(
            fused[consistent.memory_id].fused_score, fused[specialist.memory_id].fused_score
        )


if __name__ == "__main__":
    unittest.main()

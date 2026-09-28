"""Unit tests for `retrieval.reranker`: `CrossEncoderReranker` and `apply_reranker`.

Both are pure over already-fused `HybridSearchHit`s, so these run with no
database and no model download -- scoring comes from `FakeCrossEncoderModel`
or small purpose-built stubs. The same behaviour against real PostgreSQL and
the real CrossEncoder is covered in
apps/tests/integration/retrieval/test_reranking.py.
"""

import unittest
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryRecord, TemporalValidity
from apps.memory_service.embeddings.cross_encoder import FakeCrossEncoderModel
from apps.memory_service.retrieval.hybrid import HybridSearchHit
from apps.memory_service.retrieval.reranker import (
    RERANK_MAX_CANDIDATES,
    CrossEncoderReranker,
    apply_reranker,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _make_hit(content: str, fused_score: float = 0.01, **overrides: Any) -> HybridSearchHit:
    defaults: dict[str, Any] = {
        "memory_id": uuid4(),
        "tenant_id": TENANT_ID,
        "content": content,
        "confidence": 0.9,
        "provenance": [],
        "temporal_validity": TemporalValidity(valid_from=datetime(2026, 1, 1, tzinfo=UTC)),
        "memory_type": MemoryType.EPISODIC,
        "trust_level": TrustLevel.MEDIUM,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(overrides)
    # Same lightweight fixture as test_hybrid_fusion.py: provenance is never
    # inspected here, so model_construct skips its validation.
    record = MemoryRecord.model_construct(**defaults)
    return HybridSearchHit(memory=record, fused_score=fused_score, lexical_rank=1, semantic_rank=1)


def _ids(hits: Sequence[HybridSearchHit]) -> list[UUID]:
    return [hit.memory.memory_id for hit in hits]


class RecordingCrossEncoder(FakeCrossEncoderModel):
    """`FakeCrossEncoderModel` that remembers every batch of passages it scored."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append(list(passages))
        return super().score_pairs(query, passages)


class FailingCrossEncoder:
    model_version = "failing"

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        raise RuntimeError("CUDA out of memory")


class TestCrossEncoderReranker(unittest.TestCase):

    def test_reorders_by_cross_encoder_score_and_records_it(self) -> None:
        # Fused order puts the weaker match first; the CrossEncoder should flip it.
        partial = _make_hit("connection pool resized", fused_score=0.033)
        full = _make_hit("connection pool exhausted", fused_score=0.032)

        reranked = CrossEncoderReranker(FakeCrossEncoderModel()).rerank(
            "pool exhausted", [partial, full]
        )

        self.assertEqual(_ids(reranked), [full.memory.memory_id, partial.memory.memory_id])
        self.assertEqual([hit.rerank_score for hit in reranked], [1.0, 0.5])
        # Fused provenance of the score is kept alongside the new one.
        self.assertEqual(reranked[0].fused_score, 0.032)

    def test_equal_scores_keep_their_fused_order(self) -> None:
        hits = [_make_hit("pool exhausted", fused_score=0.03 - i * 0.001) for i in range(4)]

        reranked = CrossEncoderReranker(FakeCrossEncoderModel()).rerank("pool exhausted", hits)

        self.assertEqual(_ids(reranked), _ids(hits))

    def test_scores_only_max_candidates_and_passes_the_rest_through_in_order(self) -> None:
        model = RecordingCrossEncoder()
        hits = [_make_hit(f"memory number {i}") for i in range(50)]

        reranked = CrossEncoderReranker(model, max_candidates=20).rerank("memory 49", hits)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]), 20)
        self.assertEqual(model.calls[0], [hit.memory.content for hit in hits[:20]])
        # Hit 49 matches the query best but was beyond the cap: never scored, never promoted.
        self.assertEqual(_ids(reranked[20:]), _ids(hits[20:]))
        self.assertTrue(all(hit.rerank_score is None for hit in reranked[20:]))
        self.assertCountEqual(_ids(reranked), _ids(hits))

    def test_default_cap_is_the_module_constant(self) -> None:
        self.assertEqual(
            CrossEncoderReranker(FakeCrossEncoderModel()).max_candidates, RERANK_MAX_CANDIDATES
        )

    def test_rejects_a_non_positive_cap(self) -> None:
        with self.assertRaises(ValueError):
            CrossEncoderReranker(FakeCrossEncoderModel(), max_candidates=0)

    def test_empty_input_never_calls_the_model(self) -> None:
        model = RecordingCrossEncoder()

        self.assertEqual(CrossEncoderReranker(model).rerank("anything", []), [])
        self.assertEqual(model.calls, [])

    def test_a_model_returning_the_wrong_number_of_scores_is_an_error(self) -> None:
        class ShortModel(FakeCrossEncoderModel):
            def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
                return [1.0]

        with self.assertRaises(RuntimeError):
            CrossEncoderReranker(ShortModel()).rerank("q", [_make_hit("a"), _make_hit("b")])


class TestApplyReranker(unittest.TestCase):

    def test_successful_rerank_reports_its_cost(self) -> None:
        hits = [_make_hit("backup succeeded"), _make_hit("pool exhausted")]

        reordered, report = apply_reranker(
            CrossEncoderReranker(FakeCrossEncoderModel()), "pool exhausted", hits
        )

        self.assertEqual(reordered[0].memory.content, "pool exhausted")
        self.assertEqual(report.candidates_in, 2)
        self.assertEqual(report.candidates_reranked, 2)
        self.assertFalse(report.fell_back)
        self.assertGreaterEqual(report.latency_ms, 0.0)

    def test_hands_the_reranker_at_most_max_candidates(self) -> None:
        """Enforced by the stage itself, not left to each reranker: this
        reranker would happily score everything it is given.
        """
        seen: list[int] = []

        class GreedyReranker:
            max_candidates = 5

            def rerank(
                self, query_text: str, hits: Sequence[HybridSearchHit]
            ) -> list[HybridSearchHit]:
                seen.append(len(hits))
                return list(reversed(hits))

        hits = [_make_hit(f"m{i}") for i in range(12)]

        reordered, report = apply_reranker(GreedyReranker(), "q", hits)

        self.assertEqual(seen, [5])
        self.assertEqual(report.candidates_reranked, 5)
        self.assertEqual(_ids(reordered[:5]), list(reversed(_ids(hits[:5]))))
        self.assertEqual(_ids(reordered[5:]), _ids(hits[5:]))

    def test_model_failure_returns_the_fused_order_and_records_why(self) -> None:
        hits = [_make_hit("a"), _make_hit("b"), _make_hit("c")]

        with self.assertLogs("apps.memory_service.retrieval.reranker", level="WARNING") as logs:
            reordered, report = apply_reranker(
                CrossEncoderReranker(FailingCrossEncoder()), "q", hits
            )

        self.assertEqual(_ids(reordered), _ids(hits))
        self.assertTrue(report.fell_back)
        self.assertEqual(report.fallback_reason, "reranker raised RuntimeError: CUDA out of memory")
        self.assertEqual(report.candidates_reranked, 0)
        self.assertEqual(report.candidates_in, 3)
        self.assertIn("CUDA out of memory", logs.output[0])

    def test_a_reranker_that_injects_a_hit_is_rejected(self) -> None:
        """A reranker cannot add a memory -- in particular not one the hard
        filters excluded upstream.
        """
        smuggled = _make_hit("quarantined secret", status=MemoryStatus.QUARANTINED)

        class InjectingReranker:
            max_candidates = 10

            def rerank(
                self, query_text: str, hits: Sequence[HybridSearchHit]
            ) -> list[HybridSearchHit]:
                return [smuggled, *hits[1:]]

        hits = [_make_hit("a"), _make_hit("b")]

        with self.assertLogs("apps.memory_service.retrieval.reranker", level="WARNING"):
            reordered, report = apply_reranker(InjectingReranker(), "q", hits)

        self.assertEqual(_ids(reordered), _ids(hits))
        self.assertNotIn(smuggled.memory.memory_id, _ids(reordered))
        self.assertEqual(
            report.fallback_reason, "reranker output was not a reordering of its input"
        )

    def test_a_reranker_that_drops_or_duplicates_a_hit_is_rejected(self) -> None:
        class DuplicatingReranker:
            max_candidates = 10

            def rerank(
                self, query_text: str, hits: Sequence[HybridSearchHit]
            ) -> list[HybridSearchHit]:
                return [hits[0], hits[0]]

        hits = [_make_hit("a"), _make_hit("b")]

        with self.assertLogs("apps.memory_service.retrieval.reranker", level="WARNING"):
            reordered, report = apply_reranker(DuplicatingReranker(), "q", hits)

        self.assertEqual(_ids(reordered), _ids(hits))
        self.assertTrue(report.fell_back)

    def test_a_query_without_text_falls_back_instead_of_raising(self) -> None:
        model = RecordingCrossEncoder()
        hits = [_make_hit("a")]

        with self.assertLogs("apps.memory_service.retrieval.reranker", level="WARNING"):
            reordered, report = apply_reranker(CrossEncoderReranker(model), None, hits)

        self.assertEqual(_ids(reordered), _ids(hits))
        self.assertEqual(report.fallback_reason, "query has no query_text to rerank against")
        self.assertEqual(model.calls, [])

    def test_no_candidates_is_not_a_fallback(self) -> None:
        model = RecordingCrossEncoder()

        reordered, report = apply_reranker(CrossEncoderReranker(model), "q", [])

        self.assertEqual(reordered, [])
        self.assertFalse(report.fell_back)
        self.assertEqual(report.candidates_reranked, 0)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()

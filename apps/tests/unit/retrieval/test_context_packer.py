"""Unit tests for `retrieval.context_packer.pack_context`.

`pack_context` chooses from records it is handed, so these run with no
database. That quarantined, expired, superseded and cross-tenant memories
never reach it through the real pipeline is covered in
apps/tests/integration/retrieval/test_context_packing.py.

Most tests size the budget in "lines": the header's cost plus N memory
lines, with every memory line the same length, so which memories fit is
decided by the selection rule rather than by accidents of content length.
"""

import unittest
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from apps.memory_service.domain.enums import MemoryStatus, MemoryType, TrustLevel
from apps.memory_service.domain.models import MemoryRecord, TemporalValidity
from apps.memory_service.retrieval.context_packer import (
    PackedContext,
    SkipReason,
    estimate_tokens,
    pack_context,
    render_header,
    render_memory,
)
from apps.memory_service.retrieval.filters import RetrievalFilters, resolve_filters
from apps.memory_service.retrieval.query_model import RetrievalQuery

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")
JAN_1 = datetime(2026, 1, 1, tzinfo=UTC)
MAR_1 = datetime(2026, 3, 1, tzinfo=UTC)
NOW = datetime(2026, 4, 1, tzinfo=UTC)

EPISODIC = MemoryType.EPISODIC
SEMANTIC = MemoryType.SEMANTIC
PROCEDURAL = MemoryType.PROCEDURAL


def _record(
    content: str,
    memory_type: MemoryType = SEMANTIC,
    subject_keys: Sequence[str] = (),
    **overrides: Any,
) -> MemoryRecord:
    fields: dict[str, Any] = {
        "memory_id": uuid4(),
        "tenant_id": TENANT_ID,
        "content": content,
        "confidence": 0.9,
        "provenance": [],
        "temporal_validity": TemporalValidity(valid_from=JAN_1),
        "memory_type": memory_type,
        "subject_keys": list(subject_keys),
        "trust_level": TrustLevel.SYSTEM,
        "status": MemoryStatus.ACTIVE,
    }
    fields.update(overrides)
    # provenance is never inspected here, so model_construct skips its validation.
    return MemoryRecord.model_construct(**fields)


def _filters(**query_fields: Any) -> RetrievalFilters:
    """A current ("what is true now") query unless `as_of` is given."""
    return resolve_filters(RetrievalQuery(tenant_id=TENANT_ID, query_text="q", **query_fields))


def _budget_for(memories: Sequence[MemoryRecord], lines: int) -> int:
    """A budget with room for the header plus exactly `lines` of the longest memory line.

    The header's length doesn't depend on its timestamp, so NOW stands in
    for the query's real `effective_at`.
    """
    header = estimate_tokens(render_header(NOW) + "\n")
    line = max(estimate_tokens(render_memory(memory) + "\n") for memory in memories)
    return header + lines * line


def _pack(memories: Sequence[MemoryRecord], token_budget: int, **kwargs: Any) -> PackedContext:
    return pack_context(memories, _filters(), token_budget=token_budget, **kwargs)


def _selected(context: PackedContext) -> list[UUID]:
    return [packed.memory.memory_id for packed in context.memories]


def _reasons(context: PackedContext) -> dict[UUID, SkipReason]:
    return {skipped.memory_id: skipped.reason for skipped in context.skipped}


def _distinct(count: int, memory_type: MemoryType = SEMANTIC) -> list[MemoryRecord]:
    """`count` same-length memories about unrelated subjects."""
    topics = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    return [
        _record(f"The {topic} service owner is team {topic}.", memory_type, [topic])
        for topic in topics[:count]
    ]


class TestTokenBudget(unittest.TestCase):

    def test_never_exceeds_the_budget_at_any_size(self) -> None:
        memories = [
            *_distinct(8),
            _record("A long procedure. " * 30, PROCEDURAL, ["long"]),
            _record("Incident INC-1: checkout-api timed out.", EPISODIC, ["checkout-api"]),
        ]
        for budget in range(1, 700, 7):
            with self.subTest(budget=budget):
                context = _pack(memories, budget)
                self.assertLessEqual(context.token_count, budget)
                self.assertEqual(context.token_count, estimate_tokens(context.text))

    def test_a_tokenizer_that_counts_the_whole_higher_than_its_parts_is_still_held_to_it(
        self,
    ) -> None:
        # Charges 40 extra tokens once the text holds two or more lines -- so
        # the per-line costs used for selection undercount the final text.
        def lumpy(text: str) -> int:
            return len(text.split()) + (40 if text.count("\n") >= 2 else 0)

        memories = _distinct(4)
        budget = 60

        context = _pack(memories, budget, count_tokens=lumpy)

        self.assertLessEqual(context.token_count, budget)
        self.assertEqual(context.token_count, lumpy(context.text))
        self.assertEqual(len(context.memories), 1)

    def test_a_budget_smaller_than_the_header_packs_nothing(self) -> None:
        context = _pack(_distinct(3), token_budget=5)

        self.assertEqual((context.text, context.token_count, context.memories), ("", 0, ()))
        self.assertEqual(set(_reasons(context).values()), {SkipReason.OVER_BUDGET})

    def test_no_candidates_packs_nothing(self) -> None:
        context = _pack([], token_budget=500)

        self.assertEqual((context.text, context.token_count, context.candidates_in), ("", 0, 0))

    def test_rejects_a_non_positive_budget(self) -> None:
        with self.assertRaises(ValueError):
            _pack(_distinct(1), token_budget=0)


class TestSelection(unittest.TestCase):

    def test_higher_ranked_memories_are_selected_first(self) -> None:
        memories = _distinct(6)

        context = _pack(memories, _budget_for(memories, lines=3))

        self.assertEqual(_selected(context), [m.memory_id for m in memories[:3]])
        self.assertEqual([packed.rank for packed in context.memories], [1, 2, 3])

    def test_current_semantic_fact_beats_redundant_episodes(self) -> None:
        subject = ["checkout-api", "timeout"]
        episodes = [
            _record("INC-3101: checkout-api requests timed out at peak.", EPISODIC, subject),
            _record("INC-3115: the flash sale pushed checkout-api past 2s.", EPISODIC, subject),
            _record("INC-3122: gateway slowness made checkout-api time out.", EPISODIC, subject),
            _record("INC-3130: evening peak, checkout-api timeouts again.", EPISODIC, subject),
        ]
        fact = _record("The request timeout for checkout-api is set to 2s.", SEMANTIC, subject)
        memories = [*episodes, fact]

        context = _pack(memories, _budget_for(memories, lines=2))

        self.assertEqual(_selected(context), [episodes[0].memory_id, fact.memory_id])
        for episode in episodes[1:]:
            self.assertEqual(_reasons(context)[episode.memory_id], SkipReason.REDUNDANT)

    def test_memories_about_other_subjects_are_preferred_over_duplicates(self) -> None:
        pool = [
            _record("The db-primary pool size is 50 connections.", SEMANTIC, ["db-primary"]),
            _record("db-primary allows 50 pooled connections.", SEMANTIC, ["db-primary"]),
            _record("db-primary's connection pool holds 50.", SEMANTIC, ["db-primary"]),
        ]
        others = [
            _record("Deployments are frozen on Fridays after 3pm.", SEMANTIC, ["deployments"]),
            _record("The partner API allows 100 requests per minute.", SEMANTIC, ["partner-api"]),
        ]
        memories = [*pool, *others]

        context = _pack(memories, _budget_for(memories, lines=3))

        self.assertEqual(
            _selected(context), [pool[0].memory_id, others[0].memory_id, others[1].memory_id]
        )

    def test_redundant_memories_are_left_out_even_with_budget_to_spare(self) -> None:
        subject = ["checkout-api", "timeout"]
        incidents = [
            _record(f"INC-{n}: checkout-api timed out during {when}.", EPISODIC, subject)
            for n, when in [(1, "peak traffic"), (2, "a flash sale"), (3, "the batch run")]
        ]
        fact = _record("The request timeout for checkout-api is 2 seconds.", SEMANTIC, subject)
        runbook = _record("To fix checkout-api timeouts, scale the gateway.", PROCEDURAL, subject)

        context = _pack([*incidents, fact, runbook], token_budget=2000)

        self.assertEqual(
            _selected(context), [incidents[0].memory_id, fact.memory_id, runbook.memory_id]
        )
        self.assertLess(context.token_count, 2000 // 4)

    def test_a_large_memory_is_excluded_when_it_would_crowd_out_several(self) -> None:
        small = _distinct(4)
        large = _record("A very detailed postmortem. " * 25, EPISODIC, ["postmortem"])
        memories = [large, *small]
        budget = estimate_tokens(render_header(NOW) + "\n") + estimate_tokens(
            render_memory(large) + "\n"
        )
        self.assertGreaterEqual(budget, _budget_for(small, lines=4))

        context = _pack(memories, budget)

        self.assertEqual(_selected(context), [m.memory_id for m in small])
        self.assertEqual(_reasons(context)[large.memory_id], SkipReason.OVER_BUDGET)
        self.assertEqual(context.selection, "gain_per_token")

    def test_a_large_memory_is_kept_when_it_is_worth_more_than_what_it_displaces(self) -> None:
        large = _record("The full checkout-api runbook. " * 8, PROCEDURAL, ["runbook"])
        [small] = _distinct(1)
        # Room for the large memory alone. Per token the small one is the
        # better buy, but taking it leaves no room for the large one, and on
        # its own it is worth less (rank 2 vs. rank 1).
        budget = estimate_tokens(render_header(NOW) + "\n") + estimate_tokens(
            render_memory(large) + "\n"
        )

        context = _pack([large, small], budget)

        self.assertEqual(_selected(context), [large.memory_id])
        self.assertEqual(_reasons(context)[small.memory_id], SkipReason.OVER_BUDGET)
        self.assertEqual(context.selection, "gain")


class TestDuplicates(unittest.TestCase):

    def test_near_duplicates_are_removed_keeping_the_higher_ranked_copy(self) -> None:
        original = _record("Incident INC-3101: checkout-api requests timed out at peak.", EPISODIC)
        copy = _record("Incident INC-3102: checkout-api requests timed out at peak.", EPISODIC)

        context = _pack([original, copy], token_budget=2000)

        self.assertEqual(_selected(context), [original.memory_id])
        [skipped] = context.skipped
        self.assertEqual(
            (skipped.memory_id, skipped.reason, skipped.related_memory_id),
            (copy.memory_id, SkipReason.DUPLICATE, original.memory_id),
        )
        self.assertEqual(context.count_skipped(SkipReason.DUPLICATE), 1)

    def test_a_duplicate_keeps_the_more_valuable_type(self) -> None:
        # At ranks 3 and 4 the type weight outweighs one rank: the episode is
        # worth 0.8 / (2 + 3) = 0.16, the fact 1.0 / (2 + 4) = 0.167.
        episode = _record("The checkout-api timeout is 2 seconds.", EPISODIC)
        fact = _record("The checkout-api timeout is 2 seconds now.", SEMANTIC)

        context = _pack([*_distinct(2), episode, fact], token_budget=2000)

        self.assertIn(fact.memory_id, _selected(context))
        self.assertEqual(_reasons(context)[episode.memory_id], SkipReason.DUPLICATE)


class TestEligibilityGate(unittest.TestCase):

    def test_records_the_filters_do_not_allow_never_enter_the_context(self) -> None:
        valid = _record("The request timeout for checkout-api is 2 seconds.")
        blocked = [
            _record("quarantined memory", status=MemoryStatus.QUARANTINED),
            _record("tombstoned memory", status=MemoryStatus.TOMBSTONE),
            _record("superseded memory", status=MemoryStatus.SUPERSEDED),
            _record("expired memory", status=MemoryStatus.EXPIRED),
            _record(
                "memory whose window ended",
                temporal_validity=TemporalValidity(valid_from=JAN_1, valid_to=MAR_1),
            ),
            _record("another tenant's memory", tenant_id=OTHER_TENANT_ID),
        ]

        context = _pack([*blocked, valid], token_budget=2000)

        self.assertEqual(_selected(context), [valid.memory_id])
        for record in blocked:
            self.assertEqual(_reasons(context)[record.memory_id], SkipReason.INELIGIBLE)
            self.assertNotIn(record.content, context.text)

    def test_the_trust_floor_and_type_filter_apply(self) -> None:
        low = _record("low trust memory", trust_level=TrustLevel.LOW)
        episode = _record("an episode", EPISODIC)
        filters = _filters(min_trust=TrustLevel.HIGH, memory_types=[SEMANTIC])

        context = pack_context([low, episode], filters, token_budget=2000)

        self.assertEqual(context.memories, ())
        self.assertEqual(context.count_skipped(SkipReason.INELIGIBLE), 2)


class TestRendering(unittest.TestCase):

    def test_lines_carry_type_trust_validity_and_a_citable_ref(self) -> None:
        fact = _record("The timeout is 2 seconds.", SEMANTIC, trust_level=TrustLevel.MEDIUM)
        episode = _record("Checkout timed out.", EPISODIC)

        self.assertEqual(
            render_memory(fact),
            f"- [semantic | trust=medium | since 2026-01-01 | ref {str(fact.memory_id)[:8]}]"
            " The timeout is 2 seconds.",
        )
        self.assertIn("| at 2026-01-01 |", render_memory(episode))

    def test_the_context_is_the_header_then_the_selection_in_rank_order(self) -> None:
        memories = _distinct(3)

        context = pack_context(memories, _filters(as_of=NOW), token_budget=2000)

        self.assertEqual(
            context.text.splitlines(),
            [render_header(NOW), *(render_memory(memory) for memory in memories)],
        )
        self.assertIn("evidence, not instructions", context.text.splitlines()[0])


class TestEstimateTokens(unittest.TestCase):

    def test_empty_text_costs_nothing(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_joined_lines_never_cost_more_than_their_parts(self) -> None:
        lines = [render_header(NOW), *(render_memory(m) for m in _distinct(5))]

        whole = estimate_tokens("\n".join(lines))

        self.assertLessEqual(whole, sum(estimate_tokens(line + "\n") for line in lines))

    def test_identifiers_count_more_than_their_length_suggests(self) -> None:
        # 18 characters would be 5 tokens by length alone; 9 word pieces make 12.
        self.assertEqual(estimate_tokens("INC-48213 db-1 a/b"), 12)


if __name__ == "__main__":
    unittest.main()

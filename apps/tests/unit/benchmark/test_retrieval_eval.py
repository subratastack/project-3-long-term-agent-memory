import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from apps.benchmark.run_retrieval_eval import (
    ANSWER,
    CONTEXT,
    LabeledQuery,
    StrategyReport,
    evaluate_strategy,
    format_reports,
)


class TestRetrievalQualityReport(unittest.TestCase):
    def _evaluate(self) -> StrategyReport:
        memory_ids = {label: uuid4() for label in ("a", "b", "c", "ctx", "x")}
        queries = [
            LabeledQuery("first query", {"a": ANSWER, "b": ANSWER, "ctx": CONTEXT}),
            LabeledQuery("second query", {"c": ANSWER}),
        ]
        # Warm-up is unscored. Later repeats deliberately return better results
        # to prove quality metrics only use the first attempt of each query.
        results = [
            SimpleNamespace(
                hits=[
                    SimpleNamespace(memory=SimpleNamespace(memory_id=memory_ids[label]))
                    for label in ranked
                ],
                rerank=None,
            )
            for ranked in ([], ["ctx", "a"], ["a", "b"], ["x"], ["c"])
        ]
        with (
            patch("apps.benchmark.run_retrieval_eval.LABELED_QUERIES", queries),
            patch(
                "apps.benchmark.run_retrieval_eval.hybrid_search_with_report",
                side_effect=results,
            ) as search,
        ):
            report = evaluate_strategy(
                MagicMock(),
                MagicMock(),
                uuid4(),
                memory_ids,
                name="stub",
                reranker=None,
                k=3,
                repeats=2,
            )
        self.assertEqual(search.call_count, 5)
        return report

    def test_averages_first_attempts_using_only_answer_labels(self) -> None:
        report = self._evaluate()

        # One answer in three slots for query 1; no answer for query 2.
        # Grade-1 context does not earn binary relevance credit.
        self.assertAlmostEqual(report.precision_at_k, (1 / 3 + 0) / 2)
        self.assertEqual(report.recall_at_k, 0.25)
        self.assertEqual(report.mrr, 0.25)
        self.assertGreater(report.ndcg_at_k, 0.0)
        self.assertEqual(report.answer_ranks, {"first query": 2, "second query": None})

    def test_formats_precision_with_the_other_quality_metrics(self) -> None:
        report = self._evaluate()
        lines = format_reports([report]).splitlines()

        self.assertEqual(
            lines[0].split()[:5], ["strategy", "Precision@3", "Recall@3", "MRR", "nDCG@3"]
        )
        self.assertEqual(lines[2].split()[:4], ["stub", "0.167", "0.250", "0.250"])
        self.assertEqual(len(lines[0]), len(lines[1]))
        self.assertEqual(len(lines[0]), len(lines[2]))

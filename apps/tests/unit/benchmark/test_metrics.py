import math
import unittest

from apps.benchmark.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank


class TestRecallAtK(unittest.TestCase):

    def test_counts_relevant_ids_within_the_cutoff_only(self) -> None:
        self.assertEqual(recall_at_k(["a", "b", "c"], {"c"}, k=2), 0.0)
        self.assertEqual(recall_at_k(["a", "b", "c"], {"c"}, k=3), 1.0)

    def test_is_a_fraction_of_all_relevant_ids(self) -> None:
        self.assertEqual(recall_at_k(["a", "b", "c"], {"b", "z"}, k=2), 0.5)

    def test_is_undefined_without_relevant_ids(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(["a"], set(), k=1)


class TestReciprocalRank(unittest.TestCase):

    def test_uses_the_first_relevant_rank(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b", "c"], {"b", "c"}), 0.5)

    def test_is_zero_when_nothing_relevant_is_returned(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b"], {"z"}), 0.0)


class TestNdcgAtK(unittest.TestCase):

    def test_ideal_ordering_scores_one(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["answer", "ctx"], {"answer": 2, "ctx": 1}, k=2), 1.0)

    def test_swapping_graded_results_matches_the_hand_computed_value(self) -> None:
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))

        self.assertAlmostEqual(
            ndcg_at_k(["ctx", "answer"], {"answer": 2, "ctx": 1}, k=2), expected
        )

    def test_only_the_top_k_count(self) -> None:
        self.assertEqual(ndcg_at_k(["x", "answer"], {"answer": 2}, k=1), 0.0)

    def test_is_undefined_without_relevant_ids(self) -> None:
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], {}, k=1)


class TestPercentile(unittest.TestCase):

    def test_interpolates_linearly_like_numpy(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0]

        self.assertEqual(percentile(values, 50), 25.0)
        self.assertAlmostEqual(percentile(values, 95), 38.5)
        self.assertEqual(percentile(values, 0), 10.0)
        self.assertEqual(percentile(values, 100), 40.0)

    def test_order_of_input_does_not_matter(self) -> None:
        self.assertEqual(percentile([40.0, 10.0, 30.0, 20.0], 50), 25.0)

    def test_rejects_empty_input_and_out_of_range_percentiles(self) -> None:
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile([1.0], 101)


if __name__ == "__main__":
    unittest.main()

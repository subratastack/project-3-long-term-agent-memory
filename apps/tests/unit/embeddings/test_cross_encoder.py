import unittest

from apps.memory_service.embeddings.cross_encoder import FakeCrossEncoderModel


class TestFakeCrossEncoderModel(unittest.TestCase):

    def test_scores_by_fraction_of_query_tokens_present(self) -> None:
        model = FakeCrossEncoderModel()

        scores = model.score_pairs(
            "pool exhausted", ["connection pool exhausted", "pool resized", "backup ok"]
        )

        self.assertEqual(scores, [1.0, 0.5, 0.0])

    def test_returns_one_score_per_passage_in_order(self) -> None:
        model = FakeCrossEncoderModel()

        scores = model.score_pairs("a b", ["b", "a b", "c"])

        self.assertEqual(scores, [0.5, 1.0, 0.0])

    def test_is_case_and_punctuation_insensitive(self) -> None:
        model = FakeCrossEncoderModel()

        [score] = model.score_pairs("Pool EXHAUSTED?", ["pool, exhausted."])

        self.assertEqual(score, 1.0)

    def test_a_query_with_no_tokens_scores_everything_zero(self) -> None:
        model = FakeCrossEncoderModel()

        self.assertEqual(model.score_pairs("?!", ["pool", "backup"]), [0.0, 0.0])

    def test_no_passages_yields_no_scores(self) -> None:
        self.assertEqual(FakeCrossEncoderModel().score_pairs("pool", []), [])


if __name__ == "__main__":
    unittest.main()

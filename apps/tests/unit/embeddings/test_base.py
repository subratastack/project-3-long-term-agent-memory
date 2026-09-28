import unittest

from apps.memory_service.embeddings.base import FakeEmbeddingModel


class TestFakeEmbeddingModel(unittest.TestCase):

    def test_vector_has_the_configured_dimensionality(self) -> None:
        model = FakeEmbeddingModel(dimensions=32)

        [vector] = model.embed_texts(["hello world"])

        self.assertEqual(len(vector), 32)

    def test_same_text_always_embeds_to_the_same_vector(self) -> None:
        model = FakeEmbeddingModel(dimensions=32)

        first = model.embed_texts(["connection pool exhausted"])[0]
        second = model.embed_texts(["connection pool exhausted"])[0]

        self.assertEqual(first, second)

    def test_shared_vocabulary_yields_higher_similarity_than_unrelated_text(self) -> None:
        model = FakeEmbeddingModel(dimensions=256)

        query = model.embed_texts(["pool exhaustion"])[0]
        related = model.embed_texts(["The connection pool was exhausted after 30 retries."])[0]
        unrelated = model.embed_texts(["The configured request timeout is 2 seconds."])[0]

        self.assertGreater(_cosine_similarity(query, related), _cosine_similarity(query, unrelated))

    def test_empty_text_embeds_to_the_zero_vector(self) -> None:
        model = FakeEmbeddingModel(dimensions=16)

        [vector] = model.embed_texts([""])

        self.assertEqual(vector, [0.0] * 16)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


if __name__ == "__main__":
    unittest.main()

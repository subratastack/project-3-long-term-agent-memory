import unittest
from unittest import mock

from apps.memory_service.embeddings.hf_auth import huggingface_token


class TestHuggingfaceToken(unittest.TestCase):

    def test_returns_key_from_environment(self) -> None:
        with mock.patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "hf_example"}):
            self.assertEqual(huggingface_token(), "hf_example")

    def test_strips_surrounding_whitespace(self) -> None:
        with mock.patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "  hf_example\n"}):
            self.assertEqual(huggingface_token(), "hf_example")

    def test_returns_none_when_unset(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(huggingface_token())

    def test_returns_none_when_blank(self) -> None:
        with mock.patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "   "}):
            self.assertIsNone(huggingface_token())


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

from apps.memory_service.embeddings.device import model_device


class TestModelDevice(unittest.TestCase):

    def test_uses_the_configured_device(self) -> None:
        with mock.patch.dict("os.environ", {"MODEL_DEVICE": " cpu "}):
            self.assertEqual(model_device(), "cpu")

    def test_prefers_cuda_when_unset_and_available(self) -> None:
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            self.assertEqual(model_device(), "cuda")

    def test_falls_back_to_cpu_without_an_accelerator(self) -> None:
        with (
            mock.patch.dict("os.environ", {"MODEL_DEVICE": ""}),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(model_device(), "cpu")


if __name__ == "__main__":
    unittest.main()

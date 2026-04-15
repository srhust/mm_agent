from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from mm_event_agent.rag.image_encoder import Qwen3VLImageEncoder


class FakeImageBackend:
    def __init__(self, *, model_path: str, device: str, dtype: str, attn_impl: str) -> None:
        self.model_path = model_path
        self.calls: list[dict[str, object]] = []

    def encode_image_paths(self, image_paths, *, instruction: str, batch_size: int = 8):
        normalized = list(image_paths)
        self.calls.append(
            {
                "image_paths": normalized,
                "instruction": instruction,
                "batch_size": batch_size,
            }
        )
        return np.asarray([[3.0, 4.0, 12.0] for _ in normalized], dtype=np.float32)


class ImageEncoderTests(unittest.TestCase):
    def test_encoder_returns_float32_2d_and_normalizes_after_optional_slice(self) -> None:
        with patch("mm_event_agent.rag.image_encoder._Qwen3VLImageModel", FakeImageBackend), patch(
            "mm_event_agent.rag.image_encoder.verify_image_path",
            return_value=None,
        ):
            encoder = Qwen3VLImageEncoder("local-model", out_dim=2, normalize=True)
            output = encoder.encode_image_paths(["a.jpg", "b.jpg"], batch_size=2)

        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(np.allclose(output[0], np.asarray([0.6, 0.8], dtype=np.float32)))

    def test_encoder_can_disable_normalization(self) -> None:
        with patch("mm_event_agent.rag.image_encoder._Qwen3VLImageModel", FakeImageBackend), patch(
            "mm_event_agent.rag.image_encoder.verify_image_path",
            return_value=None,
        ):
            encoder = Qwen3VLImageEncoder("local-model", normalize=False)
            output = encoder.encode_image_paths(["a.jpg"])

        self.assertTrue(np.allclose(output[0], np.asarray([3.0, 4.0, 12.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from mm_event_agent.rag.text_encoder import Qwen3VLTextEncoder


class FakeQwenBackend:
    def __init__(self, *, model_path: str, device: str, dtype: str, attn_impl: str) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_impl = attn_impl

    def encode_texts(self, texts, *, instruction: str):
        if len(texts) == 1:
            return np.asarray([[3.0, 4.0, 12.0]], dtype=np.float32)
        return np.asarray([[3.0, 4.0, 12.0] for _ in texts], dtype=np.float32)


class TextEncoderTests(unittest.TestCase):
    def test_encoder_returns_float32_2d_and_normalizes_after_optional_slice(self) -> None:
        with patch("mm_event_agent.rag.text_encoder._Qwen3VLTextModel", FakeQwenBackend):
            encoder = Qwen3VLTextEncoder("local-model", out_dim=2, normalize=True)
            output = encoder.encode(["hello", "world"])

        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(np.allclose(output[0], np.asarray([0.6, 0.8], dtype=np.float32)))

    def test_encoder_can_disable_normalization(self) -> None:
        with patch("mm_event_agent.rag.text_encoder._Qwen3VLTextModel", FakeQwenBackend):
            encoder = Qwen3VLTextEncoder("local-model", normalize=False)
            output = encoder.encode(["hello"])

        self.assertTrue(np.allclose(output[0], np.asarray([3.0, 4.0, 12.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

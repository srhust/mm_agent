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
        self.calls: list[dict[str, object]] = []

    def encode_texts(self, texts, *, instruction: str, batch_size: int = 8, max_length: int = 256):
        normalized = list(texts)
        for start in range(0, len(normalized), max(1, int(batch_size))):
            self.calls.append(
                {
                    "texts": normalized[start : start + max(1, int(batch_size))],
                    "instruction": instruction,
                    "batch_size": batch_size,
                    "max_length": max_length,
                }
            )
        if len(normalized) == 1:
            return np.asarray([[3.0, 4.0, 12.0]], dtype=np.float32)
        return np.asarray([[3.0, 4.0, 12.0] for _ in normalized], dtype=np.float32)


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

    def test_encoder_batches_inputs_and_preserves_row_count(self) -> None:
        with patch("mm_event_agent.rag.text_encoder._Qwen3VLTextModel", FakeQwenBackend):
            encoder = Qwen3VLTextEncoder("local-model", normalize=False)
            output = encoder.encode(["a", "b", "c", "d", "e"], batch_size=2, max_length=128)

        self.assertEqual(output.shape, (5, 3))
        self.assertEqual(len(encoder._backend.calls), 3)
        self.assertEqual(encoder._backend.calls[0]["texts"], ["a", "b"])
        self.assertEqual(encoder._backend.calls[1]["texts"], ["c", "d"])
        self.assertEqual(encoder._backend.calls[2]["texts"], ["e"])
        self.assertEqual(encoder._backend.calls[0]["batch_size"], 2)
        self.assertEqual(encoder._backend.calls[0]["max_length"], 128)


if __name__ == "__main__":
    unittest.main()

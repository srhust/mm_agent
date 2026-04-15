from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from mm_event_agent.rag.text_encoder import SentenceTransformerTextEncoder


class FakeSentenceTransformer:
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        return np.asarray([[3.0, 4.0] for _ in texts], dtype=np.float32)


class TextEncoderTests(unittest.TestCase):
    def test_encoder_returns_float32_2d_and_normalizes(self) -> None:
        with patch("mm_event_agent.rag.text_encoder.SentenceTransformer", FakeSentenceTransformer):
            encoder = SentenceTransformerTextEncoder("local-model", normalize=True)
            output = encoder.encode(["hello", "world"])

        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(np.allclose(output[0], np.asarray([0.6, 0.8], dtype=np.float32)))

    def test_encoder_can_disable_normalization(self) -> None:
        with patch("mm_event_agent.rag.text_encoder.SentenceTransformer", FakeSentenceTransformer):
            encoder = SentenceTransformerTextEncoder("local-model", normalize=False)
            output = encoder.encode(["hello"])

        self.assertTrue(np.allclose(output[0], np.asarray([3.0, 4.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

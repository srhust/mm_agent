from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

import scripts.query_text_indexes as query_text_indexes


class FakeEncoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_impl: str = "sdpa",
        instruction: str = "Retrieve text relevant to the user's query.",
        out_dim: int = 0,
        normalize: bool = True,
    ) -> None:
        self.model_path = model_path
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size: int = 8, max_length: int = 256):
        self.calls.append(list(texts))
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


class FakeIndex:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(self, query_vector, top_k: int, filters=None):
        self.calls.append({"shape": tuple(query_vector.shape), "top_k": top_k, "filters": filters})
        return [
            {
                "score": 0.91,
                "rank": 1,
                "meta": {
                    "id": "ace-1",
                    "source_dataset": "ACE2005",
                    "event_type": "Conflict:Attack",
                    "trigger": {"text": "exploded"},
                    "doc_id": "doc-1",
                    "retrieval_text": "Conflict:Attack exploded bomb market Place",
                },
            }
        ]


class QueryTextIndexesTests(unittest.TestCase):
    def test_script_prints_readable_hit(self) -> None:
        fake_index = FakeIndex()

        with patch.object(query_text_indexes, "_resolve_index_dir", return_value="E:/fake/ace_text"), patch.object(
            query_text_indexes, "Qwen3VLTextEncoder", FakeEncoder
        ), patch.object(
            query_text_indexes.PersistentFaissIndex,
            "load",
            return_value=fake_index,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = query_text_indexes.main(
                    [
                        "--index-root",
                        "E:/fake/indexes",
                        "--model-path",
                        "E:/models/Qwen3-VL-Embedding",
                        "--index-name",
                        "ace_text",
                        "--query",
                        "market attack",
                        "--event-type",
                        "Conflict:Attack",
                    ]
                )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Rank: 1", output)
        self.assertIn("Score: 0.9100", output)
        self.assertIn("ID: ace-1", output)
        self.assertIn("Trigger: exploded", output)
        self.assertIn("Doc ID: doc-1", output)
        self.assertEqual(fake_index.calls[0]["filters"], {"event_type": "Conflict:Attack"})

    def test_script_can_print_json(self) -> None:
        with patch.object(query_text_indexes, "_resolve_index_dir", return_value="E:/fake/swig_text"), patch.object(
            query_text_indexes, "Qwen3VLTextEncoder", FakeEncoder
        ), patch.object(
            query_text_indexes.PersistentFaissIndex,
            "load",
            return_value=FakeIndex(),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = query_text_indexes.main(
                    [
                        "--index-root",
                        "E:/fake/indexes",
                        "--model-path",
                        "E:/models/Qwen3-VL-Embedding",
                        "--index-name",
                        "swig_text",
                        "--query",
                        "smoke",
                        "--json",
                    ]
                )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload[0]["id"], "ace-1")
        self.assertEqual(payload[0]["retrieval_metadata"]["index_name"], "swig_text")


if __name__ == "__main__":
    unittest.main()

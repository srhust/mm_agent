from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import scripts.query_swig_image_index as query_swig_image_index


class FakeImageEncoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_impl: str = "sdpa",
        instruction: str = "Retrieve images relevant to the user's query.",
        out_dim: int = 0,
        normalize: bool = True,
    ) -> None:
        self.model_path = model_path


class FakeRegistry:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve_swig_image_examples(self, raw_image=None, image_path: str = "", top_k: int = 5, *, event_type: str = ""):
        self.calls.append(
            {
                "raw_image": raw_image,
                "image_path": image_path,
                "top_k": top_k,
                "event_type": event_type,
            }
        )
        return [
            {
                "id": "swig-image::1",
                "source_dataset": "SWiG",
                "event_type": "Conflict:Attack",
                "image_id": "1",
                "path": "a.jpg",
                "retrieval_text": "Conflict:Attack attack weapon smoke",
                "retrieval_metadata": {"score": 0.91, "rank": 1, "index_name": "swig_image"},
            }
        ]


class QuerySwigImageIndexTests(unittest.TestCase):
    def test_script_prints_readable_hit(self) -> None:
        fake_registry = FakeRegistry()
        with patch.object(query_swig_image_index, "_resolve_index_dir", return_value="E:/fake/swig_image"), patch.object(
            query_swig_image_index,
            "verify_image_path",
            return_value=None,
        ), patch.object(
            query_swig_image_index,
            "Qwen3VLImageEncoder",
            FakeImageEncoder,
        ), patch.object(
            query_swig_image_index,
            "RagStoreRegistry",
            return_value=fake_registry,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = query_swig_image_index.main(
                    [
                        "--index-root",
                        "E:/fake/indexes",
                        "--model-path",
                        "E:/models/Qwen3-VL-Embedding",
                        "--index-name",
                        "swig_image",
                        "--image-path",
                        "query.jpg",
                        "--event-type",
                        "Conflict:Attack",
                    ]
                )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Rank: 1", output)
        self.assertIn("Score: 0.9100", output)
        self.assertIn("ID: swig-image::1", output)
        self.assertIn("Image ID: 1", output)
        self.assertIn("Path: a.jpg", output)
        self.assertEqual(fake_registry.calls[0]["image_path"], "query.jpg")
        self.assertEqual(fake_registry.calls[0]["event_type"], "Conflict:Attack")

    def test_script_can_print_json(self) -> None:
        with patch.object(query_swig_image_index, "_resolve_index_dir", return_value="E:/fake/swig_image"), patch.object(
            query_swig_image_index,
            "verify_image_path",
            return_value=None,
        ), patch.object(
            query_swig_image_index,
            "Qwen3VLImageEncoder",
            FakeImageEncoder,
        ), patch.object(
            query_swig_image_index,
            "RagStoreRegistry",
            return_value=FakeRegistry(),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = query_swig_image_index.main(
                    [
                        "--index-root",
                        "E:/fake/indexes",
                        "--model-path",
                        "E:/models/Qwen3-VL-Embedding",
                        "--index-name",
                        "swig_image",
                        "--image-path",
                        "query.jpg",
                        "--json",
                    ]
                )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload[0]["id"], "swig-image::1")
        self.assertEqual(payload[0]["retrieval_metadata"]["index_name"], "swig_image")


if __name__ == "__main__":
    unittest.main()

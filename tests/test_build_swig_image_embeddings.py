from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import scripts.build_swig_image_embeddings as build_swig_image_embeddings


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
        self.calls: list[dict[str, object]] = []

    def encode_image_paths(self, image_paths, batch_size: int = 8):
        self.calls.append({"image_paths": list(image_paths), "batch_size": batch_size})
        return np.asarray([[1.0, 0.0, 0.0] for _ in image_paths], dtype=np.float32)


class BuildSwigImageEmbeddingsTests(unittest.TestCase):
    def test_script_writes_embedding_meta_and_bad_outputs(self) -> None:
        created_encoders: list[FakeImageEncoder] = []
        saved_arrays: dict[str, np.ndarray] = {}
        written_meta: list[dict] = []
        bad_rows: list[tuple[str, str, str]] = []

        def make_encoder(*args, **kwargs):
            encoder = FakeImageEncoder(*args, **kwargs)
            created_encoders.append(encoder)
            return encoder

        def fake_validate(records):
            return (
                [
                    {"id": "swig-image::1", "image_id": "1", "path": "good1.jpg", "event_type": "Conflict:Attack"},
                    {"id": "swig-image::3", "image_id": "3", "path": "good2.jpg", "event_type": "Life:Die"},
                ],
                [("2", "bad.jpg", "BAD_IMAGE:ValueError")],
            )

        def fake_open_memmap(path, mode, dtype, shape):
            array = np.zeros(shape, dtype=dtype)
            saved_arrays[str(path)] = array
            return array

        with patch.object(build_swig_image_embeddings, "load_jsonl", return_value=[{"dummy": True}]), patch.object(
            build_swig_image_embeddings,
            "_validate_manifest_records",
            side_effect=fake_validate,
        ), patch.object(
            build_swig_image_embeddings,
            "Qwen3VLImageEncoder",
            side_effect=make_encoder,
        ), patch.object(
            build_swig_image_embeddings,
            "open_memmap",
            side_effect=fake_open_memmap,
        ), patch.object(
            build_swig_image_embeddings,
            "write_jsonl",
            side_effect=lambda path, rows: written_meta.extend(list(rows)),
        ), patch.object(
            build_swig_image_embeddings,
            "_write_bad_tsv",
            side_effect=lambda path, rows: bad_rows.extend(rows),
        ):
            exit_code = build_swig_image_embeddings.main(
                [
                    "--input",
                    "swig.image.jsonl",
                    "--out-dir",
                    ".",
                    "--model-path",
                    "E:/models/Qwen3-VL-Embedding",
                    "--batch-size",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("swig.image.emb.npy", next(iter(saved_arrays)))
        self.assertEqual(next(iter(saved_arrays.values())).shape, (2, 3))
        self.assertEqual(len(written_meta), 2)
        self.assertEqual(bad_rows, [("2", "bad.jpg", "BAD_IMAGE:ValueError")])
        self.assertEqual(created_encoders[0].calls[0]["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()

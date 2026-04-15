from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("faiss") is not None:
    import scripts.build_text_indexes as build_text_indexes
else:  # pragma: no cover - environment-specific dependency gate
    build_text_indexes = None


class FakeQwenEncoder:
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
        self.instruction = instruction
        self.out_dim = out_dim
        self.normalize = normalize
        self.calls: list[dict[str, object]] = []

    def encode(self, texts, batch_size: int = 8, max_length: int = 256):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "max_length": max_length,
            }
        )
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


@unittest.skipUnless(build_text_indexes is not None, "faiss is not installed in this environment")
class BuildTextIndexesTests(unittest.TestCase):
    def test_build_script_writes_expected_artifacts_and_build_info(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "ace2005.text.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a", "event_type": "Conflict:Attack", "retrieval_text": "attack market"}),
                        json.dumps({"id": "c", "event_type": "Conflict:Attack", "retrieval_text": "attack smoke"}),
                        json.dumps({"id": "d", "event_type": "Life:Die", "retrieval_text": "death scene"}),
                        json.dumps({"id": "b", "event_type": "Life:Die", "retrieval_text": ""}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out_root = root / "indexes"

            created_encoders: list[FakeQwenEncoder] = []

            def make_encoder(*args, **kwargs):
                encoder = FakeQwenEncoder(*args, **kwargs)
                created_encoders.append(encoder)
                return encoder

            with patch.object(build_text_indexes, "Qwen3VLTextEncoder", make_encoder):
                exit_code = build_text_indexes.main(
                    [
                        "--inputs",
                        str(input_path),
                        "--index-root",
                        str(out_root),
                        "--model-path",
                        "E:/models/Qwen3-VL-Embedding",
                        "--instruction",
                        "retrieve text",
                        "--batch-size",
                        "2",
                        "--max-length",
                        "384",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(created_encoders), 1)
            index_dir = out_root / "ace_text"
            self.assertTrue((index_dir / "index.faiss").exists())
            self.assertTrue((index_dir / "meta.jsonl").exists())
            self.assertTrue((index_dir / "build_info.json").exists())

            build_info = json.loads((index_dir / "build_info.json").read_text(encoding="utf-8"))
            self.assertEqual(build_info["encoder_type"], "qwen3_vl_embedding")
            self.assertEqual(build_info["encoder_name_or_path"], "E:/models/Qwen3-VL-Embedding")
            self.assertEqual(build_info["instruction"], "retrieve text")
            self.assertEqual(build_info["record_count"], 3)
            self.assertEqual(build_info["skip_reasons"], {"missing_retrieval_text": 1})
            self.assertEqual(build_info["batch_size"], 2)
            self.assertEqual(build_info["max_length"], 384)
            self.assertEqual(created_encoders[0].calls[0]["batch_size"], 2)
            self.assertEqual(created_encoders[0].calls[0]["max_length"], 384)
            self.assertEqual(created_encoders[0].calls[0]["texts"], ["attack market", "attack smoke"])
            self.assertEqual(created_encoders[0].calls[1]["texts"], ["death scene"])


if __name__ == "__main__":
    unittest.main()

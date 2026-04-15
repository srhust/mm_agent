from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("faiss") is not None:
    import scripts.build_swig_image_index as build_swig_image_index
else:  # pragma: no cover - environment-specific dependency gate
    build_swig_image_index = None


@unittest.skipUnless(build_swig_image_index is not None, "faiss is not installed in this environment")
class BuildSwigImageIndexTests(unittest.TestCase):
    def test_script_builds_persistent_swig_image_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            emb_path = root / "swig-image-shard.emb.npy"
            meta_path = root / "swig-image-shard.meta.jsonl"
            np.save(emb_path, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            meta_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "swig-image::1", "image_id": "1", "path": "a.jpg", "event_type": "Conflict:Attack"}),
                        json.dumps({"id": "swig-image::2", "image_id": "2", "path": "b.jpg", "event_type": "Life:Die"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code = build_swig_image_index.main(
                [
                    "--inputs",
                    str(emb_path),
                    "--index-root",
                    str(root / "indexes"),
                    "--model-path",
                    "E:/models/Qwen3-VL-Embedding",
                    "--batch-size",
                    "4",
                ]
            )

            self.assertEqual(exit_code, 0)
            build_info = json.loads((root / "indexes" / "swig_image" / "build_info.json").read_text(encoding="utf-8"))
            self.assertEqual(build_info["record_count"], 2)
            self.assertEqual(build_info["encoder_type"], "qwen3_vl_embedding")
            self.assertEqual(build_info["batch_size"], 4)


if __name__ == "__main__":
    unittest.main()

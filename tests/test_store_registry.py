from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("faiss") is not None:
    from mm_event_agent.rag.persistent_faiss import IndexArtifactPaths, PersistentFaissIndex
    from mm_event_agent.rag.store_registry import RagStoreRegistry
else:  # pragma: no cover - environment-specific dependency gate
    IndexArtifactPaths = None
    PersistentFaissIndex = None
    RagStoreRegistry = None


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeImageEncoder:
    def encode_image_paths(self, image_paths, batch_size: int = 8):
        return np.asarray([[1.0, 0.0] for _ in image_paths], dtype=np.float32)


@unittest.skipUnless(PersistentFaissIndex is not None and RagStoreRegistry is not None, "faiss is not installed in this environment")
class RagStoreRegistryTests(unittest.TestCase):
    def test_registry_retrieves_text_examples_and_attaches_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ace_dir = root / "ace_text"
            index = PersistentFaissIndex(IndexArtifactPaths.from_root(ace_dir), index_name="ace_text")
            index.build_from_embeddings(
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                [
                    {
                        "id": "ace-1",
                        "source_dataset": "ACE2005",
                        "event_type": "Conflict:Attack",
                        "retrieval_text": "attack market",
                    },
                    {
                        "id": "ace-2",
                        "source_dataset": "ACE2005",
                        "event_type": "Life:Die",
                        "retrieval_text": "death scene",
                    },
                ],
                encoder_name_or_path="local-model",
                normalized=True,
            )
            index.save()

            registry = RagStoreRegistry(
                encoder=FakeEncoder(),
                index_root=root,
            )
            results = registry.retrieve_text_examples("attack", top_k=2, event_type="Conflict:Attack")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], "ace-1")
            self.assertEqual(results[0]["retrieval_metadata"]["index_name"], "ace_text")
            self.assertEqual(results[0]["retrieval_metadata"]["rank"], 1)

    def test_registry_retrieves_swig_image_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            swig_image_dir = root / "swig_image"
            index = PersistentFaissIndex(IndexArtifactPaths.from_root(swig_image_dir), index_name="swig_image")
            index.build_from_embeddings(
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                [
                    {
                        "id": "swig-image::1",
                        "image_id": "1",
                        "source_dataset": "SWiG",
                        "event_type": "Conflict:Attack",
                        "path": "a.jpg",
                    },
                    {
                        "id": "swig-image::2",
                        "image_id": "2",
                        "source_dataset": "SWiG",
                        "event_type": "Life:Die",
                        "path": "b.jpg",
                    },
                ],
                encoder_name_or_path="local-model",
                normalized=True,
                build_info={"encoder_type": "qwen3_vl_embedding"},
            )
            index.save()

            registry = RagStoreRegistry(
                encoder=FakeEncoder(),
                image_encoder=FakeImageEncoder(),
                index_root=root,
            )
            results = registry.retrieve_swig_image_examples(image_path="query.jpg", top_k=2, event_type="Conflict:Attack")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], "swig-image::1")
            self.assertEqual(results[0]["retrieval_metadata"]["index_name"], "swig_image")
            self.assertEqual(results[0]["retrieval_metadata"]["rank"], 1)


if __name__ == "__main__":
    unittest.main()

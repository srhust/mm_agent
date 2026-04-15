from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("faiss") is not None:
    from mm_event_agent.rag.persistent_faiss import IndexArtifactPaths, PersistentFaissIndex
else:  # pragma: no cover - environment-specific dependency gate
    IndexArtifactPaths = None
    PersistentFaissIndex = None


@unittest.skipUnless(PersistentFaissIndex is not None, "faiss is not installed in this environment")
class PersistentFaissTests(unittest.TestCase):
    def test_build_save_load_roundtrip_preserves_metadata_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "ace_text"
            index = PersistentFaissIndex(IndexArtifactPaths.from_root(root), index_name="ace_text")
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            metadata = [
                {"id": "a", "event_type": "Conflict:Attack", "retrieval_text": "attack"},
                {"id": "b", "event_type": "Life:Die", "retrieval_text": "die"},
            ]
            index.build_from_embeddings(
                embeddings,
                metadata,
                encoder_name_or_path="local-model",
                normalized=True,
                build_info={"encoder_type": "qwen3_vl_embedding", "instruction": "retrieve"},
            )
            index.save()

            loaded = PersistentFaissIndex.load(root)

            self.assertEqual(loaded.index.ntotal, 2)
            self.assertEqual(loaded.metadata[0]["id"], "a")
            self.assertEqual(loaded.build_info["record_count"], 2)
            self.assertEqual(loaded.build_info["encoder_type"], "qwen3_vl_embedding")

    def test_search_returns_ranked_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "maven_text"
            index = PersistentFaissIndex(IndexArtifactPaths.from_root(root), index_name="maven_text")
            embeddings = np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
            metadata = [
                {"id": "a", "event_type": "Conflict:Attack", "retrieval_text": "attack"},
                {"id": "b", "event_type": "Conflict:Attack", "retrieval_text": "attack near market"},
            ]
            index.build_from_embeddings(
                embeddings,
                metadata,
                encoder_name_or_path="local-model",
                normalized=True,
            )

            results = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["meta"]["id"], "a")
            self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_search_filters_on_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "bridge"
            index = PersistentFaissIndex(IndexArtifactPaths.from_root(root), index_name="bridge")
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            metadata = [
                {"id": "a", "event_type": "Conflict:Attack", "retrieval_text": "attack"},
                {"id": "b", "event_type": "Life:Die", "retrieval_text": "die"},
            ]
            index.build_from_embeddings(
                embeddings,
                metadata,
                encoder_name_or_path="local-model",
                normalized=True,
            )

            results = index.search(
                np.asarray([1.0, 0.0], dtype=np.float32),
                top_k=2,
                filters={"event_type": "Life:Die"},
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["meta"]["id"], "b")

    def test_incremental_add_path_keeps_row_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "ace_text"
            index = PersistentFaissIndex(IndexArtifactPaths.from_root(root), index_name="ace_text")
            index.initialize_empty(
                vector_dim=2,
                encoder_name_or_path="local-model",
                normalized=True,
                build_info={"encoder_type": "qwen3_vl_embedding", "batch_size": 2, "max_length": 256},
            )
            index.add_embeddings(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                [{"id": "a", "event_type": "Conflict:Attack", "retrieval_text": "attack"}],
            )
            index.add_embeddings(
                np.asarray([[0.0, 1.0]], dtype=np.float32),
                [{"id": "b", "event_type": "Life:Die", "retrieval_text": "die"}],
            )

            self.assertEqual(index.index.ntotal, 2)
            self.assertEqual(len(index.metadata), 2)
            self.assertEqual(index.metadata[0]["id"], "a")
            self.assertEqual(index.metadata[1]["id"], "b")
            self.assertEqual(index.build_info["record_count"], 2)


if __name__ == "__main__":
    unittest.main()

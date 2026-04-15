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


if __name__ == "__main__":
    unittest.main()

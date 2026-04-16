from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import mm_event_agent.rag.store_registry as store_registry_module
from mm_event_agent.rag.store_registry import RagStoreRegistry, get_cached_registry


class FakeIndex:
    def __init__(self, name: str) -> None:
        self.name = name

    def search(self, query_vector, top_k: int, filters=None) -> list[dict]:
        event_type = filters.get("event_type") if isinstance(filters, dict) else ""
        if self.name == "swig_image":
            return [
                {
                    "meta": {
                        "id": "swig-image-1",
                        "source_dataset": "SWiG",
                        "event_type": event_type or "Conflict:Attack",
                        "image_id": "img-001",
                        "path": "query.jpg",
                        "retrieval_text": "market scene with smoke",
                    },
                    "score": 0.91,
                    "rank": 1,
                }
            ]
        return [
            {
                "meta": {
                    "id": "ace-1",
                    "source_dataset": "ACE2005",
                    "event_type": event_type or "Conflict:Attack",
                    "retrieval_text": "attack market",
                },
                "score": 0.82,
                "rank": 1,
            }
        ]


class FakeTextEncoder:
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1

    def encode(self, texts, batch_size: int = 8, max_length: int = 256):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeImageEncoder:
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1

    def encode_image_paths(self, image_paths, batch_size: int = 8):
        return np.asarray([[1.0, 0.0] for _ in image_paths], dtype=np.float32)


class CountingRegistry:
    instances = 0

    def __init__(self, **kwargs) -> None:
        type(self).instances += 1
        self.kwargs = kwargs


class StoreRegistryRuntimeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        store_registry_module._REGISTRY_CACHE.clear()
        FakeTextEncoder.instances = 0
        FakeImageEncoder.instances = 0
        CountingRegistry.instances = 0

    def test_get_cached_registry_reuses_same_instance_for_same_config(self) -> None:
        with patch.object(store_registry_module, "RagStoreRegistry", CountingRegistry), patch.object(
            store_registry_module,
            "settings",
            replace(
                store_registry_module.settings,
                rag_index_root="data/rag/indexes",
                rag_qwen_model_path="E:/models/qwen",
                rag_qwen_device="cuda:0",
                rag_qwen_dtype="bfloat16",
                rag_qwen_attn_impl="sdpa",
                rag_qwen_text_instruction="text instruction",
                rag_qwen_image_instruction="image instruction",
                rag_enable_image_query=True,
            ),
        ):
            first = get_cached_registry()
            second = get_cached_registry()

        self.assertIs(first, second)
        self.assertEqual(CountingRegistry.instances, 1)

    def test_repeated_text_retrieval_reuses_lazy_text_encoder(self) -> None:
        with patch.object(store_registry_module, "Qwen3VLTextEncoder", FakeTextEncoder), patch.object(
            store_registry_module, "Qwen3VLImageEncoder", FakeImageEncoder
        ), patch.object(
            store_registry_module.PersistentFaissIndex,
            "load",
            side_effect=lambda path: FakeIndex(Path(path).name),
        ), patch.object(
            Path,
            "exists",
            lambda self: self.name == "ace_text",
        ):
            registry = RagStoreRegistry(index_root="cache-root")
            registry.retrieve_text_examples("attack", top_k=1, event_type="Conflict:Attack")
            registry.retrieve_text_examples("attack", top_k=1, event_type="Conflict:Attack")

        self.assertEqual(FakeTextEncoder.instances, 1)
        self.assertEqual(FakeImageEncoder.instances, 0)

    def test_repeated_image_retrieval_reuses_lazy_image_encoder(self) -> None:
        with patch.object(store_registry_module, "Qwen3VLTextEncoder", FakeTextEncoder), patch.object(
            store_registry_module, "Qwen3VLImageEncoder", FakeImageEncoder
        ), patch.object(
            store_registry_module.PersistentFaissIndex,
            "load",
            side_effect=lambda path: FakeIndex(Path(path).name),
        ), patch.object(
            Path,
            "exists",
            lambda self: self.name == "swig_image",
        ):
            registry = RagStoreRegistry(index_root="cache-root")
            registry.retrieve_swig_image_examples(image_path="query.jpg", top_k=1, event_type="Conflict:Attack")
            registry.retrieve_swig_image_examples(image_path="query.jpg", top_k=1, event_type="Conflict:Attack")

        self.assertEqual(FakeImageEncoder.instances, 1)
        self.assertEqual(FakeTextEncoder.instances, 0)

    def test_image_encoder_stays_lazy_without_image_queries(self) -> None:
        with patch.object(store_registry_module, "Qwen3VLTextEncoder", FakeTextEncoder), patch.object(
            store_registry_module, "Qwen3VLImageEncoder", FakeImageEncoder
        ), patch.object(
            store_registry_module.PersistentFaissIndex,
            "load",
            side_effect=lambda path: FakeIndex(Path(path).name),
        ), patch.object(
            Path,
            "exists",
            lambda self: self.name == "ace_text",
        ):
            registry = RagStoreRegistry(index_root="cache-root")
            registry.available_index_names()
            registry.retrieve_text_examples("attack", top_k=1)

        self.assertEqual(FakeTextEncoder.instances, 1)
        self.assertEqual(FakeImageEncoder.instances, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from mm_event_agent.layered_rag import LayeredRagStore
from mm_event_agent.nodes.rag import rag as rag_node
from mm_event_agent.schemas import empty_layered_similar_events
import mm_event_agent.layered_rag as layered_rag_module
import mm_event_agent.nodes.rag as rag_node_module


def _make_state() -> dict:
    return {
        "text": "A bomb exploded in a market",
        "raw_image": "C:/tmp/query.jpg",
        "event": {"event_type": "Conflict:Attack"},
        "image_desc": "market area with smoke",
    }


class FakeRegistry:
    def __init__(self, available_index_names: set[str] | None = None) -> None:
        self._available_index_names = set(available_index_names or set())
        self.calls: list[tuple[str, str, int, str]] = []

    def available_index_names(self) -> set[str]:
        return set(self._available_index_names)

    def retrieve_text_examples(self, query: str, top_k: int, *, event_type: str = "") -> list[dict]:
        self.calls.append(("text", query, top_k, event_type))
        return [
            {
                "id": "ace-1",
                "source_dataset": "ACE2005",
                "event_type": "Conflict:Attack",
                "raw_text": "A bomb exploded in a market.",
                "trigger": {"text": "exploded", "span": {"start": 7, "end": 15}},
                "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
                "pattern_summary": "Attack pattern.",
                "retrieval_text": "Conflict:Attack exploded market",
                "retrieval_metadata": {"score": 0.81, "rank": 1, "index_name": "ace_text"},
            }
        ]

    def retrieve_swig_text_examples(self, query: str, top_k: int, *, event_type: str = "") -> list[dict]:
        self.calls.append(("swig_text", query, top_k, event_type))
        return [
            {
                "id": "swig-text-1",
                "source_dataset": "SWiG",
                "event_type": "Conflict:Attack",
                "visual_situation": "attack aftermath",
                "image_desc": "smoke around market stalls",
                "image_arguments": [{"role": "Place", "label": "market area"}],
                "visual_pattern_summary": "Scene-level attack cues.",
                "retrieval_text": "Conflict:Attack smoke market",
                "retrieval_metadata": {"score": 0.73, "rank": 1, "index_name": "swig_text"},
            }
        ]

    def retrieve_swig_image_examples(
        self,
        raw_image=None,
        image_path: str = "",
        top_k: int = 5,
        *,
        event_type: str = "",
    ) -> list[dict]:
        self.calls.append(("swig_image", image_path, top_k, event_type))
        return [
            {
                "id": "swig-image-1",
                "source_dataset": "SWiG",
                "event_type": "Conflict:Attack",
                "image_id": "img-001",
                "path": image_path,
                "retrieval_text": "market scene with smoke",
                "retrieval_metadata": {"score": 0.92, "rank": 1, "index_name": "swig_image"},
            }
        ]

    def retrieve_bridge_examples(self, query: str, top_k: int, *, event_type: str = "") -> list[dict]:
        self.calls.append(("bridge", query, top_k, event_type))
        return []


class LayeredRagRuntimeTests(unittest.TestCase):
    def test_feature_flag_off_keeps_demo_path(self) -> None:
        store = LayeredRagStore(persistent_registry_factory=lambda: FakeRegistry({"ace_text"}))
        expected = {
            "text_event_examples": [{"id": "demo-text"}],
            "image_semantic_examples": [{"id": "demo-image"}],
            "bridge_examples": [{"id": "demo-bridge"}],
        }

        with patch.object(layered_rag_module, "settings", replace(layered_rag_module.settings, rag_use_persistent_index=False)):
            with patch.object(store, "_retrieve_demo", return_value=expected) as mock_demo, patch.object(
                store, "_retrieve_persistent", return_value=empty_layered_similar_events()
            ) as mock_persistent:
                result = store.retrieve(raw_text="query", image_desc="scene", event_type="Conflict:Attack", top_k=3)

        self.assertEqual(result, expected)
        self.assertTrue(mock_demo.called)
        self.assertFalse(mock_persistent.called)

    def test_feature_flag_on_uses_persistent_registry_and_keeps_shape(self) -> None:
        fake_registry = FakeRegistry({"ace_text", "swig_text", "swig_image"})
        store = LayeredRagStore(persistent_registry_factory=lambda: fake_registry)

        with patch.object(
            layered_rag_module,
            "settings",
            replace(
                layered_rag_module.settings,
                rag_use_persistent_index=True,
                rag_use_demo_corpus=False,
                rag_enable_image_query=True,
                rag_text_top_k=2,
                rag_image_top_k=2,
                rag_bridge_top_k=1,
            ),
        ), patch.object(layered_rag_module, "_extract_image_query_path", return_value="C:/tmp/query.jpg"):
            result = store.retrieve(
                raw_text="A bomb exploded in a market",
                image_desc="market area with smoke",
                event_type="Conflict:Attack",
                top_k=3,
                raw_image="C:/tmp/query.jpg",
            )

        self.assertEqual(set(result.keys()), {"text_event_examples", "image_semantic_examples", "bridge_examples"})
        self.assertEqual(result["text_event_examples"][0]["id"], "ace-1")
        self.assertEqual(result["text_event_examples"][0]["modality"], "text")
        self.assertEqual(len(result["image_semantic_examples"]), 2)
        self.assertEqual(result["image_semantic_examples"][0]["id"], "swig-image-1")
        self.assertEqual(result["image_semantic_examples"][0]["modality"], "image_semantics")
        self.assertEqual(result["image_semantic_examples"][0]["path"], "C:/tmp/query.jpg")
        self.assertEqual(result["image_semantic_examples"][1]["id"], "swig-text-1")
        self.assertEqual(result["bridge_examples"], [])
        self.assertIn(("text", "A bomb exploded in a market", 2, "Conflict:Attack"), fake_registry.calls)
        self.assertIn(("swig_image", "C:/tmp/query.jpg", 2, "Conflict:Attack"), fake_registry.calls)

    def test_missing_persistent_indexes_falls_back_to_demo_when_enabled(self) -> None:
        store = LayeredRagStore(persistent_registry_factory=lambda: FakeRegistry(set()))
        expected = {
            "text_event_examples": [{"id": "demo-text"}],
            "image_semantic_examples": [],
            "bridge_examples": [],
        }

        with patch.object(
            layered_rag_module,
            "settings",
            replace(layered_rag_module.settings, rag_use_persistent_index=True, rag_use_demo_corpus=True),
        ):
            with patch.object(store, "_retrieve_demo", return_value=expected) as mock_demo:
                result = store.retrieve(raw_text="query", image_desc="", event_type="", top_k=3)

        self.assertEqual(result, expected)
        self.assertTrue(mock_demo.called)

    def test_missing_persistent_indexes_returns_empty_when_demo_disabled(self) -> None:
        store = LayeredRagStore(persistent_registry_factory=lambda: FakeRegistry(set()))

        with patch.object(
            layered_rag_module,
            "settings",
            replace(layered_rag_module.settings, rag_use_persistent_index=True, rag_use_demo_corpus=False),
        ):
            result = store.retrieve(raw_text="query", image_desc="", event_type="", top_k=3)

        self.assertEqual(result, empty_layered_similar_events())

    def test_local_rag_node_passes_raw_image_and_configured_top_k(self) -> None:
        state = _make_state()

        with patch.object(rag_node_module, "settings", replace(rag_node_module.settings, rag_default_top_k=7)):
            with patch.object(rag_node_module, "_retrieve_similar_events", return_value=empty_layered_similar_events()) as mock_retrieve:
                rag_node(state)

        _, kwargs = mock_retrieve.call_args
        self.assertEqual(kwargs["top_k"], 7)
        self.assertEqual(kwargs["raw_image"], "C:/tmp/query.jpg")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import shutil
import unittest
import uuid
import mm_event_agent.nodes.extraction as extraction_module
import mm_event_agent.main as main_module
import mm_event_agent.nodes.perception as perception_module
import mm_event_agent.runtime_config as runtime_config
import mm_event_agent.grounding.florence2_hf as grounding_module
import scripts.eval_m2e2_agent as eval_m2e2_agent_module
import scripts.run_m2e2_smoke as run_m2e2_smoke_module
from types import SimpleNamespace
from unittest.mock import patch

from mm_event_agent.graph import build_graph
from mm_event_agent.m2e2_adapter import (
    agent_output_to_m2e2_prediction,
    extract_m2e2_gold_annotations,
    extract_m2e2_gold_record,
    get_m2e2_sample_id,
    m2e2_sample_to_agent_state,
)
from mm_event_agent.evidence.debug import build_evidence_source_snapshot, summarize_evidence_sources
from mm_event_agent.grounding.florence2_hf import (
    Florence2HFGrounder,
    Florence2ServiceGrounder,
    apply_grounding_results_to_event,
    build_grounding_service_payload,
    execute_grounding_requests,
    parse_grounding_service_response,
)
from mm_event_agent.grounding.debug import compare_grounding_stages, summarize_grounding_activity
from mm_event_agent.nodes.extraction import extraction
from mm_event_agent.nodes.fusion import fusion
from mm_event_agent.nodes.perception import perception
from mm_event_agent.nodes.repair import repair
from mm_event_agent.nodes.rag import rag
from mm_event_agent.nodes.search import search
from mm_event_agent.nodes.verifier import verifier
from mm_event_agent.ontology import (
    format_image_role_visibility_guidance_for_prompt,
    get_event_schema,
    get_supported_event_types,
)
from mm_event_agent.search.tavily_client import TavilySearchClient
from mm_event_agent.schemas import (
    align_text_grounded_event,
    attach_retrieval_metadata,
    ValidationError,
    attach_text_spans,
    build_grounding_requests,
    choose_best_span,
    empty_event,
    empty_fusion_context,
    empty_layered_similar_events,
    find_all_text_occurrences,
    find_text_span,
    enforce_strict_text_grounding,
    validate_event,
)


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages):
        if not self._responses:
            raise AssertionError("No fake response configured")
        return SimpleNamespace(content=self._responses.pop(0))


class RecordingLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.contents: list[object] = []

    def invoke(self, messages):
        for message in messages:
            content = getattr(message, "content", "")
            self.contents.append(content)
            self.prompts.append(content if isinstance(content, str) else str(content))
        return SimpleNamespace(content=self.response)


def make_state() -> dict:
    return {
        "text": "A bomb exploded in a market",
        "raw_image": "tests://fixtures/market-scene.jpg",
        "event_type_mode": "closed_set",
        "run_mode": "open_world",
        "effective_search_enabled": True,
        # Current bridge representation derived from raw_image.
        "image_desc": "market area with smoke and people running",
        "perception_summary": "",
        "search_query": "",
        "similar_events": empty_layered_similar_events(),
        "evidence": [],
        "fusion_context": empty_fusion_context(),
        "event": empty_event(),
        "grounding_results": [],
        "memory": [],
        "verified": False,
        "issues": [],
        "verifier_diagnostics": [],
        "verifier_confidence": 0.0,
        "verifier_reason": "",
        "repair_attempts": 0,
    }


class SmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        extraction_module._llm = None
        perception_module._PERCEPTION_CLIENT_CACHE.clear()
        perception_module._PERCEPTION_RESULT_CACHE.clear()

    def test_load_settings_exposes_future_rag_flags_without_changing_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            loaded = runtime_config.load_settings()

        self.assertFalse(loaded.rag_use_persistent_index)
        self.assertTrue(loaded.rag_use_demo_corpus)
        self.assertEqual(loaded.rag_index_root, "data/rag/indexes")
        self.assertEqual(loaded.run_mode, "benchmark")
        self.assertFalse(loaded.enable_search)
        self.assertFalse(loaded.effective_search_enabled)
        self.assertEqual(loaded.rag_qwen_model_path, "")
        self.assertEqual(loaded.rag_qwen_device, "cuda:0")
        self.assertEqual(loaded.rag_qwen_dtype, "bfloat16")
        self.assertEqual(loaded.rag_qwen_attn_impl, "sdpa")
        self.assertFalse(loaded.use_vlm_perception)
        self.assertEqual(loaded.perception_backend, "openai_compatible")
        self.assertEqual(loaded.perception_model_name, "qwen3.5-plus")
        self.assertEqual(loaded.perception_api_base, "")
        self.assertEqual(loaded.perception_api_key, "")
        self.assertEqual(loaded.perception_timeout_seconds, 30.0)
        self.assertTrue(loaded.perception_cache_enabled)
        self.assertEqual(loaded.extraction_backend, "openai_compatible")
        self.assertEqual(loaded.extraction_model_name, "gpt-4o-mini")
        self.assertEqual(loaded.extraction_api_base, "")
        self.assertEqual(loaded.extraction_api_key, "")
        self.assertEqual(loaded.extraction_timeout_seconds, 30.0)
        self.assertFalse(loaded.extraction_stage_a_use_raw_image)
        self.assertIn('"image_desc"', loaded.perception_instruction)
        self.assertEqual(loaded.florence2_task, "<CAPTION_TO_PHRASE_GROUNDING>")
        self.assertEqual(loaded.rag_text_encoder_model, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(loaded.rag_qwen_embedding_model_path, "")
        self.assertEqual(loaded.rag_qwen_embedding_device, "cuda:0")
        self.assertEqual(loaded.rag_qwen_embedding_dtype, "bfloat16")
        self.assertEqual(loaded.rag_qwen_embedding_attn_impl, "sdpa")
        self.assertEqual(loaded.rag_qwen_text_instruction, "Retrieve text relevant to the user's query.")
        self.assertEqual(loaded.rag_qwen_image_instruction, "Retrieve images relevant to the user's query.")
        self.assertEqual(loaded.rag_qwen_embedding_out_dim, 0)
        self.assertTrue(loaded.rag_qwen_embedding_normalize)
        self.assertEqual(loaded.rag_image_encoder_model_path, "")
        self.assertEqual(loaded.rag_image_encoder_device, "")
        self.assertEqual(loaded.rag_swig_image_index_dir, "")
        self.assertEqual(loaded.rag_default_top_k, 3)
        self.assertEqual(loaded.rag_text_top_k, 3)
        self.assertEqual(loaded.rag_image_top_k, 3)
        self.assertEqual(loaded.rag_bridge_top_k, 3)
        self.assertTrue(loaded.rag_enable_image_query)

    def test_attach_retrieval_metadata_returns_copy_and_preserves_existing_fields(self) -> None:
        original = {
            "id": "ace_000123",
            "event_type": "Conflict:Attack",
            "retrieval_text": "Conflict:Attack exploded market",
        }

        enriched = attach_retrieval_metadata(
            original,
            score=0.87,
            rank=1,
            index_name="text_event_examples",
        )

        self.assertEqual(original, {
            "id": "ace_000123",
            "event_type": "Conflict:Attack",
            "retrieval_text": "Conflict:Attack exploded market",
        })
        self.assertEqual(enriched["id"], "ace_000123")
        self.assertEqual(
            enriched["retrieval_metadata"],
            {
                "score": 0.87,
                "rank": 1,
                "index_name": "text_event_examples",
            },
        )

    def test_initialize_rag_runtime_uses_persistent_mode_without_demo_corpus(self) -> None:
        with patch.object(
            main_module,
            "settings",
            replace(main_module.settings, rag_use_persistent_index=True, rag_use_demo_corpus=False),
        ), patch.object(main_module, "build_index") as mock_build_index:
            mode = main_module._initialize_rag_runtime()

        self.assertEqual(mode, "persistent")
        mock_build_index.assert_called_once_with(None)

    def test_initialize_rag_runtime_keeps_demo_init_when_enabled(self) -> None:
        with patch.object(
            main_module,
            "settings",
            replace(main_module.settings, rag_use_persistent_index=False, rag_use_demo_corpus=True),
        ), patch.object(main_module, "build_index") as mock_build_index:
            mode = main_module._initialize_rag_runtime()

        self.assertEqual(mode, "demo")
        mock_build_index.assert_called_once_with(main_module.RAG_EVENT_CORPUS)

    def test_m2e2_smoke_main_starts_in_no_rag_mode_without_demo_init(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample.jpg",
        }

        class FakeGraph:
            def invoke(self, state):
                return {
                    **state,
                    "event": empty_event(),
                    "grounding_results": [],
                    "similar_events": empty_layered_similar_events(),
                    "verified": False,
                    "issues": [],
                    "verifier_reason": "",
                    "verifier_confidence": 0.0,
                }

        stdout = io.StringIO()
        with patch.object(
            run_m2e2_smoke_module,
            "parse_args",
            return_value=argparse.Namespace(
                input="dummy.jsonl",
                image_root="dummy_images",
                sample_id=None,
                output_dir=None,
            ),
        ), patch.object(
            run_m2e2_smoke_module,
            "load_m2e2_samples",
            return_value=[sample],
        ), patch(
            "pathlib.Path.exists",
            return_value=True,
        ), patch.object(
            main_module,
            "settings",
            replace(main_module.settings, rag_use_persistent_index=False, rag_use_demo_corpus=False),
        ), patch.object(main_module, "build_index") as mock_build_index, patch.object(
            run_m2e2_smoke_module,
            "build_graph",
            return_value=FakeGraph(),
        ), redirect_stdout(stdout):
            run_m2e2_smoke_module.main()

        mock_build_index.assert_called_once_with(None)
        self.assertIn("sample_id: m2e2_001", stdout.getvalue())
        self.assertIn("similar_events_summary:", stdout.getvalue())

    def test_graph_execution_smoke(self) -> None:
        state = make_state()
        state["perception_summary"] = ""
        graph = build_graph()

        with patch(
            "mm_event_agent.nodes.rag._retrieve_similar_events",
            return_value={
                "text_event_examples": [
                    {
                        "id": "ace_000123",
                        "source_dataset": "ACE2005",
                        "modality": "text",
                        "event_type": "Conflict:Attack",
                        "raw_text": "A bomb exploded in a market",
                        "trigger": {"text": "exploded", "span": {"start": 2, "end": 3}},
                        "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
                        "pattern_summary": "Attack reports often name place and instrument.",
                        "retrieval_text": "Conflict:Attack exploded bomb market Place",
                    }
                ],
                "image_semantic_examples": [],
                "bridge_examples": [],
            },
        ), patch(
            "mm_event_agent.nodes.search.search_news",
            return_value=[
                {
                    "title": "Market attack update",
                    "snippet": "Police said a bomb exploded at the city market.",
                    "url": "https://example.com/news/market-attack",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.82,
                }
            ],
        ), patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                    '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}',
                ]
            ),
        ), patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.93,"reason":"supported by evidence"}']
            ),
        ):
            final = graph.invoke(state)

        self.assertTrue(final["verified"])
        self.assertEqual(final["event"]["event_type"], "Conflict:Attack")
        self.assertEqual(final["event"]["trigger"]["text"], "exploded")
        self.assertEqual(final["event"]["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(final["event"]["text_arguments"][0]["role"], "Place")
        self.assertEqual(final["event"]["text_arguments"][0]["text"], "market")
        self.assertEqual(final["event"]["text_arguments"][0]["span"], {"start": 5, "end": 6})
        self.assertEqual(final["event"]["image_arguments"][0]["bbox"], None)
        self.assertEqual(final["event"]["image_arguments"][0]["grounding_status"], "unresolved")
        self.assertEqual(final["search_query"], "A bomb exploded in a market")
        self.assertEqual(len(final["evidence"]), 1)

    def test_perception_raw_image_present_uses_vlm_outputs(self) -> None:
        state = make_state()
        state["raw_image"] = "fake.jpg"
        state["image_desc"] = ""

        with patch.object(
            perception_module,
            "settings",
            replace(perception_module.settings, use_vlm_perception=True, perception_cache_enabled=True),
        ), patch(
            "mm_event_agent.nodes.perception._invoke_remote_perception",
            return_value={
                "image_desc": "Smoke rises above damaged market stalls.",
                "perception_summary": "Text: A bomb exploded in a market\nImage: Smoke rises above damaged market stalls.",
            },
        ) as mock_invoke:
            result = perception(state)

        self.assertTrue(mock_invoke.called)
        self.assertEqual(result["image_desc"], "Smoke rises above damaged market stalls.")
        self.assertIn("Image: Smoke rises above damaged market stalls.", result["perception_summary"])

    def test_perception_repeated_identical_calls_reuse_cached_result(self) -> None:
        state = make_state()
        state["raw_image"] = b"fake-image-bytes"
        state["image_desc"] = ""

        with patch.object(
            perception_module,
            "settings",
            replace(perception_module.settings, use_vlm_perception=True, perception_cache_enabled=True),
        ), patch(
            "mm_event_agent.nodes.perception._invoke_remote_perception",
            return_value={
                "image_desc": "people running past smoke",
                "perception_summary": "Text: A bomb exploded in a market\nImage: people running past smoke",
            },
        ) as mock_invoke:
            first = perception(state)
            second = perception(state)

        self.assertEqual(first, second)
        self.assertEqual(mock_invoke.call_count, 1)

    def test_perception_api_client_is_reused_across_calls(self) -> None:
        class FakeClient:
            instances = 0

            def __init__(self, **kwargs) -> None:
                type(self).instances += 1
                self.kwargs = kwargs

            def invoke(self, _messages):
                return SimpleNamespace(
                    content='{"image_desc":"smoke near market stalls","perception_summary":"Text: A bomb exploded in a market\\nImage: smoke near market stalls"}'
                )

        state = make_state()
        state["raw_image"] = b"fake-image-bytes"
        state["image_desc"] = ""

        with patch.object(
            perception_module,
            "settings",
            replace(
                perception_module.settings,
                use_vlm_perception=True,
                perception_cache_enabled=False,
                perception_model_name="qwen3.5-plus",
                perception_api_base="https://example.invalid/v1",
                perception_api_key="secret",
            ),
        ), patch("mm_event_agent.nodes.perception.ChatOpenAI", FakeClient):
            first = perception(state)
            state["text"] = "A second bomb exploded near the market"
            second = perception(state)

        self.assertEqual(FakeClient.instances, 1)
        self.assertEqual(first["image_desc"], "smoke near market stalls")
        self.assertEqual(second["image_desc"], "smoke near market stalls")

    def test_perception_without_raw_image_uses_existing_image_desc(self) -> None:
        state = make_state()
        state["raw_image"] = None
        state["image_desc"] = "people running, smoke"

        result = perception(state)

        self.assertEqual(result["image_desc"], "people running, smoke")
        self.assertEqual(
            result["perception_summary"],
            "Text: A bomb exploded in a market\nImage: people running, smoke",
        )

    def test_perception_without_raw_image_or_image_desc_returns_safe_empty_image_side(self) -> None:
        state = make_state()
        state["raw_image"] = None
        state["image_desc"] = ""

        result = perception(state)

        self.assertEqual(result["image_desc"], "")
        self.assertEqual(
            result["perception_summary"],
            "Text: A bomb exploded in a market\nImage: ",
        )

    def test_supported_event_types_match_closed_set(self) -> None:
        self.assertEqual(
            get_supported_event_types(),
            [
                "Movement:Transport",
                "Conflict:Attack",
                "Conflict:Demonstrate",
                "Justice:Arrest-Jail",
                "Contact:Phone-Write",
                "Contact:Meet",
                "Life:Die",
                "Transaction:Transfer-Money",
            ],
        )

    def test_enriched_ontology_schema_exposes_semantic_guidance(self) -> None:
        schema = get_event_schema("Conflict:Attack")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["definition"], "A violent or harmful attack is carried out against a target, person, group, or location.")
        self.assertIn("exploded", schema["trigger_hint"])
        self.assertEqual(schema["role_definitions"]["Attacker"], "The person, group, or force carrying out the attack.")
        self.assertTrue(schema["extraction_notes"])

    def test_visibility_guidance_marks_weak_roles_more_conservatively(self) -> None:
        guidance = format_image_role_visibility_guidance_for_prompt("Conflict:Attack")

        self.assertIn("visually stronger roles: [Target, Instrument, Place]", guidance)
        self.assertIn("visually weaker roles: [Attacker]", guidance)
        self.assertIn("prefer omission", guidance)

    def test_visibility_guidance_keeps_clearly_visible_roles_allowed(self) -> None:
        guidance = format_image_role_visibility_guidance_for_prompt("Conflict:Demonstrate")

        self.assertIn("visually stronger roles: [Entity, Instrument, Place]", guidance)
        self.assertIn("visually weaker roles: [Police]", guidance)

    def test_invalid_json_output_falls_back_to_empty_event(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "Text: A bomb exploded in a market\nImage: market area with smoke and people running",
            "patterns": empty_layered_similar_events(),
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(["not valid json"]),
        ):
            result = extraction(state)

        self.assertEqual(result["event"], empty_event())

    def test_empty_evidence_fallback(self) -> None:
        state = make_state()

        with patch("mm_event_agent.nodes.search.search_news", side_effect=RuntimeError("search down")):
            search_result = search(state)

        self.assertEqual(search_result["evidence"], [])
        self.assertEqual(search_result["search_query"], "A bomb exploded in a market")

        fused = fusion({**state, **search_result, "perception_summary": "summary"})
        self.assertEqual(fused["fusion_context"]["evidence"], [])

    def test_stage_a_still_works_with_empty_local_rag(self) -> None:
        llm = RecordingLLM('{"event_type":"Conflict:Attack"}')

        with patch("mm_event_agent.nodes.extraction._get_llm", return_value=llm):
            event_type = extraction_module._stage_a_select_event_type(
                raw_text="A bomb exploded in a market",
                image_side_info="raw_image_desc: smoke near market",
                evidence_items=[],
                patterns=empty_layered_similar_events(),
                event_type_mode="closed_set",
            )

        self.assertEqual(event_type, "Conflict:Attack")
        prompt = "\n".join(llm.prompts)
        self.assertIn("Retrieved text event examples:\n(none)", prompt)
        self.assertIn("Retrieved cross-modal bridge hints:\n(none)", prompt)

    def test_stage_a_injects_formatted_text_examples_when_available(self) -> None:
        llm = RecordingLLM('{"event_type":"Conflict:Attack"}')
        patterns = {
            "text_event_examples": [
                {
                    "id": "ace_000123",
                    "source_dataset": "ACE2005",
                    "modality": "text",
                    "event_type": "Conflict:Attack",
                    "raw_text": "A bomb exploded in a crowded market.",
                    "trigger": {"text": "exploded", "span": {"start": 7, "end": 15}},
                    "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 20, "end": 26}}],
                    "pattern_summary": "Attack pattern with explicit place.",
                    "retrieval_text": "Conflict:Attack exploded market Place",
                }
            ],
            "image_semantic_examples": [],
            "bridge_examples": [],
        }

        with patch("mm_event_agent.nodes.extraction._get_llm", return_value=llm):
            extraction_module._stage_a_select_event_type(
                raw_text="A bomb exploded in a market",
                image_side_info="raw_image_desc: smoke near market",
                evidence_items=[],
                patterns=patterns,
                event_type_mode="closed_set",
            )

        prompt = "\n".join(llm.prompts)
        self.assertIn('"id": "ace_000123"', prompt)
        self.assertIn('"source_dataset": "ACE2005"', prompt)
        self.assertIn('"pattern_summary": "Attack pattern with explicit place."', prompt)
        self.assertIn("Use retrieved text event examples as pattern guidance only.", prompt)

    def test_stage_a_injects_formatted_bridge_examples_when_available(self) -> None:
        llm = RecordingLLM('{"event_type":"Conflict:Attack"}')
        patterns = {
            "text_event_examples": [],
            "image_semantic_examples": [],
            "bridge_examples": [
                {
                    "id": "bridge_attack_instrument_001",
                    "source_dataset": "manual_bridge",
                    "modality": "bridge",
                    "event_type": "Conflict:Attack",
                    "role": "Instrument",
                    "text_cues": ["bomb", "missile"],
                    "visual_cues": ["smoke", "flames"],
                    "note": "Instrument may be indirectly visible through smoke and fire.",
                    "retrieval_text": "Conflict:Attack Instrument bomb smoke flames",
                }
            ],
        }

        with patch("mm_event_agent.nodes.extraction._get_llm", return_value=llm):
            extraction_module._stage_a_select_event_type(
                raw_text="A bomb exploded in a market",
                image_side_info="raw_image_desc: smoke near market",
                evidence_items=[],
                patterns=patterns,
                event_type_mode="closed_set",
            )

        prompt = "\n".join(llm.prompts)
        self.assertIn("Bridge 1", prompt)
        self.assertIn("role: Instrument", prompt)
        self.assertIn("visual_cues: smoke, flames", prompt)
        self.assertIn("Use cross-modal bridge hints only as auxiliary semantic support.", prompt)

    def test_stage_a_still_respects_closed_set_and_transfer_modes(self) -> None:
        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(['{"event_type":"Conflict:Attack"}']),
        ):
            closed_set_result = extraction_module._stage_a_select_event_type(
                raw_text="A bomb exploded in a market",
                image_side_info="raw_image_desc: smoke near market",
                evidence_items=[],
                patterns=empty_layered_similar_events(),
                event_type_mode="closed_set",
            )

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(['{"event_type":"Unsupported"}']),
        ):
            transfer_result = extraction_module._stage_a_select_event_type(
                raw_text="This may not match the ontology",
                image_side_info="raw_image_desc: unclear",
                evidence_items=[],
                patterns=empty_layered_similar_events(),
                event_type_mode="transfer",
            )

        self.assertEqual(closed_set_result, "Conflict:Attack")
        self.assertEqual(transfer_result, "Unsupported")

    def test_stage_a_can_optionally_attach_raw_image(self) -> None:
        llm = RecordingLLM('{"event_type":"Conflict:Attack"}')

        with patch.object(
            extraction_module,
            "settings",
            replace(extraction_module.settings, extraction_stage_a_use_raw_image=True),
        ), patch("mm_event_agent.nodes.extraction._get_llm", return_value=llm):
            result = extraction_module._stage_a_select_event_type(
                raw_text="A bomb exploded in a market",
                image_side_info="raw_image_desc: smoke near market",
                evidence_items=[],
                patterns=empty_layered_similar_events(),
                event_type_mode="closed_set",
                raw_image=b"fake-image-bytes",
            )

        self.assertEqual(result, "Conflict:Attack")
        self.assertTrue(llm.contents)
        self.assertIsInstance(llm.contents[0], list)
        self.assertEqual(llm.contents[0][0]["type"], "text")
        self.assertEqual(llm.contents[0][1]["type"], "image_url")
        self.assertIn("raw_image direct vision is enabled", llm.contents[0][0]["text"])

    def test_stage_b_still_works_with_empty_local_rag(self) -> None:
        llm = RecordingLLM('{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[]}')

        with patch("mm_event_agent.nodes.extraction._get_llm", return_value=llm):
            result = extraction_module._stage_b_extract_text_fields(
                event_type="Conflict:Attack",
                raw_text="A bomb exploded in a market",
                image_side_info="raw_image_desc: smoke near market",
                patterns=empty_layered_similar_events(),
                evidence_items=[],
            )

        self.assertEqual(result["trigger"]["text"], "exploded")
        prompt = "\n".join(llm.prompts)
        self.assertIn("Retrieved text event examples:\n(none)", prompt)
        self.assertIn("Optional bridge hints:\n(none)", prompt)

    def test_stage_b_injects_text_examples_and_bridge_hints(self) -> None:
        llm = RecordingLLM('{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[]}')
        patterns = {
            "text_event_examples": [
                {
                    "id": "ace_000123",
                    "source_dataset": "ACE2005",
                    "modality": "text",
                    "event_type": "Conflict:Attack",
                    "raw_text": "A bomb exploded in a crowded market.",
                    "trigger": {"text": "exploded", "span": {"start": 7, "end": 15}},
                    "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 20, "end": 26}}],
                    "pattern_summary": "Attack pattern with explicit place.",
                    "retrieval_text": "Conflict:Attack exploded market Place",
                }
            ],
            "image_semantic_examples": [],
            "bridge_examples": [
                {
                    "id": "bridge_attack_target_001",
                    "source_dataset": "manual_bridge",
                    "modality": "bridge",
                    "event_type": "Conflict:Attack",
                    "role": "Target",
                    "text_cues": ["civilian"],
                    "visual_cues": ["injured person"],
                    "note": "Victim/Target disambiguation support.",
                    "retrieval_text": "Conflict:Attack Target civilian injured person",
                }
            ],
        }

        with patch("mm_event_agent.nodes.extraction._get_llm", return_value=llm):
            extraction_module._stage_b_extract_text_fields(
                event_type="Conflict:Attack",
                raw_text="A bomb exploded in a market",
                image_side_info="raw_image_desc: smoke near market",
                patterns=patterns,
                evidence_items=[],
            )

        prompt = "\n".join(llm.prompts)
        self.assertIn('"id": "ace_000123"', prompt)
        self.assertIn("Use retrieved text event examples as few-shot pattern guidance.", prompt)
        self.assertIn("Victim/Target disambiguation support.", prompt)
        self.assertIn("Use bridge hints only for limited role disambiguation support.", prompt)

    def test_local_rag_empty_layered_corpora_returns_safe_empty_output(self) -> None:
        state = make_state()

        with patch(
            "mm_event_agent.nodes.rag._retrieve_similar_events",
            return_value=empty_layered_similar_events(),
        ):
            result = rag(state)

        self.assertEqual(
            result["similar_events"],
            {
                "text_event_examples": [],
                "image_semantic_examples": [],
                "bridge_examples": [],
            },
        )

    def test_local_rag_output_structure_is_stable(self) -> None:
        state = make_state()

        with patch(
            "mm_event_agent.nodes.rag._retrieve_similar_events",
            return_value={
                "text_event_examples": [
                    {
                        "id": "ace_000123",
                        "source_dataset": "ACE2005",
                        "modality": "text",
                        "event_type": "Conflict:Attack",
                        "raw_text": "A bomb exploded in a market",
                        "trigger": {"text": "exploded", "span": {"start": 7, "end": 15}},
                        "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
                        "pattern_summary": "Attack reports often name place and instrument.",
                        "retrieval_text": "Conflict:Attack exploded bomb market Place",
                    }
                ],
                "image_semantic_examples": [
                    {
                        "id": "swig_000456",
                        "source_dataset": "SWiG",
                        "modality": "image_semantics",
                        "event_type": "Conflict:Attack",
                        "visual_situation": "explosion aftermath",
                        "image_desc": "smoke near stalls",
                        "image_arguments": [{"role": "Place", "label": "market area"}],
                        "visual_pattern_summary": "Scene-level place labels help image-side argument prediction.",
                        "retrieval_text": "Conflict:Attack smoke market area Place",
                    }
                ],
                "bridge_examples": [
                    {
                        "id": "bridge_attack_place_001",
                        "source_dataset": "manual_bridge",
                        "modality": "bridge",
                        "event_type": "Conflict:Attack",
                        "role": "Place",
                        "text_cues": ["market"],
                        "visual_cues": ["market area"],
                        "note": "Maps text place mentions to image-side scene labels.",
                        "retrieval_text": "Conflict:Attack Place market market area",
                    }
                ],
            },
        ):
            result = rag(state)

        self.assertEqual(
            set(result["similar_events"].keys()),
            {"text_event_examples", "image_semantic_examples", "bridge_examples"},
        )
        self.assertEqual(result["similar_events"]["text_event_examples"][0]["id"], "ace_000123")
        self.assertEqual(result["similar_events"]["text_event_examples"][0]["modality"], "text")
        self.assertEqual(result["similar_events"]["text_event_examples"][0]["trigger"]["text"], "exploded")
        self.assertEqual(result["similar_events"]["text_event_examples"][0]["source_dataset"], "ACE2005")
        self.assertEqual(result["similar_events"]["image_semantic_examples"][0]["source_dataset"], "SWiG")
        self.assertEqual(result["similar_events"]["image_semantic_examples"][0]["modality"], "image_semantics")
        self.assertEqual(result["similar_events"]["image_semantic_examples"][0]["image_arguments"][0]["label"], "market area")
        self.assertEqual(result["similar_events"]["bridge_examples"][0]["role"], "Place")
        self.assertEqual(result["similar_events"]["bridge_examples"][0]["modality"], "bridge")

    def test_layered_retrieval_remains_usable_by_downstream_code(self) -> None:
        state = make_state()
        state["similar_events"] = {
            "text_event_examples": [
                {
                    "id": "ace_000123",
                    "source_dataset": "ACE2005",
                    "modality": "text",
                    "event_type": "Conflict:Attack",
                    "raw_text": "A bomb exploded in a market",
                    "trigger": {"text": "exploded", "span": {"start": 7, "end": 15}},
                    "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
                    "pattern_summary": "Attack reports often name place and instrument.",
                    "retrieval_text": "Conflict:Attack exploded bomb market Place",
                }
            ],
            "image_semantic_examples": [
                {
                    "id": "swig_000456",
                    "source_dataset": "SWiG",
                    "modality": "image_semantics",
                    "event_type": "Conflict:Attack",
                    "visual_situation": "explosion aftermath",
                    "image_desc": "smoke near stalls",
                    "image_arguments": [{"role": "Place", "label": "market area"}],
                    "visual_pattern_summary": "Scene-level place labels help image-side argument prediction.",
                    "retrieval_text": "Conflict:Attack smoke market area Place",
                }
            ],
            "bridge_examples": [
                {
                    "id": "bridge_attack_place_001",
                    "source_dataset": "manual_bridge",
                    "modality": "bridge",
                    "event_type": "Conflict:Attack",
                    "role": "Place",
                    "text_cues": ["market"],
                    "visual_cues": ["market area"],
                    "note": "Maps text place mentions to image-side scene labels.",
                    "retrieval_text": "Conflict:Attack Place market market area",
                }
            ],
        }

        fused = fusion({**state, "perception_summary": "summary"})

        self.assertEqual(
            set(fused["fusion_context"]["patterns"].keys()),
            {"text_event_examples", "image_semantic_examples", "bridge_examples"},
        )
        self.assertEqual(fused["fusion_context"]["patterns"]["text_event_examples"][0]["modality"], "text")
        self.assertEqual(fused["fusion_context"]["patterns"]["bridge_examples"][0]["role"], "Place")

    def test_tavily_adapter_without_api_key_returns_empty_evidence(self) -> None:
        client = TavilySearchClient(api_key="")

        self.assertEqual(client.search("market bombing"), [])

    def test_tavily_adapter_normalizes_successful_output_shape(self) -> None:
        client = TavilySearchClient(api_key="test-key", endpoint="https://example.test/search")

        with patch.object(
            client,
            "_post",
            return_value={
                "results": [
                    {
                        "title": "Market blast latest",
                        "content": "Officials said the blast happened near the market entrance.",
                        "url": "https://example.com/blast",
                        "published_date": "2026-04-12",
                        "score": 0.87,
                    }
                ]
            },
        ):
            results = client.search("market blast")

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0],
            {
                "title": "Market blast latest",
                "snippet": "Officials said the blast happened near the market entrance.",
                "url": "https://example.com/blast",
                "source_type": "search",
                "published_at": "2026-04-12",
                "score": 0.87,
            },
        )

    def test_search_node_returns_schema_valid_evidence_items(self) -> None:
        state = make_state()

        with patch(
            "mm_event_agent.nodes.search.search_news",
            return_value=[
                {
                    "title": "Market attack update",
                    "snippet": "Police confirmed an explosion at the market.",
                    "url": "https://example.com/news/market-attack",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.91,
                }
            ],
        ):
            result = search(state)

        self.assertEqual(result["search_query"], "A bomb exploded in a market")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["source_type"], "search")
        self.assertIsInstance(result["evidence"][0]["score"], float)
        self.assertEqual(result["evidence"][0]["published_at"], "2026-04-12")

    def test_search_node_benchmark_mode_skips_external_search(self) -> None:
        state = make_state()
        state["run_mode"] = "benchmark"
        state["effective_search_enabled"] = False

        with patch("mm_event_agent.nodes.search.search_news") as mock_search_news:
            result = search(state)

        mock_search_news.assert_not_called()
        self.assertEqual(result["search_query"], "A bomb exploded in a market")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["run_mode"], "benchmark")
        self.assertFalse(result["effective_search_enabled"])

    def test_search_node_open_world_mode_respects_disabled_search_flag(self) -> None:
        state = make_state()
        state["run_mode"] = "open_world"
        state["effective_search_enabled"] = False

        with patch("mm_event_agent.nodes.search.search_news") as mock_search_news:
            result = search(state)

        mock_search_news.assert_not_called()
        self.assertEqual(result["search_query"], "A bomb exploded in a market")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["run_mode"], "open_world")
        self.assertFalse(result["effective_search_enabled"])

    def test_tavily_adapter_failure_does_not_crash_pipeline(self) -> None:
        client = TavilySearchClient(api_key="test-key", endpoint="https://example.test/search")

        with patch.object(client, "_post", side_effect=RuntimeError("boom")):
            self.assertEqual(client.search("market blast"), [])

        with patch("mm_event_agent.nodes.search.search_news", side_effect=RuntimeError("boom")):
            result = search(make_state())

        self.assertEqual(result["evidence"], [])

    def test_search_node_top_k_filtering_keeps_best_ranked_evidence(self) -> None:
        state = make_state()

        with patch("mm_event_agent.nodes.search.settings", replace(runtime_config.settings, search_top_k=2)), patch(
            "mm_event_agent.nodes.search.search_news",
            return_value=[
                {
                    "title": "Market blast kills shoppers",
                    "snippet": "A bomb exploded in a crowded market and witnesses saw smoke.",
                    "url": "https://example.com/1",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.92,
                },
                {
                    "title": "Officials probe market explosion",
                    "snippet": "Investigators said the market explosion injured civilians.",
                    "url": "https://example.com/2",
                    "source_type": "search",
                    "published_at": "2026-04-11",
                    "score": 0.83,
                },
                {
                    "title": "Stocks fall after global trade jitters",
                    "snippet": "Markets fell as investors reacted to tariff risks.",
                    "url": "https://example.com/3",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.95,
                },
            ],
        ):
            result = search(state)

        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual([item["url"] for item in result["evidence"]], ["https://example.com/1", "https://example.com/2"])

    def test_runtime_config_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = runtime_config.load_settings()

        self.assertEqual(settings.event_type_mode, "closed_set")
        self.assertFalse(settings.debug)
        self.assertEqual(settings.log_level, "INFO")
        self.assertEqual(settings.run_mode, "benchmark")
        self.assertFalse(settings.enable_search)
        self.assertFalse(settings.effective_search_enabled)
        self.assertEqual(settings.openai_model, "gpt-4o-mini")
        self.assertEqual(settings.tavily_endpoint, "https://api.tavily.com/search")
        self.assertEqual(settings.search_top_k, 3)
        self.assertEqual(settings.florence2_model_id, "microsoft/Florence-2-base-ft")

    def test_runtime_config_override_via_env_vars(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MM_EVENT_TYPE_MODE": "transfer",
                "MM_EVENT_DEBUG": "true",
                "MM_AGENT_LOG_LEVEL": "DEBUG",
                "MM_EVENT_RUN_MODE": "open_world",
                "MM_EVENT_ENABLE_SEARCH": "true",
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_MODEL": "gpt-test",
                "OPENAI_BASE_URL": "https://example.test/v1",
                "OPENAI_TIMEOUT_SECONDS": "12",
                "TAVILY_API_KEY": "tvly-test",
                "MM_EVENT_TAVILY_ENDPOINT": "https://search.example.test",
                "MM_EVENT_SEARCH_TIMEOUT_SECONDS": "5",
                "MM_EVENT_SEARCH_MAX_RESULTS": "7",
                "MM_EVENT_SEARCH_TOP_K": "2",
                "MM_EVENT_SEARCH_MIN_RELEVANCE": "0.4",
                "FLORENCE2_MODEL_ID": "florence-test",
                "FLORENCE2_TASK": "<CAPTION>",
                "FLORENCE2_DEVICE": "cpu",
                "FLORENCE2_LOCAL_ENDPOINT": "http://localhost:9000",
                "FLORENCE2_LOCAL_TIMEOUT_SECONDS": "15",
            },
            clear=True,
        ):
            settings = runtime_config.load_settings()

        self.assertEqual(settings.event_type_mode, "transfer")
        self.assertTrue(settings.debug)
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertEqual(settings.run_mode, "open_world")
        self.assertTrue(settings.enable_search)
        self.assertTrue(settings.effective_search_enabled)
        self.assertEqual(settings.openai_api_key, "sk-test")
        self.assertEqual(settings.openai_model, "gpt-test")
        self.assertEqual(settings.openai_base_url, "https://example.test/v1")
        self.assertEqual(settings.openai_timeout_seconds, 12.0)
        self.assertEqual(settings.tavily_api_key, "tvly-test")
        self.assertEqual(settings.tavily_endpoint, "https://search.example.test")
        self.assertEqual(settings.search_max_results, 7)
        self.assertEqual(settings.search_top_k, 2)
        self.assertEqual(settings.search_min_relevance, 0.4)
        self.assertEqual(settings.florence2_model_id, "florence-test")
        self.assertEqual(settings.florence2_task, "<CAPTION>")
        self.assertEqual(settings.florence2_device, "cpu")
        self.assertEqual(settings.florence2_local_endpoint, "http://localhost:9000")
        self.assertEqual(settings.florence2_local_timeout_seconds, 15.0)

    def test_missing_optional_config_remains_safe(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "gpt-test",
            },
            clear=True,
        ):
            settings = runtime_config.load_settings()

        self.assertEqual(settings.openai_api_key, "")
        self.assertEqual(settings.tavily_api_key, "")
        self.assertEqual(settings.florence2_local_endpoint, "")
        with patch("mm_event_agent.search.tavily_client.settings", settings):
            client = TavilySearchClient()
            self.assertFalse(client.configured)
            self.assertEqual(client.search("market bombing"), [])

    def test_runtime_config_benchmark_mode_forces_search_off(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MM_EVENT_RUN_MODE": "benchmark",
                "MM_EVENT_ENABLE_SEARCH": "true",
            },
            clear=True,
        ):
            settings = runtime_config.load_settings()

        self.assertEqual(settings.run_mode, "benchmark")
        self.assertTrue(settings.enable_search)
        self.assertFalse(settings.effective_search_enabled)

    def test_runtime_config_open_world_mode_respects_search_flag(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MM_EVENT_RUN_MODE": "open_world",
                "MM_EVENT_ENABLE_SEARCH": "false",
            },
            clear=True,
        ):
            disabled = runtime_config.load_settings()
        with patch.dict(
            os.environ,
            {
                "MM_EVENT_RUN_MODE": "open_world",
                "MM_EVENT_ENABLE_SEARCH": "true",
            },
            clear=True,
        ):
            enabled = runtime_config.load_settings()

        self.assertEqual(disabled.run_mode, "open_world")
        self.assertFalse(disabled.effective_search_enabled)
        self.assertEqual(enabled.run_mode, "open_world")
        self.assertTrue(enabled.effective_search_enabled)

    def test_search_node_drops_obviously_irrelevant_results_when_possible(self) -> None:
        state = make_state()

        with patch(
            "mm_event_agent.nodes.search.search_news",
            return_value=[
                {
                    "title": "Celebrity wedding stuns fans",
                    "snippet": "A singer married an actor in a beach ceremony.",
                    "url": "https://example.com/wedding",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.78,
                },
                {
                    "title": "Market bombing investigated",
                    "snippet": "Police said a bomb exploded in a market and launched an investigation.",
                    "url": "https://example.com/attack",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.62,
                },
            ],
        ):
            result = search(state)

        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["url"], "https://example.com/attack")

    def test_search_node_empty_results_remain_safe_after_filtering(self) -> None:
        state = make_state()

        with patch("mm_event_agent.nodes.search.search_news", return_value=[]):
            result = search(state)

        self.assertEqual(result["search_query"], "A bomb exploded in a market")
        self.assertEqual(result["evidence"], [])

    def test_fusion_audit_summary_includes_mode_info_with_empty_evidence(self) -> None:
        state = make_state()
        state["run_mode"] = "benchmark"
        state["effective_search_enabled"] = False
        state["perception_summary"] = "summary"
        state["prompt_trace"] = []
        state["stage_outputs"] = {}
        state["evidence"] = []

        result = fusion(state)

        summary = result["stage_outputs"]["fusion_context_summary"]
        self.assertEqual(summary["run_mode"], "benchmark")
        self.assertFalse(summary["effective_search_enabled"])
        self.assertEqual(summary["evidence_count"], 0)
        self.assertEqual(summary["evidence_summary"]["count"], 0)
        self.assertEqual(result["fusion_context"]["evidence"], [])

    def test_filtered_evidence_remains_schema_valid(self) -> None:
        state = make_state()

        with patch(
            "mm_event_agent.nodes.search.search_news",
            return_value=[
                {
                    "title": "Market bombing latest",
                    "snippet": "Authorities confirmed the bomb exploded inside the market.",
                    "url": "https://example.com/latest",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.88,
                },
                {
                    "title": "Low quality item",
                    "snippet": "noise",
                    "url": "https://example.com/noise",
                    "source_type": "search",
                    "published_at": "2020-01-01",
                    "score": 0.01,
                },
            ],
        ):
            result = search(state)

        self.assertTrue(result["evidence"])
        for item in result["evidence"]:
            self.assertEqual(set(item.keys()), {"title", "snippet", "url", "source_type", "published_at", "score"})
            self.assertEqual(item["source_type"], "search")
            self.assertIsInstance(item["score"], float)

    def test_evidence_source_summary_text_only_case(self) -> None:
        summary = summarize_evidence_sources(
            raw_event={
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
                "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
                "image_arguments": [],
            },
            raw_text="A bomb exploded in a market",
            raw_image_desc="",
            grounding_results=[],
            evidence=[],
        )

        self.assertEqual(
            summary,
            {
                "text_support": True,
                "image_support": False,
                "grounding_support": False,
                "external_evidence_support": False,
            },
        )

    def test_evidence_source_summary_text_and_grounding_case(self) -> None:
        summary = summarize_evidence_sources(
            raw_event={
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
                "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
                "image_arguments": [
                    {"role": "Place", "label": "market area", "bbox": [1.0, 2.0, 11.0, 12.0], "grounding_status": "grounded"}
                ],
            },
            raw_text="A bomb exploded in a market",
            raw_image_desc="market area with smoke",
            grounding_results=[
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": [1.0, 2.0, 11.0, 12.0],
                    "score": 0.88,
                    "grounding_status": "grounded",
                }
            ],
            evidence=[],
        )

        self.assertTrue(summary["text_support"])
        self.assertTrue(summary["image_support"])
        self.assertTrue(summary["grounding_support"])
        self.assertFalse(summary["external_evidence_support"])

    def test_evidence_source_summary_text_and_external_evidence_case(self) -> None:
        snapshot = build_evidence_source_snapshot(
            raw_event={
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
                "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
                "image_arguments": [],
            },
            raw_text="A bomb exploded in a market",
            evidence=[
                {
                    "title": "Market explosion latest",
                    "snippet": "Police said a bomb exploded in a market district.",
                    "url": "https://example.com/evidence",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.9,
                }
            ],
        )

        self.assertEqual(snapshot["event_type"], "Conflict:Attack")
        self.assertTrue(snapshot["text_support"])
        self.assertFalse(snapshot["image_support"])
        self.assertFalse(snapshot["grounding_support"])
        self.assertTrue(snapshot["external_evidence_support"])
        self.assertEqual(snapshot["final_event"]["event_type"], "Conflict:Attack")

    def test_evidence_source_summary_no_support_empty_event_case(self) -> None:
        summary = summarize_evidence_sources(
            raw_event=empty_event(),
            raw_text="A bomb exploded in a market",
            raw_image_desc="market area with smoke",
            grounding_results=[
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": [1.0, 2.0, 11.0, 12.0],
                    "score": 0.88,
                    "grounding_status": "grounded",
                }
            ],
            evidence=[
                {
                    "title": "Market explosion latest",
                    "snippet": "Police said a bomb exploded in a market district.",
                    "url": "https://example.com/evidence",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.9,
                }
            ],
        )

        self.assertEqual(
            summary,
            {
                "text_support": False,
                "image_support": False,
                "grounding_support": False,
                "external_evidence_support": False,
            },
        )

    def test_verifier_logs_evidence_source_summary_fields(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "market area with smoke",
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [
                {
                    "title": "Market explosion latest",
                    "snippet": "Police said a bomb exploded in a market district.",
                    "url": "https://example.com/evidence",
                    "source_type": "search",
                    "published_at": "2026-04-12",
                    "score": 0.9,
                }
            ],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
            "image_arguments": [],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.95,"reason":"supported"}']
            ),
        ), patch("mm_event_agent.nodes.verifier.log_node_event") as mock_log:
            verifier(state)

        _, kwargs = mock_log.call_args
        self.assertIn("text_support", kwargs)
        self.assertIn("image_support", kwargs)
        self.assertIn("grounding_support", kwargs)
        self.assertIn("external_evidence_support", kwargs)
        self.assertTrue(kwargs["text_support"])
        self.assertTrue(kwargs["external_evidence_support"])

    def test_raw_image_field_does_not_break_image_desc_fallback(self) -> None:
        state = make_state()
        state["raw_image"] = b"fake-image-bytes"

        fused = fusion({**state, "perception_summary": "summary"})

        self.assertEqual(fused["fusion_context"]["raw_image_desc"], state["image_desc"])

    def test_initial_state_contract_carries_raw_text_plus_raw_image(self) -> None:
        state = make_state()

        self.assertEqual(state["text"], "A bomb exploded in a market")
        self.assertTrue(state["raw_image"])
        self.assertEqual(state["image_desc"], "market area with smoke and people running")

    def test_verifier_flags_invalid_text_span(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 1}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.8,"reason":"looks good"}']
            ),
        ):
            result = verifier(state)

        self.assertFalse(result["verified"])
        self.assertIn("trigger text/span mismatch", result["issues"])
        self.assertTrue(any(x["field_path"] == "trigger.span" for x in result["verifier_diagnostics"]))

    def test_verifier_flags_unnormalized_broad_text_arguments(self) -> None:
        state = make_state()
        state["text"] = "Two Chicago police officers arrested the suspect"
        state["tokens"] = ["Two", "Chicago", "police", "officers", "arrested", "the", "suspect"]
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "",
            "perception_summary": "",
            "patterns": [],
            "evidence": [],
            "text_tokens": state["tokens"],
        }
        state["event"] = {
            "event_type": "Justice:Arrest-Jail",
            "trigger": {"text": "arrested", "modality": "text", "span": {"start": 4, "end": 5}},
            "text_arguments": [
                {"role": "Agent", "text": "Two Chicago police officers", "span": {"start": 0, "end": 4}}
            ],
            "image_arguments": [],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"NO","issues":["unnormalized mention"],"confidence":0.7,"reason":"broad argument span"}']
            ),
        ):
            result = verifier(state)

        issue_types = {item["issue_type"] for item in result["verifier_diagnostics"]}
        self.assertIn("contains_quantity", issue_types)
        self.assertIn("unnormalized_head_word", issue_types)

    def test_verifier_keeps_structured_diagnostics(self) -> None:
        state = make_state()
        state["event"] = empty_event()

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                [
                    '{"verdict":"NO","issues":["unsupported trigger"],"confidence":0.4,"reason":"bad trigger",'
                    '"diagnostics":[{"field_path":"trigger","issue_type":"unsupported","suggested_action":"drop_or_rebuild"}]}'
                ]
            ),
        ):
            result = verifier(state)

        self.assertFalse(result["verified"])
        self.assertTrue(any(x["field_path"] == "trigger" for x in result["verifier_diagnostics"]))

    def test_verifier_rejects_invalid_role_for_event_type(self) -> None:
        state = make_state()
        state["event"] = {
            "event_type": "Contact:Meet",
            "trigger": None,
            "text_arguments": [{"role": "Instrument", "text": "bomb", "span": None}],
            "image_arguments": [],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.8,"reason":"looks good"}']
            ),
        ):
            result = verifier(state)

        self.assertFalse(result["verified"])
        self.assertIn("invalid text role at index 0", result["issues"])
        self.assertTrue(any(x["field_path"] == "text_arguments[0].role" for x in result["verifier_diagnostics"]))

    def test_verifier_flags_semantically_wrong_role_under_valid_event_type(self) -> None:
        state = make_state()
        state["text"] = "Police arrested the suspect outside the station"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "",
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Justice:Arrest-Jail",
            "trigger": {"text": "arrested", "modality": "text", "span": {"start": 1, "end": 2}},
            "text_arguments": [
                {"role": "Agent", "text": "suspect", "span": {"start": 3, "end": 4}},
                {"role": "Person", "text": "Police", "span": {"start": 0, "end": 1}},
            ],
            "image_arguments": [],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                [
                    '{"verdict":"NO","issues":["semantic role confusion: Agent vs Person"],"confidence":0.89,"reason":"the detainee and authority are swapped",'
                    '"diagnostics":[{"field_path":"text_arguments[0].role","issue_type":"semantic_role_confusion","suggested_action":"replace_or_drop"},{"field_path":"text_arguments[1].role","issue_type":"semantic_role_confusion","suggested_action":"replace_or_drop"}]}'
                ]
            ),
        ):
            result = verifier(state)

        self.assertFalse(result["verified"])
        self.assertIn("semantic role confusion: Agent vs Person", result["issues"])
        self.assertTrue(any(x["issue_type"] == "semantic_role_confusion" for x in result["verifier_diagnostics"]))

    def test_verifier_flags_trigger_inconsistent_with_selected_event_type(self) -> None:
        state = make_state()
        state["text"] = "Leaders met in Geneva for talks"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "",
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Life:Die",
            "trigger": {"text": "met", "modality": "text", "span": {"start": 1, "end": 2}},
            "text_arguments": [{"role": "Place", "text": "Geneva", "span": {"start": 3, "end": 4}}],
            "image_arguments": [],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                [
                    '{"verdict":"NO","issues":["trigger meaning inconsistent with event_type"],"confidence":0.92,"reason":"meeting trigger does not match Life:Die semantics",'
                    '"diagnostics":[{"field_path":"trigger","issue_type":"semantic_trigger_mismatch","suggested_action":"set_supported_event_type"}]}'
                ]
            ),
        ):
            result = verifier(state)

        self.assertFalse(result["verified"])
        self.assertIn("trigger meaning inconsistent with event_type", result["issues"])
        self.assertTrue(any(x["issue_type"] == "semantic_trigger_mismatch" for x in result["verifier_diagnostics"]))

    def test_verifier_accepts_grounded_image_arguments(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": [1.0, 2.0, 11.0, 12.0], "grounding_status": "grounded"}
            ],
        }
        state["grounding_results"] = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [1.0, 2.0, 11.0, 12.0],
                "score": 0.88,
                "grounding_status": "grounded",
            }
        ]

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.95,"reason":"grounded image support is consistent"}']
            ),
        ):
            result = verifier(state)

        self.assertTrue(result["verified"])
        self.assertEqual(result["issues"], [])

    def test_grounding_results_strengthen_support_without_forcing_failure(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": [1.0, 2.0, 11.0, 12.0], "grounding_status": "grounded"}
            ],
        }
        state["grounding_results"] = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [1.0, 2.0, 11.0, 12.0],
                "score": 0.88,
                "grounding_status": "grounded",
            }
        ]

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.97,"reason":"grounding strengthens image support"}']
            ),
        ), patch("mm_event_agent.nodes.verifier.log_node_event") as mock_log:
            result = verifier(state)

        self.assertTrue(result["verified"])
        self.assertTrue(mock_log.called)
        _, kwargs = mock_log.call_args
        self.assertEqual(kwargs["grounded_support_count"], 1)

    def test_failed_grounding_does_not_invalidate_otherwise_acceptable_unresolved_image_arguments(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
            ],
        }
        state["grounding_results"] = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": None,
                "score": None,
                "grounding_status": "failed",
            }
        ]

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.84,"reason":"failed grounding is non-fatal"}']
            ),
        ):
            result = verifier(state)

        self.assertTrue(result["verified"])
        self.assertEqual(result["issues"], [])

    def test_verifier_accepts_valid_text_only_event_with_empty_image_arguments(self) -> None:
        state = make_state()
        state["raw_image"] = None
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "",
            "perception_summary": "Text: A bomb exploded in a market\nImage: ",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(['{"verdict":"YES","issues":[],"confidence":0.9,"reason":"valid text-only event"}']),
        ):
            result = verifier(state)

        self.assertTrue(result["verified"])
        self.assertEqual(result["issues"], [])

    def test_verifier_accepts_partial_multimodal_overlap_without_one_to_one_alignment(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Instrument", "text": "bomb", "span": {"start": 1, "end": 2}}],
            "image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(['{"verdict":"YES","issues":[],"confidence":0.88,"reason":"partial modality overlap is acceptable"}']),
        ):
            result = verifier(state)

        self.assertTrue(result["verified"])
        self.assertEqual(result["issues"], [])

    def test_verifier_flags_generic_weak_place_image_argument(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Justice:Arrest-Jail",
            "trigger": {"text": "arrested", "modality": "text", "span": {"start": 1, "end": 2}},
            "text_arguments": [{"role": "Person", "text": "suspect", "span": {"start": 3, "end": 4}}],
            "image_arguments": [{"role": "Place", "label": "street", "bbox": [1, 2, 3, 4], "grounding_status": "grounded"}],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(['{"verdict":"NO","issues":["weak place"],"confidence":0.6,"reason":"generic backdrop place"}']),
        ):
            result = verifier(state)

        self.assertTrue(any(item["issue_type"] == "generic_weak_place" for item in result["verifier_diagnostics"]))

    def test_verifier_flags_image_arguments_in_text_only_run(self) -> None:
        state = make_state()
        state["raw_image"] = None
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "",
            "perception_summary": "Text: A bomb exploded in a market\nImage: ",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
        }

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(['{"verdict":"YES","issues":[],"confidence":0.8,"reason":"looks good"}']),
        ):
            result = verifier(state)

        self.assertFalse(result["verified"])
        self.assertIn("image arguments present without usable image evidence in a text-only run", result["issues"])
        self.assertTrue(any(x["issue_type"] == "missing_image_evidence" for x in result["verifier_diagnostics"]))

    def test_verifier_can_flag_unresolved_argument_when_grounded_match_exists(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
            ],
        }
        state["grounding_results"] = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [1.0, 2.0, 11.0, 12.0],
                "score": 0.9,
                "grounding_status": "grounded",
            }
        ]

        with patch(
            "mm_event_agent.nodes.verifier._get_llm",
            return_value=FakeLLM(
                ['{"verdict":"YES","issues":[],"confidence":0.8,"reason":"looks okay"}']
            ),
        ):
            result = verifier(state)

        self.assertFalse(result["verified"])
        self.assertIn("grounding result available but image argument remains unresolved at index 0", result["issues"])
        self.assertTrue(
            any(x["issue_type"] == "grounding_result_not_applied" for x in result["verifier_diagnostics"])
        )

    def test_find_all_text_occurrences_returns_all_exact_matches(self) -> None:
        self.assertEqual(
            find_all_text_occurrences("market north market south", "market"),
            [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
        )

    def test_find_text_span_returns_none_when_repeated_match_is_ambiguous(self) -> None:
        self.assertIsNone(find_text_span("market north market south", "market"))

    def test_choose_best_span_uses_anchor_context_for_repeated_matches(self) -> None:
        occurrences = find_all_text_occurrences("market north market south", "market")
        self.assertEqual(
            choose_best_span(occurrences, anchor_spans=[{"start": 3, "end": 4}]),
            {"start": 2, "end": 3},
        )

    def test_attach_text_spans_drops_unaligned_text_grounded_fields(self) -> None:
        event = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "detonated", "modality": "text", "span": None},
            "text_arguments": [
                {"role": "Place", "text": "market", "span": None},
                {"role": "Instrument", "text": "pipe bomb", "span": None},
            ],
            "image_arguments": [{"role": "Place", "label": "market area", "bbox": [0, 1, 2, 3], "grounding_status": "grounded"}],
        }

        aligned = attach_text_spans(event, "A bomb exploded in a market")

        self.assertIsNone(aligned["trigger"])
        self.assertEqual(
            aligned["text_arguments"],
            [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
        )
        self.assertEqual(
            aligned["image_arguments"],
            [{"role": "Place", "label": "market area", "bbox": [0, 1, 2, 3], "grounding_status": "grounded"}],
        )

    def test_align_text_grounded_event_realigns_mismatched_spans_when_unique(self) -> None:
        aligned, issues, diagnostics = align_text_grounded_event(
            {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 1}},
                "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 0, "end": 1}}],
                "image_arguments": [],
            },
            "A bomb exploded in a market",
        )

        self.assertEqual(aligned["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(aligned["text_arguments"][0]["span"], {"start": 5, "end": 6})
        self.assertEqual(issues, [])
        self.assertEqual(diagnostics, [])

    def test_align_text_grounded_event_shrinks_a_man_to_head_word(self) -> None:
        aligned, issues, diagnostics = align_text_grounded_event(
            {
                "event_type": "Justice:Arrest-Jail",
                "trigger": None,
                "text_arguments": [{"role": "Person", "text": "a man", "span": {"start": 0, "end": 2}}],
                "image_arguments": [],
            },
            "a man was arrested",
            token_sequence=["a", "man", "was", "arrested"],
        )

        self.assertEqual(aligned["text_arguments"][0]["text"], "man")
        self.assertEqual(aligned["text_arguments"][0]["span"], {"start": 1, "end": 2})
        self.assertEqual(issues, [])
        self.assertEqual(diagnostics, [])

    def test_align_text_grounded_event_shrinks_two_chicago_police_officers_to_head_word(self) -> None:
        aligned, _, _ = align_text_grounded_event(
            {
                "event_type": "Justice:Arrest-Jail",
                "trigger": None,
                "text_arguments": [
                    {"role": "Agent", "text": "Two Chicago police officers", "span": {"start": 0, "end": 4}}
                ],
                "image_arguments": [],
            },
            "Two Chicago police officers arrested him",
            token_sequence=["Two", "Chicago", "police", "officers", "arrested", "him"],
        )

        self.assertEqual(aligned["text_arguments"][0]["text"], "officers")
        self.assertEqual(aligned["text_arguments"][0]["span"], {"start": 3, "end": 4})

    def test_align_text_grounded_event_normalizes_span_field_order(self) -> None:
        aligned, _, _ = align_text_grounded_event(
            {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": {"end": 3, "start": 2}},
                "text_arguments": [{"role": "Place", "text": "market", "span": {"end": 6, "start": 5}}],
                "image_arguments": [],
            },
            "A bomb exploded in a market",
        )

        self.assertEqual(aligned["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(list(aligned["trigger"]["span"].keys()), ["start", "end"])
        self.assertEqual(aligned["text_arguments"][0]["span"], {"start": 5, "end": 6})
        self.assertEqual(list(aligned["text_arguments"][0]["span"].keys()), ["start", "end"])

    def test_align_text_grounded_event_drops_ambiguous_or_missing_text_fields(self) -> None:
        aligned, issues, diagnostics = align_text_grounded_event(
            {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "market", "modality": "text", "span": {"start": 1, "end": 2}},
                "text_arguments": [{"role": "Place", "text": "market", "span": None}],
                "image_arguments": [],
            },
            "market north market south",
        )

        self.assertIsNone(aligned["trigger"])
        self.assertEqual(aligned["text_arguments"], [])
        self.assertEqual(len(issues), 2)
        self.assertTrue(any(x["field_path"] == "trigger.span" for x in diagnostics))
        self.assertTrue(any(x["field_path"] == "text_arguments[0].span" for x in diagnostics))

    def test_strict_text_grounding_removes_null_span_text_fields(self) -> None:
        grounded = enforce_strict_text_grounding(
            {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": None},
                "text_arguments": [
                    {"role": "Place", "text": "market", "span": None},
                    {"role": "Instrument", "text": "pipe bomb", "span": None},
                ],
                "image_arguments": [],
            },
            "A bomb exploded in a market",
        )

        self.assertEqual(grounded["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(grounded["text_arguments"], [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}])

    def test_repair_drops_text_argument_when_exact_alignment_fails(self) -> None:
        state = make_state()
        state["text"] = "A bomb exploded in a market"
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [],
        }
        state["verifier_diagnostics"] = [
            {
                "field_path": "text_arguments[0].text",
                "issue_type": "span_mismatch",
                "suggested_action": "realign_or_drop",
            }
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack","trigger":{"text":"exploded","modality":"text","span":null},'
                    '"text_arguments":[{"role":"Place","text":"downtown","span":null}],"image_arguments":[]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["text_arguments"], [])

    def test_validate_event_normalizes_legacy_list_span(self) -> None:
        from mm_event_agent.schemas import validate_event

        normalized = validate_event(
            {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": [2, 3]},
                "text_arguments": [{"role": "Place", "text": "market", "span": [5, 6]}],
                "image_arguments": [],
            }
        )

        self.assertEqual(normalized["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(normalized["text_arguments"][0]["span"], {"start": 5, "end": 6})

    def test_repair_only_trigger_span_diagnosed_preserves_non_trigger_fields(self) -> None:
        state = make_state()
        state["text"] = "A bomb exploded in a market"
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 1}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
        }
        state["verifier_diagnostics"] = [
            {"field_path": "trigger.span", "issue_type": "span_mismatch", "suggested_action": "realign_or_drop"}
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack","trigger":{"text":"exploded","modality":"text","span":null},'
                    '"text_arguments":[{"role":"Place","text":"CHANGED","span":null}],"image_arguments":[{"role":"Place","label":"changed","bbox":null,"grounding_status":"unresolved"}]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])
        self.assertEqual(result["event"]["text_arguments"], state["event"]["text_arguments"])
        self.assertEqual(result["event"]["image_arguments"], state["event"]["image_arguments"])

    def test_repair_only_one_text_argument_span_diagnosed_preserves_other_arguments(self) -> None:
        state = make_state()
        state["text"] = "A civilian died from shrapnel in the market"
        state["event"] = {
            "event_type": "Life:Die",
            "trigger": {"text": "died", "modality": "text", "span": {"start": 2, "end": 3}},
            "text_arguments": [
                {"role": "Victim", "text": "civilian", "span": {"start": 0, "end": 1}},
                {"role": "Instrument", "text": "shrapnel", "span": {"start": 4, "end": 5}},
            ],
            "image_arguments": [],
        }
        state["verifier_diagnostics"] = [
            {
                "field_path": "text_arguments[0].span",
                "issue_type": "span_mismatch",
                "suggested_action": "realign_or_drop",
            }
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Life:Die","trigger":{"text":"changed","modality":"text","span":null},'
                    '"text_arguments":[{"role":"Victim","text":"civilian","span":null},{"role":"Instrument","text":"changed","span":null}],"image_arguments":[]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["text_arguments"][0]["span"], {"start": 1, "end": 2})
        self.assertEqual(result["event"]["text_arguments"][1]["role"], state["event"]["text_arguments"][1]["role"])
        self.assertEqual(result["event"]["text_arguments"][1]["text"], state["event"]["text_arguments"][1]["text"])
        self.assertEqual(result["event"]["text_arguments"][1]["span"], {"start": 4, "end": 5})
        self.assertEqual(result["event"]["trigger"], state["event"]["trigger"])
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])

    def test_repair_only_one_image_bbox_diagnosed_preserves_unrelated_fields(self) -> None:
        state = make_state()
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"},
                {"role": "Target", "label": "market stall", "bbox": [2, 2, 3, 3], "grounding_status": "grounded"},
            ],
        }
        state["verifier_diagnostics"] = [
            {
                "field_path": "image_arguments[1].bbox",
                "issue_type": "invalid_bbox_format",
                "suggested_action": "fix_or_drop",
            }
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Movement:Transport","trigger":{"text":"exploded","modality":"text","span":null},'
                    '"text_arguments":[{"role":"Destination","text":"changed","span":null}],"image_arguments":[{"role":"Origin","label":"changed","bbox":null,"grounding_status":"unresolved"},{"role":"Target","label":"market stall","bbox":[10,11,12,13],"grounding_status":"grounded"}]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["image_arguments"][1]["bbox"], [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(result["event"]["image_arguments"][0], state["event"]["image_arguments"][0])
        self.assertEqual(result["event"]["text_arguments"], state["event"]["text_arguments"])
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])

    def test_repair_prompt_includes_field_local_repair_plan(self) -> None:
        state = make_state()
        state["text"] = "A bomb exploded in a market"
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 1}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
        }
        state["verifier_diagnostics"] = [
            {"field_path": "trigger.span", "issue_type": "span_mismatch", "suggested_action": "realign_or_drop"},
            {
                "field_path": "image_arguments[0].label",
                "issue_type": "missing_or_empty",
                "suggested_action": "fill_or_drop",
            },
        ]

        repair_llm = RecordingLLM(
            '{"event_type":"Conflict:Attack","trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}],"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}'
        )

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=repair_llm,
        ):
            repair(state)

        self.assertTrue(repair_llm.prompts)
        prompt = repair_llm.prompts[0]
        self.assertIn("Repair plan:", prompt)
        self.assertIn("field_path: trigger.span; issue_type: span_mismatch; suggested_action: realign_or_drop", prompt)
        self.assertIn(
            "field_path: image_arguments[0].label; issue_type: missing_or_empty; suggested_action: fill_or_drop",
            prompt,
        )
        self.assertIn("Modify ONLY the diagnosed fields listed in the repair plan.", prompt)
        self.assertIn("Preserve all other fields unchanged.", prompt)

    def test_grounding_result_not_applied_leads_to_targeted_image_argument_repair(self) -> None:
        state = make_state()
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
            ],
        }
        state["grounding_results"] = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [5.0, 6.0, 20.0, 22.0],
                "score": 0.8,
                "grounding_status": "grounded",
            }
        ]
        state["verifier_reason"] = "grounding result exists but was not applied"
        state["verifier_diagnostics"] = [
            {
                "field_path": "image_arguments[0].grounding_status",
                "issue_type": "grounding_result_not_applied",
                "suggested_action": "upgrade_from_grounding",
            }
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack","trigger":null,"text_arguments":[{"role":"Place","text":"market","span":null}],"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["image_arguments"][0]["bbox"], [5.0, 6.0, 20.0, 22.0])
        self.assertEqual(result["event"]["image_arguments"][0]["grounding_status"], "grounded")
        self.assertEqual(result["event"]["text_arguments"], state["event"]["text_arguments"])
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])

    def test_grounded_bbox_is_preserved_during_unrelated_repairs(self) -> None:
        state = make_state()
        state["text"] = "A bomb exploded in a market"
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 1}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": [5.0, 6.0, 20.0, 22.0], "grounding_status": "grounded"}
            ],
        }
        state["grounding_results"] = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [5.0, 6.0, 20.0, 22.0],
                "score": 0.8,
                "grounding_status": "grounded",
            }
        ]
        state["verifier_reason"] = "trigger span mismatch only"
        state["verifier_diagnostics"] = [
            {"field_path": "trigger.span", "issue_type": "span_mismatch", "suggested_action": "realign_or_drop"}
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack","trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"changed","span":null}],"image_arguments":[{"role":"Place","label":"changed","bbox":null,"grounding_status":"unresolved"}]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(result["event"]["image_arguments"][0]["bbox"], [5.0, 6.0, 20.0, 22.0])
        self.assertEqual(result["event"]["image_arguments"][0]["grounding_status"], "grounded")
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])

    def test_failed_grounding_does_not_force_dropping_otherwise_acceptable_unresolved_image_argument(self) -> None:
        state = make_state()
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
            ],
        }
        state["grounding_results"] = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": None,
                "score": None,
                "grounding_status": "failed",
            }
        ]
        state["verifier_reason"] = "grounding failed but unresolved image argument is still acceptable"
        state["verifier_diagnostics"] = [
            {
                "field_path": "text_arguments[0].span",
                "issue_type": "span_mismatch",
                "suggested_action": "realign_or_drop",
            }
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack","trigger":null,"text_arguments":[{"role":"Place","text":"market","span":null}],"image_arguments":[]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(
            result["event"]["image_arguments"][0],
            {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"},
        )
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])

    def test_repair_does_not_reintroduce_image_arguments_in_text_only_mode(self) -> None:
        state = make_state()
        state["raw_image"] = None
        state["image_desc"] = ""
        state["perception_summary"] = "Text: A bomb exploded in a market\nImage: "
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 1}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [],
        }
        state["verifier_diagnostics"] = [
            {"field_path": "trigger.span", "issue_type": "span_mismatch", "suggested_action": "realign_or_drop"}
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack","trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}],"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["trigger"]["span"], {"start": 2, "end": 3})
        self.assertEqual(result["event"]["image_arguments"], [])

    def test_repair_fixes_broad_text_argument_to_head_word_with_token_span(self) -> None:
        state = make_state()
        state["text"] = "Two Chicago police officers arrested the suspect"
        state["tokens"] = ["Two", "Chicago", "police", "officers", "arrested", "the", "suspect"]
        state["event"] = {
            "event_type": "Justice:Arrest-Jail",
            "trigger": {"text": "arrested", "modality": "text", "span": {"start": 4, "end": 5}},
            "text_arguments": [
                {"role": "Agent", "text": "Two Chicago police officers", "span": {"start": 0, "end": 4}}
            ],
            "image_arguments": [],
        }
        state["verifier_diagnostics"] = [
            {
                "field_path": "text_arguments[0].span",
                "issue_type": "unnormalized_head_word",
                "suggested_action": "shrink_to_head_word",
            }
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Justice:Arrest-Jail","trigger":{"text":"arrested","modality":"text","span":null},"text_arguments":[{"role":"Agent","text":"Two Chicago police officers","span":null}],"image_arguments":[]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["text_arguments"][0]["text"], "officers")
        self.assertEqual(result["event"]["text_arguments"][0]["span"], {"start": 3, "end": 4})

    def test_repair_removes_generic_weak_place_image_argument(self) -> None:
        state = make_state()
        state["event"] = {
            "event_type": "Justice:Arrest-Jail",
            "trigger": {"text": "arrested", "modality": "text", "span": {"start": 1, "end": 2}},
            "text_arguments": [{"role": "Person", "text": "suspect", "span": {"start": 3, "end": 4}}],
            "image_arguments": [{"role": "Place", "label": "street", "bbox": None, "grounding_status": "unresolved"}],
        }
        state["verifier_diagnostics"] = [
            {
                "field_path": "image_arguments[0].label",
                "issue_type": "generic_weak_place",
                "suggested_action": "drop_or_replace_with_directly_visible_place",
            }
        ]

        with patch(
            "mm_event_agent.nodes.repair._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Justice:Arrest-Jail","trigger":{"text":"arrested","modality":"text","span":null},"text_arguments":[{"role":"Person","text":"suspect","span":null}],"image_arguments":[{"role":"Place","label":"street","bbox":null,"grounding_status":"unresolved"}]}'
                ]
            ),
        ):
            result = repair(state)

        self.assertEqual(result["event"]["image_arguments"], [])

    def test_extraction_rejects_unsupported_closed_set_event_type(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }
        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(['{"event_type":"attack"}']),
        ):
            result = extraction(state)
        self.assertEqual(result["event"], empty_event())

    def test_closed_set_mode_forces_supported_ontology_label(self) -> None:
        state = make_state()
        state["event_type_mode"] = "closed_set"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[]}',
                    '{"image_arguments":[]}',
                ]
            ),
        ):
            result = extraction(state)

        self.assertEqual(result["event"]["event_type"], "Conflict:Attack")

    def test_closed_set_mode_never_accepts_unsupported(self) -> None:
        state = make_state()
        state["event_type_mode"] = "closed_set"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(['{"event_type":"Unsupported"}']),
        ):
            result = extraction(state)

        self.assertEqual(result["event"], empty_event())

    def test_stage_c_prompt_adds_conservative_guidance_for_weak_roles(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": empty_layered_similar_events(),
            "evidence": [],
        }
        stage_c_llm = RecordingLLM('{"image_arguments":[]}')
        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            result = _stage_c_extract_image_arguments("Conflict:Attack", state["image_desc"], [], empty_layered_similar_events(), [])

        self.assertEqual(result, [])
        self.assertTrue(stage_c_llm.prompts)
        prompt = stage_c_llm.prompts[0]
        self.assertIn("Image-role visibility guidance", prompt)
        self.assertIn("visually weaker roles: [Attacker]", prompt)
        self.assertIn("prefer omission over unsupported image-role prediction", prompt)
        self.assertIn("Do not output a weakly visible role just because it is semantically allowed", prompt)

    def test_stage_c_directly_attaches_raw_image_and_auxiliary_context(self) -> None:
        stage_c_llm = RecordingLLM(
            '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}'
        )

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            result = _stage_c_extract_image_arguments(
                "Conflict:Attack",
                "raw_image_desc: smoke near market\nperception_summary: people running from a blast",
                [{"role": "Place", "text": "market", "span": None}],
                empty_layered_similar_events(),
                [],
                raw_image=b"fake-image-bytes",
                raw_text="A bomb exploded in a market",
            )

        self.assertEqual(
            result,
            [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
        )
        self.assertTrue(stage_c_llm.contents)
        self.assertIsInstance(stage_c_llm.contents[0], list)
        self.assertEqual(stage_c_llm.contents[0][1]["type"], "image_url")
        prompt = stage_c_llm.contents[0][0]["text"]
        self.assertIn('"raw_text": "A bomb exploded in a market"', prompt)
        self.assertIn("raw_image is the primary visual evidence for this stage.", prompt)
        self.assertIn("Treat image_side_info as auxiliary context", prompt)

    def test_stage_c_text_only_fallback_still_works_without_raw_image(self) -> None:
        stage_c_llm = RecordingLLM('{"image_arguments":[]}')

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            result = _stage_c_extract_image_arguments(
                "Conflict:Attack",
                "raw_image_desc: smoke near market\nperception_summary: summary",
                [],
                empty_layered_similar_events(),
                [],
                raw_image=None,
                raw_text="A bomb exploded in a market",
            )

        self.assertEqual(result, [])
        self.assertIsInstance(stage_c_llm.contents[0], str)
        self.assertIn('"raw_text": "A bomb exploded in a market"', stage_c_llm.contents[0])

    def test_stage_c_does_not_depend_only_on_image_desc_when_raw_image_exists(self) -> None:
        stage_c_llm = RecordingLLM('{"image_arguments":[]}')

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            _stage_c_extract_image_arguments(
                "Conflict:Attack",
                "",
                [],
                empty_layered_similar_events(),
                [],
                raw_image=b"fake-image-bytes",
                raw_text="A bomb exploded in a market",
            )

        self.assertIsInstance(stage_c_llm.contents[0], list)
        self.assertEqual(stage_c_llm.contents[0][1]["type"], "image_url")
        self.assertIn('"image_side_info": ""', stage_c_llm.contents[0][0]["text"])

    def test_stage_c_prompt_still_allows_clearly_visible_roles(self) -> None:
        stage_c_llm = RecordingLLM(
            '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}'
        )

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            result = _stage_c_extract_image_arguments(
                "Conflict:Attack",
                "market area with smoke and visible blast damage",
                [],
                empty_layered_similar_events(),
                [],
            )

        self.assertEqual(
            result,
            [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
        )
        prompt = stage_c_llm.prompts[0]
        self.assertIn("visually stronger roles: [Target, Instrument, Place]", prompt)
        self.assertIn("For visually stronger roles, prediction is still optional", prompt)

    def test_stage_c_filters_generic_weak_place_labels_for_arrest(self) -> None:
        stage_c_llm = RecordingLLM(
            '{"image_arguments":[{"role":"Place","label":"street","bbox":null,"grounding_status":"unresolved"},{"role":"Person","label":"suspect","bbox":null,"grounding_status":"unresolved"}]}'
        )

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            result = _stage_c_extract_image_arguments(
                "Justice:Arrest-Jail",
                "police arresting a person outdoors",
                [],
                empty_layered_similar_events(),
                [],
            )

        self.assertEqual(
            result,
            [{"role": "Person", "label": "suspect", "bbox": None, "grounding_status": "unresolved"}],
        )

    def test_stage_c_preserves_multiple_same_role_image_arguments(self) -> None:
        stage_c_llm = RecordingLLM(
            '{"image_arguments":[{"role":"Person","label":"man","bbox":null,"grounding_status":"unresolved"},{"role":"Person","label":"woman","bbox":null,"grounding_status":"unresolved"}]}'
        )

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            result = _stage_c_extract_image_arguments(
                "Justice:Arrest-Jail",
                "officers detaining two people",
                [],
                empty_layered_similar_events(),
                [],
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "Person")
        self.assertEqual(result[1]["role"], "Person")

    def test_stage_c_injects_image_examples_and_bridge_hints(self) -> None:
        stage_c_llm = RecordingLLM('{"image_arguments":[]}')
        patterns = {
            "text_event_examples": [],
            "image_semantic_examples": [
                {
                    "id": "swig_000456",
                    "source_dataset": "SWiG",
                    "modality": "image_semantics",
                    "event_type": "Conflict:Attack",
                    "image_desc": "Smoke rises from damaged vehicles near a burning building.",
                    "visual_situation": "attack-like urban explosion scene",
                    "image_arguments": [
                        {"role": "Instrument", "label": "fire"},
                        {"role": "Target", "label": "vehicle"},
                        {"role": "Place", "label": "street"},
                    ],
                    "visual_pattern_summary": "Attack-like image with visible fire and urban damage.",
                    "retrieval_text": "Conflict:Attack smoke damaged vehicles burning building street Instrument fire Target vehicle Place street",
                }
            ],
            "bridge_examples": [
                {
                    "id": "bridge_attack_instrument_001",
                    "source_dataset": "manual_bridge",
                    "modality": "bridge",
                    "event_type": "Conflict:Attack",
                    "role": "Instrument",
                    "text_cues": ["bomb", "gun", "missile"],
                    "visual_cues": ["smoke", "flames", "debris"],
                    "note": "Instrument may be explicit in text but only indirectly visible in image.",
                    "retrieval_text": "Conflict:Attack Instrument bomb gun missile smoke flames debris",
                }
            ],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            _stage_c_extract_image_arguments(
                "Conflict:Attack",
                "people running, smoke near damaged cars",
                [{"role": "Instrument", "text": "bomb", "span": None}],
                patterns,
                [],
            )

        prompt = stage_c_llm.prompts[0]
        self.assertIn("Retrieved image semantic examples:", prompt)
        self.assertIn("Image Example 1", prompt)
        self.assertIn("- Instrument: fire", prompt)
        self.assertIn("summary: Attack-like image with visible fire and urban damage.", prompt)
        self.assertIn("Retrieved cross-modal bridge hints:", prompt)
        self.assertIn("Bridge 1", prompt)
        self.assertIn("Use image semantic examples as visual pattern guidance.", prompt)
        self.assertIn("Use bridge hints to connect text roles and visual cues.", prompt)

    def test_extraction_output_contract_stays_stable_with_multimodal_stage_c(self) -> None:
        state = make_state()
        state["raw_image"] = b"fake-image-bytes"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "Text: A bomb exploded in a market\nImage: smoke near market stalls",
            "patterns": empty_layered_similar_events(),
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                    '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}',
                ]
            ),
        ):
            result = extraction(state)

        self.assertEqual(set(result.keys()), {"event", "grounding_results"})
        self.assertEqual(
            set(result["event"].keys()),
            {"event_type", "trigger", "text_arguments", "image_arguments"},
        )
        self.assertEqual(result["event"]["event_type"], "Conflict:Attack")

    def test_text_only_extraction_forces_empty_image_arguments_and_skips_stage_c(self) -> None:
        state = make_state()
        state["raw_image"] = None
        state["image_desc"] = ""
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "",
            "perception_summary": "Text: A bomb exploded in a market\nImage: ",
            "patterns": empty_layered_similar_events(),
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                ]
            ),
        ), patch("mm_event_agent.nodes.extraction._stage_c_extract_image_arguments") as mock_stage_c, patch(
            "mm_event_agent.nodes.extraction.execute_grounding_requests"
        ) as mock_grounding:
            result = extraction(state)

        self.assertFalse(mock_stage_c.called)
        self.assertFalse(mock_grounding.called)
        self.assertEqual(result["event"]["image_arguments"], [])
        self.assertEqual(result["grounding_results"], [])

    def test_extraction_uses_provided_text_tokens_for_token_span_alignment(self) -> None:
        state = make_state()
        state["text"] = "New York exploded near downtown"
        state["raw_image"] = None
        state["image_desc"] = ""
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": "",
            "perception_summary": "Text: New York exploded near downtown\nImage: ",
            "text_tokens": ["New York", "exploded", "near", "downtown"],
            "patterns": empty_layered_similar_events(),
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":{"end":99,"start":98}},"text_arguments":[{"role":"Place","text":"downtown","span":{"end":99,"start":98}}]}',
                ]
            ),
        ), patch("mm_event_agent.nodes.extraction._stage_c_extract_image_arguments") as mock_stage_c:
            result = extraction(state)

        self.assertFalse(mock_stage_c.called)
        self.assertEqual(result["event"]["trigger"]["span"], {"start": 1, "end": 2})
        self.assertEqual(list(result["event"]["trigger"]["span"].keys()), ["start", "end"])
        self.assertEqual(result["event"]["text_arguments"][0]["span"], {"start": 3, "end": 4})
        self.assertEqual(list(result["event"]["text_arguments"][0]["span"].keys()), ["start", "end"])

    def test_transfer_mode_allows_unsupported(self) -> None:
        state = make_state()
        state["event_type_mode"] = "transfer"
        state["fusion_context"] = {
            "raw_text": "A football match ended in a draw",
            "raw_image_desc": "players on a field",
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(['{"event_type":"Unsupported"}']),
        ):
            result = extraction(state)

        self.assertEqual(result["event"], empty_event())
        self.assertEqual(result["event"]["event_type"], "")
        self.assertIsNone(result["event"]["trigger"])
        self.assertEqual(result["event"]["text_arguments"], [])
        self.assertEqual(result["event"]["image_arguments"], [])

    def test_unsupported_event_type_short_circuits_downstream_extraction(self) -> None:
        state = make_state()
        state["event_type_mode"] = "transfer"
        state["fusion_context"] = {
            "raw_text": "A football match ended in a draw",
            "raw_image_desc": "players on a field",
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        fake_llm = FakeLLM(['{"event_type":"Unsupported"}'])
        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=fake_llm,
        ):
            result = extraction(state)

        self.assertEqual(result["event"], empty_event())
        self.assertEqual(result["event"]["event_type"], "")
        self.assertIsNone(result["event"]["trigger"])
        self.assertEqual(result["event"]["text_arguments"], [])
        self.assertEqual(result["event"]["image_arguments"], [])

    def test_extraction_logs_transfer_mode_unsupported_selection(self) -> None:
        state = make_state()
        state["event_type_mode"] = "transfer"
        state["fusion_context"] = {
            "raw_text": "A football match ended in a draw",
            "raw_image_desc": "players on a field",
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(['{"event_type":"Unsupported"}']),
        ), patch("mm_event_agent.nodes.extraction.log_node_event") as mock_log:
            result = extraction(state)

        self.assertEqual(result["event"], empty_event())
        self.assertTrue(mock_log.called)
        _, kwargs = mock_log.call_args
        self.assertEqual(kwargs["event_type_mode"], "transfer")
        self.assertTrue(kwargs["stage_a_selected_unsupported"])
        self.assertEqual(kwargs["stage_a_selected_event_type"], "Unsupported")

    def test_validate_event_rejects_invalid_event_type(self) -> None:
        with self.assertRaises(ValidationError):
            validate_event(
                {
                    "event_type": "explosion",
                    "trigger": None,
                    "text_arguments": [],
                    "image_arguments": [],
                }
            )

    def test_validate_event_rejects_invalid_role_for_event_type(self) -> None:
        with self.assertRaises(ValidationError):
            validate_event(
                {
                    "event_type": "Contact:Meet",
                    "trigger": None,
                    "text_arguments": [{"role": "Instrument", "text": "phone", "span": None}],
                    "image_arguments": [],
                }
            )

    def test_validate_event_accepts_valid_m2e2_style_event_object(self) -> None:
        valid = validate_event(
            {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": [2, 3]},
                "text_arguments": [{"role": "Place", "text": "market", "span": [5, 6]}],
                "image_arguments": [
                    {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
                ],
            }
        )
        self.assertEqual(valid["event_type"], "Conflict:Attack")
        self.assertEqual(valid["text_arguments"][0]["role"], "Place")
        self.assertEqual(valid["image_arguments"][0]["grounding_status"], "unresolved")

    def test_m2e2_sample_to_agent_state_maps_text_tokens_and_image_path(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "images/sample.jpg",
        }

        state = m2e2_sample_to_agent_state(sample, "E:/dataset")

        self.assertEqual(state["text"], sample["text"])
        self.assertEqual(state["tokens"], sample["words"])
        self.assertEqual(state["fusion_context"]["text_tokens"], sample["words"])
        self.assertEqual(state["raw_image"], str(Path("E:/dataset") / "images/sample.jpg"))
        self.assertEqual(state["image_desc"], "")
        self.assertEqual(state["memory"], [])
        self.assertEqual(state["prompt_trace"], [])
        self.assertEqual(state["stage_outputs"], {})
        self.assertEqual(state["repair_history"], [])
        self.assertEqual(state["repair_attempts"], 0)

    def test_m2e2_sample_to_agent_state_excludes_gold_annotation_fields(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "images/sample.jpg",
            "event_type": "Conflict:Attack",
            "text_event_mentions": [{"event_type": "Conflict:Attack"}],
            "text_arguments_flat": [{"role": "Place", "text": "market"}],
            "image_event": {"event_type": "Conflict:Attack"},
            "image_arguments_flat": [{"role": "Place", "label": "market"}],
            "ground_truth": {"event_type": "Conflict:Attack"},
        }

        state = m2e2_sample_to_agent_state(sample, "E:/dataset")

        for forbidden_key in (
            "event_type",
            "text_event_mentions",
            "text_arguments_flat",
            "image_event",
            "image_arguments_flat",
            "ground_truth",
        ):
            self.assertNotIn(forbidden_key, state)
            self.assertNotIn(forbidden_key, state["fusion_context"])

    def test_agent_output_to_m2e2_prediction_normalizes_spans_and_bboxes(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "images/sample.jpg",
        }
        agent_output = {
            "event": {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": None},
                "text_arguments": [{"role": "Place", "text": "market", "span": None}],
                "image_arguments": [{"role": "Place", "label": "market area", "bbox": [1, 2, 3, 4], "grounding_status": "grounded"}],
            },
            "grounding_results": [
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": [1, 2, 3, 4],
                    "score": 0.9,
                    "grounding_status": "grounded",
                }
            ],
        }

        prediction = agent_output_to_m2e2_prediction(sample, agent_output)

        self.assertEqual(get_m2e2_sample_id(sample), "m2e2_001")
        self.assertEqual(prediction["prediction"]["trigger"], {"text": "exploded", "start": 2, "end": 3})
        self.assertEqual(
            prediction["prediction"]["text_arguments"][0],
            {"role": "Place", "text": "market", "start": 5, "end": 6},
        )
        self.assertEqual(
            prediction["prediction"]["image_arguments"][0]["bbox"],
            [1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(prediction["prediction"]["image_arguments"][0]["role"], "Place")

    def test_agent_output_to_m2e2_prediction_excludes_unresolved_image_arguments(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "images/sample.jpg",
        }
        agent_output = {
            "event": {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": None},
                "text_arguments": [{"role": "Place", "text": "market", "span": None}],
                "image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
            },
            "grounding_results": [],
        }

        prediction = agent_output_to_m2e2_prediction(sample, agent_output)

        self.assertEqual(prediction["prediction"]["image_arguments"], [])

    def test_extract_m2e2_gold_annotations_keeps_gold_for_offline_comparison(self) -> None:
        sample = {
            "id": "m2e2_001",
            "event_type": "Conflict:Attack",
            "event_mentions": [
                {
                    "event_type": "Conflict:Attack",
                    "trigger": {"text": "exploded", "start": 2, "end": 3},
                    "arguments": [{"role": "Place", "text": "market", "start": 5, "end": 6}],
                }
            ],
            "text_event_mentions": [{"event_type": "Conflict:Attack"}],
            "text_arguments_flat": [{"role": "Place", "text": "market"}],
            "image_event": {"event_type": "Conflict:Attack"},
            "image_arguments_flat": [{"role": "Place", "label": "market area"}],
            "ground_truth": {"event_type": "Conflict:Attack"},
        }

        gold = extract_m2e2_gold_annotations(sample)

        self.assertEqual(gold["sample_id"], "m2e2_001")
        self.assertEqual(gold["event_type"], "Conflict:Attack")
        self.assertEqual(gold["trigger"], {"text": "exploded", "start": 2, "end": 3})
        self.assertEqual(gold["arguments"], [{"role": "Place", "text": "market", "start": 5, "end": 6}])
        self.assertEqual(gold["text_arguments_flat"], [{"role": "Place", "text": "market"}])
        self.assertEqual(gold["image_arguments_flat"], [{"role": "Place", "label": "market area"}])

    def test_m2e2_smoke_main_does_not_leak_gold_fields_into_graph_input(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample.jpg",
            "event_type": "Conflict:Attack",
            "text_event_mentions": [{"event_type": "Conflict:Attack"}],
            "text_arguments_flat": [{"role": "Place", "text": "market"}],
            "image_event": {"event_type": "Conflict:Attack"},
            "image_arguments_flat": [{"role": "Place", "label": "market area"}],
            "ground_truth": {"event_type": "Conflict:Attack"},
        }

        class FakeGraph:
            def invoke(self, state):
                for forbidden_key in (
                    "event_type",
                    "text_event_mentions",
                    "text_arguments_flat",
                    "image_event",
                    "image_arguments_flat",
                    "ground_truth",
                ):
                    if forbidden_key in state or forbidden_key in state.get("fusion_context", {}):
                        raise AssertionError(f"leaked gold field: {forbidden_key}")
                return {
                    **state,
                    "event": empty_event(),
                    "grounding_results": [],
                    "similar_events": empty_layered_similar_events(),
                    "verified": False,
                    "issues": [],
                    "verifier_reason": "",
                    "verifier_confidence": 0.0,
                }

        stdout = io.StringIO()
        with patch.object(
            run_m2e2_smoke_module,
            "parse_args",
            return_value=argparse.Namespace(input="dummy.jsonl", image_root="dummy_images", sample_id=None, output_dir=None),
        ), patch.object(
            run_m2e2_smoke_module,
            "load_m2e2_samples",
            return_value=[sample],
        ), patch(
            "pathlib.Path.exists",
            return_value=True,
        ), patch.object(
            run_m2e2_smoke_module,
            "_initialize_rag_runtime",
        ), patch.object(
            run_m2e2_smoke_module,
            "build_graph",
            return_value=FakeGraph(),
        ), redirect_stdout(stdout):
            run_m2e2_smoke_module.main()

        self.assertIn("gold:", stdout.getvalue())
        self.assertIn("predicted_event:", stdout.getvalue())

    def test_eval_m2e2_agent_uses_sanitized_graph_input_and_offline_gold_comparison(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample.jpg",
            "event_type": "Conflict:Attack",
            "event_mentions": [
                {
                    "event_type": "Conflict:Attack",
                    "trigger": {"text": "exploded", "start": 2, "end": 3},
                    "arguments": [{"role": "Place", "text": "market", "start": 5, "end": 6}],
                }
            ],
            "text_event_mentions": [{"event_type": "Conflict:Attack"}],
            "text_arguments_flat": [{"role": "Place", "text": "market"}],
            "image_event": {"event_type": "Conflict:Attack"},
            "image_arguments_flat": [{"role": "Place", "label": "market area"}],
            "ground_truth": {"event_type": "Conflict:Attack"},
        }

        class FakeGraph:
            def invoke(self, state):
                for forbidden_key in (
                    "event_type",
                    "text_event_mentions",
                    "text_arguments_flat",
                    "image_event",
                    "image_arguments_flat",
                    "ground_truth",
                ):
                    if forbidden_key in state or forbidden_key in state.get("fusion_context", {}):
                        raise AssertionError(f"leaked gold field: {forbidden_key}")
                return {
                    **state,
                    "event": {
                        "event_type": "Conflict:Attack",
                        "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
                        "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
                        "image_arguments": [],
                    },
                    "grounding_results": [],
                    "verified": True,
                    "issues": [],
                }

        with patch.object(eval_m2e2_agent_module, "_initialize_rag_runtime"), patch.object(
            eval_m2e2_agent_module,
            "build_graph",
            return_value=FakeGraph(),
        ):
            results = eval_m2e2_agent_module.evaluate_samples([sample], "dummy_images")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["gold"]["event_type"], "Conflict:Attack")
        self.assertEqual(results[0]["prediction"]["prediction"]["trigger"], {"text": "exploded", "start": 2, "end": 3})
        self.assertTrue(results[0]["verified"])

    def test_build_stage_trace_record_saves_structured_stage_outputs_without_gold(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample.jpg",
            "event_type": "Conflict:Attack",
            "ground_truth": {"event_type": "Conflict:Attack"},
        }
        agent_input = m2e2_sample_to_agent_state(sample, "dummy_images")
        final_state = {
            **agent_input,
            "run_mode": "benchmark",
            "effective_search_enabled": False,
            "event": {
                "event_type": "Conflict:Attack",
                "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
                "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
                "image_arguments": [],
            },
            "prompt_trace": [
                {
                    "sample_id": "",
                    "stage": "extraction_stage_b",
                    "model_name": "gpt-test",
                    "prompt_text": "raw_text only",
                    "image_path": "dummy_images/sample.jpg",
                    "input_summary": {
                        "depends_on": ["stage_a_output"],
                        "stage_a_output": {"event_type": "Conflict:Attack"},
                    },
                    "response_text": '{"trigger":{"text":"exploded"}}',
                    "parsed_output": {"trigger": {"text": "exploded"}},
                }
            ],
            "stage_outputs": {
                "stage_a_output": {"event_type": "Conflict:Attack"},
                "stage_b_output": {"trigger": {"text": "exploded"}, "text_arguments": [{"role": "Place", "text": "market"}]},
                "stage_c_output": {"image_arguments": []},
                "grounding_requests": [],
                "verifier_output": {"verified": True, "issues": []},
                "fusion_context_summary": {"raw_text": sample["text"]},
                "similar_events_summary": {"text_event_examples": 0, "image_semantic_examples": 0, "bridge_examples": 0},
                "evidence_summary": {"count": 0, "items": []},
            },
            "verified": True,
            "issues": [],
            "repair_history": [],
        }

        trace_record = run_m2e2_smoke_module.build_stage_trace_record(sample, agent_input, final_state)

        self.assertEqual(trace_record["sample_id"], "m2e2_001")
        self.assertEqual(trace_record["run_mode"], "benchmark")
        self.assertFalse(trace_record["effective_search_enabled"])
        self.assertEqual(trace_record["stage_a_output"]["event_type"], "Conflict:Attack")
        self.assertEqual(trace_record["stage_b_output"]["text_arguments"][0]["role"], "Place")
        self.assertEqual(trace_record["stage_c_output"]["image_arguments"], [])
        self.assertEqual(trace_record["grounding_requests"], [])
        self.assertEqual(trace_record["prompt_trace"][0]["sample_id"], "m2e2_001")
        self.assertNotIn("ground_truth", trace_record["agent_input"])
        self.assertNotIn("event_type", trace_record["agent_input"])
        self.assertNotIn("ground_truth", trace_record["prompt_trace"][0]["prompt_text"])

    def test_eval_save_outputs_writes_predictions_errors_summary_and_trace(self) -> None:
        results = [
            {
                "sample_id": "m2e2_001",
                "agent_input": {"text": "A bomb exploded in a market", "tokens": ["A"], "raw_image": "sample.jpg"},
                "gold": extract_m2e2_gold_record(
                    {
                        "id": "m2e2_001",
                        "event_type": "Conflict:Attack",
                    }
                ),
                "prediction": {"id": "m2e2_001", "prediction": {"event_type": "", "trigger": None, "text_arguments": [], "image_arguments": []}},
                "verified": False,
                "issues": ["needs review"],
                "trace": {
                    "sample_id": "m2e2_001",
                    "run_mode": "open_world",
                    "effective_search_enabled": True,
                    "stage_a_output": {},
                    "prompt_trace": [],
                },
            }
        ]

        output_dir = Path(f"test_eval_outputs_{uuid.uuid4().hex}")
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with patch.object(
                run_m2e2_smoke_module,
                "settings",
                replace(run_m2e2_smoke_module.settings, run_mode="open_world", enable_search=True),
            ):
                summary = eval_m2e2_agent_module.save_evaluation_outputs(results, output_dir)
            self.assertTrue((output_dir / "predictions.jsonl").exists())
            self.assertTrue((output_dir / "errors.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "trace.jsonl").exists())
            self.assertTrue((output_dir / "per_sample_metrics.jsonl").exists())
            self.assertEqual(summary["count"], 1)
            self.assertEqual(summary["error_count"], 1)
            self.assertEqual(summary["run_mode"], "open_world")
            self.assertTrue(summary["effective_search_enabled"])

            saved_trace = json.loads((output_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
            saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_trace["run_mode"], "open_world")
            self.assertTrue(saved_trace["effective_search_enabled"])
            self.assertEqual(saved_summary["run_mode"], "open_world")
            self.assertTrue(saved_summary["effective_search_enabled"])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_eval_incremental_writing_appends_per_sample_and_updates_summary(self) -> None:
        sample_one = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample1.jpg",
        }
        sample_two = {
            "id": "m2e2_002",
            "text": "Police arrested a man",
            "words": ["Police", "arrested", "a", "man"],
            "image": "sample2.jpg",
        }

        class FakeGraph:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, state):
                self.calls += 1
                event_type = "Conflict:Attack" if self.calls == 1 else "Justice:Arrest-Jail"
                return {
                    **state,
                    "event": {
                        "event_type": event_type,
                        "trigger": {"text": "exploded" if self.calls == 1 else "arrested", "modality": "text", "span": {"start": 2 if self.calls == 1 else 1, "end": 3 if self.calls == 1 else 2}},
                        "text_arguments": [],
                        "image_arguments": [],
                    },
                    "grounding_results": [],
                    "verified": self.calls == 1,
                    "issues": [] if self.calls == 1 else ["needs review"],
                    "prompt_trace": [],
                    "stage_outputs": {},
                }

        output_dir = Path(f"test_eval_incremental_outputs_{uuid.uuid4().hex}")
        shutil.rmtree(output_dir, ignore_errors=True)
        try:
            with patch.object(eval_m2e2_agent_module, "_initialize_rag_runtime"), patch.object(
                eval_m2e2_agent_module,
                "build_graph",
                return_value=FakeGraph(),
            ):
                summary = eval_m2e2_agent_module.run_incremental_evaluation(
                    [sample_one, sample_two],
                    "dummy_images",
                    output_dir,
                    resume=False,
                )

            predictions_lines = (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            trace_lines = (output_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            metric_lines = (output_dir / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            error_lines = (output_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            summary_on_disk = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(len(predictions_lines), 2)
            self.assertEqual(len(trace_lines), 2)
            self.assertEqual(len(metric_lines), 2)
            self.assertEqual(len(error_lines), 1)
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["error_count"], 1)
            self.assertEqual(summary_on_disk["count"], 2)
            self.assertEqual(summary_on_disk["verified_count"], 1)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_eval_resume_skips_already_processed_ids(self) -> None:
        sample_one = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample1.jpg",
        }
        sample_two = {
            "id": "m2e2_002",
            "text": "Police arrested a man",
            "words": ["Police", "arrested", "a", "man"],
            "image": "sample2.jpg",
        }

        class RecordingGraph:
            def __init__(self) -> None:
                self.sample_ids: list[str] = []

            def invoke(self, state):
                raw_image = str(state.get("raw_image") or "")
                self.sample_ids.append(Path(raw_image).name)
                return {
                    **state,
                    "event": empty_event(),
                    "grounding_results": [],
                    "verified": True,
                    "issues": [],
                    "prompt_trace": [],
                    "stage_outputs": {},
                }

        output_dir = Path(f"test_eval_resume_outputs_{uuid.uuid4().hex}")
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            (output_dir / "predictions.jsonl").write_text(
                json.dumps({"id": "m2e2_001", "prediction": {"event_type": "", "trigger": None, "text_arguments": [], "image_arguments": []}}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "per_sample_metrics.jsonl").write_text(
                json.dumps({"sample_id": "m2e2_001", "verified": True, "issue_count": 0}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            graph = RecordingGraph()
            with patch.object(eval_m2e2_agent_module, "_initialize_rag_runtime"), patch.object(
                eval_m2e2_agent_module,
                "build_graph",
                return_value=graph,
            ):
                summary = eval_m2e2_agent_module.run_incremental_evaluation(
                    [sample_one, sample_two],
                    "dummy_images",
                    output_dir,
                    resume=True,
                )

            predictions_lines = (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(predictions_lines), 2)
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["skipped_count"], 1)
            self.assertEqual(graph.sample_ids, ["sample2.jpg"])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_eval_partial_outputs_remain_readable_after_interruption(self) -> None:
        sample_one = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample1.jpg",
        }
        sample_two = {
            "id": "m2e2_002",
            "text": "Police arrested a man",
            "words": ["Police", "arrested", "a", "man"],
            "image": "sample2.jpg",
        }

        class FailingGraph:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, state):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("boom")
                return {
                    **state,
                    "event": empty_event(),
                    "grounding_results": [],
                    "verified": True,
                    "issues": [],
                    "prompt_trace": [],
                    "stage_outputs": {},
                }

        output_dir = Path(f"test_eval_interruption_outputs_{uuid.uuid4().hex}")
        shutil.rmtree(output_dir, ignore_errors=True)
        try:
            with patch.object(eval_m2e2_agent_module, "_initialize_rag_runtime"), patch.object(
                eval_m2e2_agent_module,
                "build_graph",
                return_value=FailingGraph(),
            ):
                with self.assertRaises(RuntimeError):
                    eval_m2e2_agent_module.run_incremental_evaluation(
                        [sample_one, sample_two],
                        "dummy_images",
                        output_dir,
                        resume=False,
                    )

            predictions_lines = (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            trace_lines = (output_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            summary_on_disk = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(predictions_lines), 1)
            self.assertEqual(len(trace_lines), 1)
            self.assertEqual(summary_on_disk["count"], 1)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_extraction_trace_makes_stage_dependencies_explicit(self) -> None:
        state = make_state()
        state["prompt_trace"] = []
        state["stage_outputs"] = {}
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": state["perception_summary"],
            "patterns": empty_layered_similar_events(),
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                    '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}',
                ]
            ),
        ):
            result = extraction(state)

        trace = result["prompt_trace"]
        stage_b = next(item for item in trace if item["stage"] == "extraction_stage_b")
        stage_c = next(item for item in trace if item["stage"] == "extraction_stage_c")
        self.assertEqual(stage_b["input_summary"]["depends_on"], ["stage_a_output"])
        self.assertEqual(stage_b["input_summary"]["stage_a_output"]["event_type"], "Conflict:Attack")
        self.assertEqual(stage_c["input_summary"]["depends_on"], ["stage_a_output", "stage_b_output"])
        self.assertEqual(stage_c["input_summary"]["stage_b_output"]["text_arguments"][0]["role"], "Place")

    def test_trace_record_contains_stage_outputs_and_grounding_outputs(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample.jpg",
        }
        agent_input = m2e2_sample_to_agent_state(sample, "dummy_images")
        final_state = {
            **agent_input,
            "event": empty_event(),
            "grounding_results": [{"role": "Place", "label": "market area", "bbox": [1, 2, 3, 4], "grounding_status": "grounded"}],
            "stage_outputs": {
                "stage_a_output": {"event_type": "Conflict:Attack"},
                "stage_b_output": {"trigger": {"text": "exploded"}, "text_arguments": []},
                "stage_c_output": {"image_arguments": [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}]},
                "grounding_requests": [{"role": "Place", "label": "market area", "grounding_query": "Place: market area", "grounding_status": "unresolved"}],
                "verifier_output": {"verified": True, "issues": []},
            },
            "repair_history": [],
        }

        trace = run_m2e2_smoke_module.build_stage_trace_record(sample, agent_input, final_state)

        self.assertIn("stage_a_output", trace)
        self.assertIn("stage_b_output", trace)
        self.assertIn("stage_c_output", trace)
        self.assertIn("grounding_requests", trace)
        self.assertIn("grounding_results", trace)
        self.assertIn("verifier_output", trace)

    def test_smoke_runner_can_save_auditable_artifacts(self) -> None:
        sample = {
            "id": "m2e2_001",
            "text": "A bomb exploded in a market",
            "words": ["A", "bomb", "exploded", "in", "a", "market"],
            "image": "sample.jpg",
        }

        class FakeGraph:
            def invoke(self, state):
                return {
                    **state,
                    "event": {
                        "event_type": "Conflict:Attack",
                        "trigger": {"text": "exploded", "modality": "text", "span": {"start": 2, "end": 3}},
                        "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
                        "image_arguments": [],
                    },
                    "grounding_results": [],
                    "similar_events": empty_layered_similar_events(),
                    "prompt_trace": [],
                    "stage_outputs": {
                        "stage_a_output": {"event_type": "Conflict:Attack"},
                        "stage_b_output": {"trigger": {"text": "exploded"}, "text_arguments": []},
                        "stage_c_output": {"image_arguments": []},
                        "grounding_requests": [],
                    },
                    "verified": True,
                    "issues": [],
                    "verifier_reason": "",
                    "verifier_confidence": 0.0,
                }

        output_dir = Path(f"test_smoke_outputs_{uuid.uuid4().hex}")
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            stdout = io.StringIO()
            with patch.object(
                run_m2e2_smoke_module,
                "parse_args",
                return_value=argparse.Namespace(
                    input="dummy.jsonl",
                    image_root="dummy_images",
                    sample_id=None,
                    output_dir=str(output_dir),
                ),
            ), patch.object(
                run_m2e2_smoke_module,
                "load_m2e2_samples",
                return_value=[sample],
            ), patch(
                "pathlib.Path.exists",
                return_value=True,
            ), patch.object(
                run_m2e2_smoke_module,
                "_initialize_rag_runtime",
            ), patch.object(
                run_m2e2_smoke_module,
                "build_graph",
                return_value=FakeGraph(),
            ), redirect_stdout(stdout):
                run_m2e2_smoke_module.main()

            self.assertTrue((output_dir / "predictions.jsonl").exists())
            self.assertTrue((output_dir / "trace.jsonl").exists())
            self.assertTrue((output_dir / "errors.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertIn("saved_files:", stdout.getvalue())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_image_arguments_without_bbox_must_be_explicitly_unresolved(self) -> None:
        valid = validate_event(
            {
                "event_type": "Conflict:Attack",
                "trigger": None,
                "text_arguments": [],
                "image_arguments": [
                    {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
                ],
            }
        )
        self.assertEqual(valid["image_arguments"][0]["grounding_status"], "unresolved")

        with self.assertRaises(ValidationError):
            validate_event(
                {
                    "event_type": "Conflict:Attack",
                    "trigger": None,
                    "text_arguments": [],
                    "image_arguments": [
                        {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "grounded"}
                    ],
                }
            )

    def test_unresolved_image_arguments_produce_grounding_requests(self) -> None:
        event = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"},
                {"role": "Target", "label": "market stall", "bbox": None, "grounding_status": "unresolved"},
            ],
        }

        requests = build_grounding_requests(event)

        self.assertEqual(
            requests,
            [
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "grounding_status": "unresolved",
                },
                {
                    "role": "Target",
                    "label": "market stall",
                    "grounding_query": "Target: market stall",
                    "grounding_status": "unresolved",
                },
            ],
        )

    def test_grounding_requests_are_derived_from_stage_c_image_arguments_only(self) -> None:
        event = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 5, "end": 6}}],
            "image_arguments": [],
        }

        requests = build_grounding_requests(event)

        self.assertEqual(requests, [])

    def test_extraction_successful_grounding_writes_bbox_back_into_event(self) -> None:
        state = make_state()
        state["raw_image"] = b"fake-image-bytes"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                    '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}',
                ]
            ),
        ), patch(
            "mm_event_agent.nodes.extraction.execute_grounding_requests",
            return_value=[
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": [1.0, 2.0, 11.0, 12.0],
                    "score": 0.88,
                    "grounding_status": "grounded",
                }
            ],
        ):
            result = extraction(state)

        self.assertEqual(result["event"]["image_arguments"][0]["bbox"], [1.0, 2.0, 11.0, 12.0])
        self.assertEqual(result["event"]["image_arguments"][0]["grounding_status"], "grounded")
        self.assertEqual(len(result["grounding_results"]), 1)

    def test_extraction_failed_grounding_leaves_event_unchanged(self) -> None:
        state = make_state()
        state["raw_image"] = b"fake-image-bytes"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                    '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}',
                ]
            ),
        ), patch(
            "mm_event_agent.nodes.extraction.execute_grounding_requests",
            side_effect=RuntimeError("grounding failed"),
        ):
            result = extraction(state)

        self.assertIsNone(result["event"]["image_arguments"][0]["bbox"])
        self.assertEqual(result["event"]["image_arguments"][0]["grounding_status"], "unresolved")
        self.assertEqual(result["grounding_results"], [])

    def test_extraction_skips_grounding_when_no_unresolved_image_arguments(self) -> None:
        state = make_state()
        state["raw_image"] = b"fake-image-bytes"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                    '{"image_arguments":[{"role":"Place","label":"market area","bbox":[0,1,2,3],"grounding_status":"grounded"}]}',
                ]
            ),
        ), patch("mm_event_agent.nodes.extraction.execute_grounding_requests") as mock_grounding:
            result = extraction(state)

        self.assertFalse(mock_grounding.called)
        self.assertEqual(result["event"]["image_arguments"][0]["bbox"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(result["grounding_results"], [])

    def test_resolved_image_arguments_are_skipped_for_grounding_requests(self) -> None:
        event = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": [0, 1, 2, 3], "grounding_status": "grounded"},
                {"role": "Target", "label": "market stall", "bbox": [2, 2, 3, 3], "grounding_status": "grounded"},
            ],
        }

        requests = build_grounding_requests(event)

        self.assertEqual(requests, [])

    def test_empty_image_arguments_produce_no_grounding_requests(self) -> None:
        event = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [],
            "image_arguments": [],
        }

        requests = build_grounding_requests(event)

        self.assertEqual(requests, [])

    def test_grounding_executor_returns_empty_for_empty_request_list(self) -> None:
        results = execute_grounding_requests("tests://fixtures/market-scene.jpg", [])

        self.assertEqual(results, [])

    def test_grounding_executor_missing_raw_image_returns_failed_results(self) -> None:
        requests = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "grounding_status": "unresolved",
            }
        ]

        results = execute_grounding_requests(None, requests)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["role"], "Place")
        self.assertEqual(results[0]["label"], "market area")
        self.assertEqual(results[0]["grounding_query"], "Place: market area")
        self.assertIsNone(results[0]["bbox"])
        self.assertIsNone(results[0]["score"])
        self.assertEqual(results[0]["grounding_status"], "failed")

    def test_grounding_executor_unresolved_request_can_return_grounded_schema(self) -> None:
        requests = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "grounding_status": "unresolved",
            }
        ]
        grounder = Florence2HFGrounder()

        with patch.object(Florence2HFGrounder, "_load_image", return_value=object()), patch.object(
            Florence2HFGrounder, "_ensure_model_loaded", return_value=None
        ), patch.object(
            Florence2HFGrounder,
            "_run_single_request",
            return_value={
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [1.0, 2.0, 10.0, 12.0],
                "score": 0.91,
                "grounding_status": "grounded",
            },
        ):
            results = grounder.execute("fake-path.jpg", requests)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["grounding_status"], "grounded")
        self.assertEqual(results[0]["bbox"], [1.0, 2.0, 10.0, 12.0])
        self.assertEqual(results[0]["score"], 0.91)

    def test_grounding_executor_inference_failure_returns_failed_schema(self) -> None:
        requests = [
            {
                "role": "Target",
                "label": "market stall",
                "grounding_query": "Target: market stall",
                "grounding_status": "unresolved",
            }
        ]
        grounder = Florence2HFGrounder()

        with patch.object(Florence2HFGrounder, "_load_image", return_value=object()), patch.object(
            Florence2HFGrounder, "_ensure_model_loaded", return_value=None
        ), patch.object(
            Florence2HFGrounder,
            "_run_single_request",
            side_effect=RuntimeError("inference failed"),
        ):
            results = grounder.execute("fake-path.jpg", requests)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["role"], "Target")
        self.assertEqual(results[0]["label"], "market stall")
        self.assertEqual(results[0]["grounding_query"], "Target: market stall")
        self.assertIsNone(results[0]["bbox"])
        self.assertIsNone(results[0]["score"])
        self.assertEqual(results[0]["grounding_status"], "failed")

    def test_grounding_executor_output_structure_validity(self) -> None:
        requests = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "grounding_status": "unresolved",
            }
        ]
        grounder = Florence2HFGrounder()

        with patch.object(Florence2HFGrounder, "_load_image", return_value=object()), patch.object(
            Florence2HFGrounder, "_ensure_model_loaded", return_value=None
        ), patch.object(
            Florence2HFGrounder,
            "_run_single_request",
            return_value={
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [0.0, 1.0, 2.0, 3.0],
                "score": 0.75,
                "grounding_status": "grounded",
            },
        ):
            results = grounder.execute("fake-path.jpg", requests)

        self.assertEqual(set(results[0].keys()), {"role", "label", "grounding_query", "bbox", "score", "grounding_status"})
        self.assertIsInstance(results[0]["bbox"], list)
        self.assertEqual(len(results[0]["bbox"]), 4)
        self.assertIsInstance(results[0]["score"], float)

    def test_grounding_executor_phrase_grounding_parsed_format_returns_first_bbox(self) -> None:
        grounder = Florence2HFGrounder(task_prompt="<CAPTION_TO_PHRASE_GROUNDING>")

        bbox, score = grounder._extract_best_grounding(
            {
                "<CAPTION_TO_PHRASE_GROUNDING>": {
                    "bboxes": [[1, 2, 10, 12], [4, 5, 6, 7]],
                    "labels": ["helicopter", "cargo net"],
                }
            }
        )

        self.assertEqual(bbox, [1.0, 2.0, 10.0, 12.0])
        self.assertIsNone(score)

    def test_grounding_executor_prepare_inputs_casts_only_pixel_values_to_float16_on_gpu(self) -> None:
        class FakeTensor:
            def __init__(self, name: str) -> None:
                self.name = name
                self.device_moves: list[str] = []
                self.dtype_moves: list[object] = []

            def to(self, device=None, dtype=None):
                if device is not None:
                    self.device_moves.append(device)
                if dtype is not None:
                    self.dtype_moves.append(dtype)
                return self

        grounder = Florence2HFGrounder(device="cuda:0")
        grounder._torch = SimpleNamespace(float16="float16")
        input_ids = FakeTensor("input_ids")
        pixel_values = FakeTensor("pixel_values")

        prepared = grounder._prepare_inputs(
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "attention_mask": FakeTensor("attention_mask"),
            }
        )

        self.assertIs(prepared["input_ids"], input_ids)
        self.assertEqual(input_ids.device_moves, ["cuda:0"])
        self.assertEqual(input_ids.dtype_moves, [])
        self.assertIs(prepared["pixel_values"], pixel_values)
        self.assertEqual(pixel_values.device_moves, ["cuda:0"])
        self.assertEqual(pixel_values.dtype_moves, ["float16"])

    def test_grounding_executor_run_single_request_extracts_bbox_without_score(self) -> None:
        class FakeProcessor:
            def __call__(self, text, images, return_tensors):
                return {
                    "input_ids": FakeTensor("input_ids"),
                    "pixel_values": FakeTensor("pixel_values"),
                }

            def batch_decode(self, generated_ids, skip_special_tokens=False):
                return ["decoded"]

            def post_process_generation(self, generated_text, task, image_size):
                return {
                    "<CAPTION_TO_PHRASE_GROUNDING>": {
                        "bboxes": [[5, 6, 20, 22]],
                        "labels": ["helicopter"],
                    }
                }

        class FakeModel:
            def generate(self, **kwargs):
                return [[1, 2, 3]]

        class FakeTensor:
            def __init__(self, name: str) -> None:
                self.name = name

            def to(self, device=None, dtype=None):
                return self

        grounder = Florence2HFGrounder(task_prompt="<CAPTION_TO_PHRASE_GROUNDING>", device="cuda:0")
        grounder._processor = FakeProcessor()
        grounder._model = FakeModel()
        grounder._torch = SimpleNamespace(float16="float16")

        image = SimpleNamespace(width=640, height=480)
        request = {
            "role": "Instrument",
            "label": "helicopter",
            "grounding_query": "helicopter",
            "grounding_status": "unresolved",
        }

        result = grounder._run_single_request(image, request)

        self.assertEqual(result["bbox"], [5.0, 6.0, 20.0, 22.0])
        self.assertIsNone(result["score"])
        self.assertEqual(result["grounding_status"], "grounded")

    def test_grounding_executor_query_fallback_tries_label_then_optional_phrase(self) -> None:
        grounder = Florence2HFGrounder(task_prompt="<CAPTION_TO_PHRASE_GROUNDING>", device="cpu")
        grounder._processor = object()
        grounder._model = object()
        grounder._torch = object()
        seen_queries: list[str] = []

        def fake_run_phrase_query(image, query):
            seen_queries.append(query)
            if query == "suspended cargo net":
                return [9.0, 10.0, 30.0, 40.0], None
            return None, None

        with patch.object(Florence2HFGrounder, "_run_phrase_query", side_effect=fake_run_phrase_query):
            result = grounder._run_single_request(
                SimpleNamespace(width=640, height=480),
                {
                    "role": "Artifact",
                    "label": "cargo net",
                    "grounding_query": "suspended cargo net",
                    "grounding_status": "unresolved",
                },
            )

        self.assertEqual(seen_queries, ["cargo net", "suspended cargo net"])
        self.assertEqual(result["bbox"], [9.0, 10.0, 30.0, 40.0])
        self.assertEqual(result["grounding_status"], "grounded")

    def test_grounding_executor_query_fallback_supports_pipe_separated_candidates(self) -> None:
        grounder = Florence2HFGrounder(task_prompt="<CAPTION_TO_PHRASE_GROUNDING>", device="cpu")

        candidates = grounder._build_candidate_queries(
            {
                "role": "Artifact",
                "label": "cargo net",
                "grounding_query": "Artifact: cargo net || suspended cargo net || hanging cargo net",
                "grounding_status": "unresolved",
            }
        )

        self.assertEqual(candidates, ["cargo net", "Artifact: cargo net", "suspended cargo net"])

    def test_grounding_service_payload_uses_absolute_local_image_path(self) -> None:
        payload = build_grounding_service_payload(
            raw_image="tests",
            grounding_requests=[
                {
                    "role": "Vehicle",
                    "label": "helicopter",
                    "grounding_query": "helicopter",
                    "grounding_status": "unresolved",
                }
            ],
            task="<CAPTION_TO_PHRASE_GROUNDING>",
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(os.path.isabs(payload["raw_image"]))
        self.assertEqual(payload["task"], "<CAPTION_TO_PHRASE_GROUNDING>")
        self.assertEqual(payload["grounding_requests"][0]["grounding_query"], "helicopter")

    def test_grounding_service_response_is_normalized_to_existing_contract(self) -> None:
        results = parse_grounding_service_response(
            {
                "results": [
                    {
                        "role": "Vehicle",
                        "label": "helicopter",
                        "grounding_query": "helicopter",
                        "bbox": [1, 2, 10, 12],
                        "score": None,
                        "grounding_status": "grounded",
                    }
                ]
            },
            [
                {
                    "role": "Vehicle",
                    "label": "helicopter",
                    "grounding_query": "helicopter",
                    "grounding_status": "unresolved",
                }
            ],
        )

        self.assertEqual(
            results,
            [
                {
                    "role": "Vehicle",
                    "label": "helicopter",
                    "grounding_query": "helicopter",
                    "bbox": [1.0, 2.0, 10.0, 12.0],
                    "score": None,
                    "grounding_status": "grounded",
                }
            ],
        )

    def test_grounding_executor_prefers_service_endpoint_when_configured(self) -> None:
        requests = [
            {
                "role": "Vehicle",
                "label": "helicopter",
                "grounding_query": "helicopter",
                "grounding_status": "unresolved",
            }
        ]
        service_settings = replace(
            grounding_module.settings,
            florence2_local_endpoint="http://127.0.0.1:8765/ground",
        )

        with patch.object(grounding_module, "settings", service_settings), patch.object(
            Florence2ServiceGrounder,
            "execute",
            return_value=[
                {
                    "role": "Vehicle",
                    "label": "helicopter",
                    "grounding_query": "helicopter",
                    "bbox": [1.0, 2.0, 10.0, 12.0],
                    "score": None,
                    "grounding_status": "grounded",
                }
            ],
        ) as mock_service_execute, patch.object(
            Florence2HFGrounder,
            "execute",
            side_effect=AssertionError("local fallback should not be used"),
        ):
            results = execute_grounding_requests("tests", requests)

        self.assertEqual(results[0]["grounding_status"], "grounded")
        self.assertTrue(mock_service_execute.called)

    def test_grounding_service_endpoint_failure_returns_safe_failed_results(self) -> None:
        requests = [
            {
                "role": "Vehicle",
                "label": "helicopter",
                "grounding_query": "helicopter",
                "grounding_status": "unresolved",
            }
        ]
        service_settings = replace(
            grounding_module.settings,
            florence2_local_endpoint="http://127.0.0.1:8765/ground",
            florence2_local_timeout_seconds=1.5,
        )

        with patch.object(grounding_module, "settings", service_settings), patch(
            "urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ), patch.object(
            Florence2HFGrounder,
            "_ensure_model_loaded",
            side_effect=AssertionError("local fallback should not be loaded"),
        ):
            results = execute_grounding_requests("tests", requests)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["grounding_status"], "failed")
        self.assertIsNone(results[0]["bbox"])

    def test_extraction_grounding_contract_is_unchanged_with_service_results(self) -> None:
        state = make_state()
        state["raw_image"] = b"fake-image-bytes"
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "summary",
            "patterns": [],
            "evidence": [],
        }

        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"Conflict:Attack"}',
                    '{"trigger":{"text":"exploded","modality":"text","span":null},"text_arguments":[{"role":"Place","text":"market","span":null}]}',
                    '{"image_arguments":[{"role":"Place","label":"market area","bbox":null,"grounding_status":"unresolved"}]}',
                ]
            ),
        ), patch(
            "mm_event_agent.nodes.extraction.execute_grounding_requests",
            return_value=[
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": [1.0, 2.0, 11.0, 12.0],
                    "score": None,
                    "grounding_status": "grounded",
                }
            ],
        ):
            result = extraction(state)

        self.assertEqual(result["event"]["image_arguments"][0]["bbox"], [1.0, 2.0, 11.0, 12.0])
        self.assertEqual(result["event"]["image_arguments"][0]["grounding_status"], "grounded")

    def test_grounding_executor_run_single_request_safe_fails_when_phrase_grounding_has_no_bbox(self) -> None:
        class FakeProcessor:
            def __call__(self, text, images, return_tensors):
                return {
                    "input_ids": FakeTensor(),
                    "pixel_values": FakeTensor(),
                }

            def batch_decode(self, generated_ids, skip_special_tokens=False):
                return ["decoded"]

            def post_process_generation(self, generated_text, task, image_size):
                return {
                    "<CAPTION_TO_PHRASE_GROUNDING>": {
                        "bboxes": [],
                        "labels": [],
                    }
                }

        class FakeModel:
            def generate(self, **kwargs):
                return [[1, 2, 3]]

        class FakeTensor:
            def to(self, device=None, dtype=None):
                return self

        grounder = Florence2HFGrounder(task_prompt="<CAPTION_TO_PHRASE_GROUNDING>", device="cpu")
        grounder._processor = FakeProcessor()
        grounder._model = FakeModel()
        grounder._torch = SimpleNamespace(float16="float16")

        image = SimpleNamespace(width=640, height=480)
        request = {
            "role": "Instrument",
            "label": "helicopter",
            "grounding_query": "helicopter",
            "grounding_status": "unresolved",
        }

        result = grounder._run_single_request(image, request)

        self.assertIsNone(result["bbox"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["grounding_status"], "failed")

    def test_apply_grounding_results_to_event_is_future_facing_write_back_helper(self) -> None:
        event = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [],
            "image_arguments": [
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"},
                {"role": "Target", "label": "market stall", "bbox": None, "grounding_status": "unresolved"},
            ],
        }
        grounding_results = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [5.0, 6.0, 20.0, 22.0],
                "score": 0.8,
                "grounding_status": "grounded",
            }
        ]

        updated = apply_grounding_results_to_event(event, grounding_results)

        self.assertEqual(updated["image_arguments"][0]["bbox"], [5.0, 6.0, 20.0, 22.0])
        self.assertEqual(updated["image_arguments"][0]["grounding_status"], "grounded")
        self.assertEqual(updated["image_arguments"][1], event["image_arguments"][1])

    def test_apply_grounding_results_to_event_preserves_multiple_same_role_instances(self) -> None:
        event = {
            "event_type": "Justice:Arrest-Jail",
            "trigger": None,
            "text_arguments": [],
            "image_arguments": [
                {"role": "Person", "label": "suspect", "bbox": None, "grounding_status": "unresolved"},
                {"role": "Person", "label": "suspect", "bbox": None, "grounding_status": "unresolved"},
            ],
        }
        grounding_results = [
            {
                "role": "Person",
                "label": "suspect",
                "grounding_query": "Person: suspect",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "score": 0.9,
                "grounding_status": "grounded",
            },
            {
                "role": "Person",
                "label": "suspect",
                "grounding_query": "Person: suspect",
                "bbox": [5.0, 6.0, 7.0, 8.0],
                "score": 0.85,
                "grounding_status": "grounded",
            },
        ]

        updated = apply_grounding_results_to_event(event, grounding_results)

        self.assertEqual(updated["image_arguments"][0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(updated["image_arguments"][1]["bbox"], [5.0, 6.0, 7.0, 8.0])

    def test_grounding_summary_counts_are_correct(self) -> None:
        before_image_arguments = [
            {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"},
            {"role": "Target", "label": "market stall", "bbox": None, "grounding_status": "unresolved"},
        ]
        grounding_requests = [
            {"role": "Place", "label": "market area", "grounding_query": "Place: market area", "grounding_status": "unresolved"},
            {"role": "Target", "label": "market stall", "grounding_query": "Target: market stall", "grounding_status": "unresolved"},
        ]
        grounding_results = [
            {
                "role": "Place",
                "label": "market area",
                "grounding_query": "Place: market area",
                "bbox": [5.0, 6.0, 20.0, 22.0],
                "score": 0.8,
                "grounding_status": "grounded",
            },
            {
                "role": "Target",
                "label": "market stall",
                "grounding_query": "Target: market stall",
                "bbox": None,
                "score": None,
                "grounding_status": "failed",
            },
        ]
        after_image_arguments = [
            {"role": "Place", "label": "market area", "bbox": [5.0, 6.0, 20.0, 22.0], "grounding_status": "grounded"},
            {"role": "Target", "label": "market stall", "bbox": None, "grounding_status": "unresolved"},
        ]

        summary = summarize_grounding_activity(
            image_arguments_before=before_image_arguments,
            grounding_requests=grounding_requests,
            grounding_results=grounding_results,
            image_arguments_after=after_image_arguments,
        )

        self.assertEqual(
            summary,
            {
                "unresolved_image_arguments": 2,
                "grounding_requests": 2,
                "grounded_results": 1,
                "failed_grounding_results": 1,
                "applied_grounded_bboxes": 1,
            },
        )

    def test_applied_grounding_count_is_correct(self) -> None:
        summary = summarize_grounding_activity(
            image_arguments_before=[
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"},
                {"role": "Target", "label": "market stall", "bbox": None, "grounding_status": "unresolved"},
            ],
            grounding_requests=[],
            grounding_results=[
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.9,
                    "grounding_status": "grounded",
                }
            ],
            image_arguments_after=[
                {"role": "Place", "label": "market area", "bbox": [1.0, 2.0, 3.0, 4.0], "grounding_status": "grounded"},
                {"role": "Target", "label": "market stall", "bbox": None, "grounding_status": "unresolved"},
            ],
        )

        self.assertEqual(summary["applied_grounded_bboxes"], 1)

    def test_failed_grounding_is_reflected_in_summary(self) -> None:
        summary = summarize_grounding_activity(
            image_arguments_before=[
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
            ],
            grounding_requests=[
                {"role": "Place", "label": "market area", "grounding_query": "Place: market area", "grounding_status": "unresolved"}
            ],
            grounding_results=[
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": None,
                    "score": None,
                    "grounding_status": "failed",
                }
            ],
            image_arguments_after=[
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
            ],
        )

        self.assertEqual(summary["failed_grounding_results"], 1)
        self.assertEqual(summary["grounded_results"], 0)

    def test_no_grounding_activity_produces_zeroed_summaries(self) -> None:
        summary = summarize_grounding_activity([], [], [], [])

        self.assertEqual(
            summary,
            {
                "unresolved_image_arguments": 0,
                "grounding_requests": 0,
                "grounded_results": 0,
                "failed_grounding_results": 0,
                "applied_grounded_bboxes": 0,
            },
        )

    def test_compare_grounding_stages_returns_eval_friendly_snapshot(self) -> None:
        debug_snapshot = compare_grounding_stages(
            image_arguments_before=[
                {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
            ],
            grounding_results=[
                {
                    "role": "Place",
                    "label": "market area",
                    "grounding_query": "Place: market area",
                    "bbox": [5.0, 6.0, 20.0, 22.0],
                    "score": 0.8,
                    "grounding_status": "grounded",
                }
            ],
            image_arguments_after=[
                {"role": "Place", "label": "market area", "bbox": [5.0, 6.0, 20.0, 22.0], "grounding_status": "grounded"}
            ],
        )

        self.assertIn("before_image_arguments", debug_snapshot)
        self.assertIn("grounding_results", debug_snapshot)
        self.assertIn("after_image_arguments", debug_snapshot)
        self.assertIn("summary", debug_snapshot)
        self.assertEqual(debug_snapshot["summary"]["applied_grounded_bboxes"], 1)


if __name__ == "__main__":
    unittest.main()

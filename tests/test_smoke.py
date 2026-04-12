from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mm_event_agent.graph import build_graph
from mm_event_agent.grounding.florence2_hf import (
    Florence2HFGrounder,
    apply_grounding_results_to_event,
    execute_grounding_requests,
)
from mm_event_agent.grounding.debug import compare_grounding_stages, summarize_grounding_activity
from mm_event_agent.nodes.extraction import extraction
from mm_event_agent.nodes.fusion import fusion
from mm_event_agent.nodes.repair import repair
from mm_event_agent.nodes.search import search
from mm_event_agent.nodes.verifier import verifier
from mm_event_agent.ontology import (
    format_image_role_visibility_guidance_for_prompt,
    get_event_schema,
    get_supported_event_types,
)
from mm_event_agent.schemas import (
    ValidationError,
    attach_text_spans,
    build_grounding_requests,
    choose_best_span,
    empty_event,
    empty_fusion_context,
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

    def invoke(self, messages):
        for message in messages:
            content = getattr(message, "content", "")
            self.prompts.append(content if isinstance(content, str) else str(content))
        return SimpleNamespace(content=self.response)


def make_state() -> dict:
    return {
        "text": "A bomb exploded in a market",
        "raw_image": "tests://fixtures/market-scene.jpg",
        "event_type_mode": "closed_set",
        # Current bridge representation derived from raw_image.
        "image_desc": "market area with smoke and people running",
        "perception_summary": "",
        "search_query": "",
        "similar_events": [],
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
    def test_graph_execution_smoke(self) -> None:
        state = make_state()
        state["perception_summary"] = ""
        graph = build_graph()

        with patch(
            "mm_event_agent.nodes.rag._retrieve_similar_events",
            return_value=[
                '{"event_type":"Conflict:Attack","trigger":null,"text_arguments":[],"image_arguments":[]}'
            ],
        ), patch(
            "mm_event_agent.nodes.search._search_web",
            return_value=[
                {
                    "title": "Market attack update",
                    "body": "Police said a bomb exploded at the city market.",
                    "href": "https://example.com/news/market-attack",
                    "date": "2026-04-12",
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
        self.assertEqual(final["event"]["trigger"]["span"], {"start": 7, "end": 15})
        self.assertEqual(final["event"]["text_arguments"][0]["role"], "Place")
        self.assertEqual(final["event"]["text_arguments"][0]["text"], "market")
        self.assertEqual(final["event"]["text_arguments"][0]["span"], {"start": 21, "end": 27})
        self.assertEqual(final["event"]["image_arguments"][0]["bbox"], None)
        self.assertEqual(final["event"]["image_arguments"][0]["grounding_status"], "unresolved")
        self.assertEqual(final["search_query"], "A bomb exploded in a market")
        self.assertEqual(len(final["evidence"]), 1)

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
            "patterns": [],
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

        with patch("mm_event_agent.nodes.search._search_web", side_effect=RuntimeError("search down")):
            search_result = search(state)

        self.assertEqual(search_result["evidence"], [])
        self.assertEqual(search_result["search_query"], "A bomb exploded in a market")

        fused = fusion({**state, **search_result, "perception_summary": "summary"})
        self.assertEqual(fused["fusion_context"]["evidence"], [])

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
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 4}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "trigger": {"text": "arrested", "modality": "text", "span": {"start": 7, "end": 15}},
            "text_arguments": [
                {"role": "Agent", "text": "suspect", "span": {"start": 20, "end": 27}},
                {"role": "Person", "text": "Police", "span": {"start": 0, "end": 6}},
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
            "trigger": {"text": "met", "modality": "text", "span": {"start": 8, "end": 11}},
            "text_arguments": [{"role": "Place", "text": "Geneva", "span": {"start": 15, "end": 21}}],
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
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            [{"start": 0, "end": 6}, {"start": 13, "end": 19}],
        )

    def test_find_text_span_returns_none_when_repeated_match_is_ambiguous(self) -> None:
        self.assertIsNone(find_text_span("market north market south", "market"))

    def test_choose_best_span_uses_anchor_context_for_repeated_matches(self) -> None:
        occurrences = find_all_text_occurrences("market north market south", "market")
        self.assertEqual(
            choose_best_span(occurrences, anchor_spans=[{"start": 20, "end": 25}]),
            {"start": 13, "end": 19},
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
            [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
        )
        self.assertEqual(
            aligned["image_arguments"],
            [{"role": "Place", "label": "market area", "bbox": [0, 1, 2, 3], "grounding_status": "grounded"}],
        )

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

        self.assertEqual(grounded["trigger"]["span"], {"start": 7, "end": 15})
        self.assertEqual(grounded["text_arguments"], [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}])

    def test_repair_drops_text_argument_when_exact_alignment_fails(self) -> None:
        state = make_state()
        state["text"] = "A bomb exploded in a market"
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 7, "end": 15}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
                "trigger": {"text": "exploded", "modality": "text", "span": [7, 15]},
                "text_arguments": [{"role": "Place", "text": "market", "span": [21, 27]}],
                "image_arguments": [],
            }
        )

        self.assertEqual(normalized["trigger"]["span"], {"start": 7, "end": 15})
        self.assertEqual(normalized["text_arguments"][0]["span"], {"start": 21, "end": 27})

    def test_repair_only_trigger_span_diagnosed_preserves_non_trigger_fields(self) -> None:
        state = make_state()
        state["text"] = "A bomb exploded in a market"
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 4}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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

        self.assertEqual(result["event"]["trigger"]["span"], {"start": 7, "end": 15})
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])
        self.assertEqual(result["event"]["text_arguments"], state["event"]["text_arguments"])
        self.assertEqual(result["event"]["image_arguments"], state["event"]["image_arguments"])

    def test_repair_only_one_text_argument_span_diagnosed_preserves_other_arguments(self) -> None:
        state = make_state()
        state["text"] = "A civilian died from shrapnel in the market"
        state["event"] = {
            "event_type": "Life:Die",
            "trigger": {"text": "died", "modality": "text", "span": {"start": 11, "end": 15}},
            "text_arguments": [
                {"role": "Victim", "text": "civilian", "span": {"start": 0, "end": 8}},
                {"role": "Instrument", "text": "shrapnel", "span": {"start": 21, "end": 29}},
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

        self.assertEqual(result["event"]["text_arguments"][0]["span"], {"start": 2, "end": 10})
        self.assertEqual(result["event"]["text_arguments"][1]["role"], state["event"]["text_arguments"][1]["role"])
        self.assertEqual(result["event"]["text_arguments"][1]["text"], state["event"]["text_arguments"][1]["text"])
        self.assertEqual(result["event"]["text_arguments"][1]["span"], {"start": 21, "end": 29})
        self.assertEqual(result["event"]["trigger"], state["event"]["trigger"])
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])

    def test_repair_only_one_image_bbox_diagnosed_preserves_unrelated_fields(self) -> None:
        state = make_state()
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 4}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 4}},
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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

        self.assertEqual(result["event"]["trigger"]["span"], {"start": 7, "end": 15})
        self.assertEqual(result["event"]["image_arguments"][0]["bbox"], [5.0, 6.0, 20.0, 22.0])
        self.assertEqual(result["event"]["image_arguments"][0]["grounding_status"], "grounded")
        self.assertEqual(result["event"]["event_type"], state["event"]["event_type"])

    def test_failed_grounding_does_not_force_dropping_otherwise_acceptable_unresolved_image_argument(self) -> None:
        state = make_state()
        state["event"] = {
            "event_type": "Conflict:Attack",
            "trigger": None,
            "text_arguments": [{"role": "Place", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "patterns": [],
            "evidence": [],
        }
        stage_c_llm = RecordingLLM('{"image_arguments":[]}')
        with patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=stage_c_llm,
        ):
            from mm_event_agent.nodes.extraction import _stage_c_extract_image_arguments

            result = _stage_c_extract_image_arguments("Conflict:Attack", state["image_desc"], [])

        self.assertEqual(result, [])
        self.assertTrue(stage_c_llm.prompts)
        prompt = stage_c_llm.prompts[0]
        self.assertIn("Image-role visibility guidance", prompt)
        self.assertIn("visually weaker roles: [Attacker]", prompt)
        self.assertIn("prefer omission over unsupported image-role prediction", prompt)
        self.assertIn("Do not output a weakly visible role just because it is semantically allowed", prompt)

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
            )

        self.assertEqual(
            result,
            [{"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}],
        )
        prompt = stage_c_llm.prompts[0]
        self.assertIn("visually stronger roles: [Target, Instrument, Place]", prompt)
        self.assertIn("For visually stronger roles, prediction is still optional", prompt)

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
                "trigger": {"text": "exploded", "modality": "text", "span": [7, 15]},
                "text_arguments": [{"role": "Place", "text": "market", "span": [21, 27]}],
                "image_arguments": [
                    {"role": "Place", "label": "market area", "bbox": None, "grounding_status": "unresolved"}
                ],
            }
        )
        self.assertEqual(valid["event_type"], "Conflict:Attack")
        self.assertEqual(valid["text_arguments"][0]["role"], "Place")
        self.assertEqual(valid["image_arguments"][0]["grounding_status"], "unresolved")

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

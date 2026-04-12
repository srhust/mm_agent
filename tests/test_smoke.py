from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mm_event_agent.graph import build_graph
from mm_event_agent.nodes.extraction import extraction
from mm_event_agent.nodes.fusion import fusion
from mm_event_agent.nodes.repair import repair
from mm_event_agent.nodes.search import search
from mm_event_agent.nodes.verifier import verifier
from mm_event_agent.ontology import get_event_schema, get_supported_event_types
from mm_event_agent.schemas import (
    ValidationError,
    attach_text_spans,
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


def make_state() -> dict:
    return {
        "text": "A bomb exploded in a market",
        "image_desc": "market area with smoke and people running",
        "perception_summary": "",
        "search_query": "",
        "similar_events": [],
        "evidence": [],
        "fusion_context": empty_fusion_context(),
        "event": empty_event(),
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


if __name__ == "__main__":
    unittest.main()

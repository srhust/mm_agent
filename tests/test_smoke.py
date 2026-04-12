from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mm_event_agent.graph import build_graph
from mm_event_agent.nodes.extraction import extraction
from mm_event_agent.nodes.fusion import fusion
from mm_event_agent.nodes.search import search
from mm_event_agent.nodes.verifier import verifier
from mm_event_agent.schemas import (
    attach_text_spans,
    choose_best_span,
    empty_event,
    empty_fusion_context,
    find_all_text_occurrences,
    find_text_span,
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
        "image_desc": "people running, smoke",
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
                '{"event_type":"explosion","trigger":null,"text_arguments":[],"image_arguments":[]}'
            ],
        ), patch(
            "mm_event_agent.nodes.search._search_web",
            return_value=[
                {
                    "title": "Market explosion update",
                    "body": "Police said an explosion occurred at the city market.",
                    "href": "https://example.com/news/market-explosion",
                    "date": "2026-04-12",
                }
            ],
        ), patch(
            "mm_event_agent.nodes.extraction._get_llm",
            return_value=FakeLLM(
                [
                    '{"event_type":"explosion","trigger":{"text":"exploded","modality":"text","span":null},'
                    '"text_arguments":[{"role":"location","text":"market","span":null}],"image_arguments":[]}'
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
        self.assertEqual(final["event"]["event_type"], "explosion")
        self.assertEqual(final["event"]["trigger"]["text"], "exploded")
        self.assertEqual(final["event"]["trigger"]["span"], {"start": 7, "end": 15})
        self.assertEqual(final["event"]["text_arguments"][0]["text"], "market")
        self.assertEqual(final["event"]["text_arguments"][0]["span"], {"start": 21, "end": 27})
        self.assertEqual(final["search_query"], "A bomb exploded in a market")
        self.assertEqual(len(final["evidence"]), 1)

    def test_invalid_json_output_falls_back_to_empty_event(self) -> None:
        state = make_state()
        state["fusion_context"] = {
            "raw_text": state["text"],
            "raw_image_desc": state["image_desc"],
            "perception_summary": "Text: A bomb exploded in a market\nImage: people running, smoke",
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
            "event_type": "explosion",
            "trigger": {"text": "exploded", "modality": "text", "span": {"start": 0, "end": 4}},
            "text_arguments": [{"role": "location", "text": "market", "span": {"start": 21, "end": 27}}],
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
            "event_type": "explosion",
            "trigger": {"text": "detonated", "modality": "text", "span": None},
            "text_arguments": [
                {"role": "location", "text": "market", "span": None},
                {"role": "device", "text": "pipe bomb", "span": None},
            ],
            "image_arguments": [{"role": "smoke", "label": "smoke plume", "bbox": [0, 1, 2, 3]}],
        }

        aligned = attach_text_spans(event, "A bomb exploded in a market")

        self.assertIsNone(aligned["trigger"])
        self.assertEqual(
            aligned["text_arguments"],
            [{"role": "location", "text": "market", "span": {"start": 21, "end": 27}}],
        )
        self.assertEqual(aligned["image_arguments"], [{"role": "smoke", "label": "smoke plume", "bbox": [0, 1, 2, 3]}])

    def test_validate_event_normalizes_legacy_list_span(self) -> None:
        from mm_event_agent.schemas import validate_event

        normalized = validate_event(
            {
                "event_type": "explosion",
                "trigger": {"text": "exploded", "modality": "text", "span": [7, 15]},
                "text_arguments": [{"role": "location", "text": "market", "span": [21, 27]}],
                "image_arguments": [],
            }
        )

        self.assertEqual(normalized["trigger"]["span"], {"start": 7, "end": 15})
        self.assertEqual(normalized["text_arguments"][0]["span"], {"start": 21, "end": 27})


if __name__ == "__main__":
    unittest.main()

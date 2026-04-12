from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mm_event_agent.graph import build_graph
from mm_event_agent.nodes.extraction import extraction
from mm_event_agent.nodes.fusion import fusion
from mm_event_agent.nodes.search import search
from mm_event_agent.schemas import empty_event, empty_fusion_context


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
                '{"event_type":"explosion","trigger":"exploded","arguments":{"location":"market"}}'
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
                ['{"event_type":"explosion","trigger":"exploded","arguments":{"location":"market"}}']
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


if __name__ == "__main__":
    unittest.main()

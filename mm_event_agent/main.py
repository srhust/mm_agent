"""Run the LangGraph multimodal event extraction pipeline.

Current input contract: raw text + raw image.
In graph state, the raw text input lives in state["text"] and the primary raw
image input lives in state["raw_image"].
Current operational image representation: image_desc remains a derived
intermediate fallback until raw-image grounding is implemented.
"""

from __future__ import annotations

import mm_event_agent.graph as graph
from mm_event_agent.schemas import empty_event, empty_fusion_context
from mm_event_agent.vector_store import build_index

# Local RAG corpus: similar-event patterns / few-shot examples, not current news facts.
RAG_EVENT_CORPUS = [
    (
        '{"event_type":"Conflict:Attack","trigger":"detonated",'
        '"arguments":{"Place":"public market","Instrument":"improvised explosive device","Target":"market"}}'
    ),
    (
        '{"event_type":"Life:Die","trigger":"killed",'
        '"arguments":{"Victim":"civilian","Place":"residential building","Instrument":"fire"}}'
    ),
    (
        "Pattern: conflict attack reports often include Attacker, Target, Instrument, and Place; "
        "death reports often include Victim, Agent or Instrument, and Place."
    ),
]


def main() -> None:
    build_index(RAG_EVENT_CORPUS)

    g = graph.build_graph()
    state = {
        # Data fields: raw user inputs plus current intermediate representations.
        "text": "A bomb exploded in a market",
        "raw_image": "example://images/market-scene.jpg",
        "event_type_mode": "closed_set",
        # image_desc is currently derived from raw_image outside this demo entry.
        "image_desc": "people running, smoke",
        "perception_summary": "",
        "search_query": "",
        "similar_events": [],
        "evidence": [],
        "fusion_context": empty_fusion_context(),
        "event": empty_event(),
        "memory": [],
        # Control fields
        "verified": False,
        "issues": [],
        "verifier_diagnostics": [],
        "verifier_confidence": 0.0,
        "verifier_reason": "",
        "repair_attempts": 0,
    }
    final = g.invoke(state)
    print(final["event"])


if __name__ == "__main__":
    main()

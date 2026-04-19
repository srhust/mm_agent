"""Run the LangGraph multimodal event extraction pipeline.

Current input contract: raw text + raw image.
In graph state, the raw text input lives in state["text"] and the primary raw
image input lives in state["raw_image"].
Current operational image representation: image_desc remains a derived
intermediate fallback until raw-image grounding is implemented.
"""

from __future__ import annotations

import mm_event_agent.graph as graph
from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import empty_event, empty_fusion_context, empty_layered_similar_events
from mm_event_agent.layered_rag import build_index

# Local layered RAG corpora: heterogeneous retrieval sources, kept separate.
RAG_EVENT_CORPUS = {
    "text_event_examples": [
        {
            "id": "ace_000123",
            "source_dataset": "ACE2005",
            "modality": "text",
            "event_type": "Conflict:Attack",
            "raw_text": "A bomb exploded in a crowded market, killing several civilians.",
            "trigger": {"text": "exploded", "span": {"start": 7, "end": 16}},
            "text_arguments": [
                {"role": "Instrument", "text": "bomb", "span": {"start": 2, "end": 6}},
                {"role": "Place", "text": "a crowded market", "span": {"start": 20, "end": 36}},
                {"role": "Target", "text": "civilians", "span": {"start": 53, "end": 62}},
            ],
            "pattern_summary": "Attack reports often mention trigger, place, instrument, and target.",
            "retrieval_text": "Conflict:Attack exploded bomb crowded market civilians Instrument Place Target",
        },
        {
            "id": "maven_arg_000001",
            "source_dataset": "MAVEN-ARG",
            "modality": "text",
            "event_type": "Life:Die",
            "raw_text": "A civilian was killed in a residential building fire.",
            "trigger": {"text": "killed", "span": {"start": 16, "end": 22}},
            "text_arguments": [
                {"role": "Victim", "text": "civilian", "span": {"start": 2, "end": 10}},
                {"role": "Place", "text": "a residential building", "span": {"start": 26, "end": 48}},
            ],
            "pattern_summary": "Death reports emphasize victim, cause or instrument, and place.",
            "retrieval_text": "Life:Die killed civilian residential building fire Victim Place Instrument",
        },
    ],
    "image_semantic_examples": [
        {
            "id": "swig_000456",
            "source_dataset": "SWiG",
            "modality": "image_semantics",
            "event_type": "Conflict:Attack",
            "image_desc": "Smoke rises from damaged vehicles near a burning building on a city street.",
            "visual_situation": "attack-like urban explosion scene",
            "image_arguments": [
                {"role": "Instrument", "label": "fire"},
                {"role": "Target", "label": "vehicle"},
                {"role": "Place", "label": "street"},
            ],
            "visual_pattern_summary": "Attack-like image with visible fire, vehicle damage, and urban place cues.",
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
            "text_cues": ["bomb", "gun", "missile", "explosive device", "fire"],
            "visual_cues": ["smoke", "flames", "weapon", "damaged vehicle", "debris"],
            "note": "In attack events, Instrument may be explicit in text, but in images it is often supported indirectly by smoke, fire, debris, or weapon-like objects.",
            "retrieval_text": "Conflict:Attack Instrument bomb gun missile explosive device fire smoke flames weapon damaged vehicle debris",
        }
    ],
}


def _initialize_rag_runtime() -> str:
    """Initialize RAG according to runtime feature flags.

    Persistent mode must not build the demo in-memory corpus. We still clear
    any demo in-memory state when demo retrieval is disabled by passing
    ``None`` into the compatibility-layer builder.
    """
    if settings.rag_use_persistent_index:
        build_index(None)
        return "persistent"
    if settings.rag_use_demo_corpus:
        build_index(RAG_EVENT_CORPUS)
        return "demo"
    build_index(None)
    return "disabled"


def main() -> None:
    _initialize_rag_runtime()

    g = graph.build_graph()
    state = {
        # Data fields: raw user inputs plus current intermediate representations.
        "text": "A bomb exploded in a market",
        "raw_image": "example://images/market-scene.jpg",
        "event_type_mode": settings.event_type_mode,
        "run_mode": settings.run_mode,
        "effective_search_enabled": settings.effective_search_enabled,
        # image_desc is currently derived from raw_image outside this demo entry.
        "image_desc": "people running, smoke",
        "perception_summary": "",
        "search_query": "",
        "similar_events": empty_layered_similar_events(),
        "evidence": [],
        "fusion_context": empty_fusion_context(),
        "event": empty_event(),
        "grounding_results": [],
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

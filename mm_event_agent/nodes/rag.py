"""Data node: layered local RAG returns structured retrieval sources."""

from __future__ import annotations

import time
from typing import Any, Mapping

from mm_event_agent.observability import log_node_event
from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import LayeredSimilarEvents, empty_layered_similar_events


def _retrieve_similar_events(
    raw_text: str,
    image_desc: str = "",
    event_type: str = "",
    top_k: int = 3,
    *,
    raw_image: Any | None = None,
) -> LayeredSimilarEvents:
    from mm_event_agent.layered_rag import retrieve as layered_retrieve

    return layered_retrieve(
        raw_text=raw_text,
        image_desc=image_desc,
        event_type=event_type,
        top_k=top_k,
        raw_image=raw_image,
    )


def rag(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read only data field text and write only data field similar_events."""
    started_at = time.perf_counter()
    raw_text = str(state.get("text") or "").strip()
    raw_image = state.get("raw_image")
    image_desc = str(state.get("image_desc") or "").strip()
    event_type = str(state.get("event", {}).get("event_type") or "").strip() if isinstance(state.get("event"), dict) else ""
    if not raw_text and not image_desc and not event_type:
        result = {"similar_events": empty_layered_similar_events()}
        log_node_event("local_rag", state, started_at, True, patterns=0, text_examples=0, image_examples=0, bridge_examples=0)
        return result
    try:
        similar_events = _retrieve_similar_events(
            raw_text=raw_text,
            image_desc=image_desc,
            event_type=event_type,
            top_k=settings.rag_default_top_k,
            raw_image=raw_image,
        )
        total = (
            len(similar_events["text_event_examples"])
            + len(similar_events["image_semantic_examples"])
            + len(similar_events["bridge_examples"])
        )
        result = {"similar_events": similar_events}
        log_node_event(
            "local_rag",
            state,
            started_at,
            True,
            patterns=total,
            text_examples=len(similar_events["text_event_examples"]),
            image_examples=len(similar_events["image_semantic_examples"]),
            bridge_examples=len(similar_events["bridge_examples"]),
        )
        return result
    except Exception as exc:
        log_node_event(
            "local_rag",
            state,
            started_at,
            False,
            error=str(exc),
            patterns=0,
            text_examples=0,
            image_examples=0,
            bridge_examples=0,
        )
        return {"similar_events": empty_layered_similar_events()}


run = rag

"""Data node: build a structured, JSON-serializable fusion_context."""

from __future__ import annotations

import time
from typing import Any, Mapping

from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import FusionContext, empty_fusion_context


def fusion(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build fusion_context from raw text plus the current derived image_desc."""
    started_at = time.perf_counter()
    try:
        evidence = state.get("evidence")
        fusion_context: FusionContext = {
            "raw_text": str(state.get("text") or ""),
            # Keep using the derived image description in fusion_context until
            # raw_image-grounded perception is implemented.
            "raw_image_desc": str(state.get("image_desc") or ""),
            "perception_summary": str(state.get("perception_summary") or ""),
            "patterns": list(state.get("similar_events")) if isinstance(state.get("similar_events"), list) else [],
            "evidence": list(evidence) if isinstance(evidence, list) else [],
        }
        result = {"fusion_context": fusion_context}
        log_node_event(
            "fusion",
            state,
            started_at,
            True,
            fusion_patterns=len(fusion_context["patterns"]),
            returned_evidence=len(fusion_context["evidence"]),
        )
        return result
    except Exception as exc:
        result = {"fusion_context": empty_fusion_context()}
        log_node_event("fusion", state, started_at, False, error=str(exc), fusion_patterns=0, returned_evidence=0)
        return result


run = fusion

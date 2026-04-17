"""Data node: build a structured, JSON-serializable fusion_context."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import FusionContext, empty_fusion_context, empty_layered_similar_events
from mm_event_agent.trace_utils import append_prompt_trace, merge_stage_outputs


def fusion(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build fusion_context from raw text plus the current derived image_desc."""
    started_at = time.perf_counter()
    try:
        evidence = state.get("evidence")
        raw_patterns = state.get("similar_events")
        fusion_context: FusionContext = {
            "raw_text": str(state.get("text") or ""),
            # Keep using the derived image description in fusion_context until
            # raw_image-grounded perception is implemented.
            "raw_image_desc": str(state.get("image_desc") or ""),
            "perception_summary": str(state.get("perception_summary") or ""),
            "patterns": raw_patterns if isinstance(raw_patterns, dict) else empty_layered_similar_events(),
            "evidence": list(evidence) if isinstance(evidence, list) else [],
        }
        audit_enabled = "prompt_trace" in state or "stage_outputs" in state
        similar_events_summary = {
            "text_event_examples": len(fusion_context["patterns"]["text_event_examples"]),
            "image_semantic_examples": len(fusion_context["patterns"]["image_semantic_examples"]),
            "bridge_examples": len(fusion_context["patterns"]["bridge_examples"]),
        }
        evidence_summary = {
            "count": len(fusion_context["evidence"]),
            "items": fusion_context["evidence"],
        }
        fusion_context_summary = {
            "raw_text": fusion_context["raw_text"],
            "raw_image_desc": fusion_context["raw_image_desc"],
            "perception_summary": fusion_context["perception_summary"],
            "similar_events_summary": similar_events_summary,
            "evidence_summary": evidence_summary,
        }
        result = {"fusion_context": fusion_context}
        if audit_enabled:
            result["prompt_trace"] = append_prompt_trace(
                state,
                {
                    "sample_id": str(state.get("sample_id") or ""),
                    "stage": "fusion",
                    "model_name": "",
                    "prompt_text": "Assemble fusion_context from sanitized text, image-side context, retrieval patterns, and evidence.",
                    "image_path": None,
                    "input_summary": {
                        "text_length": len(fusion_context["raw_text"]),
                        "has_image_desc": bool(fusion_context["raw_image_desc"]),
                        "has_perception_summary": bool(fusion_context["perception_summary"]),
                        "similar_events_summary": similar_events_summary,
                        "evidence_count": len(fusion_context["evidence"]),
                    },
                    "response_text": json.dumps(fusion_context_summary, ensure_ascii=False),
                    "parsed_output": fusion_context_summary,
                },
            )
            result["stage_outputs"] = merge_stage_outputs(
                state,
                {
                    "similar_events_summary": similar_events_summary,
                    "evidence_summary": evidence_summary,
                    "fusion_context_summary": fusion_context_summary,
                },
            )
        log_node_event(
            "fusion",
            state,
            started_at,
            True,
            fusion_patterns=(
                len(fusion_context["patterns"]["text_event_examples"])
                + len(fusion_context["patterns"]["image_semantic_examples"])
                + len(fusion_context["patterns"]["bridge_examples"])
            ),
            returned_evidence=len(fusion_context["evidence"]),
        )
        return result
    except Exception as exc:
        result = {"fusion_context": empty_fusion_context()}
        log_node_event("fusion", state, started_at, False, error=str(exc), fusion_patterns=0, returned_evidence=0)
        return result


run = fusion

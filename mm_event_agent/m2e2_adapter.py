"""Helpers for adapting M2E2-style samples to and from the agent contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import (
    align_text_grounded_event,
    empty_event,
    empty_fusion_context,
    empty_layered_similar_events,
    find_text_span,
    normalize_text_span,
)

M2E2_AGENT_INPUT_TEXT_KEYS = ("text",)
M2E2_AGENT_INPUT_TOKEN_KEYS = ("words", "tokens")
M2E2_AGENT_INPUT_IMAGE_KEYS = ("image",)
M2E2_EVALUATION_ONLY_FIELDS = (
    "event_type",
    "text_event_mentions",
    "text_arguments_flat",
    "image_event",
    "image_arguments_flat",
    "ground_truth",
    "event_mentions",
    "gold_event_mentions",
    "events",
    "gold_events",
    "labels",
    "annotations",
)


def get_m2e2_sample_id(sample: Mapping[str, Any]) -> str:
    for key in ("id", "sample_id", "sentence_id", "sent_id", "doc_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_m2e2_model_input(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": _extract_first_string(sample, M2E2_AGENT_INPUT_TEXT_KEYS),
        "tokens": _extract_first_string_list(sample, M2E2_AGENT_INPUT_TOKEN_KEYS),
        "image": _extract_first_string(sample, M2E2_AGENT_INPUT_IMAGE_KEYS),
    }


def m2e2_sample_to_agent_state(
    sample: Mapping[str, Any],
    image_root: str | Path,
    *,
    event_type_mode: str | None = None,
) -> dict[str, Any]:
    model_input = extract_m2e2_model_input(sample)
    raw_text = model_input["text"]
    words = model_input["tokens"]
    image_value = model_input["image"]
    image_path = Path(image_value)
    if image_value and not image_path.is_absolute():
        image_path = Path(image_root) / image_value

    return {
        "text": raw_text,
        "tokens": words,
        "raw_image": str(image_path) if image_value else None,
        "event_type_mode": str(event_type_mode or settings.event_type_mode or "closed_set"),
        "run_mode": settings.run_mode,
        "effective_search_enabled": settings.effective_search_enabled,
        "image_desc": "",
        "perception_summary": "",
        "search_query": "",
        "similar_events": empty_layered_similar_events(),
        "evidence": [],
        "fusion_context": {
            **empty_fusion_context(),
            "raw_text": raw_text,
            "text_tokens": words,
        },
        "event": empty_event(),
        "grounding_results": [],
        "memory": [],
        "prompt_trace": [],
        "stage_outputs": {},
        "repair_history": [],
        "verified": False,
        "issues": [],
        "verifier_diagnostics": [],
        "verifier_confidence": 0.0,
        "verifier_reason": "",
        "repair_attempts": 0,
    }


def extract_m2e2_gold_annotations(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return gold/annotation fields for offline comparison only."""
    event_mentions = _extract_first_list(sample, ("gold_event_mentions", "event_mentions", "events", "gold_events"))
    text_event_mentions = _extract_first_list(sample, ("text_event_mentions",))
    text_arguments_flat = _extract_first_list(sample, ("text_arguments_flat",))
    image_event = sample.get("image_event")
    image_arguments_flat = _extract_first_list(sample, ("image_arguments_flat",))
    ground_truth = sample.get("ground_truth")
    labels = sample.get("labels")
    annotations = sample.get("annotations")
    first_event = event_mentions[0] if event_mentions and isinstance(event_mentions[0], Mapping) else None
    trigger = first_event.get("trigger") if isinstance(first_event, Mapping) else None
    arguments = first_event.get("arguments") if isinstance(first_event, Mapping) and isinstance(first_event.get("arguments"), list) else []
    if not arguments and isinstance(first_event, Mapping) and isinstance(first_event.get("args"), list):
        arguments = first_event.get("args")

    return {
        "sample_id": get_m2e2_sample_id(sample),
        "event_type": (
            first_event.get("event_type") or first_event.get("type")
            if isinstance(first_event, Mapping)
            else sample.get("event_type")
        ),
        "trigger": trigger,
        "arguments": arguments,
        "text_event_mentions": text_event_mentions,
        "text_arguments_flat": text_arguments_flat,
        "image_event": image_event,
        "image_arguments_flat": image_arguments_flat,
        "ground_truth": ground_truth,
        "labels": labels,
        "annotations": annotations,
    }


def extract_m2e2_gold_record(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return evaluation-only gold fields for offline comparison."""
    return extract_m2e2_gold_annotations(sample)


def agent_output_to_m2e2_prediction(sample: Mapping[str, Any], agent_output: Mapping[str, Any]) -> dict[str, Any]:
    model_input = extract_m2e2_model_input(sample)
    raw_text = model_input["text"]
    words = model_input["tokens"]
    raw_event = agent_output.get("event")
    if isinstance(raw_event, dict) and raw_event.get("event_type"):
        aligned_event, _, _ = align_text_grounded_event(raw_event, raw_text, token_sequence=words)
    else:
        aligned_event = empty_event()

    prediction = {
        "event_type": aligned_event["event_type"],
        "trigger": _normalize_trigger(aligned_event.get("trigger"), raw_text, words),
        "text_arguments": _normalize_text_arguments(aligned_event.get("text_arguments"), raw_text, words),
        "image_arguments": _normalize_image_arguments(
            aligned_event.get("image_arguments"),
            agent_output.get("grounding_results"),
        ),
    }
    similar_events = agent_output.get("similar_events")
    return {
        "id": get_m2e2_sample_id(sample),
        "prediction": prediction,
        "verified": bool(agent_output.get("verified")),
        "issues": list(agent_output.get("issues", [])) if isinstance(agent_output.get("issues"), list) else [],
        "verifier_reason": str(agent_output.get("verifier_reason") or ""),
        "similar_events_summary": {
            "text": len(similar_events.get("text_event_examples", []))
            if isinstance(similar_events, Mapping) and isinstance(similar_events.get("text_event_examples"), list)
            else 0,
            "image": len(similar_events.get("image_semantic_examples", []))
            if isinstance(similar_events, Mapping) and isinstance(similar_events.get("image_semantic_examples"), list)
            else 0,
            "bridge": len(similar_events.get("bridge_examples", []))
            if isinstance(similar_events, Mapping) and isinstance(similar_events.get("bridge_examples"), list)
            else 0,
        },
    }


def _extract_first_string(sample: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_first_string_list(sample: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, list):
            normalized = [str(item) for item in value if str(item)]
            if normalized:
                return normalized
    return []


def _extract_first_list(sample: Mapping[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, list):
            return list(value)
    return []


def _normalize_trigger(trigger: Any, raw_text: str, words: list[str]) -> dict[str, Any] | None:
    if not isinstance(trigger, Mapping):
        return None
    text = str(trigger.get("text") or "").strip()
    if not text:
        return None
    span = normalize_text_span(trigger.get("span"))
    if span is None:
        span = find_text_span(raw_text, text, token_sequence=words)
    if span is None:
        return {"text": text, "start": None, "end": None}
    return {"text": text, "start": span["start"], "end": span["end"]}


def _normalize_text_arguments(arguments: Any, raw_text: str, words: list[str]) -> list[dict[str, Any]]:
    if not isinstance(arguments, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in arguments:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        role = str(item.get("role") or "").strip()
        if not text or not role:
            continue
        span = normalize_text_span(item.get("span"))
        if span is None:
            span = find_text_span(raw_text, text, token_sequence=words)
        normalized.append(
            {
                "role": role,
                "text": text,
                "start": span["start"] if span is not None else None,
                "end": span["end"] if span is not None else None,
            }
        )
    return normalized


def _normalize_image_arguments(arguments: Any, grounding_results: Any) -> list[dict[str, Any]]:
    if not isinstance(arguments, list):
        return []
    grounded_lookup = _build_grounded_bbox_lookup(grounding_results)
    normalized: list[dict[str, Any]] = []
    for item in arguments:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        if not role or not label:
            continue
        bbox = _normalize_bbox(item.get("bbox"))
        if bbox is None:
            bbox = grounded_lookup.get((role, label))
        grounding_status = str(item.get("grounding_status") or "").strip() or "unresolved"
        if bbox is None or grounding_status != "grounded":
            continue
        normalized.append(
            {
                "role": role,
                "bbox": bbox,
            }
        )
    return normalized


def _build_grounded_bbox_lookup(grounding_results: Any) -> dict[tuple[str, str], list[float]]:
    lookup: dict[tuple[str, str], list[float]] = {}
    if not isinstance(grounding_results, list):
        return lookup
    for item in grounding_results:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        bbox = _normalize_bbox(item.get("bbox"))
        if role and label and bbox is not None:
            lookup[(role, label)] = bbox
    return lookup


def _normalize_bbox(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    normalized: list[float] = []
    for value in bbox:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            return None
    return normalized

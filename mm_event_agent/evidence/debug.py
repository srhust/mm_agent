"""Lightweight multi-source evidence analysis helpers."""

from __future__ import annotations

import re
from typing import Any

from mm_event_agent.schemas import EvidenceSourceSnapshot, EvidenceSourceSummary, Event, empty_event


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) > 2}


def _normalized_event(raw_event: Any) -> Event:
    if not isinstance(raw_event, dict):
        return empty_event()
    event_type = str(raw_event.get("event_type") or "").strip()
    trigger = raw_event.get("trigger") if isinstance(raw_event.get("trigger"), dict) or raw_event.get("trigger") is None else None
    text_arguments = raw_event.get("text_arguments") if isinstance(raw_event.get("text_arguments"), list) else []
    image_arguments = raw_event.get("image_arguments") if isinstance(raw_event.get("image_arguments"), list) else []
    return {
        "event_type": event_type,
        "trigger": trigger,
        "text_arguments": text_arguments,
        "image_arguments": image_arguments,
    }


def _has_text_support(event: Event, raw_text: str) -> bool:
    source_text = str(raw_text or "")
    trigger = event.get("trigger")
    if isinstance(trigger, dict):
        trigger_text = str(trigger.get("text") or "").strip()
        if trigger_text and trigger_text in source_text:
            return True

    for item in event.get("text_arguments", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text and text in source_text:
            return True
    return False


def _has_image_support(event: Event, raw_image_desc: str, perception_summary: str = "") -> bool:
    image_side_available = bool(str(raw_image_desc or "").strip() or str(perception_summary or "").strip())
    image_arguments = event.get("image_arguments", [])
    return image_side_available and any(
        isinstance(item, dict) and str(item.get("role") or "").strip() and str(item.get("label") or "").strip()
        for item in image_arguments
    )


def _grounded_pairs(grounding_results: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not isinstance(grounding_results, list):
        return pairs
    for item in grounding_results:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        bbox = item.get("bbox")
        status = str(item.get("grounding_status") or "").strip()
        if role and label and status == "grounded" and isinstance(bbox, list) and len(bbox) == 4:
            pairs.add((role, label))
    return pairs


def _has_grounding_support(event: Event, grounding_results: Any) -> bool:
    grounded_pairs = _grounded_pairs(grounding_results)
    if not grounded_pairs:
        return False
    for item in event.get("image_arguments", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        bbox = item.get("bbox")
        status = str(item.get("grounding_status") or "").strip()
        if role and label and (role, label) in grounded_pairs:
            return True
        if status == "grounded" and isinstance(bbox, list) and len(bbox) == 4:
            return True
    return False


def _event_surface_tokens(event: Event) -> set[str]:
    parts: list[str] = [event.get("event_type", "")]
    trigger = event.get("trigger")
    if isinstance(trigger, dict):
        parts.append(str(trigger.get("text") or ""))
    for item in event.get("text_arguments", []):
        if isinstance(item, dict):
            parts.append(str(item.get("text") or ""))
            parts.append(str(item.get("role") or ""))
    for item in event.get("image_arguments", []):
        if isinstance(item, dict):
            parts.append(str(item.get("label") or ""))
            parts.append(str(item.get("role") or ""))
    return _tokenize(" ".join(parts))


def _has_external_evidence_support(event: Event, evidence: Any) -> bool:
    if not isinstance(evidence, list) or not evidence or not event.get("event_type"):
        return False
    event_tokens = _event_surface_tokens(event)
    if not event_tokens:
        return False
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_tokens = _tokenize(f'{item.get("title", "")} {item.get("snippet", "")}')
        if event_tokens & evidence_tokens:
            return True
    return False


def summarize_evidence_sources(
    raw_event: Any,
    raw_text: str = "",
    raw_image_desc: str = "",
    perception_summary: str = "",
    grounding_results: Any = None,
    evidence: Any = None,
) -> EvidenceSourceSummary:
    """Return compact booleans describing which evidence sources support the event."""
    event = _normalized_event(raw_event)
    if not event["event_type"]:
        return {
            "text_support": False,
            "image_support": False,
            "grounding_support": False,
            "external_evidence_support": False,
        }
    return {
        "text_support": _has_text_support(event, raw_text),
        "image_support": _has_image_support(event, raw_image_desc, perception_summary),
        "grounding_support": _has_grounding_support(event, grounding_results),
        "external_evidence_support": _has_external_evidence_support(event, evidence),
    }


def build_evidence_source_snapshot(
    raw_event: Any,
    raw_text: str = "",
    raw_image_desc: str = "",
    perception_summary: str = "",
    grounding_results: Any = None,
    evidence: Any = None,
) -> EvidenceSourceSnapshot:
    """Build a compact evaluation snapshot for multi-source support analysis."""
    event = _normalized_event(raw_event)
    summary = summarize_evidence_sources(
        raw_event=event,
        raw_text=raw_text,
        raw_image_desc=raw_image_desc,
        perception_summary=perception_summary,
        grounding_results=grounding_results,
        evidence=evidence,
    )
    return {
        "event_type": event["event_type"],
        "text_support": summary["text_support"],
        "image_support": summary["image_support"],
        "grounding_support": summary["grounding_support"],
        "external_evidence_support": summary["external_evidence_support"],
        "final_event": event,
    }

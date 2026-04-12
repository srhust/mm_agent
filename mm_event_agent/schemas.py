"""Typed data contracts and validation helpers."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from mm_event_agent.ontology import is_supported_event_type


class ValidationError(ValueError):
    """Raised when a payload does not match the expected schema."""


class TextSpan(TypedDict):
    start: int
    end: int


class Trigger(TypedDict):
    text: str
    modality: str
    span: TextSpan | None


class TextArgument(TypedDict):
    role: str
    text: str
    span: TextSpan | None


class ImageArgument(TypedDict):
    role: str
    label: str
    bbox: list[float] | None
    grounding_status: str


class Event(TypedDict):
    event_type: str
    trigger: Trigger | None
    text_arguments: list[TextArgument]
    image_arguments: list[ImageArgument]


class EvidenceItem(TypedDict):
    title: str
    snippet: str
    url: str
    source_type: str
    published_at: str | None
    score: float


class VerificationDiagnostic(TypedDict):
    field_path: str
    issue_type: str
    suggested_action: str


class FusionContext(TypedDict):
    raw_text: str
    raw_image_desc: str
    perception_summary: str
    patterns: list[dict[str, Any]]
    evidence: list[EvidenceItem]


def empty_event() -> Event:
    return {
        "event_type": "",
        "trigger": None,
        "text_arguments": [],
        "image_arguments": [],
    }


def empty_fusion_context() -> FusionContext:
    return {
        "raw_text": "",
        "raw_image_desc": "",
        "perception_summary": "",
        "patterns": [],
        "evidence": [],
    }


def extract_json_object(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start, end = payload.find("{"), payload.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(payload[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_event(data: Any) -> Event:
    if not isinstance(data, dict):
        raise ValidationError("event must be a JSON object")

    event_type = data.get("event_type")
    trigger = data.get("trigger")
    text_arguments = data.get("text_arguments")
    image_arguments = data.get("image_arguments")

    if not isinstance(event_type, str):
        raise ValidationError("event.event_type must be a string")
    if not is_supported_event_type(event_type.strip()):
        raise ValidationError("event.event_type must be in the supported ontology")
    if trigger is not None and not isinstance(trigger, dict):
        raise ValidationError("event.trigger must be an object or null")
    if not isinstance(text_arguments, list):
        raise ValidationError("event.text_arguments must be a list")
    if not isinstance(image_arguments, list):
        raise ValidationError("event.image_arguments must be a list")

    normalized_trigger = _validate_trigger(trigger)
    normalized_text_arguments = [_validate_text_argument(item) for item in text_arguments]
    normalized_image_arguments = [_validate_image_argument(item) for item in image_arguments]

    return {
        "event_type": event_type.strip(),
        "trigger": normalized_trigger,
        "text_arguments": normalized_text_arguments,
        "image_arguments": normalized_image_arguments,
    }


def parse_event_json(text: str) -> Event:
    data = extract_json_object(text)
    if data is None:
        raise ValidationError("event output is not valid JSON")
    return validate_event(data)


def enforce_strict_text_grounding(event: Event, raw_text: str) -> Event:
    anchor_spans: list[TextSpan] = []
    trigger = event["trigger"]
    if trigger is not None:
        trigger_span = find_text_span(raw_text, trigger["text"], anchor_spans=anchor_spans)
        if trigger_span is not None:
            trigger = {
                "text": trigger["text"],
                "modality": "text",
                "span": trigger_span,
            }
            anchor_spans.append(trigger_span)
        else:
            trigger = None

    text_arguments: list[TextArgument] = []
    for item in event["text_arguments"]:
        span = find_text_span(raw_text, item["text"], anchor_spans=anchor_spans)
        if span is not None:
            text_arguments.append(
                {
                    "role": item["role"],
                    "text": item["text"],
                    "span": span,
                }
            )
            anchor_spans.append(span)

    return {
        "event_type": event["event_type"],
        "trigger": trigger,
        "text_arguments": text_arguments,
        "image_arguments": event["image_arguments"],
    }


def attach_text_spans(event: Event, raw_text: str) -> Event:
    """Backward-compatible alias for strict text grounding post-processing."""
    return enforce_strict_text_grounding(event, raw_text)


def validate_evidence_item(data: Any) -> EvidenceItem:
    if not isinstance(data, dict):
        raise ValidationError("evidence item must be an object")

    title = data.get("title")
    snippet = data.get("snippet")
    url = data.get("url")
    source_type = data.get("source_type")
    published_at = data.get("published_at")
    raw_score = data.get("score", 0.0)

    if not isinstance(title, str) or not title.strip():
        raise ValidationError("evidence.title must be a non-empty string")
    if not isinstance(snippet, str) or not snippet.strip():
        raise ValidationError("evidence.snippet must be a non-empty string")
    if not isinstance(url, str) or not url.strip():
        raise ValidationError("evidence.url must be a non-empty string")
    if not isinstance(source_type, str) or not source_type.strip():
        raise ValidationError("evidence.source_type must be a non-empty string")
    if published_at is not None and not isinstance(published_at, str):
        raise ValidationError("evidence.published_at must be a string or null")

    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValidationError("evidence.score must be numeric") from exc

    return {
        "title": title.strip(),
        "snippet": snippet.strip(),
        "url": url.strip(),
        "source_type": source_type.strip(),
        "published_at": published_at.strip() if isinstance(published_at, str) and published_at.strip() else None,
        "score": max(0.0, min(1.0, score)),
    }


def _validate_trigger(data: Any) -> Trigger | None:
    if data is None:
        return None
    text = data.get("text")
    modality = data.get("modality")
    span = data.get("span")
    if not isinstance(text, str):
        raise ValidationError("event.trigger.text must be a string")
    if modality != "text":
        raise ValidationError('event.trigger.modality must be "text"')
    return {
        "text": text,
        "modality": "text",
        "span": _validate_span(span),
    }


def _validate_text_argument(data: Any) -> TextArgument:
    if not isinstance(data, dict):
        raise ValidationError("event.text_arguments items must be objects")
    role = data.get("role")
    text = data.get("text")
    span = data.get("span")
    if not isinstance(role, str) or not role.strip():
        raise ValidationError("text argument role must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("text argument text must be a non-empty string")
    return {
        "role": role.strip(),
        "text": text,
        "span": _validate_span(span),
    }


def _validate_image_argument(data: Any) -> ImageArgument:
    if not isinstance(data, dict):
        raise ValidationError("event.image_arguments items must be objects")
    role = data.get("role")
    label = data.get("label")
    bbox = data.get("bbox")
    grounding_status = data.get("grounding_status")
    if not isinstance(role, str) or not role.strip():
        raise ValidationError("image argument role must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise ValidationError("image argument label must be a non-empty string")
    if not isinstance(grounding_status, str) or not grounding_status.strip():
        raise ValidationError("image argument grounding_status must be a non-empty string")

    norm_bbox: list[float] | None
    if bbox is None:
        if grounding_status != "unresolved":
            raise ValidationError("image argument without bbox must be marked unresolved")
        norm_bbox = None
    else:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValidationError("image argument bbox must be a list of four numbers or null")
        norm_bbox = []
        for value in bbox:
            try:
                norm_bbox.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValidationError("image argument bbox values must be numeric") from exc
    return {
        "role": role.strip(),
        "label": label.strip(),
        "bbox": norm_bbox,
        "grounding_status": grounding_status.strip(),
    }


def _validate_span(span: Any) -> list[int] | None:
    if span is None:
        return None
    if isinstance(span, dict):
        start = span.get("start")
        end = span.get("end")
    elif isinstance(span, list) and len(span) == 2:
        start, end = span
    else:
        raise ValidationError('span must be null, {"start": int, "end": int}, or [start, end]')
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValidationError("span values must be valid non-negative integers")
    return {"start": start, "end": end}


def find_all_text_occurrences(source_text: str, value: str) -> list[TextSpan]:
    source = str(source_text or "")
    target = str(value or "")
    if not source or not target:
        return []

    matches: list[TextSpan] = []
    start = 0
    while True:
        index = source.find(target, start)
        if index < 0:
            break
        matches.append({"start": index, "end": index + len(target)})
        start = index + 1
    return matches


def choose_best_span(
    occurrences: list[TextSpan],
    anchor_spans: list[TextSpan] | None = None,
) -> TextSpan | None:
    if not occurrences:
        return None
    if len(occurrences) == 1:
        return occurrences[0]

    anchors = [
        span
        for span in (anchor_spans or [])
        if isinstance(span, dict)
        and isinstance(span.get("start"), int)
        and isinstance(span.get("end"), int)
    ]
    if not anchors:
        return None

    scored: list[tuple[float, TextSpan]] = []
    for occurrence in occurrences:
        occ_mid = (occurrence["start"] + occurrence["end"]) / 2.0
        min_distance = min(
            abs(occ_mid - ((anchor["start"] + anchor["end"]) / 2.0))
            for anchor in anchors
        )
        scored.append((min_distance, occurrence))

    scored.sort(key=lambda item: item[0])
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def find_text_span(
    source_text: str,
    value: str,
    anchor_spans: list[TextSpan] | None = None,
) -> TextSpan | None:
    occurrences = find_all_text_occurrences(source_text, value)
    return choose_best_span(occurrences, anchor_spans=anchor_spans)

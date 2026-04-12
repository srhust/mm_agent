"""Typed data contracts and validation helpers."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict


class ValidationError(ValueError):
    """Raised when a payload does not match the expected schema."""


class Event(TypedDict):
    event_type: str
    trigger: str
    arguments: dict[str, Any]


class EvidenceItem(TypedDict):
    title: str
    snippet: str
    url: str
    source_type: str
    published_at: str | None
    score: float


class FusionContext(TypedDict):
    raw_text: str
    raw_image_desc: str
    perception_summary: str
    patterns: list[dict[str, Any]]
    evidence: list[EvidenceItem]


def empty_event() -> Event:
    return {
        "event_type": "",
        "trigger": "",
        "arguments": {},
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
    arguments = data.get("arguments")

    if not isinstance(event_type, str):
        raise ValidationError("event.event_type must be a string")
    if not isinstance(trigger, str):
        raise ValidationError("event.trigger must be a string")
    if not isinstance(arguments, dict):
        raise ValidationError("event.arguments must be an object")

    return {
        "event_type": event_type.strip(),
        "trigger": trigger.strip(),
        "arguments": arguments,
    }


def parse_event_json(text: str) -> Event:
    data = extract_json_object(text)
    if data is None:
        raise ValidationError("event output is not valid JSON")
    return validate_event(data)


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

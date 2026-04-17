"""Helpers for audit-friendly prompt traces and stage output snapshots."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping


def make_json_safe(value: Any) -> Any:
    """Return a JSON-serializable clone, falling back to string conversion."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except Exception:
        if isinstance(value, Mapping):
            return {str(key): make_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [make_json_safe(item) for item in value]
        return str(value)


def safe_image_reference(raw_image: Any) -> str | None:
    """Expose a safe image reference without embedding image bytes or data URLs."""
    if isinstance(raw_image, str):
        candidate = raw_image.strip()
        if not candidate:
            return None
        if candidate.startswith("data:"):
            return "[data-url omitted]"
        return candidate
    if isinstance(raw_image, (bytes, bytearray)):
        return "[image-bytes omitted]"
    return None


def append_prompt_trace(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    existing = state.get("prompt_trace")
    trace: list[dict[str, Any]] = list(existing) if isinstance(existing, list) else []
    trace.append(dict(make_json_safe(record)))
    return trace


def merge_stage_outputs(
    state: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    existing = state.get("stage_outputs")
    merged = deepcopy(existing) if isinstance(existing, dict) else {}
    for key, value in updates.items():
        merged[str(key)] = make_json_safe(value)
    return merged


def append_repair_history(
    state: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    existing = state.get("repair_history")
    history: list[dict[str, Any]] = list(existing) if isinstance(existing, list) else []
    history.append(dict(make_json_safe(entry)))
    return history

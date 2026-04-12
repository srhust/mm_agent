"""Runtime memory: append raw text plus the current derived image_desc."""

from __future__ import annotations

from typing import Any, Mapping


def memory(state: Mapping[str, Any]) -> dict[str, Any]:
    """Append the current raw text and derived image_desc to memory."""
    text = "" if state.get("text") is None else str(state.get("text"))
    image_desc = "" if state.get("image_desc") is None else str(state.get("image_desc"))
    return {"memory": [{"text": text, "image_desc": image_desc}]}


run = memory

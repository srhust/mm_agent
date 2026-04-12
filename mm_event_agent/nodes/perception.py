"""Data node: merge raw text plus image-side description into perception_summary.

The original user image is carried in state["raw_image"], but the current
pipeline still summarizes image information through state["image_desc"] as an
intermediate representation. Raw-image perception / detector grounding is not
implemented here yet.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from mm_event_agent.observability import log_node_event


def perception(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read text plus image_desc and write the intermediate perception summary."""
    started_at = time.perf_counter()
    try:
        text = "" if state.get("text") is None else str(state.get("text"))
        image_desc = "" if state.get("image_desc") is None else str(state.get("image_desc"))
        perception_summary = f"Text: {text}\nImage: {image_desc}"
        result = {"perception_summary": perception_summary}
        log_node_event("perception", state, started_at, True)
        return result
    except Exception as exc:
        log_node_event("perception", state, started_at, False, error=str(exc))
        return {"perception_summary": ""}


run = perception

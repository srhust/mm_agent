"""Data node: build a structured, JSON-serializable fusion_context."""

from __future__ import annotations

from typing import Any, Mapping


def fusion(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read only data fields and write only data field fusion_context."""
    fusion_context = {
        "input": str(state.get("perception_summary") or ""),
        "patterns": list(state.get("similar_events")) if isinstance(state.get("similar_events"), list) else [],
        "evidence": list(state.get("evidence")) if isinstance(state.get("evidence"), list) else [],
    }
    return {"fusion_context": fusion_context}


run = fusion

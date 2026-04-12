"""LangGraph global state definition."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """State for the multimodal event extraction agent.

    Keep data fields and control fields separate across nodes.
    """

    # Data fields: business input, intermediate context, and final output.
    text: str
    image_desc: str
    perception_summary: str
    similar_events: list[dict[str, Any]]
    evidence: list[dict[str, str]]
    fusion_context: dict[str, Any]
    event: str
    memory: list[Any]

    # Control fields: verifier/repair loop only.
    verified: bool
    issues: list[str]
    verifier_confidence: float
    verifier_reason: str
    repair_attempts: int

"""LangGraph global state definition."""

from __future__ import annotations

from typing import Any, TypedDict

from mm_event_agent.schemas import Event, EvidenceItem, FusionContext, VerificationDiagnostic


class AgentState(TypedDict):
    """State for the multimodal event extraction agent.

    Keep data fields and control fields separate across nodes.
    """

    # Data fields: business input, intermediate context, and final output.
    text: str
    image_desc: str
    perception_summary: str
    search_query: str
    similar_events: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    fusion_context: FusionContext
    event: Event
    memory: list[Any]

    # Control fields: verifier/repair loop only.
    verified: bool
    issues: list[str]
    verifier_diagnostics: list[VerificationDiagnostic]
    verifier_confidence: float
    verifier_reason: str
    repair_attempts: int

"""LangGraph global state definition."""

from __future__ import annotations

from typing import Any, TypedDict

from mm_event_agent.schemas import Event, EvidenceItem, FusionContext, VerificationDiagnostic


class AgentState(TypedDict):
    """State for the multimodal event extraction agent.

    The user-facing input contract is raw text plus raw image.
    In the graph state, state["text"] stores the raw text input and
    state["raw_image"] stores the primary image input object.
    The current pipeline still uses state["image_desc"] as a derived
    intermediate image-side representation until raw-image perception /
    grounding is implemented.
    Keep data fields and control fields separate across nodes.
    """

    # Data fields: business input, intermediate context, and final output.
    # Raw user text input.
    text: str
    # Primary raw user image input. This can hold bytes, a local path, or a URI,
    # but is not consumed for detector grounding yet.
    raw_image: Any
    # Stage A event type selection mode: "closed_set" or "transfer".
    event_type_mode: str
    # Current derived intermediate representation of raw_image used by the
    # existing image-side extraction and verification path.
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

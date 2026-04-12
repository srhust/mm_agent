"""Optional image grounding executors.

raw_image is the primary image input carried in graph state.
image_desc remains the current intermediate representation used by the
existing extraction / verifier path.
Grounding executors live here as a next-stage spatial grounding layer.
"""

from mm_event_agent.grounding.florence2_hf import (
    Florence2HFGrounder,
    apply_grounding_results_to_event,
    execute_grounding_requests,
)

__all__ = [
    "Florence2HFGrounder",
    "apply_grounding_results_to_event",
    "execute_grounding_requests",
]

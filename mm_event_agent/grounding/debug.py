"""Grounding observability and evaluation-friendly helpers.

These helpers keep grounding analysis lightweight:
- summarize grounding activity with compact counts
- compare image arguments before grounding, grounding results, and later states

raw_image remains the primary image input.
image_desc remains the current intermediate representation used by the
existing extraction / verifier path.
"""

from __future__ import annotations

from typing import Any

from mm_event_agent.schemas import GroundingSummary


def summarize_grounding_activity(
    image_arguments_before: Any,
    grounding_requests: Any = None,
    grounding_results: Any = None,
    image_arguments_after: Any = None,
) -> GroundingSummary:
    """Return compact grounding counters for logging and analysis."""
    before_items = image_arguments_before if isinstance(image_arguments_before, list) else []
    request_items = grounding_requests if isinstance(grounding_requests, list) else []
    result_items = grounding_results if isinstance(grounding_results, list) else []
    after_items = image_arguments_after if isinstance(image_arguments_after, list) else []

    unresolved_image_arguments = 0
    for item in before_items:
        if isinstance(item, dict) and item.get("bbox") is None and item.get("grounding_status") == "unresolved":
            unresolved_image_arguments += 1

    grounded_results = 0
    failed_grounding_results = 0
    for item in result_items:
        if not isinstance(item, dict):
            continue
        status = item.get("grounding_status")
        bbox = item.get("bbox")
        if status == "grounded" and isinstance(bbox, list) and len(bbox) == 4:
            grounded_results += 1
        elif status == "failed":
            failed_grounding_results += 1

    applied_grounded_bboxes = 0
    max_len = min(len(before_items), len(after_items))
    for index in range(max_len):
        before_item = before_items[index]
        after_item = after_items[index]
        if not isinstance(before_item, dict) or not isinstance(after_item, dict):
            continue
        before_bbox = before_item.get("bbox")
        after_bbox = after_item.get("bbox")
        before_status = before_item.get("grounding_status")
        after_status = after_item.get("grounding_status")
        if (
            before_bbox is None
            and before_status == "unresolved"
            and isinstance(after_bbox, list)
            and len(after_bbox) == 4
            and after_status == "grounded"
        ):
            applied_grounded_bboxes += 1

    return {
        "unresolved_image_arguments": unresolved_image_arguments,
        "grounding_requests": len(request_items),
        "grounded_results": grounded_results,
        "failed_grounding_results": failed_grounding_results,
        "applied_grounded_bboxes": applied_grounded_bboxes,
    }


def compare_grounding_stages(
    image_arguments_before: Any,
    grounding_results: Any,
    image_arguments_after: Any,
) -> dict[str, Any]:
    """Build a small debug snapshot for before/results/after comparison."""
    summary = summarize_grounding_activity(
        image_arguments_before=image_arguments_before,
        grounding_requests=None,
        grounding_results=grounding_results,
        image_arguments_after=image_arguments_after,
    )
    return {
        "before_image_arguments": image_arguments_before if isinstance(image_arguments_before, list) else [],
        "grounding_results": grounding_results if isinstance(grounding_results, list) else [],
        "after_image_arguments": image_arguments_after if isinstance(image_arguments_after, list) else [],
        "summary": summary,
    }

"""Control node: validate event against fused context and update control fields."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.ontology import (
    format_event_schema_for_prompt,
    format_full_ontology_for_prompt,
    get_allowed_image_roles,
    get_allowed_text_roles,
    is_supported_event_type,
)
from mm_event_agent.observability import log_node_event
from mm_event_agent.runtime_config import settings
from mm_event_agent.evidence.debug import summarize_evidence_sources
from mm_event_agent.grounding.debug import summarize_grounding_activity
from mm_event_agent.schemas import (
    VerificationDiagnostic,
    describe_text_argument_normalization,
    empty_event,
    empty_fusion_context,
    empty_layered_similar_events,
    extract_json_object,
    resolve_text_token_sequence,
    tokenize_text,
    validate_event,
)
from mm_event_agent.trace_utils import append_prompt_trace, merge_stage_outputs, safe_image_reference

_llm: ChatOpenAI | None = None
_GENERIC_WEAK_PLACE_LABELS = {
    "street",
    "road",
    "outdoors",
    "outdoor",
    "outside",
    "scene",
    "background",
    "area",
    "crowd",
}


ROLE_CONFUSION_GUIDANCE = [
    "Attacker vs Target: the initiator of violence is not the entity harmed or attacked.",
    "Agent vs Person: in arrest events, Agent is the authority and Person is the detainee.",
    "Destination vs Origin vs Place: Destination is where movement ends, Origin is where movement starts, and Place is a general event location role only when defined for that event type.",
    "Entity vs Participant: Entity is used for demonstrators or communicators depending on the event schema, while Participant is used for people or groups in a meeting.",
    "Victim vs Target: Victim is the person who dies in Life:Die, while Target is the person, object, or place under attack in Conflict:Attack.",
]


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "temperature": 0,
            "timeout": settings.openai_timeout_seconds,
        }
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _llm = ChatOpenAI(**kwargs)
    return _llm


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _perception_image_signal(perception_summary: str) -> str:
    summary = str(perception_summary or "").strip()
    if not summary:
        return ""
    match = re.search(r"(?im)^image:\s*(.*)$", summary)
    if match is not None:
        return match.group(1).strip()
    return summary


def _has_usable_image_evidence(raw_image: Any, raw_image_desc: str, perception_summary: str) -> bool:
    if isinstance(raw_image, (bytes, bytearray)):
        return True
    if isinstance(raw_image, str) and raw_image.strip():
        return True
    if str(raw_image_desc or "").strip():
        return True
    return bool(_perception_image_signal(perception_summary))


def _normalize_verdict_payload(
    data: dict[str, Any],
) -> tuple[str, list[str], bool, float, str, list[VerificationDiagnostic]]:
    v = str(data.get("verdict", "NO")).strip().upper()
    verdict = "YES" if v == "YES" else "NO"
    raw_issues = data.get("issues")
    if raw_issues is None:
        issues: list[str] = []
    elif isinstance(raw_issues, list):
        issues = [str(x) for x in raw_issues]
    else:
        issues = [str(raw_issues)]
    if verdict == "YES":
        issues = []
    raw_confidence = data.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason") or "").strip()
    diagnostics = _normalize_diagnostics(data.get("diagnostics"))
    verified = verdict == "YES"
    return verdict, issues, verified, confidence, reason, diagnostics


def _normalize_diagnostics(raw: Any) -> list[VerificationDiagnostic]:
    if not isinstance(raw, list):
        return []
    out: list[VerificationDiagnostic] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        issue_type = str(item.get("issue_type") or "").strip()
        suggested_action = str(item.get("suggested_action") or "").strip()
        if field_path and issue_type and suggested_action:
            out.append(
                {
                    "field_path": field_path,
                    "issue_type": issue_type,
                    "suggested_action": suggested_action,
                }
            )
    return out


def _get_text_token_sequence(state: Mapping[str, Any], fusion_context: Mapping[str, Any], raw_text: str) -> list[str]:
    for container in (fusion_context, state):
        for key in ("text_tokens", "token_sequence", "tokens"):
            value = container.get(key) if isinstance(container, Mapping) else None
            if isinstance(value, list) and value:
                return resolve_text_token_sequence(raw_text, value)
    return resolve_text_token_sequence(raw_text)


def _is_valid_span(span: Any, source_tokens: list[str]) -> bool:
    if not isinstance(span, dict):
        return False
    start = span.get("start")
    end = span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < start or end > len(source_tokens):
        return False
    return True


def _validate_trigger_fields(
    event: Mapping[str, Any],
    raw_text: str,
    text_tokens: list[str],
) -> tuple[list[str], list[VerificationDiagnostic]]:
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    trigger = event.get("trigger")
    if trigger is None:
        return issues, diagnostics
    if not isinstance(trigger, dict):
        return ["invalid trigger object"], [
            {
                "field_path": "trigger",
                "issue_type": "invalid_object",
                "suggested_action": "drop_or_rebuild",
            }
        ]

    if trigger.get("modality") != "text":
        issues.append("invalid trigger modality")
        diagnostics.append(
            {
                "field_path": "trigger.modality",
                "issue_type": "invalid_modality",
                "suggested_action": "set_text_modality_or_drop",
            }
        )

    span = trigger.get("span")
    text = str(trigger.get("text") or "")
    if span is not None:
        if not _is_valid_span(span, text_tokens):
            issues.append("invalid trigger span")
            diagnostics.append(
                {
                    "field_path": "trigger.span",
                    "issue_type": "invalid_span",
                    "suggested_action": "realign_or_drop",
                }
            )
        elif text_tokens[span["start"] : span["end"]] != tokenize_text(text):
            issues.append("trigger text/span mismatch")
            diagnostics.append(
                {
                    "field_path": "trigger.span",
                    "issue_type": "span_mismatch",
                    "suggested_action": "realign_or_drop",
                }
            )
    return issues, diagnostics


def _validate_event_type_and_roles(
    event: Mapping[str, Any],
) -> tuple[list[str], list[VerificationDiagnostic]]:
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    event_type = str(event.get("event_type") or "").strip()
    if not is_supported_event_type(event_type):
        issues.append("invalid event_type for ontology")
        diagnostics.append(
            {
                "field_path": "event_type",
                "issue_type": "unsupported_event_type",
                "suggested_action": "set_supported_event_type",
            }
        )
        return issues, diagnostics

    allowed_text_roles = set(get_allowed_text_roles(event_type))
    allowed_image_roles = set(get_allowed_image_roles(event_type))

    text_arguments = event.get("text_arguments")
    if isinstance(text_arguments, list):
        for index, item in enumerate(text_arguments):
            if isinstance(item, dict):
                role = str(item.get("role") or "").strip()
                if role and role not in allowed_text_roles:
                    issues.append(f"invalid text role at index {index}")
                    diagnostics.append(
                        {
                            "field_path": f"text_arguments[{index}].role",
                            "issue_type": "invalid_role",
                            "suggested_action": "replace_or_drop",
                        }
                    )

    image_arguments = event.get("image_arguments")
    if isinstance(image_arguments, list):
        for index, item in enumerate(image_arguments):
            if isinstance(item, dict):
                role = str(item.get("role") or "").strip()
                if role and role not in allowed_image_roles:
                    issues.append(f"invalid image role at index {index}")
                    diagnostics.append(
                        {
                            "field_path": f"image_arguments[{index}].role",
                            "issue_type": "invalid_role",
                            "suggested_action": "replace_or_drop",
                        }
                    )
    return issues, diagnostics


def _validate_text_argument_fields(
    event: Mapping[str, Any],
    raw_text: str,
    text_tokens: list[str],
) -> tuple[list[str], list[VerificationDiagnostic]]:
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    text_arguments = event.get("text_arguments")
    if not isinstance(text_arguments, list):
        return ["invalid text_arguments list"], [
            {
                "field_path": "text_arguments",
                "issue_type": "invalid_list",
                "suggested_action": "drop_or_rebuild",
            }
        ]

    for index, item in enumerate(text_arguments):
        if not isinstance(item, dict):
            issues.append(f"invalid text argument object at index {index}")
            diagnostics.append(
                {
                    "field_path": f"text_arguments[{index}]",
                    "issue_type": "invalid_object",
                    "suggested_action": "drop_or_rebuild",
                }
            )
            continue
        span = item.get("span")
        text = str(item.get("text") or "")
        if span is not None:
            if not _is_valid_span(span, text_tokens):
                issues.append(f"invalid text argument span at index {index}")
                diagnostics.append(
                    {
                        "field_path": f"text_arguments[{index}].span",
                        "issue_type": "invalid_span",
                        "suggested_action": "realign_or_drop",
                    }
                )
            elif text_tokens[span["start"] : span["end"]] != tokenize_text(text):
                issues.append(f"text argument span mismatch at index {index}")
                diagnostics.append(
                    {
                        "field_path": f"text_arguments[{index}].span",
                        "issue_type": "span_mismatch",
                        "suggested_action": "realign_or_drop",
                    }
                )
            else:
                normalization = describe_text_argument_normalization(text, span, text_tokens)
                if normalization["has_determiner"]:
                    issues.append(f"text argument includes determiner at index {index}")
                    diagnostics.append(
                        {
                            "field_path": f"text_arguments[{index}].text",
                            "issue_type": "contains_determiner",
                            "suggested_action": "shrink_to_head_word",
                        }
                    )
                if normalization["has_quantity"]:
                    issues.append(f"text argument includes quantity word at index {index}")
                    diagnostics.append(
                        {
                            "field_path": f"text_arguments[{index}].text",
                            "issue_type": "contains_quantity",
                            "suggested_action": "shrink_to_head_word",
                        }
                    )
                if normalization["is_broader_than_preferred"]:
                    issues.append(f"text argument is not normalized to preferred head-word span at index {index}")
                    diagnostics.append(
                        {
                            "field_path": f"text_arguments[{index}].span",
                            "issue_type": "unnormalized_head_word",
                            "suggested_action": "shrink_to_head_word",
                        }
                    )
    return issues, diagnostics


def _validate_image_argument_fields(event: Mapping[str, Any]) -> tuple[list[str], list[VerificationDiagnostic]]:
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    image_arguments = event.get("image_arguments")
    if not isinstance(image_arguments, list):
        return ["invalid image_arguments list"], [
            {
                "field_path": "image_arguments",
                "issue_type": "invalid_list",
                "suggested_action": "drop_or_rebuild",
            }
        ]

    for index, item in enumerate(image_arguments):
        if not isinstance(item, dict):
            issues.append(f"invalid image argument object at index {index}")
            diagnostics.append(
                {
                    "field_path": f"image_arguments[{index}]",
                    "issue_type": "invalid_object",
                    "suggested_action": "drop_or_rebuild",
                }
            )
            continue
        role = item.get("role")
        label = item.get("label")
        bbox = item.get("bbox")
        grounding_status = item.get("grounding_status")
        if not isinstance(role, str) or not role.strip():
            issues.append(f"invalid image argument role at index {index}")
            diagnostics.append(
                {
                    "field_path": f"image_arguments[{index}].role",
                    "issue_type": "missing_or_empty",
                    "suggested_action": "fill_or_drop",
                }
            )
        if not isinstance(label, str) or not label.strip():
            issues.append(f"invalid image argument label at index {index}")
            diagnostics.append(
                {
                    "field_path": f"image_arguments[{index}].label",
                    "issue_type": "missing_or_empty",
                    "suggested_action": "fill_or_drop",
                }
            )
        normalized_role = str(role or "").strip()
        normalized_label = str(label or "").strip().lower()
        if normalized_role == "Place" and normalized_label in _GENERIC_WEAK_PLACE_LABELS:
            issues.append(f"weak generic Place image argument at index {index}")
            diagnostics.append(
                {
                    "field_path": f"image_arguments[{index}].label",
                    "issue_type": "generic_weak_place",
                    "suggested_action": "drop_or_replace_with_directly_visible_place",
                }
            )
        if bbox is None:
            if grounding_status != "unresolved":
                issues.append(f"image argument unresolved grounding missing at index {index}")
                diagnostics.append(
                    {
                        "field_path": f"image_arguments[{index}].grounding_status",
                        "issue_type": "invalid_grounding_status",
                        "suggested_action": "mark_unresolved_or_drop",
                    }
                )
            continue
        if grounding_status != "grounded":
            issues.append(f"image argument grounded bbox missing grounded status at index {index}")
            diagnostics.append(
                {
                    "field_path": f"image_arguments[{index}].grounding_status",
                    "issue_type": "invalid_grounding_status",
                    "suggested_action": "mark_grounded_or_drop_bbox",
                }
            )
        if not isinstance(bbox, list) or len(bbox) != 4:
            issues.append(f"invalid image argument bbox format at index {index}")
            diagnostics.append(
                {
                    "field_path": f"image_arguments[{index}].bbox",
                    "issue_type": "invalid_bbox_format",
                    "suggested_action": "fix_or_drop",
                }
            )
            continue
        for value in bbox:
            if not isinstance(value, (int, float)):
                issues.append(f"invalid image argument bbox format at index {index}")
                diagnostics.append(
                    {
                        "field_path": f"image_arguments[{index}].bbox",
                        "issue_type": "invalid_bbox_format",
                        "suggested_action": "fix_or_drop",
                    }
                )
                break
    return issues, diagnostics


def _collect_grounding_support(
    grounding_results: Any,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], list[dict[str, Any]]]:
    grounded_pairs: set[tuple[str, str]] = set()
    failed_pairs: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    if not isinstance(grounding_results, list):
        return grounded_pairs, failed_pairs, normalized

    for item in grounding_results:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        status = str(item.get("grounding_status") or "").strip()
        bbox = item.get("bbox")
        normalized.append(item)
        if not role or not label:
            continue
        pair = (role, label)
        if status == "grounded" and isinstance(bbox, list) and len(bbox) == 4:
            grounded_pairs.add(pair)
        elif status == "failed":
            failed_pairs.add(pair)
    return grounded_pairs, failed_pairs, normalized


def _validate_grounding_awareness(
    event: Mapping[str, Any],
    grounding_results: Any,
) -> tuple[list[str], list[VerificationDiagnostic], int]:
    """Apply conservative grounding-aware checks without penalizing failures."""
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    grounded_pairs, failed_pairs, _ = _collect_grounding_support(grounding_results)
    image_arguments = event.get("image_arguments")
    grounded_support_count = 0
    if not isinstance(image_arguments, list):
        return issues, diagnostics, grounded_support_count

    for index, item in enumerate(image_arguments):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        bbox = item.get("bbox")
        grounding_status = str(item.get("grounding_status") or "").strip()
        pair = (role, label)

        if grounding_status == "grounded" and isinstance(bbox, list) and len(bbox) == 4:
            grounded_support_count += 1
            continue

        if grounding_status == "unresolved" and pair in grounded_pairs:
            issues.append(f"grounding result available but image argument remains unresolved at index {index}")
            diagnostics.append(
                {
                    "field_path": f"image_arguments[{index}].grounding_status",
                    "issue_type": "grounding_result_not_applied",
                    "suggested_action": "upgrade_from_grounding",
                }
            )
            continue

        if grounding_status == "unresolved" and pair in failed_pairs:
            # Grounding failure is informative but should not count against an
            # otherwise acceptable unresolved image argument.
            continue

    return issues, diagnostics, grounded_support_count


def _validate_modality_specific_consistency(
    event: Mapping[str, Any],
    *,
    raw_image: Any,
    raw_image_desc: str,
    perception_summary: str,
) -> tuple[list[str], list[VerificationDiagnostic]]:
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    image_arguments = event.get("image_arguments")
    if not isinstance(image_arguments, list) or not image_arguments:
        return issues, diagnostics

    if not _has_usable_image_evidence(raw_image, raw_image_desc, perception_summary):
        issues.append("image arguments present without usable image evidence in a text-only run")
        diagnostics.append(
            {
                "field_path": "image_arguments",
                "issue_type": "missing_image_evidence",
                "suggested_action": "drop_or_rebuild",
            }
        )
    return issues, diagnostics


def _collect_field_level_issues(
    raw_event: Any,
    raw_text: str,
    text_tokens: list[str],
    raw_image: Any = None,
    raw_image_desc: str = "",
    perception_summary: str = "",
    grounding_results: Any = None,
) -> tuple[list[str], list[VerificationDiagnostic]]:
    if not isinstance(raw_event, dict):
        return ["invalid event object"], [
            {
                "field_path": "event",
                "issue_type": "invalid_object",
                "suggested_action": "drop_or_rebuild",
            }
        ]
    issues: list[str] = []
    diagnostics: list[VerificationDiagnostic] = []
    ontology_issues, ontology_diagnostics = _validate_event_type_and_roles(raw_event)
    trigger_issues, trigger_diagnostics = _validate_trigger_fields(raw_event, raw_text, text_tokens)
    text_issues, text_diagnostics = _validate_text_argument_fields(raw_event, raw_text, text_tokens)
    image_issues, image_diagnostics = _validate_image_argument_fields(raw_event)
    modality_issues, modality_diagnostics = _validate_modality_specific_consistency(
        raw_event,
        raw_image=raw_image,
        raw_image_desc=raw_image_desc,
        perception_summary=perception_summary,
    )
    grounding_issues, grounding_diagnostics, _ = _validate_grounding_awareness(raw_event, grounding_results)
    issues.extend(ontology_issues)
    issues.extend(trigger_issues)
    issues.extend(text_issues)
    issues.extend(image_issues)
    issues.extend(modality_issues)
    issues.extend(grounding_issues)
    diagnostics.extend(ontology_diagnostics)
    diagnostics.extend(trigger_diagnostics)
    diagnostics.extend(text_diagnostics)
    diagnostics.extend(image_diagnostics)
    diagnostics.extend(modality_diagnostics)
    diagnostics.extend(grounding_diagnostics)
    return issues, diagnostics


def _merge_issues(field_issues: list[str], llm_issues: list[str]) -> list[str]:
    merged: list[str] = []
    for issue in field_issues + llm_issues:
        text = str(issue).strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def _merge_diagnostics(
    field_diagnostics: list[VerificationDiagnostic],
    llm_diagnostics: list[VerificationDiagnostic],
) -> list[VerificationDiagnostic]:
    merged: list[VerificationDiagnostic] = []
    seen: set[tuple[str, str, str]] = set()
    for item in field_diagnostics + llm_diagnostics:
        key = (item["field_path"], item["issue_type"], item["suggested_action"])
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def verifier(state: Mapping[str, Any]) -> dict[str, Any]:
    """Verify against fused raw_text plus derived raw_image_desc context."""
    started_at = time.perf_counter()
    fusion_context = state.get("fusion_context")
    if not isinstance(fusion_context, dict):
        fusion_context = empty_fusion_context()
        fusion_context.update(
            {
                "raw_text": str(state.get("text") or ""),
                "raw_image_desc": str(state.get("image_desc") or ""),
                "perception_summary": str(state.get("perception_summary") or ""),
                "patterns": state.get("similar_events") if isinstance(state.get("similar_events"), dict) else empty_layered_similar_events(),
                "evidence": list(state.get("evidence")) if isinstance(state.get("evidence"), list) else [],
            }
        )
    raw_text = str(fusion_context.get("raw_text") or "")
    raw_image_desc = str(fusion_context.get("raw_image_desc") or "")
    perception_summary = str(fusion_context.get("perception_summary") or "")
    raw_image = state.get("raw_image")
    text_tokens = _get_text_token_sequence(state, fusion_context, raw_text)
    evidence_items = fusion_context.get("evidence")
    if not isinstance(evidence_items, list):
        evidence_items = []
    raw_event = state.get("event")
    grounding_results = state.get("grounding_results")
    grounding_summary = summarize_grounding_activity(
        image_arguments_before=raw_event.get("image_arguments") if isinstance(raw_event, dict) else [],
        grounding_requests=None,
        grounding_results=grounding_results,
        image_arguments_after=raw_event.get("image_arguments") if isinstance(raw_event, dict) else [],
    )
    raw_event_type = str(raw_event.get("event_type") or "").strip() if isinstance(raw_event, dict) else ""
    field_issues, field_diagnostics = _collect_field_level_issues(
        raw_event,
        raw_text,
        text_tokens,
        raw_image,
        raw_image_desc,
        perception_summary,
        grounding_results,
    )
    _, _, grounded_support_count = _validate_grounding_awareness(raw_event, grounding_results)
    try:
        event = validate_event(raw_event)
    except Exception as exc:
        field_issues.append(str(exc))
        field_diagnostics.append(
            {
                "field_path": "event",
                "issue_type": "schema_validation_failed",
                "suggested_action": "drop_or_rebuild",
            }
        )
        event = empty_event()
    support_summary = summarize_evidence_sources(
        raw_event=event,
        raw_text=raw_text,
        raw_image_desc=raw_image_desc,
        perception_summary=perception_summary,
        grounding_results=grounding_results,
        evidence=evidence_items,
    )
    audit_enabled = "prompt_trace" in state or "stage_outputs" in state

    ontology_block = format_full_ontology_for_prompt()
    if is_supported_event_type(raw_event_type):
        selected_schema_block = format_event_schema_for_prompt(raw_event_type)
    else:
        selected_schema_block = "(event_type is missing or unsupported)"
    confusion_block = "\n".join(f"- {line}" for line in ROLE_CONFUSION_GUIDANCE)

    prompt = (
        "You verify an extracted event JSON against the SAME structured fusion_context used at extraction time.\n\n"
        "Ontology semantics:\n"
        f"{ontology_block}\n\n"
        "Predicted event_type schema focus:\n"
        f"{selected_schema_block}\n\n"
        "Role confusion checks:\n"
        f"{confusion_block}\n\n"
        "Checks:\n"
        "1) Ontology: Is event.event_type in the supported ontology, does it semantically match the event definition and trigger hints, and are text/image roles valid for that event_type?\n"
        "Use the role definitions and extraction notes to decide whether arguments fit the intended meaning of each role.\n"
        "Decide whether the trigger meaning fits the predicted event type, not just whether the trigger token appears in text.\n"
        "Check whether text argument roles are semantically appropriate for their mentions.\n"
        "Check whether image argument roles are semantically appropriate for their labels.\n"
        "Pay special attention to the listed closely related role confusions and flag them when the event uses the wrong role despite using an allowed role name.\n"
        "2) Text support: Are event.trigger.text and event.text_arguments supported by fusion_context.raw_text? "
        "Interpret spans as token spans [start, end), not character offsets.\n"
        "Text arguments should be normalized to the preferred canonical head-word form: no determiners, no obvious quantity words, and no broad modifier-heavy spans when a shorter head-word span is available.\n"
        "3) Modality-specific support: Treat text_arguments and image_arguments as separate evidence views, not a one-to-one alignment requirement.\n"
        "A valid text-only run may have empty image_arguments.\n"
        "A valid multimodal run may have partial overlap across modalities.\n"
        "Only flag cross-modal issues when there is a direct contradiction for the same role or when image arguments appear without usable image evidence.\n"
        "4) Image support: Are event.image_arguments supported by fusion_context.raw_image_desc and perception_summary, the derived image-side representations of the primary raw image input when available? "
        'Unresolved image arguments are allowed only when grounding_status is "unresolved".\n'
        "If an image argument is marked grounded and has a bbox, treat that as stronger image-side support.\n"
        "Generic weak Place-like labels such as street, road, outdoors, outside, scene, background, area, or crowd should still be treated conservatively even when grounded, because they may only describe backdrop context rather than an event-relevant place role.\n"
        "If grounding_results contain a grounded match for a role/label pair, unresolved image arguments for that same pair may indicate an inconsistency.\n"
        "If grounding_results show failed grounding, do not over-penalize otherwise acceptable unresolved image arguments.\n"
        "5) Evidence support: Are event claims supported by fusion_context.evidence item snippets when evidence items are available? "
        "Use evidence snippets as the primary factual basis for externally supported facts.\n"
        "6) Structure: Is the event schema valid, including trigger/text_arguments/image_arguments shape and types, and is it consistent with "
        "fusion_context.patterns when patterns are available?\n"
        "7) If text, image, and evidence conflict with patterns, prefer grounded support over patterns.\n\n"
        "Return ONLY one JSON object (no markdown), exactly this shape:\n"
        '{"verdict": "YES" or "NO", "issues": ["unsupported argument", "wrong event type", ...], "confidence": 0.0, "reason": "short explanation", "diagnostics": [{"field_path": "trigger.span", "issue_type": "span_mismatch", "suggested_action": "realign_or_drop"}]}\n'
        "Use an empty issues array when verdict is YES. Confidence must be a float from 0 to 1. "
        "Reason must be a short explanation. diagnostics is optional but should be included when you can localize a field-level problem.\n\n"
        f"fusion_context:\n{json.dumps(fusion_context, ensure_ascii=False)}\n\n"
        f"grounding_results:\n{json.dumps(grounding_results if isinstance(grounding_results, list) else [], ensure_ascii=False)}\n\n"
        f"Structured event:\n{json.dumps(event, ensure_ascii=False)}"
    )

    try:
        raw = _msg_text(_get_llm().invoke([HumanMessage(content=prompt)]).content)
        parsed = extract_json_object(raw)
        if parsed is None:
            issues = _merge_issues(field_issues, ["invalid verifier output (not valid JSON object)"])
            diagnostics = _merge_diagnostics(
                field_diagnostics,
                [
                    {
                        "field_path": "event",
                        "issue_type": "invalid_verifier_output",
                        "suggested_action": "retry_or_repair",
                    }
                ],
            )
            verified = False
            confidence = 0.0
            reason = "invalid verifier output"
            success = False
        else:
            _, llm_issues, llm_verified, confidence, reason, llm_diagnostics = _normalize_verdict_payload(parsed)
            issues = _merge_issues(field_issues, llm_issues)
            diagnostics = _merge_diagnostics(field_diagnostics, llm_diagnostics)
            verified = llm_verified and not issues
            success = True
            if issues and not reason:
                reason = "field-level or evidence-aware verification failed"
        result = {
            "verified": verified,
            "issues": issues,
            "verifier_diagnostics": diagnostics,
            "verifier_confidence": confidence,
            "verifier_reason": reason,
        }
        if audit_enabled:
            result["prompt_trace"] = append_prompt_trace(
                state,
                {
                    "sample_id": "",
                    "stage": "verifier",
                    "model_name": settings.openai_model,
                    "prompt_text": prompt,
                    "image_path": safe_image_reference(raw_image),
                    "input_summary": {
                        "fusion_context_summary": {
                            "raw_text": raw_text,
                            "raw_image_desc": raw_image_desc,
                            "perception_summary": perception_summary,
                            "evidence_count": len(evidence_items),
                        },
                        "event": event,
                        "grounding_results": grounding_results if isinstance(grounding_results, list) else [],
                        "depends_on": ["stage_c_output", "grounding_results"],
                    },
                    "response_text": raw,
                    "parsed_output": parsed if parsed is not None else {},
                },
            )
            result["stage_outputs"] = merge_stage_outputs(
                state,
                {
                    "verifier_output": {
                        "verified": verified,
                        "issues": issues,
                        "verifier_reason": reason,
                        "verifier_confidence": confidence,
                        "verifier_diagnostics": diagnostics,
                    }
                },
            )
        log_node_event(
            "verifier",
            state,
            started_at,
            success,
            verdict="YES" if verified else "NO",
            confidence=confidence,
            grounded_support_count=grounded_support_count,
            grounding_unresolved_image_arguments=grounding_summary["unresolved_image_arguments"],
            grounding_results=grounding_summary["grounded_results"],
            grounding_failed_results=grounding_summary["failed_grounding_results"],
            text_support=support_summary["text_support"],
            image_support=support_summary["image_support"],
            grounding_support=support_summary["grounding_support"],
            external_evidence_support=support_summary["external_evidence_support"],
        )
        return result
    except Exception as exc:
        result = {
            "verified": False,
            "issues": [str(exc)],
            "verifier_diagnostics": [
                {
                    "field_path": "event",
                    "issue_type": "verifier_failure",
                    "suggested_action": "retry_or_repair",
                }
            ],
            "verifier_confidence": 0.0,
            "verifier_reason": "verifier failure",
        }
        if audit_enabled:
            result["prompt_trace"] = append_prompt_trace(
                state,
                {
                    "sample_id": "",
                    "stage": "verifier",
                    "model_name": settings.openai_model,
                    "prompt_text": prompt,
                    "image_path": safe_image_reference(raw_image),
                    "input_summary": {
                        "fusion_context_summary": {
                            "raw_text": raw_text,
                            "raw_image_desc": raw_image_desc,
                            "perception_summary": perception_summary,
                            "evidence_count": len(evidence_items),
                        },
                        "event": event,
                        "grounding_results": grounding_results if isinstance(grounding_results, list) else [],
                        "depends_on": ["stage_c_output", "grounding_results"],
                    },
                    "response_text": str(exc),
                    "parsed_output": {},
                },
            )
            result["stage_outputs"] = merge_stage_outputs(
                state,
                {
                    "verifier_output": {
                        "verified": False,
                        "issues": [str(exc)],
                        "verifier_reason": "verifier failure",
                        "verifier_confidence": 0.0,
                        "verifier_diagnostics": [
                            {
                                "field_path": "event",
                                "issue_type": "verifier_failure",
                                "suggested_action": "retry_or_repair",
                            }
                        ],
                    }
                },
            )
        log_node_event(
            "verifier",
            state,
            started_at,
            False,
            error=str(exc),
            verdict="NO",
            confidence=0.0,
            grounded_support_count=grounded_support_count,
            grounding_unresolved_image_arguments=grounding_summary["unresolved_image_arguments"],
            grounding_results=grounding_summary["grounded_results"],
            grounding_failed_results=grounding_summary["failed_grounding_results"],
            text_support=support_summary["text_support"],
            image_support=support_summary["image_support"],
            grounding_support=support_summary["grounding_support"],
            external_evidence_support=support_summary["external_evidence_support"],
        )
        return result


run = verifier

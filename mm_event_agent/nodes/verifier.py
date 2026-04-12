"""Control node: validate event against fused context and update control fields."""

from __future__ import annotations

import json
import os
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
from mm_event_agent.schemas import (
    VerificationDiagnostic,
    empty_event,
    empty_fusion_context,
    extract_json_object,
    validate_event,
)

_llm: ChatOpenAI | None = None


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
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    return _llm


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


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


def _is_valid_span(span: Any, source_text: str) -> bool:
    if not isinstance(span, dict):
        return False
    start = span.get("start")
    end = span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < start or end > len(source_text):
        return False
    return True


def _validate_trigger_fields(
    event: Mapping[str, Any],
    raw_text: str,
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
        if not _is_valid_span(span, raw_text):
            issues.append("invalid trigger span")
            diagnostics.append(
                {
                    "field_path": "trigger.span",
                    "issue_type": "invalid_span",
                    "suggested_action": "realign_or_drop",
                }
            )
        elif raw_text[span["start"] : span["end"]] != text:
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
            if not _is_valid_span(span, raw_text):
                issues.append(f"invalid text argument span at index {index}")
                diagnostics.append(
                    {
                        "field_path": f"text_arguments[{index}].span",
                        "issue_type": "invalid_span",
                        "suggested_action": "realign_or_drop",
                    }
                )
            elif raw_text[span["start"] : span["end"]] != text:
                issues.append(f"text argument span mismatch at index {index}")
                diagnostics.append(
                    {
                        "field_path": f"text_arguments[{index}].span",
                        "issue_type": "span_mismatch",
                        "suggested_action": "realign_or_drop",
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


def _collect_field_level_issues(
    raw_event: Any,
    raw_text: str,
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
    trigger_issues, trigger_diagnostics = _validate_trigger_fields(raw_event, raw_text)
    text_issues, text_diagnostics = _validate_text_argument_fields(raw_event, raw_text)
    image_issues, image_diagnostics = _validate_image_argument_fields(raw_event)
    issues.extend(ontology_issues)
    issues.extend(trigger_issues)
    issues.extend(text_issues)
    issues.extend(image_issues)
    diagnostics.extend(ontology_diagnostics)
    diagnostics.extend(trigger_diagnostics)
    diagnostics.extend(text_diagnostics)
    diagnostics.extend(image_diagnostics)
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
                "patterns": list(state.get("similar_events")) if isinstance(state.get("similar_events"), list) else [],
                "evidence": list(state.get("evidence")) if isinstance(state.get("evidence"), list) else [],
            }
        )
    raw_text = str(fusion_context.get("raw_text") or "")
    raw_event = state.get("event")
    raw_event_type = str(raw_event.get("event_type") or "").strip() if isinstance(raw_event, dict) else ""
    field_issues, field_diagnostics = _collect_field_level_issues(raw_event, raw_text)
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
        "Check quoted text and spans.\n"
        "3) Image support: Are event.image_arguments supported by fusion_context.raw_image_desc, the current derived representation of the primary raw image input? "
        'Unresolved image arguments are allowed only when grounding_status is "unresolved".\n'
        "4) Evidence support: Are event claims supported by fusion_context.evidence item snippets when evidence items are available? "
        "Use evidence snippets as the primary factual basis for externally supported facts.\n"
        "5) Structure: Is the event schema valid, including trigger/text_arguments/image_arguments shape and types, and is it consistent with "
        "fusion_context.patterns when patterns are available?\n"
        "6) If text, image, and evidence conflict with patterns, prefer grounded support over patterns.\n\n"
        "Return ONLY one JSON object (no markdown), exactly this shape:\n"
        '{"verdict": "YES" or "NO", "issues": ["unsupported argument", "wrong event type", ...], "confidence": 0.0, "reason": "short explanation", "diagnostics": [{"field_path": "trigger.span", "issue_type": "span_mismatch", "suggested_action": "realign_or_drop"}]}\n'
        "Use an empty issues array when verdict is YES. Confidence must be a float from 0 to 1. "
        "Reason must be a short explanation. diagnostics is optional but should be included when you can localize a field-level problem.\n\n"
        f"fusion_context:\n{json.dumps(fusion_context, ensure_ascii=False)}\n\n"
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
        log_node_event(
            "verifier",
            state,
            started_at,
            success,
            verdict="YES" if verified else "NO",
            confidence=confidence,
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
        log_node_event("verifier", state, started_at, False, error=str(exc), verdict="NO", confidence=0.0)
        return result


run = verifier

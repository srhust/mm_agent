"""Control node: validate event against fused context and update control fields."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import empty_event, empty_fusion_context, extract_json_object, validate_event

_llm: ChatOpenAI | None = None


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


def _normalize_verdict_payload(data: dict[str, Any]) -> tuple[str, list[str], bool, float, str]:
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
    verified = verdict == "YES"
    return verdict, issues, verified, confidence, reason


def _is_valid_span(span: Any, source_text: str) -> bool:
    if not isinstance(span, list) or len(span) != 2:
        return False
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < start or end > len(source_text):
        return False
    return True


def _validate_trigger_fields(event: Mapping[str, Any], raw_text: str) -> list[str]:
    issues: list[str] = []
    trigger = event.get("trigger")
    if trigger is None:
        return issues
    if not isinstance(trigger, dict):
        return ["invalid trigger object"]

    if trigger.get("modality") != "text":
        issues.append("invalid trigger modality")

    span = trigger.get("span")
    text = str(trigger.get("text") or "")
    if span is not None:
        if not _is_valid_span(span, raw_text):
            issues.append("invalid trigger span")
        elif raw_text[span[0] : span[1]] != text:
            issues.append("trigger text/span mismatch")
    return issues


def _validate_text_argument_fields(event: Mapping[str, Any], raw_text: str) -> list[str]:
    issues: list[str] = []
    text_arguments = event.get("text_arguments")
    if not isinstance(text_arguments, list):
        return ["invalid text_arguments list"]

    for index, item in enumerate(text_arguments):
        if not isinstance(item, dict):
            issues.append(f"invalid text argument object at index {index}")
            continue
        span = item.get("span")
        text = str(item.get("text") or "")
        if span is not None:
            if not _is_valid_span(span, raw_text):
                issues.append(f"invalid text argument span at index {index}")
            elif raw_text[span[0] : span[1]] != text:
                issues.append(f"text argument span mismatch at index {index}")
    return issues


def _validate_image_argument_fields(event: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    image_arguments = event.get("image_arguments")
    if not isinstance(image_arguments, list):
        return ["invalid image_arguments list"]

    for index, item in enumerate(image_arguments):
        if not isinstance(item, dict):
            issues.append(f"invalid image argument object at index {index}")
            continue
        role = item.get("role")
        label = item.get("label")
        bbox = item.get("bbox")
        if not isinstance(role, str) or not role.strip():
            issues.append(f"invalid image argument role at index {index}")
        if not isinstance(label, str) or not label.strip():
            issues.append(f"invalid image argument label at index {index}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            issues.append(f"invalid image argument bbox format at index {index}")
            continue
        for value in bbox:
            if not isinstance(value, (int, float)):
                issues.append(f"invalid image argument bbox format at index {index}")
                break
    return issues


def _collect_field_level_issues(raw_event: Any, raw_text: str) -> list[str]:
    if not isinstance(raw_event, dict):
        return ["invalid event object"]
    issues: list[str] = []
    issues.extend(_validate_trigger_fields(raw_event, raw_text))
    issues.extend(_validate_text_argument_fields(raw_event, raw_text))
    issues.extend(_validate_image_argument_fields(raw_event))
    return issues


def _merge_issues(field_issues: list[str], llm_issues: list[str]) -> list[str]:
    merged: list[str] = []
    for issue in field_issues + llm_issues:
        text = str(issue).strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def verifier(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read data fields and write only control fields verified/issues/confidence/reason."""
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
    field_issues = _collect_field_level_issues(raw_event, raw_text)
    try:
        event = validate_event(raw_event)
    except Exception as exc:
        field_issues.append(str(exc))
        event = empty_event()

    prompt = (
        "You verify an extracted event JSON against the SAME structured fusion_context used at extraction time.\n\n"
        "Checks:\n"
        "1) Text support: Are event.trigger.text and event.text_arguments supported by fusion_context.raw_text? "
        "Check quoted text and spans.\n"
        "2) Image support: Are event.image_arguments supported by fusion_context.raw_image_desc?\n"
        "3) Evidence support: Are event claims supported by fusion_context.evidence item snippets when evidence items are available? "
        "Use evidence snippets as the primary factual basis for externally supported facts.\n"
        "4) Structure: Is the event schema valid, including trigger/text_arguments/image_arguments shape and types, and is it consistent with "
        "fusion_context.patterns when patterns are available?\n"
        "5) If text, image, and evidence conflict with patterns, prefer grounded support over patterns.\n\n"
        "Return ONLY one JSON object (no markdown), exactly this shape:\n"
        '{"verdict": "YES" or "NO", "issues": ["unsupported argument", "wrong event type", ...], "confidence": 0.0, "reason": "short explanation"}\n'
        "Use an empty issues array when verdict is YES. Confidence must be a float from 0 to 1. "
        "Reason must be a short explanation.\n\n"
        f"fusion_context:\n{json.dumps(fusion_context, ensure_ascii=False)}\n\n"
        f"Structured event:\n{json.dumps(event, ensure_ascii=False)}"
    )

    try:
        raw = _msg_text(_get_llm().invoke([HumanMessage(content=prompt)]).content)
        parsed = extract_json_object(raw)
        if parsed is None:
            issues = _merge_issues(field_issues, ["invalid verifier output (not valid JSON object)"])
            verified = False
            confidence = 0.0
            reason = "invalid verifier output"
            success = False
        else:
            _, llm_issues, llm_verified, confidence, reason = _normalize_verdict_payload(parsed)
            issues = _merge_issues(field_issues, llm_issues)
            verified = llm_verified and not issues
            success = True
            if issues and not reason:
                reason = "field-level or evidence-aware verification failed"
        result = {
            "verified": verified,
            "issues": issues,
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
            "verifier_confidence": 0.0,
            "verifier_reason": "verifier failure",
        }
        log_node_event("verifier", state, started_at, False, error=str(exc), verdict="NO", confidence=0.0)
        return result


run = verifier

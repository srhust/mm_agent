"""Data node: extract one structured event JSON from fusion_context only."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.ontology import get_allowed_image_roles, get_allowed_text_roles, get_supported_event_types
from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import empty_event, empty_fusion_context, enforce_strict_text_grounding, parse_event_json

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )
    return _llm


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _extract_stage_json(prompt: str) -> dict[str, Any]:
    out = _get_llm().invoke([HumanMessage(content=prompt)])
    raw = _msg_text(out.content)
    from mm_event_agent.schemas import extract_json_object

    parsed = extract_json_object(raw)
    if parsed is None:
        raise ValueError("stage output is not valid JSON")
    return parsed


def _classify_event_type(raw_text: str, raw_image_desc: str, evidence_items: list[Any]) -> str:
    allowed_event_types = get_supported_event_types()
    prompt = (
        "Classify exactly one event_type from this closed set only:\n"
        f"{json.dumps(allowed_event_types, ensure_ascii=False)}\n\n"
        "Use raw_text, raw_image_desc, and evidence. Prefer evidence when available.\n"
        'Return ONLY JSON: {"event_type": "<one of the allowed labels>"}\n\n'
        f'{{"raw_text": {json.dumps(raw_text, ensure_ascii=False)}, '
        f'"raw_image_desc": {json.dumps(raw_image_desc, ensure_ascii=False)}, '
        f'"evidence": {json.dumps(evidence_items, ensure_ascii=False)}}}'
    )
    parsed = _extract_stage_json(prompt)
    event_type = str(parsed.get("event_type") or "").strip()
    if event_type not in allowed_event_types:
        raise ValueError("unsupported event_type")
    return event_type


def _extract_text_fields(
    event_type: str,
    raw_text: str,
    raw_image_desc: str,
    evidence_items: list[Any],
) -> dict[str, Any]:
    allowed_roles = get_allowed_text_roles(event_type)
    prompt = (
        "Extract text-grounded event fields from raw_text.\n"
        f"event_type: {event_type}\n"
        f"allowed text roles: {json.dumps(allowed_roles, ensure_ascii=False)}\n\n"
        "Requirements:\n"
        "- trigger.text must be copied directly from raw_text or trigger must be null.\n"
        "- each text argument text must be copied directly from raw_text.\n"
        "- do not paraphrase trigger or text arguments.\n"
        "- if evidence is insufficient, prefer omission over hallucination.\n"
        "- span must be null in the model output; post-processing will align spans.\n"
        'Return ONLY JSON: {"trigger": {"text": string, "modality": "text", "span": null} | null, '
        '"text_arguments": [{"role": string, "text": string, "span": null}]}\n\n'
        f'{{"raw_text": {json.dumps(raw_text, ensure_ascii=False)}, '
        f'"raw_image_desc": {json.dumps(raw_image_desc, ensure_ascii=False)}, '
        f'"evidence": {json.dumps(evidence_items, ensure_ascii=False)}}}'
    )
    parsed = _extract_stage_json(prompt)
    trigger = parsed.get("trigger")
    text_arguments = parsed.get("text_arguments")
    return {
        "trigger": trigger if isinstance(trigger, dict) or trigger is None else None,
        "text_arguments": text_arguments if isinstance(text_arguments, list) else [],
    }


def _extract_image_arguments(
    event_type: str,
    raw_image_desc: str,
    text_arguments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allowed_roles = get_allowed_image_roles(event_type)
    prompt = (
        "Extract image argument semantic candidates from raw_image_desc.\n"
        f"event_type: {event_type}\n"
        f"allowed image roles: {json.dumps(allowed_roles, ensure_ascii=False)}\n\n"
        "Requirements:\n"
        "- output semantic image arguments only.\n"
        '- bbox must be null and grounding_status must be "unresolved".\n'
        "- do not pretend to know a precise bbox.\n"
        "- use role + label only when supported by raw_image_desc.\n"
        'Return ONLY JSON: {"image_arguments": [{"role": string, "label": string, "bbox": null, "grounding_status": "unresolved"}]}\n\n'
        f'{{"raw_image_desc": {json.dumps(raw_image_desc, ensure_ascii=False)}, '
        f'"text_arguments": {json.dumps(text_arguments, ensure_ascii=False)}}}'
    )
    parsed = _extract_stage_json(prompt)
    image_arguments = parsed.get("image_arguments")
    return image_arguments if isinstance(image_arguments, list) else []


def extraction(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read only data field fusion_context and write only data field event."""
    started_at = time.perf_counter()
    raw_context = state.get("fusion_context")
    if isinstance(raw_context, dict):
        fusion_context = raw_context
    else:
        fusion_context = empty_fusion_context()
        fusion_context.update(
            {
                "raw_text": str(state.get("text") or ""),
                "raw_image_desc": str(state.get("image_desc") or ""),
                "perception_summary": str(state.get("perception_summary") or ""),
                "evidence": list(state.get("evidence")) if isinstance(state.get("evidence"), list) else [],
            }
        )

    raw_text = str(fusion_context.get("raw_text") or "")
    raw_image_desc = str(fusion_context.get("raw_image_desc") or "")
    perception_summary = str(fusion_context.get("perception_summary") or "")
    patterns = fusion_context.get("patterns")
    evidence_items = fusion_context.get("evidence")

    if not isinstance(patterns, list):
        patterns = []
    if not isinstance(evidence_items, list):
        evidence_items = []
    try:
        event_type = _classify_event_type(raw_text, raw_image_desc, evidence_items)
        text_fields = _extract_text_fields(event_type, raw_text, raw_image_desc, evidence_items)
        image_arguments = _extract_image_arguments(event_type, raw_image_desc, text_fields["text_arguments"])
        assembled_event = {
            "event_type": event_type,
            "trigger": text_fields["trigger"],
            "text_arguments": text_fields["text_arguments"],
            "image_arguments": image_arguments,
        }
        event = enforce_strict_text_grounding(
            parse_event_json(json.dumps(assembled_event, ensure_ascii=False)),
            raw_text,
        )
        result = {"event": event}
        log_node_event(
            "extraction",
            state,
            started_at,
            True,
            event_type=event["event_type"],
            evidence_used=len(evidence_items),
        )
        return result
    except Exception as exc:
        fallback = empty_event()
        result = {"event": fallback}
        log_node_event(
            "extraction",
            state,
            started_at,
            False,
            error=str(exc),
            evidence_used=len(evidence_items),
        )
        return result


run = extraction

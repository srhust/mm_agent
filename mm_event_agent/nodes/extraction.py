"""Data node: staged multimodal extraction from fusion_context only."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.ontology import (
    format_event_schema_for_prompt,
    format_full_ontology_for_prompt,
    format_image_role_visibility_guidance_for_prompt,
    get_allowed_image_roles,
    get_allowed_text_roles,
    get_supported_event_types,
)
from mm_event_agent.observability import log_node_event
from mm_event_agent.runtime_config import settings
from mm_event_agent.evidence.debug import summarize_evidence_sources
from mm_event_agent.grounding.debug import summarize_grounding_activity
from mm_event_agent.schemas import (
    align_text_grounded_event,
    build_grounding_requests,
    empty_event,
    empty_fusion_context,
    parse_event_json,
    resolve_text_token_sequence,
)
from mm_event_agent.trace_utils import append_prompt_trace, merge_stage_outputs, safe_image_reference
from mm_event_agent.grounding.florence2_hf import (
    apply_grounding_results_to_event,
    execute_grounding_requests,
)

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


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        kwargs: dict[str, Any] = {
            "model": settings.extraction_model_name,
            "temperature": 0.2,
            "timeout": settings.extraction_timeout_seconds,
        }
        if settings.extraction_api_key:
            kwargs["api_key"] = settings.extraction_api_key
        if settings.extraction_api_base:
            kwargs["base_url"] = settings.extraction_api_base
        _llm = ChatOpenAI(**kwargs)
    return _llm


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text") or ""))
        return " ".join(chunks)
    return str(content)


def _load_image_bytes(raw_image: Any) -> tuple[bytes, str] | None:
    if raw_image is None:
        return None
    if isinstance(raw_image, (bytes, bytearray)):
        return bytes(raw_image), "image/png"
    if isinstance(raw_image, str):
        candidate = raw_image.strip()
        if not candidate:
            return None
        if candidate.startswith("data:"):
            header, _, payload = candidate.partition(",")
            if not payload:
                return None
            mime_type = header.split(";", 1)[0].replace("data:", "") or "image/png"
            try:
                return base64.b64decode(payload), mime_type
            except Exception:
                return None
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate.encode("utf-8"), "url"
        if not os.path.exists(candidate):
            return None
        mime_type = mimetypes.guess_type(candidate)[0] or "image/png"
        return Path(candidate).read_bytes(), mime_type
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore[assignment]
    if Image is not None and isinstance(raw_image, Image.Image):
        buffer = io.BytesIO()
        raw_image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"
    return None


def _build_image_content_block(raw_image: Any) -> dict[str, Any] | None:
    if isinstance(raw_image, str):
        candidate = raw_image.strip()
        if candidate.startswith("http://") or candidate.startswith("https://") or candidate.startswith("data:"):
            return {"type": "image_url", "image_url": {"url": candidate}}
    payload = _load_image_bytes(raw_image)
    if payload is None:
        return None
    image_bytes, mime_type = payload
    if mime_type == "url":
        return {"type": "image_url", "image_url": {"url": image_bytes.decode("utf-8")}}
    data_url = "data:" + mime_type + ";base64," + base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": data_url}}


def _build_stage_message(prompt: str, *, raw_image: Any = None) -> HumanMessage:
    image_block = _build_image_content_block(raw_image)
    if image_block is None:
        return HumanMessage(content=prompt)
    return HumanMessage(content=[{"type": "text", "text": prompt}, image_block])


def _extract_stage_json(prompt: str, *, raw_image: Any = None) -> dict[str, Any]:
    out = _get_llm().invoke([_build_stage_message(prompt, raw_image=raw_image)])
    raw = _msg_text(out.content)
    from mm_event_agent.schemas import extract_json_object

    parsed = extract_json_object(raw)
    if parsed is None:
        raise ValueError("stage output is not valid JSON")
    return parsed


def _invoke_traced_stage(
    state: Mapping[str, Any],
    *,
    stage: str,
    prompt: str,
    input_summary: Mapping[str, Any],
    raw_image: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = _get_llm().invoke([_build_stage_message(prompt, raw_image=raw_image)])
    raw = _msg_text(out.content)
    from mm_event_agent.schemas import extract_json_object

    parsed = extract_json_object(raw)
    if parsed is None:
        raise ValueError("stage output is not valid JSON")
    trace_record = {
        "sample_id": "",
        "stage": stage,
        "model_name": settings.extraction_model_name,
        "prompt_text": prompt,
        "image_path": safe_image_reference(raw_image),
        "input_summary": dict(input_summary),
        "response_text": raw,
        "parsed_output": parsed,
    }
    return parsed, trace_record


def _build_image_side_info(raw_image_desc: str, perception_summary: str) -> str:
    """Build compact auxiliary image-side context from intermediate signals."""
    summary = str(perception_summary or "").strip()
    image_desc = str(raw_image_desc or "").strip()
    if summary:
        return f"raw_image_desc: {image_desc}\nperception_summary: {summary}"
    return f"raw_image_desc: {image_desc}"


def _perception_image_signal(perception_summary: str) -> str:
    summary = str(perception_summary or "").strip()
    if not summary:
        return ""
    match = re.search(r"(?im)^image:\s*(.*)$", summary)
    if match is not None:
        return match.group(1).strip()
    return summary


def _has_valid_image_side_context(raw_image_desc: str, perception_summary: str) -> bool:
    if str(raw_image_desc or "").strip():
        return True
    return bool(_perception_image_signal(perception_summary))


def _has_usable_image_evidence(raw_image: Any, raw_image_desc: str, perception_summary: str) -> bool:
    if _build_image_content_block(raw_image) is not None:
        return True
    return _has_valid_image_side_context(raw_image_desc, perception_summary)


def _is_generic_weak_place_label(label: str) -> bool:
    normalized = " ".join(str(label or "").strip().lower().split())
    return normalized in _GENERIC_WEAK_PLACE_LABELS


def _filter_weak_image_arguments(event_type: str, image_arguments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in image_arguments:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        if event_type == "Justice:Arrest-Jail" and role == "Place" and _is_generic_weak_place_label(label):
            continue
        filtered.append(item)
    return filtered


def _get_text_token_sequence(state: Mapping[str, Any], fusion_context: Mapping[str, Any], raw_text: str) -> list[str]:
    for container in (fusion_context, state):
        for key in ("text_tokens", "token_sequence", "tokens"):
            value = container.get(key) if isinstance(container, Mapping) else None
            if isinstance(value, list) and value:
                return resolve_text_token_sequence(raw_text, value)
    return resolve_text_token_sequence(raw_text)


def format_text_event_examples_for_prompt(examples: list[dict[str, Any]], top_k: int = 2) -> str:
    if not isinstance(examples, list) or not examples:
        return "(none)"
    lines: list[str] = []
    for index, item in enumerate(examples[: max(1, top_k)], start=1):
        if not isinstance(item, dict):
            continue
        payload = {
            "rank": index,
            "id": str(item.get("id") or "").strip(),
            "source_dataset": str(item.get("source_dataset") or "").strip(),
            "event_type": str(item.get("event_type") or "").strip(),
            "raw_text": str(item.get("raw_text") or "").strip(),
            "trigger": item.get("trigger"),
            "text_arguments": item.get("text_arguments") if isinstance(item.get("text_arguments"), list) else [],
            "pattern_summary": str(item.get("pattern_summary") or "").strip(),
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines) if lines else "(none)"


def format_image_semantic_examples_for_prompt(examples: list[dict[str, Any]], top_k: int = 2) -> str:
    if not isinstance(examples, list) or not examples:
        return "(none)"
    blocks: list[str] = []
    for index, item in enumerate(examples[: max(1, top_k)], start=1):
        if not isinstance(item, dict):
            continue
        lines = [
            f"Image Example {index}",
            f'event_type: {str(item.get("event_type") or "").strip()}',
            f'image_desc: {str(item.get("image_desc") or "").strip()}',
            "image_arguments:",
        ]
        image_arguments = item.get("image_arguments")
        if isinstance(image_arguments, list) and image_arguments:
            for argument in image_arguments:
                if not isinstance(argument, dict):
                    continue
                role = str(argument.get("role") or "").strip()
                label = str(argument.get("label") or "").strip()
                if role or label:
                    lines.append(f"- {role}: {label}")
        else:
            lines.append("- (none)")
        lines.append(f'summary: {str(item.get("visual_pattern_summary") or "").strip()}')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(none)"


def format_bridge_examples_for_prompt(examples: list[dict[str, Any]], top_k: int = 2) -> str:
    if not isinstance(examples, list) or not examples:
        return "(none)"
    blocks: list[str] = []
    for index, item in enumerate(examples[: max(1, top_k)], start=1):
        if not isinstance(item, dict):
            continue
        text_cues = item.get("text_cues") if isinstance(item.get("text_cues"), list) else []
        visual_cues = item.get("visual_cues") if isinstance(item.get("visual_cues"), list) else []
        blocks.append(
            "\n".join(
                [
                    f"Bridge {index}",
                    f'event_type: {str(item.get("event_type") or "").strip()}',
                    f'role: {str(item.get("role") or "").strip()}',
                    f'text_cues: {", ".join(str(x) for x in text_cues) if text_cues else "(none)"}',
                    f'visual_cues: {", ".join(str(x) for x in visual_cues) if visual_cues else "(none)"}',
                    f'note: {str(item.get("note") or "").strip()}',
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "(none)"


def _format_text_event_examples_topk(patterns: Any) -> str:
    if not isinstance(patterns, dict):
        return "(none)"
    examples = patterns.get("text_event_examples")
    return format_text_event_examples_for_prompt(examples if isinstance(examples, list) else [])


def _format_bridge_examples_topk(patterns: Any) -> str:
    if not isinstance(patterns, dict):
        return "(none)"
    examples = patterns.get("bridge_examples")
    return format_bridge_examples_for_prompt(examples if isinstance(examples, list) else [], top_k=2)


def _format_image_semantic_examples_topk(patterns: Any) -> str:
    if not isinstance(patterns, dict):
        return "(none)"
    examples = patterns.get("image_semantic_examples")
    return format_image_semantic_examples_for_prompt(examples if isinstance(examples, list) else [], top_k=2)


def _stage_a_select_event_type(
    raw_text: str,
    image_side_info: str,
    evidence_items: list[Any],
    patterns: Any,
    event_type_mode: str,
    raw_image: Any = None,
    *,
    state: Mapping[str, Any] | None = None,
    return_trace: bool = False,
) -> Any:
    allowed_event_types = get_supported_event_types()
    ontology_block = format_full_ontology_for_prompt()
    formatted_text_event_examples_topk = _format_text_event_examples_topk(patterns)
    formatted_bridge_examples_topk = _format_bridge_examples_topk(patterns)
    normalized_mode = str(event_type_mode or "closed_set").strip() or "closed_set"
    if normalized_mode not in {"closed_set", "transfer"}:
        normalized_mode = "closed_set"

    if normalized_mode == "transfer":
        selection_rule = (
            "Stage A mode: transfer.\n"
            "Choose exactly one supported ontology event_type, or return \"Unsupported\" if the input does not clearly fit the current ontology.\n"
            'Return ONLY JSON: {"event_type": "<one supported label or Unsupported>"}\n\n'
        )
    else:
        selection_rule = (
            "Stage A mode: closed_set.\n"
            "Choose exactly one supported ontology event_type from the closed set.\n"
            'Return ONLY JSON: {"event_type": "<one of the allowed labels>"}\n\n'
        )
    prompt = (
        "Stage A: event type selection.\n"
        "Classify the input using the configured event type selection mode.\n"
        f"{selection_rule}"
        "Supported ontology event types:\n"
        f"{json.dumps(allowed_event_types, ensure_ascii=False)}\n\n"
        "Ontology semantics:\n"
        f"{ontology_block}\n\n"
        "Retrieved text event examples:\n"
        f"{formatted_text_event_examples_topk}\n\n"
        "Retrieved cross-modal bridge hints:\n"
        f"{formatted_bridge_examples_topk}\n\n"
        "Use raw_text, image-side information, and evidence. Prefer evidence when available.\n"
        "- Use event definitions and trigger hints to choose the best matching event type.\n"
        "- Use retrieved text event examples as pattern guidance only.\n"
        "- Use cross-modal bridge hints only as auxiliary semantic support.\n"
        "- If evidence conflicts with retrieved examples, prefer evidence and ontology over retrieved examples.\n"
        "- Do not invent new labels or open-set variants.\n"
        f"- raw_image direct vision is {'enabled' if settings.extraction_stage_a_use_raw_image else 'disabled'} for this stage.\n"
        f'{{"raw_text": {json.dumps(raw_text, ensure_ascii=False)}, '
        f'"image_side_info": {json.dumps(image_side_info, ensure_ascii=False)}, '
        f'"evidence": {json.dumps(evidence_items, ensure_ascii=False)}}}'
    )
    parsed, trace_record = _invoke_traced_stage(
        state or {},
        stage="extraction_stage_a",
        prompt=prompt,
        input_summary={
            "event_type_mode": normalized_mode,
            "raw_text": raw_text,
            "image_side_info": image_side_info,
            "evidence_count": len(evidence_items),
            "retrieved_text_event_examples": formatted_text_event_examples_topk,
            "retrieved_bridge_examples": formatted_bridge_examples_topk,
        },
        raw_image=raw_image if settings.extraction_stage_a_use_raw_image else None,
    )
    event_type = str(parsed.get("event_type") or "").strip()
    if normalized_mode == "transfer" and event_type == "Unsupported":
        return (event_type, trace_record) if return_trace else event_type
    if event_type not in allowed_event_types:
        raise ValueError("unsupported event_type")
    return (event_type, trace_record) if return_trace else event_type


def _stage_b_extract_text_fields(
    event_type: str,
    raw_text: str,
    image_side_info: str,
    patterns: Any,
    evidence_items: list[Any],
    *,
    state: Mapping[str, Any] | None = None,
    return_trace: bool = False,
) -> Any:
    allowed_roles = get_allowed_text_roles(event_type)
    schema_block = format_event_schema_for_prompt(event_type)
    formatted_text_event_examples_topk = _format_text_event_examples_topk(patterns)
    formatted_bridge_examples_topk = _format_bridge_examples_topk(patterns)
    prompt = (
        "Stage B: extract text trigger and text arguments from raw_text.\n"
        "Extract text-grounded event fields from raw_text.\n"
        "Event ontology for the selected event_type:\n"
        f"{schema_block}\n\n"
        "Retrieved text event examples:\n"
        f"{formatted_text_event_examples_topk}\n\n"
        "Optional bridge hints:\n"
        f"{formatted_bridge_examples_topk}\n\n"
        "Requirements:\n"
        "- event_type is fixed; use only the allowed text roles for this event_type.\n"
        f"- allowed text roles for this stage: {json.dumps(allowed_roles, ensure_ascii=False)}\n"
        "- Use retrieved text event examples as few-shot pattern guidance.\n"
        "- Use bridge hints only for limited role disambiguation support.\n"
        "- Use the role definitions to decide which extracted mention belongs to which role.\n"
        "- Use the trigger_hint only as semantic guidance; trigger.text must still be copied from raw_text exactly.\n"
        "- trigger.text must be copied directly from raw_text or trigger must be null.\n"
        "- each text argument text must be copied directly from raw_text.\n"
        "- Prefer the canonical head-word mention for text arguments: no determiners, no obvious quantity words, and avoid broad modifier-heavy spans when a shorter head preserves the role meaning.\n"
        "- do not paraphrase trigger or text arguments.\n"
        "- if evidence is insufficient, prefer omission over hallucination.\n"
        "- span must be null in the model output; post-processing will align token spans.\n"
        'Return ONLY JSON: {"trigger": {"text": string, "modality": "text", "span": null} | null, '
        '"text_arguments": [{"role": string, "text": string, "span": null}]}\n\n'
        f'{{"raw_text": {json.dumps(raw_text, ensure_ascii=False)}, '
        f'"image_side_info": {json.dumps(image_side_info, ensure_ascii=False)}, '
        f'"evidence": {json.dumps(evidence_items, ensure_ascii=False)}}}'
    )
    parsed, trace_record = _invoke_traced_stage(
        state or {},
        stage="extraction_stage_b",
        prompt=prompt,
        input_summary={
            "depends_on": ["stage_a_output"],
            "stage_a_output": {"event_type": event_type},
            "raw_text": raw_text,
            "image_side_info": image_side_info,
            "evidence_count": len(evidence_items),
            "retrieved_text_event_examples": formatted_text_event_examples_topk,
            "retrieved_bridge_examples": formatted_bridge_examples_topk,
        },
    )
    trigger = parsed.get("trigger")
    text_arguments = parsed.get("text_arguments")
    result = {
        "trigger": trigger if isinstance(trigger, dict) or trigger is None else None,
        "text_arguments": text_arguments if isinstance(text_arguments, list) else [],
    }
    return (result, trace_record) if return_trace else result


def _stage_c_extract_image_arguments(
    event_type: str,
    image_side_info: str,
    text_arguments: list[dict[str, Any]],
    patterns: Any,
    evidence_items: list[Any],
    raw_image: Any = None,
    raw_text: str = "",
    *,
    state: Mapping[str, Any] | None = None,
    return_trace: bool = False,
) -> Any:
    allowed_roles = get_allowed_image_roles(event_type)
    schema_block = format_event_schema_for_prompt(event_type)
    visibility_block = format_image_role_visibility_guidance_for_prompt(event_type)
    formatted_image_semantic_examples_topk = _format_image_semantic_examples_topk(patterns)
    formatted_bridge_examples_topk = _format_bridge_examples_topk(patterns)
    prompt = (
        "Stage C: extract image argument semantic candidates.\n"
        "Extract image argument semantic candidates from the raw image with auxiliary image-side context.\n"
        "Event ontology for the selected event_type:\n"
        f"{schema_block}\n\n"
        "Retrieved image semantic examples:\n"
        f"{formatted_image_semantic_examples_topk}\n\n"
        "Retrieved cross-modal bridge hints:\n"
        f"{formatted_bridge_examples_topk}\n\n"
        "Image-role visibility guidance:\n"
        f"{visibility_block}\n\n"
        "Requirements:\n"
        "- event_type is fixed; use only the allowed image roles for this event_type.\n"
        f"- allowed image roles for this stage: {json.dumps(allowed_roles, ensure_ascii=False)}\n"
        "- Use image semantic examples as visual pattern guidance.\n"
        "- Use bridge hints to connect text roles and visual cues.\n"
        "- Use the role definitions and extraction notes to map visible evidence to semantic roles.\n"
        "- condition on the selected event_type and the extracted text arguments.\n"
        "- raw_image is the primary visual evidence for this stage.\n"
        "- Treat image_side_info as auxiliary context, redundancy, and fallback support rather than the only visual input.\n"
        "- raw_text can be used for cross-modal disambiguation, but image arguments still require direct visual support.\n"
        "- output semantic image arguments only.\n"
        "- Be conservative: prefer omission over unsupported image-role prediction.\n"
        "- For visually weaker roles, require direct visual evidence rather than generic scene context.\n"
        "- Do not output a weakly visible role just because it is semantically allowed by the ontology.\n"
        "- For weak Place-like roles, do not use generic background-only labels such as street, road, outdoors, outside, scene, background, area, or crowd unless the place itself is a clearly depicted event-relevant target.\n"
        "- In Justice:Arrest-Jail scenes, Agent and Person are usually visually stronger than Place; prefer omitting Place unless the location is directly and specifically depicted.\n"
        '- bbox must be null and grounding_status must be "unresolved".\n'
        "- do not pretend to know a precise bbox.\n"
        "- use role + label only when supported by image-side information.\n"
        'Return ONLY JSON: {"image_arguments": [{"role": string, "label": string, "bbox": null, "grounding_status": "unresolved"}]}\n\n'
        f'{{"raw_text": {json.dumps(raw_text, ensure_ascii=False)}, '
        f'"image_side_info": {json.dumps(image_side_info, ensure_ascii=False)}, '
        f'"text_arguments": {json.dumps(text_arguments, ensure_ascii=False)}, '
        f'"evidence": {json.dumps(evidence_items, ensure_ascii=False)}}}'
    )
    parsed, trace_record = _invoke_traced_stage(
        state or {},
        stage="extraction_stage_c",
        prompt=prompt,
        input_summary={
            "depends_on": ["stage_a_output", "stage_b_output"],
            "stage_a_output": {"event_type": event_type},
            "stage_b_output": {"text_arguments": text_arguments},
            "raw_text": raw_text,
            "image_side_info": image_side_info,
            "evidence_count": len(evidence_items),
            "retrieved_image_semantic_examples": formatted_image_semantic_examples_topk,
            "retrieved_bridge_examples": formatted_bridge_examples_topk,
        },
        raw_image=raw_image,
    )
    image_arguments = parsed.get("image_arguments")
    result = image_arguments if isinstance(image_arguments, list) else []
    result = _filter_weak_image_arguments(event_type, result)
    return (result, trace_record) if return_trace else result


def _run_staged_extraction(
    state: Mapping[str, Any],
    raw_text: str,
    raw_image: Any,
    raw_image_desc: str,
    perception_summary: str,
    patterns: Any,
    evidence_items: list[Any],
    event_type_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    image_side_info = _build_image_side_info(raw_image_desc, perception_summary)
    has_usable_image_evidence = _has_usable_image_evidence(raw_image, raw_image_desc, perception_summary)
    has_valid_image_side_context = _has_valid_image_side_context(raw_image_desc, perception_summary)
    trace_records: list[dict[str, Any]] = []
    event_type, stage_a_trace = _stage_a_select_event_type(
        raw_text,
        image_side_info,
        evidence_items,
        patterns,
        event_type_mode,
        raw_image,
        state=state,
        return_trace=True,
    )
    trace_records.append(stage_a_trace)
    stage_a_output = {"event_type": event_type}
    stage_a_info = {
        "stage_a_event_type_mode": str(event_type_mode or "closed_set"),
        "stage_a_selected_event_type": event_type,
        "stage_a_selected_unsupported": event_type == "Unsupported",
    }
    if event_type == "Unsupported":
        return empty_event(), stage_a_info, trace_records, {
            "stage_a_output": stage_a_output,
            "stage_b_output": {"trigger": None, "text_arguments": []},
            "stage_c_output": {"image_arguments": []},
        }
    text_fields, stage_b_trace = _stage_b_extract_text_fields(
        event_type,
        raw_text,
        image_side_info,
        patterns,
        evidence_items,
        state=state,
        return_trace=True,
    )
    trace_records.append(stage_b_trace)
    stage_b_output = {
        "trigger": text_fields["trigger"],
        "text_arguments": text_fields["text_arguments"],
    }
    if has_usable_image_evidence and (raw_image is not None or has_valid_image_side_context):
        image_arguments, stage_c_trace = _stage_c_extract_image_arguments(
            event_type,
            image_side_info,
            text_fields["text_arguments"],
            patterns,
            evidence_items,
            raw_image,
            raw_text,
            state=state,
            return_trace=True,
        )
        trace_records.append(stage_c_trace)
    else:
        image_arguments = []
        trace_records.append(
            {
                "sample_id": "",
                "stage": "extraction_stage_c",
                "model_name": settings.extraction_model_name,
                "prompt_text": "",
                "image_path": safe_image_reference(raw_image),
                "input_summary": {
                    "depends_on": ["stage_a_output", "stage_b_output"],
                    "stage_a_output": stage_a_output,
                    "stage_b_output": stage_b_output,
                    "usable_image_evidence": False,
                    "valid_image_side_context": has_valid_image_side_context,
                },
                "response_text": "skipped: no usable image evidence",
                "parsed_output": {"image_arguments": []},
            }
        )
    stage_c_output = {"image_arguments": image_arguments}
    return (
        {
            "event_type": event_type,
            "trigger": text_fields["trigger"],
            "text_arguments": text_fields["text_arguments"],
            "image_arguments": image_arguments,
        },
        stage_a_info,
        trace_records,
        {
            "stage_a_output": stage_a_output,
            "stage_b_output": stage_b_output,
            "stage_c_output": stage_c_output,
        },
    )


def _maybe_run_grounding(
    raw_image: Any,
    event: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool, list[dict[str, Any]]]:
    """Optionally ground unresolved image arguments against raw_image.

    raw_image is the primary image input.
    image_desc remains the current intermediate representation used earlier in
    extraction; grounding here is an optional next-stage spatial layer.
    """
    grounding_requests = build_grounding_requests(event)
    if not grounding_requests or _build_image_content_block(raw_image) is None:
        return event, [], False, grounding_requests

    try:
        grounding_results = execute_grounding_requests(raw_image, grounding_requests)
        grounded_event = apply_grounding_results_to_event(event, grounding_results)
        return grounded_event, grounding_results, True, grounding_requests
    except Exception:
        return event, [], True, grounding_requests


def extraction(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read fusion_context and write event using the current image_desc bridge."""
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
    raw_image = state.get("raw_image")
    perception_summary = str(fusion_context.get("perception_summary") or "")
    event_type_mode = str(state.get("event_type_mode") or "closed_set")
    patterns = fusion_context.get("patterns")
    evidence_items = fusion_context.get("evidence")
    text_tokens = _get_text_token_sequence(state, fusion_context, raw_text)

    if not isinstance(patterns, dict):
        patterns = {}
    if not isinstance(evidence_items, list):
        evidence_items = []
    stage_a_info = {
        "stage_a_event_type_mode": event_type_mode,
        "stage_a_selected_event_type": "",
        "stage_a_selected_unsupported": False,
    }
    grounding_attempted = False
    grounding_results: list[dict[str, Any]] = []
    grounding_requests: list[dict[str, Any]] = []
    grounding_summary = summarize_grounding_activity([], [], [], [])
    alignment_issues: list[str] = []
    alignment_diagnostics: list[dict[str, Any]] = []
    stage_trace_records: list[dict[str, Any]] = []
    stage_outputs_update: dict[str, Any] = {}
    audit_enabled = "prompt_trace" in state or "stage_outputs" in state
    try:
        assembled_event, stage_a_info, stage_trace_records, stage_outputs_update = _run_staged_extraction(
            state,
            raw_text,
            raw_image,
            raw_image_desc,
            perception_summary,
            patterns,
            evidence_items,
            event_type_mode,
        )
        if not assembled_event["event_type"]:
            event = empty_event()
        else:
            parsed_event = parse_event_json(json.dumps(assembled_event, ensure_ascii=False))
            event, alignment_issues, alignment_diagnostics = align_text_grounded_event(
                parsed_event,
                raw_text,
                token_sequence=text_tokens,
            )
        image_arguments_before = list(event.get("image_arguments")) if isinstance(event.get("image_arguments"), list) else []
        event, grounding_results, grounding_attempted, grounding_requests = _maybe_run_grounding(raw_image, event)
        grounding_summary = summarize_grounding_activity(
            image_arguments_before=image_arguments_before,
            grounding_requests=grounding_requests,
            grounding_results=grounding_results,
            image_arguments_after=event.get("image_arguments"),
        )
        support_summary = summarize_evidence_sources(
            raw_event=event,
            raw_text=raw_text,
            raw_image_desc=raw_image_desc,
            perception_summary=perception_summary,
            grounding_results=grounding_results,
            evidence=evidence_items,
        )
        result = {"event": event, "grounding_results": grounding_results}
        if audit_enabled:
            prompt_trace = state.get("prompt_trace")
            if not isinstance(prompt_trace, list):
                prompt_trace = []
            for record in stage_trace_records:
                prompt_trace = append_prompt_trace({"prompt_trace": prompt_trace}, record)
            stage_outputs = merge_stage_outputs(
                state,
                {
                    **stage_outputs_update,
                    "grounding_requests": grounding_requests,
                    "grounding_results": grounding_results,
                },
            )
            result["prompt_trace"] = prompt_trace
            result["stage_outputs"] = stage_outputs
        log_node_event(
            "extraction",
            state,
            started_at,
            True,
            event_type=event["event_type"],
            event_type_mode=event_type_mode,
            stage_a_selected_event_type=stage_a_info["stage_a_selected_event_type"],
            stage_a_selected_unsupported=stage_a_info["stage_a_selected_unsupported"],
            grounding_attempted=grounding_attempted,
            grounding_unresolved_image_arguments=grounding_summary["unresolved_image_arguments"],
            grounding_requests=grounding_summary["grounding_requests"],
            grounding_results=grounding_summary["grounded_results"],
            grounding_failed_results=grounding_summary["failed_grounding_results"],
            grounding_applied_bboxes=grounding_summary["applied_grounded_bboxes"],
            staged=True,
            evidence_used=len(evidence_items),
            text_alignment_issues=alignment_issues,
            text_alignment_diagnostics=alignment_diagnostics,
            text_support=support_summary["text_support"],
            image_support=support_summary["image_support"],
            grounding_support=support_summary["grounding_support"],
            external_evidence_support=support_summary["external_evidence_support"],
        )
        return result
    except Exception as exc:
        fallback = empty_event()
        support_summary = summarize_evidence_sources(
            raw_event=fallback,
            raw_text=raw_text,
            raw_image_desc=raw_image_desc,
            perception_summary=perception_summary,
            grounding_results=grounding_results,
            evidence=evidence_items,
        )
        result = {"event": fallback, "grounding_results": grounding_results}
        if audit_enabled:
            prompt_trace = state.get("prompt_trace")
            if not isinstance(prompt_trace, list):
                prompt_trace = []
            for record in stage_trace_records:
                prompt_trace = append_prompt_trace({"prompt_trace": prompt_trace}, record)
            stage_outputs = merge_stage_outputs(
                state,
                {
                    **stage_outputs_update,
                    "grounding_requests": grounding_requests,
                    "grounding_results": grounding_results,
                },
            )
            result["prompt_trace"] = prompt_trace
            result["stage_outputs"] = stage_outputs
        log_node_event(
            "extraction",
            state,
            started_at,
            False,
            error=str(exc),
            event_type_mode=event_type_mode,
            stage_a_selected_event_type=stage_a_info["stage_a_selected_event_type"],
            stage_a_selected_unsupported=stage_a_info["stage_a_selected_unsupported"],
            grounding_attempted=grounding_attempted,
            grounding_unresolved_image_arguments=grounding_summary["unresolved_image_arguments"],
            grounding_requests=grounding_summary["grounding_requests"],
            grounding_results=grounding_summary["grounded_results"],
            grounding_failed_results=grounding_summary["failed_grounding_results"],
            grounding_applied_bboxes=grounding_summary["applied_grounded_bboxes"],
            staged=True,
            evidence_used=len(evidence_items),
            text_alignment_issues=alignment_issues,
            text_alignment_diagnostics=alignment_diagnostics,
            text_support=support_summary["text_support"],
            image_support=support_summary["image_support"],
            grounding_support=support_summary["grounding_support"],
            external_evidence_support=support_summary["external_evidence_support"],
        )
        return result


run = extraction

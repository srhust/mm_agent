"""Control-guided repair node: minimally repair event without deciding verification."""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.ontology import (
    get_allowed_image_roles,
    get_allowed_text_roles,
    get_supported_event_types,
    is_supported_event_type,
)
from mm_event_agent.observability import log_node_event
from mm_event_agent.grounding.florence2_hf import apply_grounding_results_to_event
from mm_event_agent.grounding.debug import compare_grounding_stages, summarize_grounding_activity
from mm_event_agent.schemas import (
    empty_event,
    enforce_strict_text_grounding,
    extract_json_object,
    find_text_span,
    parse_event_json,
    validate_event,
)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
        )
    return _llm


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _format_similar_events(raw: Any) -> str:
    if not raw or not isinstance(raw, list):
        return "(none)"
    lines: list[str] = []
    for ev in raw:
        if isinstance(ev, dict):
            lines.append(json.dumps(ev, ensure_ascii=False))
        else:
            lines.append(str(ev))
    return "\n".join(lines) if lines else "(none)"


def _format_issues(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        return "\n".join(f"- {x}" for x in raw)
    if raw:
        return f"- {raw!r}"
    return "- (none listed)"


def _format_diagnostics(raw: Any) -> str:
    if not isinstance(raw, list) or not raw:
        return "(none)"
    lines: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            lines.append(json.dumps(item, ensure_ascii=False))
        else:
            lines.append(str(item))
    return "\n".join(lines) if lines else "(none)"


def _build_repair_plan(raw: Any) -> list[dict[str, str]]:
    """Build a lightweight field-local repair plan from verifier diagnostics."""
    if not isinstance(raw, list):
        return []

    plan: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        issue_type = str(item.get("issue_type") or "").strip()
        suggested_action = str(item.get("suggested_action") or "").strip()
        if not field_path:
            continue
        key = (field_path, issue_type, suggested_action)
        if key in seen:
            continue
        seen.add(key)
        plan.append(
            {
                "field_path": field_path,
                "issue_type": issue_type or "unspecified_issue",
                "suggested_action": suggested_action or "inspect_and_fix_locally",
            }
        )
    return plan


def _format_repair_plan(plan: list[dict[str, str]]) -> str:
    """Format the repair plan so the LLM sees field-local targets explicitly."""
    if not plan:
        return "(none)"
    lines: list[str] = []
    for item in plan:
        lines.append(
            f'- field_path: {item["field_path"]}; '
            f'issue_type: {item["issue_type"]}; '
            f'suggested_action: {item["suggested_action"]}'
        )
    return "\n".join(lines)


def _collect_target_field_paths(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        if field_path and field_path not in out:
            out.append(field_path)
    return out


def _summarize_target_field_paths(field_paths: list[str]) -> str:
    if not field_paths:
        return "(none)"
    return "\n".join(f"- {path}" for path in field_paths)


def _merge_targeted_event_fields(
    current_event: dict[str, Any],
    proposed_event: dict[str, Any],
    target_field_paths: list[str],
) -> dict[str, Any]:
    if not target_field_paths:
        return proposed_event

    merged = deepcopy(current_event)
    for field_path in target_field_paths:
        _apply_targeted_field(merged, proposed_event, field_path)
    return merged


def _apply_targeted_field(target: dict[str, Any], source: dict[str, Any], field_path: str) -> None:
    if field_path == "trigger":
        target["trigger"] = deepcopy(source.get("trigger"))
        return
    if field_path == "trigger.span":
        if isinstance(target.get("trigger"), dict) and isinstance(source.get("trigger"), dict):
            target["trigger"]["span"] = deepcopy(source["trigger"].get("span"))
        return
    if field_path == "trigger.modality":
        if isinstance(target.get("trigger"), dict) and isinstance(source.get("trigger"), dict):
            target["trigger"]["modality"] = deepcopy(source["trigger"].get("modality"))
        return

    match = re.fullmatch(r"text_arguments\[(\d+)\](?:\.(span|role|text))?", field_path)
    if match:
        index = int(match.group(1))
        subfield = match.group(2)
        _apply_list_field(target, source, "text_arguments", index, subfield)
        return

    match = re.fullmatch(r"image_arguments\[(\d+)\](?:\.(bbox|role|label))?", field_path)
    if match:
        index = int(match.group(1))
        subfield = match.group(2)
        _apply_list_field(target, source, "image_arguments", index, subfield)
        return


def _apply_list_field(
    target: dict[str, Any],
    source: dict[str, Any],
    list_name: str,
    index: int,
    subfield: str | None,
) -> None:
    target_list = target.get(list_name)
    source_list = source.get(list_name)
    if not isinstance(target_list, list) or not isinstance(source_list, list):
        return
    if index < 0 or index >= len(target_list) or index >= len(source_list):
        return
    if subfield is None:
        target_list[index] = deepcopy(source_list[index])
        return
    if isinstance(target_list[index], dict) and isinstance(source_list[index], dict):
        target_list[index][subfield] = deepcopy(source_list[index].get(subfield))


def _finalize_targeted_text_grounding(
    event: dict[str, Any],
    target_field_paths: list[str],
    raw_text: str,
) -> dict[str, Any]:
    finalized = deepcopy(event)

    if any(path == "trigger" or path.startswith("trigger.") for path in target_field_paths):
        trigger = finalized.get("trigger")
        if isinstance(trigger, dict):
            span = find_text_span(raw_text, str(trigger.get("text") or ""))
            if span is None:
                finalized["trigger"] = None
            else:
                trigger["span"] = span
                trigger["modality"] = "text"

    targeted_text_indices: set[int] = set()
    for path in target_field_paths:
        match = re.match(r"text_arguments\[(\d+)\]", path)
        if match:
            targeted_text_indices.add(int(match.group(1)))

    text_arguments = finalized.get("text_arguments")
    if isinstance(text_arguments, list) and targeted_text_indices:
        rebuilt: list[Any] = []
        for index, item in enumerate(text_arguments):
            if index not in targeted_text_indices:
                rebuilt.append(item)
                continue
            if not isinstance(item, dict):
                continue
            span = find_text_span(raw_text, str(item.get("text") or ""))
            if span is None:
                continue
            item = deepcopy(item)
            item["span"] = span
            rebuilt.append(item)
        finalized["text_arguments"] = rebuilt

    return finalized


def _format_evidence_items(raw: Any) -> str:
    if not isinstance(raw, list) or not raw:
        return "(none)"
    lines: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            lines.append(json.dumps(item, ensure_ascii=False))
        else:
            lines.append(str(item))
    return "\n".join(lines) if lines else "(none)"


def _format_grounding_results(raw: Any) -> str:
    if not isinstance(raw, list) or not raw:
        return "(none)"
    lines: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            lines.append(json.dumps(item, ensure_ascii=False))
        else:
            lines.append(str(item))
    return "\n".join(lines) if lines else "(none)"


def _collect_grounded_pairs(raw: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not isinstance(raw, list):
        return pairs
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        label = str(item.get("label") or "").strip()
        bbox = item.get("bbox")
        if role and label and isinstance(bbox, list) and len(bbox) == 4 and item.get("grounding_status") == "grounded":
            pairs.add((role, label))
    return pairs


def _apply_targeted_grounding_alignment(
    event: dict[str, Any],
    grounding_results: Any,
    repair_plan: list[dict[str, str]],
) -> dict[str, Any]:
    """Apply known grounded results only to diagnosed image fields when useful."""
    if not isinstance(grounding_results, list) or not grounding_results or not repair_plan:
        return event

    should_apply = False
    for item in repair_plan:
        field_path = item.get("field_path", "")
        issue_type = item.get("issue_type", "")
        if issue_type == "grounding_result_not_applied":
            should_apply = True
            break
        if field_path.startswith("image_arguments["):
            should_apply = True
    if not should_apply:
        return event

    return apply_grounding_results_to_event(event, grounding_results)


def _restore_grounded_bboxes_from_current_event(
    original_event: dict[str, Any],
    repaired_event: dict[str, Any],
) -> dict[str, Any]:
    """Preserve existing grounded bboxes unless repair explicitly changes them."""
    original_items = original_event.get("image_arguments")
    repaired_items = repaired_event.get("image_arguments")
    if not isinstance(original_items, list) or not isinstance(repaired_items, list):
        return repaired_event

    restored = deepcopy(repaired_event)
    restored_items = restored.get("image_arguments")
    if not isinstance(restored_items, list):
        return restored

    for index, original_item in enumerate(original_items):
        if index >= len(restored_items):
            break
        repaired_item = restored_items[index]
        if not isinstance(original_item, dict) or not isinstance(repaired_item, dict):
            continue
        original_bbox = original_item.get("bbox")
        original_status = original_item.get("grounding_status")
        repaired_bbox = repaired_item.get("bbox")
        repaired_status = repaired_item.get("grounding_status")
        if (
            isinstance(original_bbox, list)
            and len(original_bbox) == 4
            and original_status == "grounded"
            and repaired_bbox is None
            and repaired_status == "unresolved"
        ):
            repaired_item["bbox"] = list(original_bbox)
            repaired_item["grounding_status"] = "grounded"
    return restored


def repair(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read data + control context, write repaired event and repair_attempts only."""
    started_at = time.perf_counter()
    if state.get("verified"):
        log_node_event("repair", state, started_at, True, skipped=True)
        return {}

    try:
        current_event = validate_event(state.get("event"))
    except Exception:
        current_event = empty_event()

    evidence = _format_evidence_items(state.get("evidence"))
    grounding_results = state.get("grounding_results")
    grounding_results_block = _format_grounding_results(grounding_results)
    similar_block = _format_similar_events(state.get("similar_events"))
    issue_block = _format_issues(state.get("issues"))
    diagnostics_block = _format_diagnostics(state.get("verifier_diagnostics"))
    verifier_reason = str(state.get("verifier_reason") or "").strip()
    repair_plan = _build_repair_plan(state.get("verifier_diagnostics"))
    repair_plan_block = _format_repair_plan(repair_plan)
    target_field_paths = _collect_target_field_paths(repair_plan)
    target_field_summary = _summarize_target_field_paths(target_field_paths)
    raw_text = str(state.get("text") or "")
    supported_event_types = get_supported_event_types()
    current_event_type = str(current_event.get("event_type") or "").strip()
    if is_supported_event_type(current_event_type):
        ontology_guidance = (
            f"Supported event types: {json.dumps(supported_event_types, ensure_ascii=False)}\n"
            f"Current event_type text roles: {json.dumps(get_allowed_text_roles(current_event_type), ensure_ascii=False)}\n"
            f"Current event_type image roles: {json.dumps(get_allowed_image_roles(current_event_type), ensure_ascii=False)}"
        )
    else:
        ontology_guidance = (
            f"Supported event types: {json.dumps(supported_event_types, ensure_ascii=False)}\n"
            "Current event_type is invalid; choose one supported event_type and use only that type's roles."
        )

    prompt = (
        "Repair the extracted event JSON with MINIMAL changes.\n"
        "- Use the repair plan derived from verifier_diagnostics as the PRIMARY repair plan.\n"
        "- Modify ONLY the diagnosed fields listed in the repair plan.\n"
        "- Preserve all other fields unchanged.\n"
        "- Do not rewrite unrelated arguments.\n"
        "- If only one field_path is diagnosed, keep every other field exactly as-is.\n"
        "- Fix ONLY what the verifier issues point to (wrong type, unsupported facts, bad structure).\n"
        "- For factual fixes, use External evidence; for shape/granularity, follow Similar events patterns.\n"
        "- Use grounding_results for image-argument repair when they provide a matching grounded bbox.\n"
        "- If verifier reports grounding_result_not_applied, prioritize aligning that image argument to the matching grounded result.\n"
        "- If grounding failed, do not force deletion of an otherwise acceptable unresolved image argument.\n"
        "- If a grounded bbox already exists, preserve it unless the diagnosed issue specifically requires changing it.\n"
        "- event_type must stay within the supported closed ontology.\n"
        "- text_arguments roles must be valid for the chosen event_type.\n"
        "- image_arguments roles must be valid for the chosen event_type.\n"
        "- Output the COMPLETE JSON object using this schema: event_type, trigger, text_arguments, image_arguments.\n"
        '- trigger must be {"text": string, "modality": "text", "span": null} or null.\n'
        '- text_arguments items must be {"role": string, "text": string, "span": null}.\n'
        '- image_arguments items must be {"role": string, "label": string, "bbox": null, "grounding_status": "unresolved"} unless you have grounded bbox data.\n'
        "- Keep trigger.text and text_arguments[].text copied from the original text when possible; do not paraphrase.\n"
        "- Use verifier diagnostics for targeted repair when available; prefer local fixes over full regeneration.\n"
        "No markdown fences or commentary.\n\n"
        f"Ontology:\n{ontology_guidance}\n\n"
        f"External evidence:\n{evidence}\n\n"
        f"Similar events (structural patterns):\n{similar_block}\n\n"
        f"Original text:\n{raw_text}\n\n"
        f"Verifier issues:\n{issue_block}\n\n"
        f"Verifier reason:\n{verifier_reason or '(none)'}\n\n"
        f"Verifier diagnostics:\n{diagnostics_block}\n\n"
        f"Grounding results:\n{grounding_results_block}\n\n"
        f"Repair plan:\n{repair_plan_block}\n\n"
        f"Target field paths:\n{target_field_summary}\n\n"
        f"Current event (JSON object):\n{json.dumps(current_event, ensure_ascii=False)}"
    )

    attempts = int(state.get("repair_attempts") or 0) + 1
    grounding_debug = compare_grounding_stages(
        current_event.get("image_arguments"),
        grounding_results,
        current_event.get("image_arguments"),
    )
    try:
        raw = _msg_text(_get_llm().invoke([HumanMessage(content=prompt)]).content).strip()
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        parsed_object = extract_json_object(raw)
        if parsed_object is None:
            raise ValueError("repair output is not valid JSON")
        try:
            proposed_event = parse_event_json(raw)
            merged_candidate = _merge_targeted_event_fields(current_event, proposed_event, target_field_paths)
        except Exception:
            merged_candidate = _merge_targeted_event_fields(current_event, parsed_object, target_field_paths)
        merged_candidate = _apply_targeted_grounding_alignment(
            merged_candidate,
            grounding_results,
            repair_plan,
        )
        repaired_event = enforce_strict_text_grounding(
            validate_event(
                _finalize_targeted_text_grounding(
                    validate_event(merged_candidate),
                    target_field_paths,
                    raw_text,
                )
            ),
            raw_text,
        )
        repaired_event = _restore_grounded_bboxes_from_current_event(current_event, repaired_event)
        grounding_debug = compare_grounding_stages(
            current_event.get("image_arguments"),
            grounding_results,
            repaired_event.get("image_arguments"),
        )
        grounding_summary = grounding_debug["summary"]
        result = {
            "event": repaired_event,
            "repair_attempts": attempts,
        }
        log_node_event(
            "repair",
            state,
            started_at,
            True,
            repair_attempts=attempts,
            event_type=repaired_event["event_type"],
            grounded_pairs=len(_collect_grounded_pairs(grounding_results)),
            grounding_unresolved_image_arguments=grounding_summary["unresolved_image_arguments"],
            grounding_results=grounding_summary["grounded_results"],
            grounding_failed_results=grounding_summary["failed_grounding_results"],
            grounding_applied_bboxes=grounding_summary["applied_grounded_bboxes"],
        )
        return result
    except Exception as exc:
        fallback_event = current_event
        grounding_summary = summarize_grounding_activity(
            image_arguments_before=current_event.get("image_arguments"),
            grounding_requests=None,
            grounding_results=grounding_results,
            image_arguments_after=fallback_event.get("image_arguments"),
        )
        result = {
            "event": fallback_event,
            "repair_attempts": attempts,
        }
        log_node_event(
            "repair",
            state,
            started_at,
            False,
            error=str(exc),
            repair_attempts=attempts,
            grounded_pairs=len(_collect_grounded_pairs(grounding_results)),
            grounding_unresolved_image_arguments=grounding_summary["unresolved_image_arguments"],
            grounding_results=grounding_summary["grounded_results"],
            grounding_failed_results=grounding_summary["failed_grounding_results"],
            grounding_applied_bboxes=grounding_summary["applied_grounded_bboxes"],
        )
        return result


run = repair

"""Control-guided repair node: minimally repair event without deciding verification."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import attach_text_spans, empty_event, parse_event_json, validate_event

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
    similar_block = _format_similar_events(state.get("similar_events"))
    issue_block = _format_issues(state.get("issues"))
    raw_text = str(state.get("text") or "")

    prompt = (
        "Repair the extracted event JSON with MINIMAL changes.\n"
        "- Fix ONLY what the verifier issues point to (wrong type, unsupported facts, bad structure).\n"
        "- Preserve correct fields and any correct subfields unchanged; do not rewrite them.\n"
        "- For factual fixes, use External evidence; for shape/granularity, follow Similar events patterns.\n"
        "- Output the COMPLETE JSON object using this schema: event_type, trigger, text_arguments, image_arguments.\n"
        '- trigger must be {"text": string, "modality": "text", "span": null} or null.\n'
        '- text_arguments items must be {"role": string, "text": string, "span": null}.\n'
        '- image_arguments items must be {"role": string, "label": string, "bbox": [x1, y1, x2, y2]}.\n'
        "- Keep trigger.text and text_arguments[].text copied from the original text when possible; do not paraphrase.\n"
        "No markdown fences or commentary.\n\n"
        f"External evidence:\n{evidence}\n\n"
        f"Similar events (structural patterns):\n{similar_block}\n\n"
        f"Original text:\n{raw_text}\n\n"
        f"Verifier issues:\n{issue_block}\n\n"
        f"Current event (JSON object):\n{json.dumps(current_event, ensure_ascii=False)}"
    )

    attempts = int(state.get("repair_attempts") or 0) + 1
    try:
        raw = _msg_text(_get_llm().invoke([HumanMessage(content=prompt)]).content).strip()
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        repaired_event = attach_text_spans(parse_event_json(raw), raw_text)
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
        )
        return result
    except Exception as exc:
        fallback_event = current_event
        result = {
            "event": fallback_event,
            "repair_attempts": attempts,
        }
        log_node_event("repair", state, started_at, False, error=str(exc), repair_attempts=attempts)
        return result


run = repair

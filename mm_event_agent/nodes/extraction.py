"""Data node: extract one structured event JSON from fusion_context only."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import attach_text_spans, empty_event, empty_fusion_context, parse_event_json

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

    prompt = (
        "Extract exactly ONE structured event using ONLY the fusion_context below. "
        "Do not use or assume any information outside this block.\n\n"
        "Output a single JSON object with exactly this structure:\n"
        '{'
        '"event_type": string, '
        '"trigger": {"text": string, "modality": "text", "span": {"start": int, "end": int} | null} | null, '
        '"text_arguments": [{"role": string, "text": string, "span": {"start": int, "end": int} | null}], '
        '"image_arguments": [{"role": string, "label": string, "bbox": [x1, y1, x2, y2]}]'
        '}\n\n'
        "Rules:\n"
        "1. Event facts must be grounded in evidence items when evidence is available.\n"
        "2. Event structure should follow similar_events patterns when patterns are available.\n"
        "3. If evidence conflicts with patterns, trust evidence over patterns.\n"
        "4. If patterns are empty, skip pattern guidance.\n"
        "5. If evidence is empty, rely on raw_text, raw_image_desc, and perception_summary only.\n"
        "6. Use raw_text and raw_image_desc as multimodal provenance; do not ignore either when they provide usable incident cues.\n"
        "7. trigger.text must be copied from raw_text when possible; do not paraphrase it.\n"
        "8. Each text_arguments[i].text must be copied from raw_text when possible; do not paraphrase it.\n"
        "9. For trigger and text arguments, set span to null in model output; spans will be computed after validation.\n"
        "10. Image arguments must use role + label + bbox only.\n"
        "11. If evidence is insufficient, prefer omission over hallucination.\n"
        "12. Ensure extraction still returns one valid JSON object even when patterns or evidence are empty.\n\n"
        "Evidence format:\n"
        "- fusion_context.evidence is a list of evidence items.\n"
        "- Each item contains title, snippet, url, source_type, published_at, and score.\n"
        "- Treat evidence snippets as the primary factual content, with title and url as supporting provenance.\n\n"
        "Guidance:\n"
        f"- Raw text present: {'YES' if raw_text else 'NO'}\n"
        f"- Raw image description present: {'YES' if raw_image_desc else 'NO'}\n"
        f"- Perception summary present: {'YES' if perception_summary else 'NO'}\n"
        f"- Pattern guidance available: {'YES' if patterns else 'NO'}\n"
        f"- Evidence available: {'YES' if evidence_items else 'NO'}\n\n"
        "Reply with ONLY valid JSON. No markdown fences or extra text.\n\n"
        f"{json.dumps(fusion_context, ensure_ascii=False)}"
    )

    try:
        out = _get_llm().invoke([HumanMessage(content=prompt)])
        event = attach_text_spans(parse_event_json(_msg_text(out.content)), raw_text)
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

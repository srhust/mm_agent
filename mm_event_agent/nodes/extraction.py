"""Data node: extract one structured event JSON from fusion_context only."""

from __future__ import annotations

import os
import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

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
    raw_context = state.get("fusion_context")
    if isinstance(raw_context, dict):
        fusion_context = raw_context
    else:
        fusion_context = {
            "input": str(state.get("perception_summary") or ""),
            "patterns": [],
            "evidence": list(state.get("evidence")) if isinstance(state.get("evidence"), list) else [],
        }

    input_summary = str(fusion_context.get("input") or "")
    patterns = fusion_context.get("patterns")
    evidence_items = fusion_context.get("evidence")

    if not isinstance(patterns, list):
        patterns = []
    if not isinstance(evidence_items, list):
        evidence_items = []

    prompt = (
        "Extract exactly ONE structured event using ONLY the fusion_context below. "
        "Do not use or assume any information outside this block.\n\n"
        "Output a single JSON object with keys: "
        "event_type (string), trigger (string), arguments (object or array).\n\n"
        "Rules:\n"
        "1. Event facts must be grounded in evidence when evidence is available.\n"
        "2. Event structure should follow similar_events when similar_events are available.\n"
        "3. If evidence conflicts with patterns, trust evidence over patterns.\n"
        "4. If similar_events is empty, skip pattern guidance and infer only a reasonable structure from the input.\n"
        "5. If evidence is empty, rely on input only and do not invent unsupported facts.\n"
        "6. Ensure extraction still returns one valid JSON object even when patterns or evidence are empty.\n\n"
        "Evidence format:\n"
        "- fusion_context.evidence is a list of evidence items.\n"
        "- Each item may contain title, snippet, url, and source_type.\n"
        "- Treat snippets as the primary factual content unless title/url add support.\n\n"
        "Guidance:\n"
        f"- Input summary present: {'YES' if input_summary else 'NO'}\n"
        f"- Pattern guidance available: {'YES' if patterns else 'NO'}\n"
        f"- Evidence available: {'YES' if evidence_items else 'NO'}\n\n"
        "Reply with ONLY valid JSON. No markdown fences or extra text.\n\n"
        f"{json.dumps(fusion_context, ensure_ascii=False)}"
    )

    out = _get_llm().invoke([HumanMessage(content=prompt)])
    raw = _msg_text(out.content)
    return {"event": raw}


run = extraction

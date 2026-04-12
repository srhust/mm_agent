"""Control-guided repair node: minimally repair event without deciding verification."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

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
    if state.get("verified"):
        return {}

    event = str(state.get("event") or "")
    evidence = _format_evidence_items(state.get("evidence"))
    similar_block = _format_similar_events(state.get("similar_events"))
    issue_block = _format_issues(state.get("issues"))

    prompt = (
        "Repair the extracted event JSON with MINIMAL changes.\n"
        "- Fix ONLY what the verifier issues point to (wrong type, unsupported facts, bad structure).\n"
        "- Preserve correct fields and any correct subfields inside `arguments` unchanged; do not rewrite them.\n"
        "- For factual fixes, use External evidence; for shape/granularity, follow Similar events patterns.\n"
        "- Output the COMPLETE JSON object (event_type, trigger, arguments) after repair, not a patch or diff.\n"
        "No markdown fences or commentary.\n\n"
        f"External evidence:\n{evidence}\n\n"
        f"Similar events (structural patterns):\n{similar_block}\n\n"
        f"Verifier issues:\n{issue_block}\n\n"
        f"Current event (JSON string):\n{event}"
    )

    raw = _msg_text(_get_llm().invoke([HumanMessage(content=prompt)]).content).strip()
    raw = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    attempts = int(state.get("repair_attempts") or 0) + 1
    return {
        "event": raw.strip(),
        "repair_attempts": attempts,
    }


run = repair

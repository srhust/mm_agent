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
    try:
        event = validate_event(state.get("event"))
    except Exception:
        event = empty_event()

    prompt = (
        "You verify an extracted event JSON against the SAME structured fusion_context used at extraction time.\n\n"
        "Checks:\n"
        "1) Text support: Are event claims supported by fusion_context.raw_text?\n"
        "2) Image support: Are event claims supported by fusion_context.raw_image_desc?\n"
        "3) Evidence support: Are event claims supported by fusion_context.evidence item snippets when evidence items are available? "
        "Use evidence snippets as the primary factual basis for externally supported facts.\n"
        "4) Structure: Is the event (event_type, trigger, arguments shape/granularity) consistent with "
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
            issues = ["invalid verifier output (not valid JSON object)"]
            verified = False
            confidence = 0.0
            reason = "invalid verifier output"
            success = False
        else:
            _, issues, verified, confidence, reason = _normalize_verdict_payload(parsed)
            success = True
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

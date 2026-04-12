"""Control node: validate event against fused context and update control fields."""

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
            temperature=0,
        )
    return _llm


def _msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            out = json.loads(t[start : end + 1])
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            return None
    return None


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
    fusion_context = state.get("fusion_context")
    if not isinstance(fusion_context, dict):
        fusion_context = {
            "input": str(state.get("perception_summary") or ""),
            "patterns": [],
            "evidence": list(state.get("evidence")) if isinstance(state.get("evidence"), list) else [],
        }
    event = str(state.get("event") or "")

    prompt = (
        "You verify an extracted event JSON against the SAME structured fusion_context used at extraction time.\n\n"
        "Checks:\n"
        "1) Evidence: Are factual claims in the event supported by fusion_context.evidence when evidence items are available? "
        "Use the evidence item snippets as the primary factual basis and flag unsupported or over-specific facts.\n"
        "2) Structure: Is the event (event_type, trigger, arguments shape/granularity) consistent with "
        "fusion_context.patterns when patterns are available?\n"
        "3) If evidence conflicts with patterns, prefer evidence.\n\n"
        "Return ONLY one JSON object (no markdown), exactly this shape:\n"
        '{"verdict": "YES" or "NO", "issues": ["unsupported argument", "wrong event type", ...], "confidence": 0.0, "reason": "short explanation"}\n'
        "Use an empty issues array when verdict is YES. Confidence must be a float from 0 to 1. "
        "Reason must be a short explanation.\n\n"
        f"fusion_context:\n{json.dumps(fusion_context, ensure_ascii=False)}\n\nExtracted event:\n{event}"
    )

    raw = _msg_text(_get_llm().invoke([HumanMessage(content=prompt)]).content)
    parsed = _extract_json_object(raw)
    if parsed is None:
        issues = ["invalid verifier output (not valid JSON object)"]
        verified = False
        confidence = 0.0
        reason = "invalid verifier output"
    else:
        _, issues, verified, confidence, reason = _normalize_verdict_payload(parsed)

    return {
        "verified": verified,
        "issues": issues,
        "verifier_confidence": confidence,
        "verifier_reason": reason,
    }


run = verifier

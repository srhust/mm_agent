"""校验：证据支撑 + 与 similar_events 结构一致；输出 verdict / issues JSON。"""

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


def _normalize_verdict_payload(data: dict[str, Any]) -> tuple[str, list[str], bool]:
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
    verified = verdict == "YES"
    return verdict, issues, verified


def verifier(state: Mapping[str, Any]) -> dict[str, Any]:
    """依据 fusion_context 检查 event_json：证据支撑 + 与 [Similar Events] 模式一致。"""
    fusion_context = str(state.get("fusion_context") or "").strip() or "(none)"
    event = str(state.get("event_json") or "")

    prompt = (
        "You verify an extracted event JSON against the SAME fused context used at extraction time.\n\n"
        "Checks:\n"
        "1) Evidence: Are factual claims in the event supported by [External Evidence]? "
        "Flag unsupported or over-specific facts.\n"
        "2) Structure: Is the event (event_type, trigger, arguments shape/granularity) consistent with "
        "patterns shown under [Similar Events]?\n\n"
        "Return ONLY one JSON object (no markdown), exactly this shape:\n"
        '{"verdict": "YES" or "NO", "issues": ["unsupported argument", "wrong event type", ...]}\n'
        "Use an empty issues array when verdict is YES.\n\n"
        f"Context:\n{fusion_context}\n\nExtracted event JSON:\n{event}"
    )

    raw = _msg_text(_get_llm().invoke([HumanMessage(content=prompt)]).content)
    parsed = _extract_json_object(raw)
    if parsed is None:
        verdict, issues = "NO", ["invalid verifier output (not valid JSON object)"]
        verified = False
    else:
        verdict, issues, verified = _normalize_verdict_payload(parsed)

    payload = {"verdict": verdict, "issues": issues}
    return {
        "verified": verified,
        "verdict": verdict,
        "issues": issues,
        "verifier_feedback": json.dumps(payload, ensure_ascii=False),
    }


run = verifier

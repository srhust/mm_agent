"""事件抽取：仅依据 fusion_context 生成结构化事件 JSON（不单独喂 raw text / evidence）。"""

from __future__ import annotations

import os
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
    """只读 fusion_context，输出 event_json（单条 JSON 字符串）。"""
    fusion_context = str(state.get("fusion_context") or "").strip()
    if not fusion_context:
        fusion_context = "Context:\n(none)"

    prompt = (
        "Extract exactly ONE structured event using ONLY the Context below. "
        "Do not use or assume any information outside this block.\n\n"
        "Output a single JSON object with keys: "
        "event_type (string), trigger (string), arguments (object or array—"
        "follow the shape and granularity of entries under [Similar Events]).\n\n"
        "Constraints:\n"
        "- Pattern: align event_type / trigger / arguments structure with [Similar Events] examples.\n"
        "- Grounding: put factual claims (entities, place, time, casualties, etc.) only when supported by "
        "[External Evidence]; if evidence is absent or insufficient, use null or empty values—do not fabricate.\n"
        "- [Input] is the incident summary; reconcile it with [External Evidence] when both apply.\n\n"
        "Reply with ONLY valid JSON. No markdown fences or extra text.\n\n"
        f"{fusion_context}"
    )

    out = _get_llm().invoke([HumanMessage(content=prompt)])
    raw = _msg_text(out.content)
    return {"event_json": raw}


run = extraction

"""本地 RAG：FAISS + 向量检索，返回结构化相似事件（无 LLM）。"""

from __future__ import annotations

import json
from typing import Any, Mapping

from mm_event_agent.vector_store import retrieve as vector_retrieve


def _arguments_to_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        return [f"{k}: {v}" for k, v in raw.items()]
    return [str(raw)]


def _normalize_event(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": str(obj.get("event_type") or ""),
        "trigger": str(obj.get("trigger") or ""),
        "arguments": _arguments_to_list(obj.get("arguments")),
    }


def _parse_retrieved_doc(doc: str) -> dict[str, Any]:
    text = doc.strip()
    if not text:
        return {"event_type": "", "trigger": "", "arguments": []}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _normalize_event(data)
    except json.JSONDecodeError:
        pass
    return {"event_type": "", "trigger": text[:500], "arguments": []}


def rag(state: Mapping[str, Any]) -> dict[str, Any]:
    """以 state["text"] 为 query，FAISS 检索 top-3，解析为 similar_events。"""
    q = str(state.get("text") or "").strip()
    if not q:
        return {"similar_events": []}
    raw_docs = vector_retrieve(q, top_k=3)
    similar_events = [_parse_retrieved_doc(d) for d in raw_docs]
    return {"similar_events": similar_events}


run = rag

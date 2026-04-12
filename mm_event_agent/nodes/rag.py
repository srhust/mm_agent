"""Data node: local RAG returns structured similar_events."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from mm_event_agent.observability import log_node_event


def _retrieve_similar_events(query: str, top_k: int = 3) -> list[str]:
    from mm_event_agent.vector_store import retrieve as vector_retrieve

    return vector_retrieve(query, top_k=top_k)


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
    """Read only data field text and write only data field similar_events."""
    started_at = time.perf_counter()
    q = str(state.get("text") or "").strip()
    if not q:
        result = {"similar_events": []}
        log_node_event("local_rag", state, started_at, True, patterns=0)
        return result
    try:
        raw_docs = _retrieve_similar_events(q, top_k=3)
        similar_events = [_parse_retrieved_doc(d) for d in raw_docs]
        result = {"similar_events": similar_events}
        log_node_event("local_rag", state, started_at, True, patterns=len(similar_events))
        return result
    except Exception as exc:
        log_node_event("local_rag", state, started_at, False, error=str(exc), patterns=0)
        return {"similar_events": []}


run = rag

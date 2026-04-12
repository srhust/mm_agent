"""Data node: external search writes evidence."""

from __future__ import annotations

from typing import Any, Mapping

import time

from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import EvidenceItem, ValidationError, validate_evidence_item


def _search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    from ddgs import DDGS

    return list(DDGS().text(query, max_results=max_results))


def _normalize_result(result: Any, rank: int) -> EvidenceItem | None:
    if not isinstance(result, dict):
        return None

    title = str(result.get("title") or "").strip()
    snippet = str(result.get("body") or result.get("snippet") or "").strip()
    url = str(result.get("href") or result.get("url") or "").strip()
    if not title or not snippet or not url:
        return None

    raw_score = result.get("score")
    if raw_score is None:
        score = 0.0
    else:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0

    item = {
        "title": title,
        "snippet": snippet,
        "url": url,
        "source_type": "search",
        "published_at": result.get("date") or result.get("published") or result.get("published_at"),
        "score": score,
    }
    try:
        return validate_evidence_item(item)
    except ValidationError:
        return None


def search(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read only data field text and write only data field evidence."""
    started_at = time.perf_counter()
    q = str(state.get("text") or "").strip()
    if not q:
        result = {"search_query": "", "evidence": []}
        log_node_event("search", state, started_at, True, search_query="", returned_evidence=0)
        return result
    try:
        raw_results = _search_web(q)
        evidence: list[EvidenceItem] = []
        for rank, item in enumerate(raw_results):
            normalized = _normalize_result(item, rank)
            if normalized is not None:
                evidence.append(normalized)
        result = {"search_query": q, "evidence": evidence}
        log_node_event("search", state, started_at, True, search_query=q, returned_evidence=len(evidence))
        return result
    except Exception as exc:
        result = {"search_query": q, "evidence": []}
        log_node_event("search", state, started_at, False, search_query=q, error=str(exc), returned_evidence=0)
        return result


run = search

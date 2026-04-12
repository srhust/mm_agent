"""Data node: external search writes evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
from typing import Any, Mapping

import time

from mm_event_agent.observability import log_node_event
from mm_event_agent.schemas import EvidenceItem, ValidationError, validate_evidence_item
from mm_event_agent.search.tavily_client import search_news


def _rewrite_query(text: str, max_terms: int = 12) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    normalized = re.sub(r"[^\w\s:/.-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""
    return " ".join(normalized.split(" ")[:max_terms])


def _validate_evidence_list(items: Any) -> list[EvidenceItem]:
    if not isinstance(items, list):
        return []
    evidence: list[EvidenceItem] = []
    for item in items:
        try:
            evidence.append(validate_evidence_item(item))
        except ValidationError:
            continue
    return evidence


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) > 2}


def _keyword_overlap_score(reference: set[str], candidate: set[str]) -> float:
    if not reference or not candidate:
        return 0.0
    return len(reference & candidate) / len(reference)


def _recency_score(published_at: str | None) -> float:
    if not published_at:
        return 0.0
    value = str(published_at).strip()
    if not value:
        return 0.0

    parsed: datetime | None = None
    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value, pattern)
            break
        except ValueError:
            continue

    if parsed is None:
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return 0.0

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    age_days = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0)
    if age_days <= 3:
        return 1.0
    if age_days <= 14:
        return 0.7
    if age_days <= 60:
        return 0.3
    return 0.1


def _evidence_rank_score(item: EvidenceItem, query_text: str, source_text: str) -> float:
    query_tokens = _tokenize(query_text)
    source_tokens = _tokenize(source_text)
    title_tokens = _tokenize(item["title"])
    snippet_tokens = _tokenize(item["snippet"])
    combined_tokens = title_tokens | snippet_tokens

    query_overlap = _keyword_overlap_score(query_tokens, combined_tokens)
    source_overlap = _keyword_overlap_score(source_tokens, combined_tokens)
    title_overlap = _keyword_overlap_score(query_tokens, title_tokens)
    tavily_score = max(0.0, min(1.0, float(item.get("score", 0.0))))
    recency = _recency_score(item.get("published_at"))

    return (
        0.35 * query_overlap
        + 0.25 * source_overlap
        + 0.15 * title_overlap
        + 0.15 * tavily_score
        + 0.10 * recency
    )


def _passes_basic_relevance(item: EvidenceItem, query_text: str, source_text: str) -> bool:
    query_tokens = _tokenize(query_text)
    source_tokens = _tokenize(source_text)
    combined_tokens = _tokenize(item["title"]) | _tokenize(item["snippet"])
    return bool((query_tokens & combined_tokens) or (source_tokens & combined_tokens))


def _min_relevance_threshold() -> float:
    raw = os.getenv("MM_EVENT_SEARCH_MIN_RELEVANCE", "0.18")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.18


def _top_k_limit() -> int:
    raw = os.getenv("MM_EVENT_SEARCH_TOP_K", "3")
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _filter_and_rerank_evidence(evidence: list[EvidenceItem], query_text: str, source_text: str) -> list[EvidenceItem]:
    if not evidence:
        return []

    threshold = _min_relevance_threshold()
    ranked: list[tuple[float, int, EvidenceItem]] = []
    for index, item in enumerate(evidence):
        if not _passes_basic_relevance(item, query_text=query_text, source_text=source_text):
            continue
        rank_score = _evidence_rank_score(item, query_text=query_text, source_text=source_text)
        if rank_score >= threshold:
            ranked.append((rank_score, index, item))

    ranked.sort(key=lambda entry: (-entry[0], -entry[2]["score"], entry[1]))
    top_k = _top_k_limit()
    return [item for _, _, item in ranked[:top_k]]


def search(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read only data field text and write only data field evidence."""
    started_at = time.perf_counter()
    source_text = str(state.get("text") or "")
    q = _rewrite_query(source_text)
    if not q:
        result = {"search_query": "", "evidence": []}
        log_node_event("search", state, started_at, True, search_query="", returned_evidence=0)
        return result
    try:
        evidence = _filter_and_rerank_evidence(_validate_evidence_list(search_news(q)), query_text=q, source_text=source_text)
        result = {"search_query": q, "evidence": evidence}
        log_node_event("search", state, started_at, True, search_query=q, returned_evidence=len(evidence))
        return result
    except Exception as exc:
        result = {"search_query": q, "evidence": []}
        log_node_event("search", state, started_at, False, search_query=q, error=str(exc), returned_evidence=0)
        return result


run = search

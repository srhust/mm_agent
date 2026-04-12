"""Minimal Tavily-backed external search adapter."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import EvidenceItem, ValidationError, validate_evidence_item


class TavilySearchClient:
    """Small adapter that turns Tavily search results into EvidenceItem objects."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: float | None = None,
        max_results: int | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else settings.tavily_api_key).strip()
        self.endpoint = (endpoint if endpoint is not None else settings.tavily_endpoint).strip()
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.search_timeout_seconds
        self.max_results = max_results if max_results is not None else settings.search_max_results

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.endpoint)

    def search(self, query: str) -> list[EvidenceItem]:
        normalized_query = str(query or "").strip()
        if not normalized_query or not self.configured:
            return []
        try:
            payload = {
                "api_key": self.api_key,
                "query": normalized_query,
                "topic": "news",
                "search_depth": "basic",
                "max_results": max(1, int(self.max_results)),
                "include_answer": False,
                "include_raw_content": False,
            }
            response = self._post(payload)
            results = response.get("results")
            if not isinstance(results, list):
                return []

            evidence: list[EvidenceItem] = []
            for item in results:
                normalized = self._normalize_result(item)
                if normalized is not None:
                    evidence.append(normalized)
            return evidence
        except Exception:
            return []

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("tavily search request failed") from exc

        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("tavily search returned invalid json") from exc

        return data if isinstance(data, dict) else {}

    def _normalize_result(self, result: Any) -> EvidenceItem | None:
        if not isinstance(result, dict):
            return None

        item = {
            "title": str(result.get("title") or "").strip(),
            "snippet": str(result.get("content") or result.get("snippet") or "").strip(),
            "url": str(result.get("url") or "").strip(),
            "source_type": "search",
            "published_at": result.get("published_date") or result.get("published_at"),
            "score": result.get("score", 0.0),
        }
        try:
            return validate_evidence_item(item)
        except ValidationError:
            return None
def search_news(query: str) -> list[EvidenceItem]:
    """Module-level helper for the default configured search adapter."""
    return TavilySearchClient().search(query)

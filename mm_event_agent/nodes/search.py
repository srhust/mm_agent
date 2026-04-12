"""Data node: external search writes evidence."""

from __future__ import annotations

import re
from typing import Any, Mapping

_ddg = None


def _get_ddg():
    global _ddg
    if _ddg is None:
        from langchain_community.tools import DuckDuckGoSearchRun

        _ddg = DuckDuckGoSearchRun()
    return _ddg


def _to_evidence_items(raw: Any) -> list[dict[str, str]]:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return []

    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
    if not chunks:
        chunks = [text]

    items: list[dict[str, str]] = []
    for chunk in chunks[:5]:
        items.append(
            {
                "title": "",
                "snippet": chunk,
                "url": "",
                "source_type": "search",
            }
        )
    return items


def search(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read only data field text and write only data field evidence."""
    q = str(state.get("text") or "").strip()
    if not q:
        return {"evidence": []}
    try:
        out = _get_ddg().invoke(q)
        return {"evidence": _to_evidence_items(out)}
    except Exception:
        return {"evidence": []}


run = search

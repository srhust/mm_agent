"""DuckDuckGo 检索（LangChain DuckDuckGoSearchRun）。"""

from __future__ import annotations

from typing import Any, Mapping

_ddg = None


def _get_ddg():
    global _ddg
    if _ddg is None:
        from langchain_community.tools import DuckDuckGoSearchRun

        _ddg = DuckDuckGoSearchRun()
    return _ddg


def search(state: Mapping[str, Any]) -> dict[str, Any]:
    """用 state["text"] 作为查询，写入 search_result。"""
    q = str(state.get("text") or "").strip()
    if not q:
        return {"search_result": ""}
    try:
        out = _get_ddg().invoke(q)
        return {"search_result": str(out) if out is not None else ""}
    except Exception:
        return {"search_result": ""}


run = search

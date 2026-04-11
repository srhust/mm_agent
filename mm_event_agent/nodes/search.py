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
    """用 state["text"] 作为查询，写入 evidence（外部检索摘要）。"""
    q = str(state.get("text") or "").strip()
    if not q:
        return {"evidence": ""}
    try:
        out = _get_ddg().invoke(q)
        return {"evidence": str(out) if out is not None else ""}
    except Exception:
        return {"evidence": ""}


run = search

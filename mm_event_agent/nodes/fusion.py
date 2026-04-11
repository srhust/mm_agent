"""融合相似事件、外部证据与感知摘要为 fusion_context（无 LLM）。"""

from __future__ import annotations

import json
from typing import Any, Mapping


def _block_similar_events(raw: Any) -> str:
    if not raw or not isinstance(raw, list):
        return "(none)"
    lines: list[str] = []
    for i, ev in enumerate(raw, start=1):
        if isinstance(ev, dict):
            try:
                lines.append(f"{i}. {json.dumps(ev, ensure_ascii=False)}")
            except (TypeError, ValueError):
                lines.append(f"{i}. {ev!r}")
        else:
            lines.append(f"{i}. {ev!r}")
    return "\n".join(lines) if lines else "(none)"


def _nonempty_block(value: str) -> str:
    v = value.strip()
    return v if v else "(none)"


def fusion(state: Mapping[str, Any]) -> dict[str, Any]:
    """读取 perception_summary、similar_events、evidence，写入固定版式的 fusion_context。"""
    similar = _block_similar_events(state.get("similar_events"))
    evidence = _nonempty_block(str(state.get("evidence") or ""))
    perception = _nonempty_block(str(state.get("perception_summary") or ""))

    fusion_context = (
        "Context:\n"
        "[Similar Events]\n"
        f"{similar}\n\n"
        "[External Evidence]\n"
        f"{evidence}\n\n"
        "[Input]\n"
        f"{perception}"
    )
    return {"fusion_context": fusion_context}


run = fusion

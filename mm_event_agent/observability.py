"""Lightweight structured logging helpers."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping


_LOGGER = logging.getLogger("mm_event_agent")

if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("MM_AGENT_LOG_LEVEL", "INFO"))


def log_node_event(
    node_name: str,
    state: Mapping[str, Any],
    started_at: float,
    success: bool,
    **extra: Any,
) -> None:
    evidence = state.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    repair_attempts = int(state.get("repair_attempts") or 0)
    payload = {
        "node": node_name,
        "success": success,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        "evidence_count": evidence_count,
        "repair_attempts": repair_attempts,
    }
    payload.update(extra)
    line = json.dumps(payload, ensure_ascii=False, default=str)
    if success:
        _LOGGER.info(line)
    else:
        _LOGGER.warning(line)

"""结构化信息抽取节点。"""

from __future__ import annotations

from typing import Any


def run(state: dict[str, Any]) -> dict[str, Any]:
    """从上下文抽取事件或字段并更新 state。"""
    return state

"""LangGraph 全局状态定义。"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """多模态事件抽取 agent 的状态。

    核心字段（业务输入/输出）：text, image_desc, perception_summary, similar_events,
    evidence, fusion_context, event, verified。

    issues / repair_attempts：校验与修复环使用，保证图可运行（与既有逻辑一致）。
    """

    text: str
    image_desc: str
    perception_summary: str
    similar_events: list[dict[str, Any]]
    evidence: str
    fusion_context: str
    event: str
    verified: bool
    issues: list[str]
    repair_attempts: int

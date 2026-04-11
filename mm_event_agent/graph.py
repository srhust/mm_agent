"""多模态事件抽取 LangGraph。

检索分工：本地 RAG = 相似事件先验；在线 Search = 新闻与事实 grounding；Fusion = 合并为统一上下文。
流程：perception → memory → local_rag → search → fusion → extraction → verifier ⟷ repair。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from mm_event_agent.nodes.memory import memory as memory_node
from mm_event_agent.nodes.perception import perception as perception_node
from mm_event_agent.nodes.extraction import extraction as extraction_node
from mm_event_agent.nodes.fusion import fusion as fusion_node
from mm_event_agent.nodes.rag import rag as local_rag_node
from mm_event_agent.nodes.search import search as search_node
from mm_event_agent.nodes.repair import repair as repair_node
from mm_event_agent.nodes.verifier import verifier as verifier_node

MAX_REPAIR_ATTEMPTS = 3


class GraphState(TypedDict):
    """流水线共享状态。"""

    text: str
    image_desc: str
    perception_summary: str
    memory: Annotated[list[dict[str, str]], operator.add]
    similar_events: list[dict[str, Any]]
    evidence: str
    fusion_context: str
    event_json: str
    verified: bool
    verdict: str
    issues: list[str]
    verifier_feedback: str
    repair_attempts: int


def route_after_verifier(
    state: GraphState,
) -> Literal["repair", "end"]:
    """未通过校验且未超次则进入修复，否则结束。"""
    if state["verified"]:
        return "end"
    if state["repair_attempts"] >= MAX_REPAIR_ATTEMPTS:
        return "end"
    return "repair"


def build_graph():
    """构建并编译 StateGraph。"""
    g = StateGraph(GraphState)
    g.add_node("perception", perception_node)
    g.add_node("memory", memory_node)
    g.add_node("local_rag", local_rag_node)
    g.add_node("search", search_node)
    g.add_node("fusion", fusion_node)
    g.add_node("extraction", extraction_node)
    g.add_node("verifier", verifier_node)
    g.add_node("repair", repair_node)

    g.add_edge(START, "perception")
    g.add_edge("perception", "memory")
    g.add_edge("memory", "local_rag")
    g.add_edge("local_rag", "search")
    g.add_edge("search", "fusion")
    g.add_edge("fusion", "extraction")
    g.add_edge("extraction", "verifier")
    g.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"repair": "repair", "end": END},
    )
    g.add_edge("repair", "verifier")

    return g.compile()

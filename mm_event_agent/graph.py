"""LangGraph 多模态事件抽取 agent。

流程：perception → local_rag → search → fusion → extraction → verifier；
verifier 未通过则 repair → 回到 verifier。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from mm_event_agent.state import AgentState
from mm_event_agent.nodes.extraction import extraction as extraction_node
from mm_event_agent.nodes.fusion import fusion as fusion_node
from mm_event_agent.nodes.perception import perception as perception_node
from mm_event_agent.nodes.rag import rag as local_rag_node
from mm_event_agent.nodes.repair import repair as repair_node
from mm_event_agent.nodes.search import search as search_node
from mm_event_agent.nodes.verifier import verifier as verifier_node

MAX_REPAIR_ATTEMPTS = 3


def route_after_verifier(state: AgentState) -> Literal["repair", "end"]:
    if state["verified"]:
        return "end"
    if state["repair_attempts"] >= MAX_REPAIR_ATTEMPTS:
        return "end"
    return "repair"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("perception", perception_node)
    g.add_node("local_rag", local_rag_node)
    g.add_node("search", search_node)
    g.add_node("fusion", fusion_node)
    g.add_node("extraction", extraction_node)
    g.add_node("verifier", verifier_node)
    g.add_node("repair", repair_node)

    g.add_edge(START, "perception")
    g.add_edge("perception", "local_rag")
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


# 兼容旧引用
GraphState = AgentState

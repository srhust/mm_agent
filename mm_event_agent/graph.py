"""多模态事件抽取 LangGraph：感知 → 记忆 → 检索 → 抽取 → 校验 ⟷ 修复。"""

from __future__ import annotations

import json
import operator
import os
import re
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from mm_event_agent.nodes.memory import memory as memory_node
from mm_event_agent.nodes.perception import perception as perception_node
from mm_event_agent.nodes.search import search as search_node

MAX_REPAIR_ATTEMPTS = 3
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class GraphState(TypedDict):
    """流水线共享状态。"""

    text: str
    image_desc: str
    perception_summary: str
    memory: Annotated[list[dict[str, str]], operator.add]
    search_result: str
    event_json: str
    verified: bool
    verifier_feedback: str
    repair_attempts: int


def _client() -> OpenAI:
    return OpenAI()


def extraction(state: GraphState) -> dict:
    """从摘要与检索结果生成结构化事件 JSON。"""
    ctx = (
        f"场景摘要：\n{state['perception_summary']}\n\n"
        f"检索摘要：\n{state['search_result'] or '（无）'}"
    )
    schema_hint = (
        'JSON 对象，键仅包含：'
        '"event_type","summary","time_expression","location","participants"（participants 为字符串数组）。'
        "不要 markdown 代码块，只输出一行合法 JSON。"
    )
    r = _client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"从下列内容抽取一个主要事件。{schema_hint}\n\n{ctx}",
            }
        ],
        temperature=0.1,
    )
    raw = (r.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL)
    return {"event_json": raw.strip()}


def verifier(state: GraphState) -> dict:
    """校验 JSON 可解析、字段齐全，并与上下文做一致性检查。"""
    required = {"event_type", "summary", "time_expression", "location", "participants"}
    try:
        data = json.loads(state["event_json"])
    except json.JSONDecodeError as e:
        return {
            "verified": False,
            "verifier_feedback": f"JSON 无法解析：{e}",
        }
    if not isinstance(data, dict):
        return {"verified": False, "verifier_feedback": "根节点必须是 JSON 对象。"}
    keys = set(data.keys())
    if not required.issubset(keys):
        missing = required - keys
        return {
            "verified": False,
            "verifier_feedback": f"缺少字段：{sorted(missing)}",
        }
    if not isinstance(data.get("participants"), list):
        return {"verified": False, "verifier_feedback": "participants 必须是数组。"}

    check_prompt = (
        "你是校验器。根据「场景与检索」判断下方「事件 JSON」是否自洽、无编造明显矛盾。"
        '只输出一行 JSON：{"ok":true或false,"reason":"简短中文理由"}。\n\n'
        f"场景与检索：\n{state['perception_summary']}\n{state['search_result'] or ''}\n\n"
        f"事件 JSON：\n{state['event_json']}"
    )
    r = _client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": check_prompt}],
        temperature=0,
    )
    txt = (r.choices[0].message.content or "").strip()
    m = re.search(r"\{[^{}]*\}", txt, re.DOTALL)
    ok, reason = False, txt[:500]
    if m:
        try:
            j = json.loads(m.group())
            ok = bool(j.get("ok"))
            reason = str(j.get("reason", reason))
        except json.JSONDecodeError:
            ok = False
    return {"verified": ok, "verifier_feedback": reason}


def repair(state: GraphState) -> dict:
    """根据校验反馈重写 event_json，并增加修复次数。"""
    fb = state["verifier_feedback"] or "请修正。"
    prompt = (
        "根据校验意见修正事件 JSON，键集不变："
        "event_type, summary, time_expression, location, participants。"
        "只输出一行合法 JSON，不要代码块。\n\n"
        f"当前 JSON：{state['event_json']}\n\n校验意见：{fb}"
    )
    r = _client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    raw = (r.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL)
    return {
        "event_json": raw.strip(),
        "repair_attempts": state["repair_attempts"] + 1,
    }


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
    g.add_node("search", search_node)
    g.add_node("extraction", extraction)
    g.add_node("verifier", verifier)
    g.add_node("repair", repair)

    g.add_edge(START, "perception")
    g.add_edge("perception", "memory")
    g.add_edge("memory", "search")
    g.add_edge("search", "extraction")
    g.add_edge("extraction", "verifier")
    g.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"repair": "repair", "end": END},
    )
    g.add_edge("repair", "verifier")

    return g.compile()

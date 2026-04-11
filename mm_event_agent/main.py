"""运行 LangGraph 多模态事件抽取流水线。"""

from __future__ import annotations

import mm_event_agent.graph as graph
from mm_event_agent.vector_store import build_index

# 本地 RAG 语料：相似事件的模式 / few-shot（非当前新闻事实）。可按业务换成 JSONL、数据库导出等。
RAG_EVENT_CORPUS = [
    (
        '{"event_type":"explosion","trigger":"improvised explosive device detonated",'
        '"arguments":{"location":"public market","perpetrator":"unknown","casualties":"multiple"}}'
    ),
    (
        '{"event_type":"explosion","trigger":"gas leak ignition",'
        '"arguments":{"location":"residential building","cause":"gas","casualties":"unknown"}}'
    ),
    (
        "Pattern: explosion + market/crowded place → arguments often include location, device/cause, "
        "casualties, emergency response."
    ),
]


def main() -> None:
    build_index(RAG_EVENT_CORPUS)

    g = graph.build_graph()
    state = {
        "text": "A bomb exploded in a market",
        "image_desc": "people running, smoke",
        "perception_summary": "",
        "memory": [],
        "similar_events": [],
        "evidence": "",
        "fusion_context": "",
        "event_json": "",
        "verified": False,
        "verdict": "",
        "issues": [],
        "verifier_feedback": "",
        "repair_attempts": 0,
    }
    final = g.invoke(state)
    print(final["event_json"])


if __name__ == "__main__":
    main()

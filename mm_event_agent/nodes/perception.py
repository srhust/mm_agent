"""输入理解与预处理：合并文本与图像描述到 perception_summary。"""

from __future__ import annotations

from typing import Any, Mapping


def perception(state: Mapping[str, Any]) -> dict[str, Any]:
    """不修改 text / image_desc，仅写入 perception_summary。"""
    text = "" if state.get("text") is None else str(state.get("text"))
    image_desc = "" if state.get("image_desc") is None else str(state.get("image_desc"))
    perception_summary = f"Text: {text}\nImage: {image_desc}"
    return {"perception_summary": perception_summary}


run = perception

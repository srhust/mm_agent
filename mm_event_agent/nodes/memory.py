"""运行期记忆：用 Python 列表追加当前 text / image_desc。"""

from __future__ import annotations

from typing import Any, Mapping


def memory(state: Mapping[str, Any]) -> dict[str, Any]:
    """将本轮 text 与 image_desc 作为一条记录追加到 memory。"""
    text = "" if state.get("text") is None else str(state.get("text"))
    image_desc = "" if state.get("image_desc") is None else str(state.get("image_desc"))
    return {"memory": [{"text": text, "image_desc": image_desc}]}


run = memory

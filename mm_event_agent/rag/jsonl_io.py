"""Small reusable JSONL helpers for offline corpus builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            data = json.loads(payload)
            if isinstance(data, dict):
                records.append(data)
    return records


def write_jsonl(path: str | Path, records: Sequence[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

#!/usr/bin/env python3
"""Normalize ACE2005 samples into `data/rag/normalized/ace2005.text.jsonl`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mm_event_agent.rag.jsonl_io import load_jsonl, write_jsonl
from mm_event_agent.rag.normalizers import clean_text
from mm_event_agent.rag.normalizers import Ace2005Normalizer
from mm_event_agent.rag.ontology_mapper import OntologyMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw ACE2005 JSON or JSONL path.")
    parser.add_argument(
        "--output",
        default="data/rag/normalized/ace2005.text.jsonl",
        help="Normalized output JSONL path.",
    )
    return parser.parse_args()


def _load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return load_jsonl(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        candidates = data.get("records") or data.get("data") or data.get("examples") or []
        return [item for item in candidates if isinstance(item, dict)]
    return []


def flatten_ace_record(
    raw_record: dict[str, Any],
    *,
    split_name: str,
    line_number: int,
) -> list[dict[str, Any]]:
    raw_text = clean_text(raw_record.get("text") or raw_record.get("sentence"))
    base_doc_id = clean_text(
        raw_record.get("doc_id") or raw_record.get("document_id") or raw_record.get("docid")
    ) or f"{clean_text(split_name) or 'ace'}::{line_number:06d}"

    flattened: list[dict[str, Any]] = []
    for event_index, event in enumerate(raw_record.get("event", [])):
        if not isinstance(event, dict):
            continue
        trigger_text = clean_text(event.get("text"))
        arguments: list[dict[str, Any]] = []
        for argument in event.get("args", []):
            if not isinstance(argument, dict):
                continue
            argument_text = clean_text(argument.get("text"))
            if not argument_text:
                continue
            arguments.append(
                {
                    "role": clean_text(argument.get("type")),
                    "text": argument_text,
                    "span": None,
                }
            )

        event_id = f"{base_doc_id}::event::{event_index}"
        flattened.append(
            {
                "id": event_id,
                "event_id": event_id,
                "doc_id": base_doc_id,
                "event_type": clean_text(event.get("type")),
                "raw_text": raw_text,
                "trigger": {"text": trigger_text, "span": None},
                "arguments": arguments,
            }
        )
    return flattened


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = _load_records(input_path)
    normalizer = Ace2005Normalizer(OntologyMapper())

    flattened_records: list[dict[str, Any]] = []
    split_name = input_path.stem
    for line_number, record in enumerate(records):
        flattened_records.extend(flatten_ace_record(record, split_name=split_name, line_number=line_number))

    normalized: list[dict] = []
    for record in flattened_records:
        output = normalizer.normalize(record)
        if output is not None:
            normalized.append(output)

    write_jsonl(args.output, normalized)
    total = len(flattened_records)
    kept = len(normalized)
    skipped = total - kept
    print(f"raw_total={len(records)}")
    print(f"total={total}")
    print(f"kept={kept}")
    print(f"skipped={skipped}")
    print(f"skipped_by_reason={dict(normalizer.skipped_by_reason)}")


if __name__ == "__main__":
    main()

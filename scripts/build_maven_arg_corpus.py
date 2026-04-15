#!/usr/bin/env python3
"""Normalize MAVEN-ARG samples into `data/rag/normalized/maven_arg.text.jsonl`."""

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
from mm_event_agent.rag.normalizers import MavenArgNormalizer
from mm_event_agent.rag.ontology_mapper import OntologyMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw MAVEN-ARG JSON or JSONL path.")
    parser.add_argument(
        "--output",
        default="data/rag/normalized/maven_arg.text.jsonl",
        help="Normalized output JSONL path.",
    )
    return parser.parse_args()


def _load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        candidates = data.get("records") or data.get("data") or data.get("examples") or []
        return [item for item in candidates if isinstance(item, dict)]
    return []


def _build_entity_lookup(raw_document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for entity in raw_document.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = clean_text(entity.get("id"))
        if not entity_id:
            continue
        mentions = entity.get("mention")
        if not isinstance(mentions, list) or not mentions:
            continue
        mention = mentions[0]
        if not isinstance(mention, dict):
            continue
        text = clean_text(mention.get("mention") or mention.get("text") or mention.get("content"))
        offset = mention.get("offset")
        span = (
            {"start": offset[0], "end": offset[1]}
            if isinstance(offset, list) and len(offset) >= 2 and all(isinstance(item, int) for item in offset[:2])
            else None
        )
        if text:
            lookup[entity_id] = {"text": text, "span": span}
    return lookup


def _normalize_argument_role(role: Any) -> str:
    value = clean_text(role)
    if not value:
        return ""
    return value.split("_", 1)[0]


def _resolve_argument(argument: dict[str, Any], entity_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    text = clean_text(argument.get("content") or argument.get("text") or argument.get("mention"))
    offset = argument.get("offset")
    span = (
        {"start": offset[0], "end": offset[1]}
        if isinstance(offset, list) and len(offset) >= 2 and all(isinstance(item, int) for item in offset[:2])
        else None
    )
    if text:
        return {"text": text, "span": span}

    entity_id = clean_text(argument.get("entity_id"))
    if entity_id:
        return entity_lookup.get(entity_id)
    return None


def flatten_maven_document(raw_document: dict[str, Any]) -> list[dict[str, Any]]:
    entity_lookup = _build_entity_lookup(raw_document)
    doc_id = clean_text(raw_document.get("id"))
    raw_text = clean_text(raw_document.get("document"))
    flattened: list[dict[str, Any]] = []

    for event_index, event in enumerate(raw_document.get("events", [])):
        if not isinstance(event, dict):
            continue
        mentions = event.get("mention")
        if not isinstance(mentions, list) or not mentions:
            mentions = [{}]
        first_mention = mentions[0] if isinstance(mentions[0], dict) else {}
        trigger_text = clean_text(first_mention.get("trigger_word") or first_mention.get("text"))
        offset = first_mention.get("offset")
        trigger_span = (
            {"start": offset[0], "end": offset[1]}
            if isinstance(offset, list) and len(offset) >= 2 and all(isinstance(item, int) for item in offset[:2])
            else None
        )

        arguments: list[dict[str, Any]] = []
        raw_arguments = event.get("argument")
        if isinstance(raw_arguments, dict):
            for role, values in raw_arguments.items():
                normalized_role = _normalize_argument_role(role)
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict):
                        continue
                    resolved = _resolve_argument(value, entity_lookup)
                    if resolved is None:
                        continue
                    arguments.append(
                        {
                            "role": normalized_role or clean_text(role),
                            "text": resolved["text"],
                            "span": resolved.get("span"),
                        }
                    )

        mention_id = clean_text(first_mention.get("id"))
        event_id = clean_text(event.get("id")) or f"{doc_id}::event::{event_index}"
        flattened.append(
            {
                "id": mention_id or event_id,
                "event_id": event_id,
                "doc_id": doc_id,
                "event_type": clean_text(event.get("type")),
                "raw_text": raw_text,
                "trigger": {"text": trigger_text, "span": trigger_span},
                "arguments": arguments,
            }
        )

    return flattened


def main() -> None:
    args = parse_args()
    records = _load_records(Path(args.input))
    normalizer = MavenArgNormalizer(OntologyMapper())

    flattened_records: list[dict[str, Any]] = []
    for record in records:
        flattened_records.extend(flatten_maven_document(record))

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

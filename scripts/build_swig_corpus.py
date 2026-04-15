#!/usr/bin/env python3
"""Normalize SWiG samples into text and image JSONL corpora."""

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
from mm_event_agent.rag.normalizers import SwigNormalizer, build_retrieval_text, clean_text
from mm_event_agent.rag.ontology_mapper import OntologyMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw SWiG JSON or JSONL path.")
    parser.add_argument("--images-root", default="", help="Optional root used to resolve relative image paths.")
    parser.add_argument(
        "--text-output",
        default="data/rag/normalized/swig.text.jsonl",
        help="Normalized text-side SWiG JSONL path.",
    )
    parser.add_argument(
        "--image-output",
        default="data/rag/normalized/swig.image.jsonl",
        help="Normalized image-manifest SWiG JSONL path.",
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
        if candidates:
            return [item for item in candidates if isinstance(item, dict)]
        return [
            {"image_id": clean_text(image_id), **record}
            for image_id, record in data.items()
            if isinstance(record, dict)
        ]
    return []


def _select_first_frame(raw_record: dict[str, Any]) -> dict[str, Any]:
    frames = raw_record.get("frames")
    if not isinstance(frames, list):
        return {}
    for frame in frames:
        if isinstance(frame, dict):
            return frame
    return {}


def flatten_swig_record(
    image_id: str,
    raw_record: dict[str, Any],
    *,
    mapper: OntologyMapper | None = None,
) -> dict[str, Any]:
    verb = clean_text(raw_record.get("verb"))
    selected_frame = _select_first_frame(raw_record)
    arguments = [
        {"role": clean_text(role), "label": clean_text(label)}
        for role, label in selected_frame.items()
        if clean_text(role) and clean_text(label)
    ]
    canonical_event_type = mapper.map_event_type("swig", verb) if mapper is not None else None
    swig_frame = ", ".join(
        f"{role}={label}"
        for role, label in sorted(
            ((clean_text(role), clean_text(label)) for role, label in selected_frame.items()),
            key=lambda item: item[0],
        )
        if role and label
    )
    fallback_desc = build_retrieval_text(
        canonical_event_type or "",
        verb,
        [f"{item['role']} {item['label']}" for item in arguments],
    )
    return {
        "image_id": clean_text(image_id),
        "verb": verb,
        "swig_frame": swig_frame,
        "image_desc": fallback_desc,
        "visual_situation": fallback_desc or verb,
        "arguments": arguments,
        "path": clean_text(image_id),
        "selected_frame": selected_frame,
    }


def flatten_swig_records(records: list[dict[str, Any]], *, mapper: OntologyMapper) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        image_id = clean_text(record.get("image_id"))
        if not image_id:
            continue
        flattened.append(flatten_swig_record(image_id, record, mapper=mapper))
    return flattened


def main() -> None:
    args = parse_args()
    records = _load_records(Path(args.input))
    mapper = OntologyMapper()
    normalizer = SwigNormalizer(mapper)
    flattened_records = flatten_swig_records(records, mapper=mapper)

    text_records: list[dict] = []
    image_records: list[dict] = []
    for record in flattened_records:
        output = normalizer.normalize(record, images_root=args.images_root or None)
        if output is None:
            continue
        text_records.append(output.text_example)
        image_records.append(output.image_manifest)

    write_jsonl(args.text_output, text_records)
    write_jsonl(args.image_output, image_records)
    total = len(flattened_records)
    kept_text = len(text_records)
    kept_image = len(image_records)
    skipped = total - kept_text
    print(f"raw_total={len(records)}")
    print(f"total={total}")
    print(f"kept_text={kept_text}")
    print(f"kept_image={kept_image}")
    print(f"skipped={skipped}")
    print(f"skipped_by_reason={dict(normalizer.skipped_by_reason)}")


if __name__ == "__main__":
    main()

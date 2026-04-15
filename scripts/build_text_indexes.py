#!/usr/bin/env python3
"""Build persistent FAISS text indexes from normalized JSONL corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more normalized JSONL files.")
    parser.add_argument("--index-names", nargs="*", default=[], help="Optional index names matching the input order.")
    parser.add_argument("--index-root", default="data/rag/indexes", help="Root directory for persistent indexes.")
    parser.add_argument("--model-name-or-path", required=True, help="SentenceTransformer model name or local path.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization.")
    return parser.parse_args()


def infer_index_name(path: Path) -> str:
    stem = path.stem.lower()
    mapping = {
        "ace2005.text": "ace_text",
        "maven_arg.text": "maven_text",
        "swig.text": "swig_text",
        "bridge": "bridge",
    }
    return mapping.get(stem, stem.replace(".", "_"))


def main() -> None:
    args = parse_args()
    from mm_event_agent.rag.jsonl_io import load_jsonl
    from mm_event_agent.rag.persistent_faiss import IndexArtifactPaths, PersistentFaissIndex
    from mm_event_agent.rag.text_encoder import SentenceTransformerTextEncoder

    input_paths = [Path(value) for value in args.inputs]
    if args.index_names and len(args.index_names) != len(input_paths):
        raise ValueError("--index-names must match the number of --inputs")

    encoder = SentenceTransformerTextEncoder(
        args.model_name_or_path,
        normalize=not args.no_normalize,
    )
    index_root = Path(args.index_root)

    for idx, input_path in enumerate(input_paths):
        records = load_jsonl(input_path)
        usable_records: list[dict] = []
        texts: list[str] = []
        skipped = 0
        for record in records:
            retrieval_text = str(record.get("retrieval_text") or "").strip()
            if not retrieval_text:
                skipped += 1
                continue
            texts.append(retrieval_text)
            usable_records.append(dict(record))

        if not usable_records:
            raise ValueError(f"no indexable records with retrieval_text found in {input_path}")

        vectors = encoder.encode(texts)
        index_name = args.index_names[idx] if idx < len(args.index_names) and args.index_names else infer_index_name(input_path)
        output_dir = index_root / index_name
        metadata = []
        for record in usable_records:
            meta = dict(record)
            meta.setdefault("id", "")
            meta.setdefault("source_dataset", "")
            meta.setdefault("event_type", "")
            meta.setdefault("retrieval_text", "")
            metadata.append(meta)

        index = PersistentFaissIndex(
            IndexArtifactPaths.from_root(output_dir),
            index_name=index_name,
        )
        index.build_from_embeddings(
            vectors,
            metadata,
            encoder_name_or_path=args.model_name_or_path,
            normalized=not args.no_normalize,
        )
        index.save()

        print(f"index_name={index_name}")
        print(f"total_input_records={len(records)}")
        print(f"indexed_records={len(usable_records)}")
        print(f"skipped_records={skipped}")
        print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

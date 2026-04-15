#!/usr/bin/env python3
"""Merge SWiG image embedding shards into one persistent FAISS index."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mm_event_agent.runtime_config import settings
from mm_event_agent.rag.jsonl_io import load_jsonl
from mm_event_agent.rag.persistent_faiss import IndexArtifactPaths, PersistentFaissIndex


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more .emb.npy SWiG image embedding shards.")
    parser.add_argument("--index-root", default="data/rag/indexes", help="Persistent index root directory.")
    parser.add_argument("--index-name", default="swig_image", help="Persistent index directory name.")
    parser.add_argument("--model-path", default=settings.rag_qwen_embedding_model_path, help="Local Qwen3-VL-Embedding model path.")
    parser.add_argument(
        "--instruction",
        default=settings.rag_qwen_image_instruction,
        help="Instruction prompt used during image embedding.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Image encoding batch size recorded in build_info.")
    parser.add_argument("--no-normalize", action="store_true", help="Mark embeddings as non-normalized in build_info.")
    return parser.parse_args(argv)


def _meta_path_for_embedding_path(emb_path: Path) -> Path:
    if not emb_path.name.endswith(".emb.npy"):
        raise ValueError(f"expected shard path ending with .emb.npy, got {emb_path}")
    return emb_path.with_name(emb_path.name.replace(".emb.npy", ".meta.jsonl"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.model_path or "").strip():
        raise ValueError("--model-path is required to record SWiG image index build_info")

    output_dir = Path(args.index_root) / str(args.index_name)
    index = PersistentFaissIndex(IndexArtifactPaths.from_root(output_dir), index_name=str(args.index_name))

    initialized = False
    total_records = 0
    for input_value in args.inputs:
        emb_path = Path(input_value)
        meta_path = _meta_path_for_embedding_path(emb_path)
        if not emb_path.exists():
            raise FileNotFoundError(f"embedding shard does not exist: {emb_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"matching shard metadata does not exist: {meta_path}")

        vectors = np.load(emb_path)
        metadata = load_jsonl(meta_path)
        if vectors.ndim != 2:
            raise ValueError(f"embedding shard must be 2D, got shape {vectors.shape}")
        if len(metadata) != vectors.shape[0]:
            raise ValueError(f"shard metadata row count does not match embeddings for {emb_path}")
        if vectors.shape[0] == 0:
            continue

        if not initialized:
            index.initialize_empty(
                vector_dim=int(vectors.shape[1]),
                encoder_name_or_path=args.model_path,
                normalized=not args.no_normalize,
                build_info={
                    "index_name": str(args.index_name),
                    "encoder_type": "qwen3_vl_embedding",
                    "encoder_name_or_path": args.model_path,
                    "instruction": args.instruction,
                    "normalized": not args.no_normalize,
                    "index_type": "IndexFlatIP",
                    "built_at": datetime.now(timezone.utc).isoformat(),
                    "batch_size": max(1, int(args.batch_size)),
                },
            )
            initialized = True

        index.add_embeddings(vectors.astype(np.float32, copy=False), metadata)
        total_records += len(metadata)

    if not initialized or total_records == 0:
        raise ValueError("no non-empty SWiG image embedding shards were provided")

    index.save()
    print(f"index_name={args.index_name}")
    print(f"record_count={total_records}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

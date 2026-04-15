#!/usr/bin/env python3
"""Build SWiG image embedding shards from normalized image manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mm_event_agent.runtime_config import settings
from mm_event_agent.rag.image_encoder import Qwen3VLImageEncoder, verify_image_path
from mm_event_agent.rag.jsonl_io import load_jsonl, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normalized SWiG image manifest JSONL.")
    parser.add_argument("--out-dir", required=True, help="Output directory for embedding shard files.")
    parser.add_argument("--model-path", default=settings.rag_qwen_embedding_model_path, help="Local Qwen3-VL-Embedding model path.")
    parser.add_argument("--device", default=settings.rag_qwen_embedding_device, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--dtype", default=settings.rag_qwen_embedding_dtype, help="Torch dtype, e.g. bfloat16.")
    parser.add_argument("--attn-impl", default=settings.rag_qwen_embedding_attn_impl, help="Attention implementation.")
    parser.add_argument(
        "--instruction",
        default=settings.rag_qwen_image_instruction,
        help="Instruction prompt used for image embedding.",
    )
    parser.add_argument("--out-dim", type=int, default=settings.rag_qwen_embedding_out_dim, help="Optional output dimension cap.")
    parser.add_argument("--batch-size", type=int, default=8, help="Image encoding batch size.")
    return parser.parse_args(argv)


def _validate_manifest_records(records: list[dict]) -> tuple[list[dict], list[tuple[str, str, str]]]:
    valid_records: list[dict] = []
    bad_rows: list[tuple[str, str, str]] = []
    for record in records:
        image_id = str(record.get("image_id") or record.get("id") or "").strip()
        image_path = str(record.get("path") or "").strip()
        if not image_path:
            bad_rows.append((image_id, image_path, "NO_PATH"))
            continue
        try:
            verify_image_path(image_path)
        except FileNotFoundError:
            bad_rows.append((image_id, image_path, "NOT_FOUND"))
            continue
        except Exception as exc:
            bad_rows.append((image_id, image_path, f"BAD_IMAGE:{type(exc).__name__}"))
            continue
        valid_records.append(dict(record))
    return valid_records, bad_rows


def _write_bad_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["image_id", "path", "error"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.model_path or "").strip():
        raise ValueError("--model-path is required for offline SWiG image embedding")

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    emb_path = out_dir / f"{stem}.emb.npy"
    meta_path = out_dir / f"{stem}.meta.jsonl"
    bad_path = out_dir / f"{stem}.bad.tsv"

    records = load_jsonl(input_path)
    valid_records, bad_rows = _validate_manifest_records(records)
    _write_bad_tsv(bad_path, bad_rows)
    if not valid_records:
        np.save(emb_path, np.zeros((0, max(0, int(args.out_dim))), dtype=np.float32))
        write_jsonl(meta_path, [])
        print(f"valid_records=0")
        print(f"bad_records={len(bad_rows)}")
        print(f"emb_path={emb_path}")
        print(f"meta_path={meta_path}")
        print(f"bad_path={bad_path}")
        return 0

    encoder = Qwen3VLImageEncoder(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        instruction=args.instruction,
        out_dim=max(0, int(args.out_dim)),
        normalize=settings.rag_qwen_embedding_normalize,
    )

    effective_batch_size = max(1, int(args.batch_size))
    first_batch = valid_records[:effective_batch_size]
    first_vectors = encoder.encode_image_paths([str(record["path"]) for record in first_batch], batch_size=effective_batch_size)
    if first_vectors.ndim != 2 or first_vectors.shape[0] != len(first_batch):
        raise ValueError("encoder returned unexpected image embedding shape for first batch")

    mmap = open_memmap(
        emb_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(valid_records), int(first_vectors.shape[1])),
    )
    mmap[: len(first_batch)] = first_vectors
    for start in range(len(first_batch), len(valid_records), effective_batch_size):
        batch = valid_records[start : start + effective_batch_size]
        batch_vectors = encoder.encode_image_paths([str(record["path"]) for record in batch], batch_size=effective_batch_size)
        mmap[start : start + len(batch)] = batch_vectors
    del mmap

    shard_meta: list[dict] = []
    for row, record in enumerate(valid_records):
        meta = dict(record)
        meta.setdefault("id", "")
        meta.setdefault("image_id", "")
        meta.setdefault("source_dataset", "SWiG")
        meta.setdefault("event_type", "")
        meta.setdefault("path", "")
        meta.setdefault("retrieval_text", "")
        meta["row"] = row
        shard_meta.append(meta)
    write_jsonl(meta_path, shard_meta)

    print(f"valid_records={len(valid_records)}")
    print(f"bad_records={len(bad_rows)}")
    print(f"emb_path={emb_path}")
    print(f"meta_path={meta_path}")
    print(f"bad_path={bad_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build offline Qwen3-VL image embedding shards from `swig.image.jsonl`.

Output contract per shard:

- `{stem}.emb.npy`: float16 array with shape `[rows, dim]`
- `{stem}.meta.jsonl`: one JSON object per embedding row, in the same row order
- `{stem}.bad.tsv`: tab-separated log with columns `line_number`, `record_id`, `path`, `error`

`meta.jsonl` is intentionally stable for future persistent FAISS indexing:
each line keeps the row-to-record mapping plus original record payload so a
later index builder can reconstruct ids/filters without re-reading the source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INSTRUCTION = "Retrieve images relevant to the user's query."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Path to swig.image.jsonl.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for shard outputs.")
    parser.add_argument("--model-path", type=str, required=True, help="Qwen3-VL embedding checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, for example `cuda:0` or `cpu`.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Model dtype.")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"], help="Attention backend to request from Transformers.")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="Instruction string passed to Qwen3-VL.")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per forward pass.")
    parser.add_argument("--shard-size", type=int, default=10000, help="Maximum input JSONL rows per output shard.")
    parser.add_argument("--out-dim", type=int, default=4096, help="If smaller than native dim, keep the leading slice only.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize pooled embeddings before saving.")
    parser.add_argument("--path-field", type=str, default="path", help="JSON field containing the image path.")
    parser.add_argument("--id-field", type=str, default="img_id", help="Primary record id field for meta/log output.")
    parser.add_argument("--max-items", type=int, default=-1, help="Maximum number of JSONL rows to read. -1 means all.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip output shards that already have emb+meta+bad files.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise immediately on model inference failure.")
    return parser.parse_args()


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def _make_record_id(record: dict[str, Any], id_field: str, line_number: int) -> str:
    value = record.get(id_field)
    if value is None or str(value).strip() == "":
        return f"line_{line_number:08d}"
    return str(value)


def _resolve_image_path(raw_path: Any, input_path: Path) -> str:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return ""
    candidate = Path(path_text)
    if candidate.is_absolute():
        return str(candidate)
    return str((input_path.parent / candidate).resolve())


def _shard_paths(out_dir: Path, input_stem: str, shard_index: int) -> tuple[Path, Path, Path]:
    shard_name = f"{input_stem}.shard_{shard_index:05d}"
    return (
        out_dir / f"{shard_name}.emb.npy",
        out_dir / f"{shard_name}.meta.jsonl",
        out_dir / f"{shard_name}.bad.tsv",
    )


def _write_shard(
    *,
    out_dir: Path,
    input_stem: str,
    shard_index: int,
    embeddings: list[np.ndarray],
    metadata: list[dict[str, Any]],
    bad_rows: list[dict[str, str]],
    skip_existing: bool,
) -> bool:
    emb_path, meta_path, bad_path = _shard_paths(out_dir, input_stem, shard_index)
    if skip_existing and emb_path.exists() and meta_path.exists() and bad_path.exists():
        return False

    if embeddings:
        merged = np.concatenate(embeddings, axis=0).astype(np.float16, copy=False)
    else:
        dim = metadata[0]["embedding_dim"] if metadata else 0
        merged = np.zeros((0, dim), dtype=np.float16)
    np.save(emb_path, merged)

    with meta_path.open("w", encoding="utf-8") as meta_file:
        for item in metadata:
            meta_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with bad_path.open("w", encoding="utf-8") as bad_file:
        bad_file.write("line_number\trecord_id\tpath\terror\n")
        for row in bad_rows:
            bad_file.write(
                f"{row['line_number']}\t{row['record_id']}\t{row['path']}\t{row['error']}\n"
            )
    return True


def main() -> None:
    args = parse_args()

    from mm_event_agent.rag import Qwen3VLImageEncoder
    from mm_event_agent.rag.qwen3vl_image_encoder import verify_image

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    encoder = Qwen3VLImageEncoder(
        model_name_or_path=args.model_path,
        device=args.device,
        torch_dtype=_dtype_from_name(args.dtype),
        attn_implementation=args.attn_impl,
    )

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if args.max_items != -1 and line_number > args.max_items:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rows.append(
                    {
                        "kind": "bad",
                        "line_number": line_number,
                        "record_id": f"line_{line_number:08d}",
                        "path": "",
                        "error": "BAD_JSON",
                    }
                )
                continue

            if not isinstance(record, dict):
                rows.append(
                    {
                        "kind": "bad",
                        "line_number": line_number,
                        "record_id": f"line_{line_number:08d}",
                        "path": "",
                        "error": "NON_OBJECT_JSON",
                    }
                )
                continue

            record_id = _make_record_id(record, args.id_field, line_number)
            resolved_path = _resolve_image_path(record.get(args.path_field), input_path)
            if not resolved_path:
                rows.append(
                    {
                        "kind": "bad",
                        "line_number": line_number,
                        "record_id": record_id,
                        "path": "",
                        "error": "NO_PATH",
                    }
                )
                continue
            if not os.path.exists(resolved_path):
                rows.append(
                    {
                        "kind": "bad",
                        "line_number": line_number,
                        "record_id": record_id,
                        "path": resolved_path,
                        "error": "NOT_FOUND",
                    }
                )
                continue

            image_error = verify_image(resolved_path)
            if image_error is not None:
                rows.append(
                    {
                        "kind": "bad",
                        "line_number": line_number,
                        "record_id": record_id,
                        "path": resolved_path,
                        "error": f"BAD_IMAGE:{image_error}",
                    }
                )
                continue

            rows.append(
                {
                    "kind": "good",
                    "line_number": line_number,
                    "record_id": record_id,
                    "path": resolved_path,
                    "record": record,
                }
            )

    if not rows:
        rows = [
            {
                "kind": "bad",
                "line_number": 0,
                "record_id": "empty_input",
                "path": "",
                "error": "EMPTY_INPUT",
            }
        ]

    shard_size = len(rows) if args.shard_size <= 0 else args.shard_size

    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        shard_good_rows = [row for row in shard_rows if row["kind"] == "good"]
        shard_bad_output = [
            {
                "line_number": str(row["line_number"]),
                "record_id": row["record_id"],
                "path": row["path"],
                "error": row["error"],
            }
            for row in shard_rows
            if row["kind"] == "bad"
        ]

        shard_embeddings: list[np.ndarray] = []
        shard_metadata: list[dict[str, Any]] = []
        current_row_offset = 0

        for batch_start in range(0, len(shard_good_rows), args.batch_size):
            batch_rows = shard_good_rows[batch_start : batch_start + args.batch_size]
            batch_paths = [row["path"] for row in batch_rows]
            try:
                batch_embeddings = encoder.encode_image_paths(
                    batch_paths,
                    instruction=args.instruction,
                    normalize=args.normalize,
                    out_dim=args.out_dim,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                for row in batch_rows:
                    shard_bad_output.append(
                        {
                            "line_number": str(row["line_number"]),
                            "record_id": row["record_id"],
                            "path": row["path"],
                            "error": f"MODEL_ERROR:{repr(exc)}",
                        }
                    )
                continue

            shard_embeddings.append(batch_embeddings)
            for offset, row in enumerate(batch_rows):
                shard_metadata.append(
                    {
                        "row": current_row_offset + offset,
                        "record_id": row["record_id"],
                        "path": row["path"],
                        "line_number": row["line_number"],
                        "source_dataset": str(row["record"].get("source_dataset") or "SWiG"),
                        "modality": "image_semantics",
                        "embedding_model": str(args.model_path),
                        "instruction": args.instruction,
                        "normalized": bool(args.normalize),
                        "embedding_dim": int(batch_embeddings.shape[1]),
                        "record": row["record"],
                    }
                )
            current_row_offset += int(batch_embeddings.shape[0])

        wrote = _write_shard(
            out_dir=out_dir,
            input_stem=input_path.stem,
            shard_index=shard_index,
            embeddings=shard_embeddings,
            metadata=shard_metadata,
            bad_rows=shard_bad_output,
            skip_existing=args.skip_existing,
        )
        emb_path, meta_path, bad_path = _shard_paths(out_dir, input_path.stem, shard_index)
        status = "SKIP" if not wrote else "DONE"
        print(f"[{status}] emb={emb_path}")
        print(f"[{status}] meta={meta_path} rows={len(shard_metadata)}")
        print(f"[{status}] bad={bad_path} rows={len(shard_bad_output)}")


if __name__ == "__main__":
    main()

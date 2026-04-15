#!/usr/bin/env python3
"""Build persistent FAISS text indexes from normalized JSONL corpora."""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mm_event_agent.runtime_config import settings
from mm_event_agent.rag.text_encoder import Qwen3VLTextEncoder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more normalized JSONL files.")
    parser.add_argument("--index-names", nargs="*", default=[], help="Optional index names matching the input order.")
    parser.add_argument("--index-root", default="data/rag/indexes", help="Root directory for persistent indexes.")
    parser.add_argument(
        "--model-path",
        default=settings.rag_qwen_embedding_model_path,
        help="Local filesystem path to the Qwen3-VL-Embedding checkpoint.",
    )
    parser.add_argument("--device", default=settings.rag_qwen_embedding_device, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--dtype", default=settings.rag_qwen_embedding_dtype, help="Torch dtype, e.g. bfloat16.")
    parser.add_argument(
        "--attn-impl",
        default=settings.rag_qwen_embedding_attn_impl,
        help="Attention implementation to request from transformers.",
    )
    parser.add_argument(
        "--instruction",
        default=settings.rag_qwen_text_instruction,
        help="Instruction prefix used when encoding retrieval_text.",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=settings.rag_qwen_embedding_out_dim,
        help="Optional output dimension cap. Zero keeps the native embedding width.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization.")
    return parser.parse_args(argv)


def infer_index_name(path: Path) -> str:
    stem = path.stem.lower()
    mapping = {
        "ace2005.text": "ace_text",
        "maven_arg.text": "maven_text",
        "swig.text": "swig_text",
        "bridge": "bridge",
    }
    return mapping.get(stem, stem.replace(".", "_"))


def _prepare_records(records: list[dict]) -> tuple[list[dict], list[str], Counter[str]]:
    usable_records: list[dict] = []
    texts: list[str] = []
    skipped_by_reason: Counter[str] = Counter()
    for record in records:
        retrieval_text = str(record.get("retrieval_text") or "").strip()
        if not retrieval_text:
            skipped_by_reason["missing_retrieval_text"] += 1
            continue
        texts.append(retrieval_text)
        usable_records.append(dict(record))
    return usable_records, texts, skipped_by_reason


def _build_single_index(
    *,
    input_path: Path,
    index_name: str,
    index_root: Path,
    encoder,
    model_path: str,
    instruction: str,
    normalize: bool,
) -> dict[str, object]:
    from mm_event_agent.rag.jsonl_io import load_jsonl
    from mm_event_agent.rag.persistent_faiss import IndexArtifactPaths, PersistentFaissIndex
    records = load_jsonl(input_path)
    usable_records, texts, skipped_by_reason = _prepare_records(records)
    if not usable_records:
        raise ValueError(f"no indexable records with retrieval_text found in {input_path}")

    vectors = encoder.encode(texts)
    output_dir = index_root / index_name
    metadata: list[dict] = []
    for row, record in enumerate(usable_records):
        meta = dict(record)
        meta.setdefault("id", "")
        meta.setdefault("source_dataset", "")
        meta.setdefault("event_type", "")
        meta.setdefault("retrieval_text", "")
        meta.setdefault("row", row)
        metadata.append(meta)

    index = PersistentFaissIndex(
        IndexArtifactPaths.from_root(output_dir),
        index_name=index_name,
    )
    index.build_from_embeddings(
        vectors,
        metadata,
        encoder_name_or_path=model_path,
        normalized=normalize,
        build_info={
            "index_name": index_name,
            "encoder_type": "qwen3_vl_embedding",
            "encoder_name_or_path": model_path,
            "instruction": instruction,
            "normalized": normalize,
            "vector_dim": int(vectors.shape[1]),
            "index_type": "IndexFlatIP",
            "record_count": len(metadata),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(input_path),
            "skip_reasons": dict(skipped_by_reason),
        },
    )
    index.save()
    return {
        "index_name": index_name,
        "total_input_records": len(records),
        "indexed_records": len(usable_records),
        "skipped_records": sum(skipped_by_reason.values()),
        "skip_reasons": dict(skipped_by_reason),
        "output_dir": output_dir,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_paths = [Path(value) for value in args.inputs]
    if args.index_names and len(args.index_names) != len(input_paths):
        raise ValueError("--index-names must match the number of --inputs")
    if not str(args.model_path or "").strip():
        raise ValueError("--model-path is required for offline Qwen3-VL text indexing")

    encoder = Qwen3VLTextEncoder(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        instruction=args.instruction,
        out_dim=max(0, int(args.out_dim)),
        normalize=not args.no_normalize,
    )
    index_root = Path(args.index_root)

    for idx, input_path in enumerate(input_paths):
        index_name = args.index_names[idx] if idx < len(args.index_names) and args.index_names else infer_index_name(input_path)
        summary = _build_single_index(
            input_path=input_path,
            index_name=index_name,
            index_root=index_root,
            encoder=encoder,
            model_path=args.model_path,
            instruction=args.instruction,
            normalize=not args.no_normalize,
        )
        print(f"index_name={summary['index_name']}")
        print(f"total_input_records={summary['total_input_records']}")
        print(f"indexed_records={summary['indexed_records']}")
        print(f"skipped_records={summary['skipped_records']}")
        print(f"skip_reasons={summary['skip_reasons']}")
        print(f"output_dir={summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

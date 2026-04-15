#!/usr/bin/env python3
"""Query one persistent text index and print readable retrieval hits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mm_event_agent.runtime_config import settings
from mm_event_agent.rag.persistent_faiss import PersistentFaissIndex
from mm_event_agent.rag.text_encoder import Qwen3VLTextEncoder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", default=settings.rag_index_root, help="Root directory containing persistent indexes.")
    parser.add_argument("--model-path", default=settings.rag_qwen_embedding_model_path, help="Local Qwen3-VL-Embedding model path.")
    parser.add_argument("--device", default=settings.rag_qwen_embedding_device, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--dtype", default=settings.rag_qwen_embedding_dtype, help="Torch dtype, e.g. bfloat16.")
    parser.add_argument("--attn-impl", default=settings.rag_qwen_embedding_attn_impl, help="Attention implementation.")
    parser.add_argument(
        "--instruction",
        default=settings.rag_qwen_text_instruction,
        help="Instruction prompt used for text embedding.",
    )
    parser.add_argument("--top-k", type=int, default=settings.rag_text_top_k, help="Number of hits to return.")
    parser.add_argument("--index-name", required=True, choices=["ace_text", "maven_text", "swig_text"], help="Persistent text index to query.")
    parser.add_argument("--query", required=True, help="Query string.")
    parser.add_argument("--event-type", default="", help="Optional metadata-side event_type filter.")
    parser.add_argument("--json", action="store_true", help="Print raw hits as JSON.")
    parser.add_argument("--show-full-text", action="store_true", help="Print full retrieval_text instead of a preview.")
    return parser.parse_args(argv)


def _resolve_index_dir(index_root: str | Path, index_name: str) -> Path:
    index_dir = Path(index_root) / str(index_name)
    if not index_dir.exists():
        raise FileNotFoundError(f"persistent index directory does not exist: {index_dir}")
    return index_dir


def _trigger_text(record: dict[str, Any]) -> str:
    trigger = record.get("trigger")
    if isinstance(trigger, dict):
        return str(trigger.get("text") or "").strip()
    return str(trigger or "").strip()


def _preview_text(text: str, *, limit: int = 120) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _build_output_hits(results: list[dict[str, Any]], *, index_name: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in results:
        meta = dict(item.get("meta") or {})
        hit = dict(meta)
        hit["retrieval_metadata"] = {
            "score": float(item.get("score", 0.0)),
            "rank": int(item.get("rank", 0)),
            "index_name": index_name,
        }
        hits.append(hit)
    return hits


def _print_readable_hits(hits: list[dict[str, Any]], *, show_full_text: bool) -> None:
    if not hits:
        print("(no hits)")
        return
    for hit in hits:
        retrieval_meta = hit.get("retrieval_metadata", {}) if isinstance(hit, dict) else {}
        retrieval_text = str(hit.get("retrieval_text") or "")
        preview = retrieval_text if show_full_text else _preview_text(retrieval_text)
        trigger_text = _trigger_text(hit) if isinstance(hit, dict) else ""
        print(f"Rank: {retrieval_meta.get('rank', '')}")
        print(f"Score: {float(retrieval_meta.get('score', 0.0)):.4f}")
        print(f"ID: {hit.get('id', '')}")
        print(f"Source: {hit.get('source_dataset', '')}")
        print(f"Event Type: {hit.get('event_type', '')}")
        if trigger_text:
            print(f"Trigger: {trigger_text}")
        if str(hit.get('doc_id') or '').strip():
            print(f"Doc ID: {hit.get('doc_id')}")
        if str(hit.get('image_id') or '').strip():
            print(f"Image ID: {hit.get('image_id')}")
        print(f"Retrieval Text: {preview}")
        print("")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.model_path or "").strip():
        raise ValueError("--model-path is required for Qwen text query encoding")

    index_dir = _resolve_index_dir(args.index_root, args.index_name)
    index = PersistentFaissIndex.load(index_dir)
    encoder = Qwen3VLTextEncoder(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        instruction=args.instruction,
        out_dim=settings.rag_qwen_embedding_out_dim,
        normalize=settings.rag_qwen_embedding_normalize,
    )

    query_vector = encoder.encode([args.query])[0]
    filters = {"event_type": args.event_type} if str(args.event_type or "").strip() else None
    results = index.search(query_vector, top_k=max(1, int(args.top_k)), filters=filters)
    hits = _build_output_hits(results, index_name=args.index_name)

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0

    _print_readable_hits(hits, show_full_text=bool(args.show_full_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

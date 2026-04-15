"""Persistent FAISS index wrapper for offline text-side RAG."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mm_event_agent.rag.jsonl_io import load_jsonl, write_jsonl

try:  # pragma: no cover - dependency availability varies by environment
    import faiss
except ImportError:  # pragma: no cover - dependency availability varies by environment
    faiss = None


def _require_faiss():
    if faiss is None:
        raise ImportError("faiss is required for persistent index build/load/search operations")
    return faiss


@dataclass(frozen=True)
class IndexArtifactPaths:
    root_dir: Path
    index_path: Path
    meta_path: Path
    build_info_path: Path

    @classmethod
    def from_root(cls, root_dir: str | Path) -> "IndexArtifactPaths":
        root = Path(root_dir)
        return cls(
            root_dir=root,
            index_path=root / "index.faiss",
            meta_path=root / "meta.jsonl",
            build_info_path=root / "build_info.json",
        )


class PersistentFaissIndex:
    def __init__(
        self,
        artifact_paths: IndexArtifactPaths,
        *,
        index_name: str,
        index: faiss.Index | None = None,
        metadata: list[dict[str, Any]] | None = None,
        build_info: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_paths = artifact_paths
        self.index_name = str(index_name)
        self.index = index
        self.metadata = list(metadata or [])
        self.build_info = dict(build_info or {})

    def build_from_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Sequence[dict[str, Any]],
        *,
        encoder_name_or_path: str,
        normalized: bool,
        index_type: str = "IndexFlatIP",
        build_info: dict[str, Any] | None = None,
    ) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {vectors.shape}")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("metadata row count must match embedding rows")
        if vectors.shape[0] == 0:
            raise ValueError("cannot build persistent index from zero embeddings")
        if not np.isfinite(vectors).all():
            raise ValueError("embeddings must contain only finite float values")

        faiss_module = _require_faiss()
        index = faiss_module.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index = index
        self.metadata = [dict(item) for item in metadata]
        merged_build_info = dict(build_info or {})
        merged_build_info.update(
            {
            "index_name": self.index_name,
            "encoder_name": str(encoder_name_or_path),
            "encoder_name_or_path": str(encoder_name_or_path),
            "normalized": bool(normalized),
            "vector_dim": int(vectors.shape[1]),
            "index_type": str(index_type),
            "record_count": int(vectors.shape[0]),
            }
        )
        self.build_info = merged_build_info

    @classmethod
    def load(cls, root_dir: str | Path) -> "PersistentFaissIndex":
        paths = IndexArtifactPaths.from_root(root_dir)
        missing = [
            str(path)
            for path in [paths.index_path, paths.meta_path, paths.build_info_path]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing persistent FAISS artifacts: {missing}")

        faiss_module = _require_faiss()
        index = faiss_module.read_index(str(paths.index_path))
        metadata = load_jsonl(paths.meta_path)
        build_info = json.loads(paths.build_info_path.read_text(encoding="utf-8"))
        if not isinstance(build_info, dict):
            raise ValueError("build_info.json must contain a JSON object")
        if index.ntotal != len(metadata):
            raise ValueError("FAISS row count does not match meta.jsonl row count")

        index_name = str(build_info.get("index_name") or paths.root_dir.name)
        return cls(
            artifact_paths=paths,
            index_name=index_name,
            index=index,
            metadata=metadata,
            build_info=build_info,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            raise ValueError("index is not loaded")
        if top_k <= 0:
            return []

        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError(f"query_vector must have shape [dim] or [1, dim], got {vector.shape}")
        if self.index.d != vector.shape[1]:
            raise ValueError(f"query dimension {vector.shape[1]} does not match index dimension {self.index.d}")

        search_k = len(self.metadata) if filters is not None else min(top_k, len(self.metadata))
        scores, indices = self.index.search(vector, search_k)

        def _passes(meta: dict[str, Any]) -> bool:
            if filters is None:
                return True
            if callable(filters):
                return bool(filters(meta))
            for key, expected in filters.items():
                if meta.get(key) != expected:
                    return False
            return True

        results: list[dict[str, Any]] = []
        for rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
            if row_index < 0:
                continue
            meta = dict(self.metadata[int(row_index)])
            if not _passes(meta):
                continue
            results.append(
                {
                    "row": int(row_index),
                    "score": float(score),
                    "rank": len(results) + 1,
                    "meta": meta,
                }
            )
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        if self.index is None:
            raise ValueError("index is not loaded")
        if not self.metadata:
            raise ValueError("metadata is empty")
        if self.index.ntotal != len(self.metadata):
            raise ValueError("cannot save: FAISS row count does not match metadata length")

        self.artifact_paths.root_dir.mkdir(parents=True, exist_ok=True)
        faiss_module = _require_faiss()
        faiss_module.write_index(self.index, str(self.artifact_paths.index_path))
        write_jsonl(self.artifact_paths.meta_path, self.metadata)
        self.artifact_paths.build_info_path.write_text(
            json.dumps(self.build_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

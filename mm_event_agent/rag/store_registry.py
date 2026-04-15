"""Registry for loading persistent text-side RAG stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import attach_retrieval_metadata
from mm_event_agent.rag.persistent_faiss import PersistentFaissIndex
from mm_event_agent.rag.text_encoder import SentenceTransformerTextEncoder


class RagStoreRegistry:
    def __init__(
        self,
        *,
        encoder: SentenceTransformerTextEncoder | None = None,
        index_root: str | Path | None = None,
        ace_text_dir: str | Path | None = None,
        maven_text_dir: str | Path | None = None,
        swig_text_dir: str | Path | None = None,
        bridge_dir: str | Path | None = None,
    ) -> None:
        self._index_root = Path(index_root or settings.rag_index_root)
        encoder_source = settings.rag_text_encoder_model_path or settings.rag_text_encoder_model
        self._encoder = encoder or SentenceTransformerTextEncoder(encoder_source, normalize=True)
        self._indexes: dict[str, PersistentFaissIndex] = {}

        configured_paths = {
            "ace_text": self._resolve_dir("ace_text", ace_text_dir, settings.rag_ace_text_index_dir),
            "maven_text": self._resolve_dir("maven_text", maven_text_dir, settings.rag_maven_text_index_dir),
            "swig_text": self._resolve_dir("swig_text", swig_text_dir, settings.rag_swig_text_index_dir),
            "bridge": self._resolve_dir("bridge", bridge_dir, settings.rag_bridge_index_dir),
        }
        for name, path in configured_paths.items():
            if path.exists():
                self._indexes[name] = PersistentFaissIndex.load(path)

    def retrieve_text_examples(self, query: str, top_k: int, *, event_type: str = "") -> list[dict[str, Any]]:
        indexes = [name for name in ["ace_text", "maven_text"] if name in self._indexes]
        return self._search_indexes(indexes, query, top_k, event_type=event_type)

    def retrieve_bridge_examples(self, query: str, top_k: int, *, event_type: str = "") -> list[dict[str, Any]]:
        return self._search_indexes(["bridge"], query, top_k, event_type=event_type)

    def retrieve_swig_text_examples(self, query: str, top_k: int, *, event_type: str = "") -> list[dict[str, Any]]:
        return self._search_indexes(["swig_text"], query, top_k, event_type=event_type)

    def _search_indexes(self, index_names: list[str], query: str, top_k: int, *, event_type: str = "") -> list[dict[str, Any]]:
        normalized_query = str(query or "").strip()
        if not normalized_query or top_k <= 0:
            return []

        query_vector = self._encoder.encode([normalized_query])[0]
        filters = {"event_type": event_type} if str(event_type or "").strip() else None
        combined: list[dict[str, Any]] = []
        for index_name in index_names:
            index = self._indexes.get(index_name)
            if index is None:
                continue
            for item in index.search(query_vector, top_k=top_k, filters=filters):
                combined.append(
                    attach_retrieval_metadata(
                        item["meta"],
                        score=item["score"],
                        rank=item["rank"],
                        index_name=index_name,
                    )
                )

        combined.sort(
            key=lambda item: float(item.get("retrieval_metadata", {}).get("score", 0.0)),
            reverse=True,
        )
        trimmed = combined[:top_k]
        for rank, item in enumerate(trimmed, start=1):
            item["retrieval_metadata"]["rank"] = rank
        return trimmed

    def _resolve_dir(self, index_name: str, override: str | Path | None, configured: str) -> Path:
        if override is not None:
            return Path(override)
        if str(configured or "").strip():
            return Path(configured)
        return self._index_root / index_name

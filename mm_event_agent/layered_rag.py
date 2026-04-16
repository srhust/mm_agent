"""Layered local retrieval for heterogeneous multimodal event resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mm_event_agent.runtime_config import settings
from mm_event_agent.schemas import LayeredSimilarEvents, empty_layered_similar_events

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

try:  # pragma: no cover - dependency availability varies by environment
    import faiss
except ImportError:  # pragma: no cover - dependency availability varies by environment
    faiss = None

try:  # pragma: no cover - dependency availability varies by environment
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - dependency availability varies by environment
    SentenceTransformer = None


class InMemoryFaissCollection:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._index: faiss.Index | None = None
        self._docs: list[dict[str, Any]] = []

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if faiss is None:
            raise ImportError("faiss is required to use the demo in-memory RAG path")
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required to use the demo in-memory RAG path")
            self._model = SentenceTransformer(self._model_name)
        emb = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        out = np.asarray(emb, dtype=np.float32)
        faiss.normalize_L2(out)
        return out

    def build(self, docs: Sequence[dict[str, Any]], text_builder) -> None:
        self._docs = [dict(item) for item in docs if isinstance(item, dict)]
        if not self._docs:
            self._index = None
            return
        vectors = self._encode([text_builder(doc) for doc in self._docs])
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def retrieve(self, query: str, top_k: int, filter_fn=None) -> list[dict[str, Any]]:
        if self._index is None or not self._docs or not str(query or "").strip():
            return []
        k = min(max(1, int(top_k)), len(self._docs))
        q = self._encode([query])
        _, indices = self._index.search(q, k)
        results: list[dict[str, Any]] = []
        for index in indices[0]:
            doc = self._docs[int(index)]
            if filter_fn is not None and not filter_fn(doc):
                continue
            results.append(dict(doc))
        return results


class LayeredRagStore:
    def __init__(self, persistent_registry_factory: Callable[[], Any] | None = None) -> None:
        self._text_store = InMemoryFaissCollection()
        self._image_store = InMemoryFaissCollection()
        self._bridge_store = InMemoryFaissCollection()
        self._persistent_registry_factory = persistent_registry_factory
        self._persistent_registry: Any | None = None

    def build_index(self, corpora: dict[str, Sequence[dict[str, Any]]] | None) -> None:
        layered = _normalize_corpora(corpora)
        self._text_store.build(layered["text_event_examples"], _text_event_repr)
        self._image_store.build(layered["image_semantic_examples"], _image_semantic_repr)
        self._bridge_store.build(layered["bridge_examples"], _bridge_repr)

    def retrieve(
        self,
        raw_text: str,
        image_desc: str = "",
        event_type: str = "",
        top_k: int = 3,
        *,
        raw_image: Any | None = None,
    ) -> LayeredSimilarEvents:
        if settings.rag_use_persistent_index:
            persistent_results = self._retrieve_persistent(
                raw_text=raw_text,
                image_desc=image_desc,
                event_type=event_type,
                top_k=top_k,
                raw_image=raw_image,
            )
            if persistent_results is not None:
                return persistent_results
            if not settings.rag_use_demo_corpus:
                return empty_layered_similar_events()
        return self._retrieve_demo(raw_text=raw_text, image_desc=image_desc, event_type=event_type, top_k=top_k)

    def _retrieve_demo(
        self,
        *,
        raw_text: str,
        image_desc: str,
        event_type: str,
        top_k: int,
    ) -> LayeredSimilarEvents:
        normalized_raw_text = str(raw_text or "").strip()
        normalized_image_desc = str(image_desc or "").strip()
        normalized_event_type = str(event_type or "").strip()
        if not normalized_raw_text and not normalized_image_desc and not normalized_event_type:
            return empty_layered_similar_events()

        primary_query = normalized_raw_text or normalized_event_type or normalized_image_desc
        image_query = " ".join(part for part in [normalized_raw_text, normalized_image_desc, normalized_event_type] if part).strip()
        bridge_query = " ".join(part for part in [normalized_event_type, normalized_raw_text, normalized_image_desc] if part).strip()

        def _event_filter(doc: dict[str, Any]) -> bool:
            if not normalized_event_type:
                return True
            return str(doc.get("event_type") or "").strip() in {"", normalized_event_type}

        def _bridge_filter(doc: dict[str, Any]) -> bool:
            if not normalized_event_type:
                return True
            return str(doc.get("event_type") or "").strip() in {"", normalized_event_type}

        return {
            "text_event_examples": self._text_store.retrieve(primary_query, top_k=top_k, filter_fn=_event_filter),
            "image_semantic_examples": self._image_store.retrieve(image_query or primary_query, top_k=top_k),
            "bridge_examples": self._bridge_store.retrieve(bridge_query or primary_query, top_k=top_k, filter_fn=_bridge_filter),
        }

    def _retrieve_persistent(
        self,
        *,
        raw_text: str,
        image_desc: str,
        event_type: str,
        top_k: int,
        raw_image: Any | None,
    ) -> LayeredSimilarEvents | None:
        registry = self._get_persistent_registry()
        if registry is None:
            return None
        available_index_names = set(getattr(registry, "available_index_names", lambda: set())())
        if not available_index_names:
            return None

        normalized_raw_text = str(raw_text or "").strip()
        normalized_image_desc = str(image_desc or "").strip()
        normalized_event_type = str(event_type or "").strip()
        if not normalized_raw_text and not normalized_image_desc and not normalized_event_type and raw_image is None:
            return empty_layered_similar_events()

        effective_top_k = max(1, int(top_k or settings.rag_default_top_k))
        text_top_k = max(1, int(settings.rag_text_top_k or effective_top_k))
        image_top_k = max(1, int(settings.rag_image_top_k or effective_top_k))
        bridge_top_k = max(1, int(settings.rag_bridge_top_k or effective_top_k))

        primary_query = normalized_raw_text or normalized_event_type or normalized_image_desc
        image_text_query = " ".join(
            part for part in [normalized_raw_text, normalized_image_desc, normalized_event_type] if part
        ).strip()
        bridge_query = " ".join(
            part for part in [normalized_event_type, normalized_raw_text, normalized_image_desc] if part
        ).strip()

        text_examples = registry.retrieve_text_examples(
            primary_query,
            top_k=text_top_k,
            event_type=normalized_event_type,
        ) if primary_query else []

        image_examples: list[dict[str, Any]] = []
        if image_text_query and "swig_text" in available_index_names:
            image_examples.extend(
                registry.retrieve_swig_text_examples(
                    image_text_query,
                    top_k=image_top_k,
                    event_type=normalized_event_type,
                )
            )
        image_query_path = _extract_image_query_path(raw_image)
        if (
            settings.rag_enable_image_query
            and image_query_path
            and "swig_image" in available_index_names
        ):
            image_examples.extend(
                registry.retrieve_swig_image_examples(
                    raw_image=raw_image,
                    image_path=image_query_path,
                    top_k=image_top_k,
                    event_type=normalized_event_type,
                )
            )

        bridge_examples = registry.retrieve_bridge_examples(
            bridge_query or primary_query,
            top_k=bridge_top_k,
            event_type=normalized_event_type,
        ) if (bridge_query or primary_query) and "bridge" in available_index_names else []

        return {
            "text_event_examples": [
                _normalize_text_event_example(item)
                for item in _rank_and_trim_examples(text_examples, text_top_k)
            ],
            "image_semantic_examples": [
                _normalize_image_semantic_example(item)
                for item in _rank_and_trim_examples(image_examples, image_top_k)
            ],
            "bridge_examples": [
                _normalize_bridge_example(item)
                for item in _rank_and_trim_examples(bridge_examples, bridge_top_k)
            ],
        }

    def _get_persistent_registry(self) -> Any | None:
        if self._persistent_registry is not None:
            return self._persistent_registry
        try:
            if self._persistent_registry_factory is not None:
                self._persistent_registry = self._persistent_registry_factory()
            else:
                from mm_event_agent.rag.store_registry import get_cached_registry

                self._persistent_registry = get_cached_registry()
        except Exception:
            self._persistent_registry = None
        return self._persistent_registry


def _normalize_corpora(corpora: dict[str, Sequence[dict[str, Any]]] | None) -> LayeredSimilarEvents:
    if not isinstance(corpora, dict):
        return empty_layered_similar_events()
    return {
        "text_event_examples": [_normalize_text_event_example(item) for item in corpora.get("text_event_examples", []) if isinstance(item, dict)],
        "image_semantic_examples": [_normalize_image_semantic_example(item) for item in corpora.get("image_semantic_examples", []) if isinstance(item, dict)],
        "bridge_examples": [_normalize_bridge_example(item) for item in corpora.get("bridge_examples", []) if isinstance(item, dict)],
    }


def _text_event_repr(doc: dict[str, Any]) -> str:
    retrieval_text = str(doc.get("retrieval_text") or "").strip()
    if retrieval_text:
        return retrieval_text
    payload = {
        "id": doc.get("id", ""),
        "source_dataset": doc.get("source_dataset", ""),
        "modality": doc.get("modality", "text"),
        "event_type": doc.get("event_type", ""),
        "raw_text": doc.get("raw_text", ""),
        "trigger": doc.get("trigger", ""),
        "text_arguments": doc.get("text_arguments", []),
        "pattern_summary": doc.get("pattern_summary", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def _image_semantic_repr(doc: dict[str, Any]) -> str:
    retrieval_text = str(doc.get("retrieval_text") or "").strip()
    if retrieval_text:
        return retrieval_text
    payload = {
        "id": doc.get("id", ""),
        "source_dataset": doc.get("source_dataset", ""),
        "modality": doc.get("modality", "image_semantics"),
        "event_type": doc.get("event_type", ""),
        "visual_situation": doc.get("visual_situation", ""),
        "image_desc": doc.get("image_desc", ""),
        "image_arguments": doc.get("image_arguments", []),
        "visual_pattern_summary": doc.get("visual_pattern_summary", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def _bridge_repr(doc: dict[str, Any]) -> str:
    retrieval_text = str(doc.get("retrieval_text") or "").strip()
    if retrieval_text:
        return retrieval_text
    payload = {
        "id": doc.get("id", ""),
        "source_dataset": doc.get("source_dataset", ""),
        "modality": doc.get("modality", "bridge"),
        "event_type": doc.get("event_type", ""),
        "role": doc.get("role", ""),
        "text_cues": doc.get("text_cues", []),
        "visual_cues": doc.get("visual_cues", []),
        "note": doc.get("note", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


_default_store = LayeredRagStore()


def _extract_image_query_path(raw_image: Any | None) -> str:
    if not isinstance(raw_image, str):
        return ""
    candidate = raw_image.strip()
    if not candidate:
        return ""
    return candidate if Path(candidate).exists() else ""


def _rank_and_trim_examples(items: Sequence[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    normalized = [dict(item) for item in items if isinstance(item, dict)]
    normalized.sort(
        key=lambda item: float(item.get("retrieval_metadata", {}).get("score", 0.0)),
        reverse=True,
    )
    trimmed = normalized[: max(0, int(top_k))]
    for rank, item in enumerate(trimmed, start=1):
        retrieval_metadata = dict(item.get("retrieval_metadata") or {})
        retrieval_metadata["rank"] = rank
        item["retrieval_metadata"] = retrieval_metadata
    return trimmed


def _normalize_text_event_example(item: dict[str, Any]) -> dict[str, Any]:
    example = dict(item)
    example["id"] = str(example.get("id") or "").strip()
    example["source_dataset"] = str(example.get("source_dataset") or "").strip()
    example["modality"] = "text"
    example["event_type"] = str(example.get("event_type") or "").strip()
    example["raw_text"] = str(example.get("raw_text") or "").strip()
    trigger = example.get("trigger")
    if not isinstance(trigger, dict):
        trigger = {"text": str(trigger or "").strip(), "span": None}
    else:
        trigger = {
            "text": str(trigger.get("text") or "").strip(),
            "span": trigger.get("span") if isinstance(trigger.get("span"), dict) else None,
        }
    example["trigger"] = trigger
    raw_arguments = example.get("text_arguments")
    example["text_arguments"] = [dict(arg) for arg in raw_arguments] if isinstance(raw_arguments, list) else []
    example["pattern_summary"] = str(example.get("pattern_summary") or "").strip()
    example["retrieval_text"] = str(example.get("retrieval_text") or "").strip() or _default_text_event_retrieval_text(example)
    return example


def _normalize_image_semantic_example(item: dict[str, Any]) -> dict[str, Any]:
    example = dict(item)
    example["id"] = str(example.get("id") or "").strip()
    example["source_dataset"] = str(example.get("source_dataset") or "").strip()
    example["modality"] = "image_semantics"
    example["event_type"] = str(example.get("event_type") or "").strip()
    example["image_desc"] = str(example.get("image_desc") or "").strip()
    example["visual_situation"] = str(example.get("visual_situation") or "").strip()
    raw_arguments = example.get("image_arguments") or example.get("image_roles")
    example["image_arguments"] = [dict(arg) for arg in raw_arguments] if isinstance(raw_arguments, list) else []
    example["visual_pattern_summary"] = str(example.get("visual_pattern_summary") or "").strip()
    example["retrieval_text"] = str(example.get("retrieval_text") or "").strip() or _default_image_semantic_retrieval_text(example)
    return example


def _normalize_bridge_example(item: dict[str, Any]) -> dict[str, Any]:
    example = dict(item)
    example["id"] = str(example.get("id") or "").strip()
    example["source_dataset"] = str(example.get("source_dataset") or "").strip()
    example["modality"] = "bridge"
    example["event_type"] = str(example.get("event_type") or "").strip()
    example["role"] = str(example.get("role") or "").strip()
    example["text_cues"] = [str(value) for value in example.get("text_cues", [])] if isinstance(example.get("text_cues"), list) else []
    example["visual_cues"] = [str(value) for value in example.get("visual_cues", [])] if isinstance(example.get("visual_cues"), list) else []
    example["note"] = str(example.get("note") or "").strip()
    example["retrieval_text"] = str(example.get("retrieval_text") or "").strip() or _default_bridge_retrieval_text(example)
    return example


def _default_text_event_retrieval_text(example: dict[str, Any]) -> str:
    parts: list[str] = [example.get("event_type", ""), example.get("raw_text", ""), example.get("pattern_summary", "")]
    trigger = example.get("trigger")
    if isinstance(trigger, dict):
        parts.append(str(trigger.get("text") or ""))
    for argument in example.get("text_arguments", []):
        if isinstance(argument, dict):
            parts.append(str(argument.get("role") or ""))
            parts.append(str(argument.get("text") or ""))
    return " ".join(part for part in parts if str(part).strip())


def _default_image_semantic_retrieval_text(example: dict[str, Any]) -> str:
    parts: list[str] = [
        example.get("event_type", ""),
        example.get("visual_situation", ""),
        example.get("image_desc", ""),
        example.get("visual_pattern_summary", ""),
    ]
    for argument in example.get("image_arguments", []):
        if isinstance(argument, dict):
            parts.append(str(argument.get("role") or ""))
            parts.append(str(argument.get("label") or ""))
    return " ".join(part for part in parts if str(part).strip())


def _default_bridge_retrieval_text(example: dict[str, Any]) -> str:
    parts: list[str] = [example.get("event_type", ""), example.get("role", ""), example.get("note", "")]
    parts.extend(example.get("text_cues", []))
    parts.extend(example.get("visual_cues", []))
    return " ".join(part for part in parts if str(part).strip())


def build_index(corpora: dict[str, Sequence[dict[str, Any]]] | None) -> None:
    _default_store.build_index(corpora)


def retrieve(
    raw_text: str,
    image_desc: str = "",
    event_type: str = "",
    top_k: int = 3,
    *,
    raw_image: Any | None = None,
) -> LayeredSimilarEvents:
    return _default_store.retrieve(
        raw_text=raw_text,
        image_desc=image_desc,
        event_type=event_type,
        top_k=top_k,
        raw_image=raw_image,
    )

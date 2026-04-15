"""Reusable dense text encoder for persistent text-side RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - dependency availability varies by environment
    SentenceTransformer = None


class SentenceTransformerTextEncoder:
    """Small predictable wrapper over sentence-transformers."""

    def __init__(self, model_name_or_path: str, normalize: bool = True) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.normalize = bool(normalize)
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required to use SentenceTransformerTextEncoder")
        model_path = Path(self.model_name_or_path)
        kwargs = {"local_files_only": True} if model_path.exists() else {}
        self._model = SentenceTransformer(self.model_name_or_path, **kwargs)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            [str(text) for text in texts],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"expected 2D embeddings, got shape {array.shape}")
        if self.normalize and array.size:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            array = array / norms
        return array

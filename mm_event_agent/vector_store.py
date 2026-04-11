"""内存 FAISS 向量库：sentence-transformers + IndexFlatIP（归一化后等价于余弦相似度）。"""

from __future__ import annotations

from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class InMemoryFaissStore:
    """纯内存，不落盘。"""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._docs: list[str] = []

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME)
        emb = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out = np.asarray(emb, dtype=np.float32)
        faiss.normalize_L2(out)
        return out

    def build_index(self, docs: Sequence[str]) -> None:
        """用文档列表构建（或重建）FAISS 索引。"""
        self._docs = [str(d) for d in docs]
        if not self._docs:
            self._index = None
            return
        vectors = self._encode(self._docs)
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """按与 query 的余弦相似度返回 top_k 条原文。"""
        if self._index is None or not self._docs:
            return []
        k = min(int(top_k), len(self._docs))
        q = self._encode([query])
        _, indices = self._index.search(q, k)
        return [self._docs[int(i)] for i in indices[0]]


_default_store = InMemoryFaissStore()


def build_index(docs: Sequence[str]) -> None:
    """在默认单例上构建索引。"""
    _default_store.build_index(docs)


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """在默认单例上检索。"""
    return _default_store.retrieve(query, top_k)

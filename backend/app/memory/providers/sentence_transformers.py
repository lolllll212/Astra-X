"""Sentence-Transformers embedding provider.

Runs models locally using the ``sentence-transformers`` package::

    pip install sentence-transformers
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.memory.embedder import Embedder
from app.memory.models.embedding import EmbeddingResult


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dimensions: int = 384

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers") from exc
        self._model = SentenceTransformer(self._model_name)
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        await self._ensure_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return list(vector)

    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        await self._ensure_model()
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return EmbeddingResult(
            embeddings=[list(v) for v in vectors],
            model=self._model_name,
            dimensions=self._dimensions,
        )

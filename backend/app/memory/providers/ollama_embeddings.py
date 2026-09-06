"""Ollama embedding provider.

Calls the Ollama embeddings API (``/api/embed``) on a local Ollama instance.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.memory.embedder import Embedder
from app.memory.models.embedding import EmbeddingResult


class OllamaEmbedder(Embedder):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions: int = 768

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings:
            self._dimensions = len(embeddings[0])
        return embeddings[0] if embeddings else []

    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": list(texts)},
            )
            resp.raise_for_status()
            data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings:
            self._dimensions = len(embeddings[0])
        return EmbeddingResult(
            embeddings=embeddings,
            model=self._model,
            dimensions=self._dimensions,
        )

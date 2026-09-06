"""OpenAI embedding provider.

Calls the OpenAI embeddings API (``/v1/embeddings``).
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.memory.embedder import Embedder
from app.memory.models.embedding import EmbeddingResult


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions: int = 1536

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
        vector: list[float] = data["data"][0]["embedding"]
        self._dimensions = len(vector)
        return vector

    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"model": self._model, "input": list(texts)},
            )
            resp.raise_for_status()
            data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        if embeddings:
            self._dimensions = len(embeddings[0])
        return EmbeddingResult(
            embeddings=embeddings,
            model=self._model,
            dimensions=self._dimensions,
        )

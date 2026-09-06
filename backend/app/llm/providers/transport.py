"""Shared HTTP transport for LLM provider adapters.

The :class:`Transport` wraps an ``httpx.AsyncClient`` with sensible
defaults (connection pooling, timeouts, error mapping) so that provider
adapters don't create their own clients.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from app.llm.exceptions import (
    GenerationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
)

__all__ = [
    "Transport",
    "TransportError",
]


class TransportError(Exception):
    """Wrapper for HTTP-level transport failures."""


class Transport:
    """HTTP transport with connection pooling and error mapping.

    Usage::

        transport = Transport(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            timeout=60.0,
        )
        data = await transport.request("POST", "/chat/completions", json=payload)

        async for chunk in transport.stream("POST", "/chat/completions", json=payload):
            ...
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")

        if client is not None:
            self._client = client
        else:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                **(extra_headers or {}),
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(timeout),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request and return the parsed JSON response.

        Args:
            method: HTTP method (e.g. ``"POST"``).
            path: URL path (e.g. ``"/chat/completions"``).
            json_data: Optional JSON-serialisable request body.

        Returns:
            The parsed JSON response body.

        Raises:
            ProviderTimeoutError: If the request exceeds the configured timeout.
            ProviderConnectionError: If the provider is unreachable.
            ProviderAuthenticationError: If authentication fails (401).
            GenerationError: For other HTTP errors.
        """
        try:
            resp = await self._client.request(method, path, json=json_data)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Provider request timed out after {self._client.timeout.read}s",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Could not connect to {self._base_url}",
            ) from exc

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise ProviderAuthenticationError(
                    "Provider returned 401 Unauthorized",
                ) from exc
            raise GenerationError(
                f"Provider returned {exc.response.status_code}: {exc.response.text[:200]}",
            ) from exc

        return cast(dict[str, Any], resp.json())

    async def stream(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream SSE lines from the provider, yielding parsed JSON chunks.

        ``[DONE]`` sentinels are consumed automatically and not yielded.

        Args:
            method: HTTP method (e.g. ``"POST"``).
            path: URL path (e.g. ``"/chat/completions"``).
            json_data: Optional JSON-serialisable request body.

        Yields:
            Parsed JSON dicts from ``data: `` SSE lines.

        Raises:
            ProviderTimeoutError: If the request exceeds the configured timeout.
            ProviderConnectionError: If the provider is unreachable.
            GenerationError: For other HTTP errors.
        """
        try:
            async with self._client.stream(method, path, json=json_data) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_data = line.removeprefix("data: ").strip()
                    if chunk_data == "[DONE]":
                        return
                    if not chunk_data:
                        continue
                    yield json.loads(chunk_data)

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Provider stream timed out after {self._client.timeout.read}s",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Could not connect to {self._base_url}",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"Provider stream returned {exc.response.status_code}",
            ) from exc

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

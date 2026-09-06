"""Core SDK client for the Astra X API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class AstraSDK:
    """Python client for the Astra X agent API.

    Wraps the REST and streaming endpoints with typed helpers.
    Thread-safe for concurrent use.

    Args:
        base_url: Root URL of the API (e.g. ``http://localhost:8000/api/v1``).
        api_key: Optional API key for authentication.
        timeout: Default request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        conversation_id: str,
        message: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Send a message and receive the full response.

        Args:
            conversation_id: Existing or new conversation ID.
            message: The user's message text.
            model: Optional model override.
            provider: Optional provider override.

        Returns:
            The full ``ChatResponse`` JSON.
        """
        body: dict[str, Any] = {
            "conversation_id": conversation_id,
            "content": [{"type": "text", "text": message}],
        }
        if model:
            body["model"] = model
        if provider:
            body["provider"] = provider

        response = await self._client.post("/chat", json=body)
        response.raise_for_status()
        return response.json()

    async def chat_stream(
        self,
        conversation_id: str,
        message: str,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat response as a sequence of events.

        Yields:
            Parsed ``StreamEvent`` dicts with ``type``, ``data``, etc.
        """
        body: dict[str, Any] = {
            "conversation_id": conversation_id,
            "content": [{"type": "text", "text": message}],
        }
        if model:
            body["model"] = model
        if provider:
            body["provider"] = provider

        async with self._client.stream(
            "POST", "/chat/stream", json=body
        ) as stream:
            async for line in stream.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def list_conversations(
        self,
        page: int = 1,
        per_page: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent conversations.

        Args:
            page: Page number (1-indexed).
            per_page: Items per page.

        Returns:
            A list of conversation summary dicts.
        """
        response = await self._client.get(
            "/conversations",
            params={"page": page, "per_page": per_page},
        )
        response.raise_for_status()
        return response.json()

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Get full conversation details.

        Args:
            conversation_id: The conversation to retrieve.

        Returns:
            The conversation dict with messages.
        """
        response = await self._client.get(f"/conversations/{conversation_id}")
        response.raise_for_status()
        return response.json()

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and its messages.

        Args:
            conversation_id: The conversation to delete.
        """
        response = await self._client.delete(f"/conversations/{conversation_id}")
        response.raise_for_status()

    # ------------------------------------------------------------------
    # Dashboard / observability
    # ------------------------------------------------------------------

    async def get_dashboard(self) -> dict[str, Any]:
        """Return runtime dashboard data (learning + provider stats).

        Returns:
            The full ``DashboardResponse`` JSON.
        """
        response = await self._client.get("/dashboard")
        response.raise_for_status()
        return response.json()

    async def get_health(self) -> dict[str, Any]:
        """Return the health-check status of providers.

        Returns:
            A dict mapping provider IDs to health status.
        """
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def get_system_info(self) -> dict[str, Any]:
        """Return application version and configuration info.

        Returns:
            A dict with version and config fields.
        """
        response = await self._client.get("/system/info")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    async def list_providers(self) -> list[dict[str, Any]]:
        """List all configured LLM providers.

        Returns:
            A list of provider configuration dicts.
        """
        response = await self._client.get("/providers")
        response.raise_for_status()
        return response.json()

    async def test_provider(self, provider_id: str) -> dict[str, Any]:
        """Test connectivity to a provider.

        Args:
            provider_id: The provider to test.

        Returns:
            A dict with the test result.
        """
        response = await self._client.post(f"/providers/{provider_id}/test")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Plugins
    # ------------------------------------------------------------------

    async def list_plugins(self) -> list[dict[str, Any]]:
        """List installed plugins.

        Returns:
            A list of plugin metadata dicts.
        """
        response = await self._client.get("/plugins")
        response.raise_for_status()
        return response.json()

    async def install_plugin(
        self,
        source: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Install a plugin from a source URL or path.

        Args:
            source: Plugin source (URL, file path, or package name).
            config: Optional plugin configuration.

        Returns:
            The installed plugin metadata.
        """
        body: dict[str, Any] = {"source": source}
        if config:
            body["config"] = config
        response = await self._client.post("/plugins", json=body)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

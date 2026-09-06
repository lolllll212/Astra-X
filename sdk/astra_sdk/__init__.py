"""

``astra_sdk`` — Python client for the Astra X agent API.

Usage::

    from astra_sdk import AstraSDK

    client = AstraSDK(base_url="http://localhost:8000/api/v1")
    result = client.chat("conversation_id", "Hello!")
    print(result["response"])

For streaming::

    for event in client.chat_stream("conversation_id", "Tell me a story"):
        if event["type"] == "text_delta":
            print(event["delta"], end="")
"""

from __future__ import annotations

from astra_sdk.client import AstraSDK

__all__ = ["AstraSDK"]

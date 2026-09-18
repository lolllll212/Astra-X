"""LM Studio local API tool - OpenAI-compatible inference on locally installed GGUF models (DeepSeek R1 Distill Qwen 7B, Qwen2.5 Coder 7B, Llava v1.6 Mistral 7B, ...) at http://127.0.0.1:1234."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from urllib.request import Request, urlopen

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.lmstudio.models import (
    LmStudioChatResult,
    LmStudioCompletionResult,
    LmStudioEmbedding,
    LmStudioModelInfo,
    LmStudioStatus,
)
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class LmStudioTool(Tool):
    """LM Studio local API - chat/completions/embeddings on locally installed GGUF models (DeepSeek R1, Qwen2.5 Coder, LLaVA, Gemma, ...) served by the LM Studio OpenAI-compatible server."""

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def description(self) -> str:
        return "LM Studio local inference (OpenAI-compatible): chat, completion, embeddings, list models, status. Default base http://127.0.0.1:1234. Great for local DeepSeek R1 Distill Qwen 7B, Qwen2.5 Coder 7B, Llava v1.6 Mistral 7B GGUF models."

    @property
    def capabilities(self) -> list[str]:
        return ["local_inference", "openai_compatible", "gguf_models", "nlp"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action: chat, completion, embeddings, list_models, get_status",
                    required=True,
                    enum=["chat", "completion", "embeddings", "list_models", "get_status"],
                ),
                ToolParameter(
                    name="model",
                    type_="string",
                    description="Model id (default: deepseek-r1-distill-qwen-7b). Options: deepseek-r1-distill-qwen-7b, qwen2.5-coder-7b-instruct, llava-v1.6-mistral-7b, mistral-7b-instruct-v0.2, deepseek-coder-6.7b-instruct, google/gemma-3-4b",
                    required=False,
                    default="deepseek-r1-distill-qwen-7b",
                ),
                ToolParameter(
                    name="prompt",
                    type_="string",
                    description="User message / prompt",
                    required=False,
                ),
                ToolParameter(
                    name="system",
                    type_="string",
                    description="Optional system prompt",
                    required=False,
                ),
                ToolParameter(
                    name="messages",
                    type_="array",
                    description="Full message list [{role, content}, ...] (overrides prompt+system)",
                    required=False,
                    default=[],
                ),
                ToolParameter(
                    name="temperature",
                    type_="number",
                    description="Sampling temperature",
                    required=False,
                    default=0.7,
                ),
                ToolParameter(
                    name="max_tokens",
                    type_="integer",
                    description="Maximum response tokens",
                    required=False,
                    default=1024,
                ),
                ToolParameter(
                    name="top_p",
                    type_="number",
                    description="Nucleus sampling",
                    required=False,
                    default=1.0,
                ),
                ToolParameter(
                    name="base_url",
                    type_="string",
                    description="Base URL of LM Studio server (default http://127.0.0.1:1234)",
                    required=False,
                    default="http://127.0.0.1:1234",
                ),
                ToolParameter(
                    name="stream",
                    type_="boolean",
                    description="Stream the response (reported via metadata)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="text",
                    type_="string",
                    description="Input text (for embeddings)",
                    required=False,
                ),
                ToolParameter(
                    name="greeting",
                    type_="string",
                    description="N/A",
                    required=False,
                ),
            ],
        )

    def __init__(self) -> None:
        self._base_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")

    def _resolve_base(self, kwargs: dict) -> str:
        return kwargs.get("base_url", "") or self._base_url

    def _call(self, base_url: str, path: str, payload: dict | None = None, timeout: float = 300.0) -> tuple[int, Any, str]:
        url = base_url.rstrip("/") + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.status, json.loads(body), ""
        except Exception as exc:
            return 0, None, str(exc)

    async def _call_async(self, base_url: str, path: str, payload: dict | None = None, timeout: float = 300.0) -> tuple[int, Any, str]:  # noqa: ASYNC109
        return await asyncio.get_event_loop().run_in_executor(None, self._call, base_url, path, payload, timeout)

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        if action == "chat":
            return await self._chat(kwargs)
        elif action == "completion":
            return await self._completion(kwargs)
        elif action == "embeddings":
            return await self._embeddings(kwargs)
        elif action == "list_models":
            return await self._list_models(kwargs)
        elif action == "get_status":
            return await self._get_status(kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _chat(self, kwargs: dict) -> ToolResult:
        base = self._resolve_base(kwargs)
        model = kwargs.get("model", "deepseek-r1-distill-qwen-7b")
        prompt = kwargs.get("prompt", "")
        system = kwargs.get("system", "")
        messages = kwargs.get("messages", [])
        if not messages:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if not prompt:
                return ToolResult(success=False, error="prompt is required for chat")
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": kwargs.get("stream", False),
        }

        start = time.monotonic()
        status, data, err = await self._call_async(base, "/v1/chat/completions", payload)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        if status != 200:
            return ToolResult(
                success=False,
                error=err or (data.get("error", {}).get("message") if isinstance(data, dict) else f"HTTP {status}"),
                metadata={"base_url": base, "status": status},
            )

        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        result = LmStudioChatResult(
            ok=True,
            content=message.get("content", ""),
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason", ""),
            usage_prompt_tokens=usage.get("prompt_tokens", 0),
            usage_completion_tokens=usage.get("completion_tokens", 0),
            usage_total_tokens=usage.get("total_tokens", 0),
            reasoning_content=message.get("reasoning_content", ""),
        )
        content = result.content
        used_reasoning_fallback = False
        if not content and result.reasoning_content:
            content = result.reasoning_content
            used_reasoning_fallback = True
        output = content
        if result.finish_reason == "length":
            note = (
                "\n\n[response truncated at max_tokens; "
                "reasoning models like qwen3.5/deepseek-r1 need a larger max_tokens (>=2048)]"
            )
            output = (output + note).strip() if output else note.strip()
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "model": result.model,
                "finish_reason": result.finish_reason,
                "usage_total_tokens": result.usage_total_tokens,
                "usage_completion_tokens": result.usage_completion_tokens,
                "usage_prompt_tokens": result.usage_prompt_tokens,
                "latency_ms": elapsed_ms,
                "reasoning_content": result.reasoning_content,
                "used_reasoning_fallback": used_reasoning_fallback,
                "base_url": base,
            },
        )

    async def _completion(self, kwargs: dict) -> ToolResult:
        base = self._resolve_base(kwargs)
        model = kwargs.get("model", "deepseek-r1-distill-qwen-7b")
        prompt = kwargs.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, error="prompt is required for completion")

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": kwargs.get("stream", False),
        }

        start = time.monotonic()
        status, data, err = await self._call_async(base, "/v1/completions", payload)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        if status != 200:
            return ToolResult(
                success=False,
                error=err or (data.get("error", {}).get("message") if isinstance(data, dict) else f"HTTP {status}"),
                metadata={"base_url": base, "status": status},
            )

        choice = data["choices"][0]
        usage = data.get("usage", {})
        result = LmStudioCompletionResult(
            ok=True,
            text=choice.get("text", ""),
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason", ""),
            usage_prompt_tokens=usage.get("prompt_tokens", 0),
            usage_completion_tokens=usage.get("completion_tokens", 0),
        )
        return ToolResult(
            success=True,
            output=result.text,
            metadata={
                "model": result.model,
                "finish_reason": result.finish_reason,
                "usage_total_tokens": result.usage_prompt_tokens + result.usage_completion_tokens,
                "latency_ms": elapsed_ms,
            },
        )

    async def _embeddings(self, kwargs: dict) -> ToolResult:
        base = self._resolve_base(kwargs)
        model = kwargs.get("model", "text-embedding-nomic-embed-text-v1.5")
        text = kwargs.get("text") or kwargs.get("prompt", "")
        if not text:
            return ToolResult(success=False, error="text is required for embeddings")

        payload = {"model": model, "input": text}
        status, data, err = await self._call_async(base, "/v1/embeddings", payload)

        if status != 200:
            return ToolResult(
                success=False,
                error=err or (data.get("error", {}).get("message") if isinstance(data, dict) else f"HTTP {status}"),
                metadata={"base_url": base, "status": status},
            )

        raw = data.get("data", [])
        emb: LmStudioEmbedding | None = None
        if raw:
            emb = LmStudioEmbedding(vector=raw[0].get("embedding", []), index=raw[0].get("index", 0))
        dims = len(emb.vector) if emb else 0
        return ToolResult(
            success=True,
            output=json.dumps(emb.vector if emb else [], indent=2),
            metadata={
                "model": data.get("model", model),
                "dimensions": dims,
                "base_url": base,
            },
        )

    async def _list_models(self, kwargs: dict) -> ToolResult:
        base = self._resolve_base(kwargs)
        status, data, err = await self._call_async(base, "/v1/models", timeout=30.0)

        if status != 200:
            return ToolResult(success=False, error=err or f"HTTP {status}", metadata={"base_url": base})

        models = []
        for m in data.get("data", []):
            models.append(LmStudioModelInfo(id=m.get("id", ""), object=m.get("object", "model"), created=m.get("created", 0), owned_by=m.get("owned_by", "lm-studio")))
        return ToolResult(
            success=True,
            output=json.dumps([m.__dict__ for m in models], indent=2),
            metadata={"model_count": len(models), "models": [m.id for m in models]},
        )

    async def _get_status(self, kwargs: dict) -> ToolResult:
        base = self._resolve_base(kwargs)
        start = time.monotonic()
        status, data, err = await self._call_async(base, "/v1/models", timeout=30.0)
        latency = round((time.monotonic() - start) * 1000)

        lm_status = LmStudioStatus(base_url=base, reachable=status == 200, latencies_ms=latency)
        if status != 200:
            lm_status.server_version = err
        else:
            lm_status.models = [LmStudioModelInfo(id=m.get("id", "")) for m in data.get("data", [])]

        return ToolResult(
            success=True,
            output=json.dumps({
                "reachable": lm_status.reachable,
                "base_url": lm_status.base_url,
                "model_count": len(lm_status.models),
                "models": [m.id for m in lm_status.models],
                "latency_ms": lm_status.latencies_ms,
            }, indent=2),
        )

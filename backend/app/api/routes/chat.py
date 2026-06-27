"""Chat endpoints — the primary AI interaction layer.

Provides non-streaming (``POST /chat``) and streaming (``POST /chat/stream``)
chat completions, as well as conversation continuation after tool calls.

All business logic is delegated to :class:`app.services.chat_service.ChatService`.
Routes only validate input, call the service, and format the response.
"""

# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ContentBlockSchema,
    ContinueRequest,
    GenerationParamsSchema,
    ImageBlockSchema,
    MessageResponse,
    TextBlockSchema,
    ToolCallBlockSchema,
    ToolResultBlockSchema,
    UsageResponse,
)
from app.api.schemas.conversation import ConversationResponse
from app.domain.conversation import Conversation
from app.domain.message import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from app.domain.usage import Usage
from app.llm.models import GenerationParams
from app.services.chat_service import ChatResult, ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


# ------------------------------------------------------------------ #
# Content block conversion helpers
# ------------------------------------------------------------------ #


def _content_schema_to_domain(block: ContentBlockSchema) -> ContentBlock:
    """Convert an API content-block schema to a domain content block.

    Args:
        block: The API schema variant.

    Returns:
        The equivalent domain object.

    Raises:
        ValueError: If the block type is unrecognised.
    """
    if isinstance(block, TextBlockSchema):
        return TextBlock(text=block.text)
    if isinstance(block, ImageBlockSchema):
        return ImageBlock(data_uri=block.data_uri, mime_type=block.mime_type)
    if isinstance(block, ToolCallBlockSchema):
        return ToolCallBlock(
            tool_call_id=block.tool_call_id,
            tool_name=block.tool_name,
            arguments=block.arguments,
        )
    if isinstance(block, ToolResultBlockSchema):
        return ToolResultBlock(
            tool_call_id=block.tool_call_id,
            tool_name=block.tool_name,
            output=block.output,
            is_error=block.is_error,
        )
    raise ValueError(f"Unrecognised content block type: {type(block).__name__}")


def _content_domain_to_schema(block: ContentBlock) -> ContentBlockSchema:
    """Convert a domain content block to its API schema equivalent.

    Args:
        block: The domain content block.

    Returns:
        The serialisable schema variant.
    """
    if isinstance(block, TextBlock):
        return TextBlockSchema(text=block.text)
    if isinstance(block, ImageBlock):
        return ImageBlockSchema(data_uri=block.data_uri, mime_type=block.mime_type)
    if isinstance(block, ToolCallBlock):
        return ToolCallBlockSchema(
            tool_call_id=block.tool_call_id,
            tool_name=block.tool_name,
            arguments=block.arguments,
        )
    if isinstance(block, ToolResultBlock):
        return ToolResultBlockSchema(
            tool_call_id=block.tool_call_id,
            tool_name=block.tool_name,
            output=block.output,
            is_error=block.is_error,
        )
    raise ValueError(f"Unrecognised content block type: {type(block).__name__}")


def _params_schema_to_domain(params: GenerationParamsSchema | None) -> GenerationParams | None:
    """Convert an API generation-params schema to a domain object.

    Only non-``None`` fields are passed through, allowing partial overrides.

    Args:
        params: The API schema, or ``None``.

    Returns:
        A domain ``GenerationParams``, or ``None`` if the schema was ``None``.
    """
    if params is None:
        return None
    kwargs = {k: v for k, v in params.model_dump().items() if v is not None}
    return GenerationParams(**kwargs)


# ------------------------------------------------------------------ #
# Response helpers
# ------------------------------------------------------------------ #


def _usage_to_response(usage: Usage | None) -> UsageResponse | None:
    """Convert a domain ``Usage`` value object to its API schema."""
    if usage is None:
        return None
    return UsageResponse(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def _message_to_response(msg: Message) -> MessageResponse:
    """Convert a domain ``Message`` to its API response schema."""
    return MessageResponse(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=[_content_domain_to_schema(b) for b in msg.content],
        created_at=msg.created_at,
        parent_id=msg.parent_id,
    )


def _conversation_to_response(conv: Conversation) -> ConversationResponse:
    """Convert a domain ``Conversation`` to its API response schema."""
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        status=conv.status,
        participants=[p.model_dump() for p in conv.participants],
        metadata=conv.metadata.model_dump(),
        message_count=conv.message_count,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


def _chat_result_to_response(result: ChatResult) -> ChatResponse:
    """Convert a service-level ``ChatResult`` to an API ``ChatResponse``."""
    return ChatResponse(
        message=_message_to_response(result.assistant_message),
        conversation=_conversation_to_response(result.conversation),
        usage=_usage_to_response(result.usage),
    )


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a chat message",
)
async def chat(
    body: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Process a user message and return the assistant's response.

    The full chat pipeline is handled by :class:`ChatService`:
    load conversation, persist user message, build context, count tokens,
    route to the LLM, persist the response, and return the result.
    """
    content_domain = [_content_schema_to_domain(b) for b in body.content]
    params = _params_schema_to_domain(body.params)

    result = await chat_service.process_message(
        conversation_id=body.conversation_id,
        user_content=content_domain,
        model=body.model,
        provider=body.provider,
        params=params,
    )
    return _chat_result_to_response(result)


@router.post(
    "/stream",
    summary="Stream a chat response",
    responses={
        200: {
            "description": "Server-sent event stream",
            "content": {"text/event-stream": {}},
        },
    },
)
async def chat_stream(
    body: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Process a user message and stream the assistant's response via SSE.

    The response is sent as a ``text/event-stream`` with a single
    ``done`` event containing the complete result once generation
    finishes.
    """
    content_domain = [_content_schema_to_domain(b) for b in body.content]
    params = _params_schema_to_domain(body.params)

    result = await chat_service.process_message_stream(
        conversation_id=body.conversation_id,
        user_content=content_domain,
        model=body.model,
        provider=body.provider,
        params=params,
    )

    response_schema = _chat_result_to_response(result)

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"event: done\ndata: {response_schema.model_dump_json()}\n\n"

    return StreamingResponse(
        content=event_generator(),
        media_type="text/event-stream",
    )


@router.post(
    "/continue",
    response_model=ChatResponse,
    summary="Continue after tool results",
)
async def chat_continue(
    body: ContinueRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Feed tool results back to the model and continue generation.

    The provided tool results are sent as content blocks to the existing
    conversation, and the model generates a follow-up response.
    """
    tool_results = [_content_schema_to_domain(r) for r in body.tool_results]
    params = _params_schema_to_domain(body.params)

    result = await chat_service.process_message(
        conversation_id=body.conversation_id,
        user_content=tool_results,
        model=body.model,
        provider=body.provider,
        params=params,
    )
    return _chat_result_to_response(result)

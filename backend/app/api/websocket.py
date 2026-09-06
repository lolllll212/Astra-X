"""WebSocket endpoint for real-time chat.

Provides a bidirectional WebSocket at ``/ws`` that clients can use to
send messages and receive assistant responses over a persistent
connection.

The endpoint resolves services directly from ``app.state`` rather than
through the FastAPI ``Depends`` system, which has limited support for
request-scoped dependencies in WebSocket handlers.
"""

from __future__ import annotations

from contextlib import suppress

from fastapi import APIRouter, WebSocket

from app.api.routes.chat import _chat_result_to_response, _content_schema_to_domain
from app.api.schemas.chat import ChatRequest
from app.core.logging import get_logger
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.repositories.message_repository import MessageRepository
from app.database.repositories.usage_repository import UsageRepository
from app.database.session import create_session_factory, session_context
from app.llm.exceptions import LLMError
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and stream chat responses.

    Protocol
    --------
    1. Client connects.
    2. Client sends a JSON ``ChatRequest`` as the first message.
    3. Server processes the request and sends a JSON response with the
       assistant's reply.
    4. On completion, the server sends a ``"done"`` event and closes
       the connection.

    Future iterations will stream per-token events for lower latency.
    """
    await websocket.accept()
    logger.info("websocket.connected", client=websocket.client)

    try:
        raw = await websocket.receive_json()
        request = ChatRequest(**raw)
    except Exception as exc:
        await websocket.send_json({"event": "error", "message": f"Invalid request format: {exc}"})
        await websocket.close(code=1003)
        return

    content_domain = [_content_schema_to_domain(b) for b in request.content]

    engine = websocket.app.state.db_engine
    llm_router = websocket.app.state.llm_router
    factory = create_session_factory(engine)

    async with session_context(factory) as session:
        try:
            conv_repo = ConversationRepository(session)
            msg_repo = MessageRepository(session)
            usage_repo = UsageRepository(session)
            memory = MemoryService(msg_repo)

            chat_service = ChatService(
                conversation_repo=conv_repo,
                message_repo=msg_repo,
                usage_repo=usage_repo,
                llm_router=llm_router,
                memory_service=memory,
            )

            result = await chat_service.process_message_stream(
                conversation_id=request.conversation_id,
                user_content=content_domain,
                model=request.model,
                provider=request.provider,
            )
            response = _chat_result_to_response(result)
            await websocket.send_json(
                {
                    "event": "done",
                    "data": response.model_dump(mode="json"),
                }
            )
        except LLMError as exc:
            logger.warning("websocket.llm_error", error=str(exc))
            await websocket.send_json({"event": "error", "message": str(exc)})
        except Exception as exc:
            logger.exception("websocket.error", error=str(exc))
            await websocket.send_json({"event": "error", "message": "Internal server error."})
        finally:
            with suppress(Exception):
                await websocket.close()

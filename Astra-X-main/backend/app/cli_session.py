"""Session management, stats, and export/import for the ``astra`` CLI.

Mirrors opencode's ``session``/``stats``/``export``/``import`` commands,
backed by the Astra X database (conversations, messages, usage).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.settings import get_settings
from app.database.base import Base
from app.database.models.conversation import ConversationModel
from app.database.models.message import MessageModel
from app.database.models.usage import UsageModel
from app.database.repositories.conversation_repository import ConversationRepository
from app.database.repositories.message_repository import MessageRepository
from app.database.session import create_session_factory, session_context

if TYPE_CHECKING:
    from app.domain.message import Message

__all__ = [
    "cmd_session_export",
    "cmd_session_import",
    "cmd_session_list",
    "cmd_session_delete",
    "cmd_stats",
]


async def _engine_and_session():
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    return engine, factory


async def cmd_session_list(max_count: int | None = None, as_json: bool = False) -> int:
    """List conversations (sessions) from the database."""
    engine, factory = await _engine_and_session()
    try:
        async with session_context(factory) as session:
            repo = ConversationRepository(session)
            conversations = await repo.list_all()
    finally:
        await engine.dispose()

    conversations.sort(key=lambda c: c.updated_at or datetime.min, reverse=True)
    if max_count:
        conversations = conversations[:max_count]

    if not conversations:
        print("  No sessions found.")
        return 0

    if as_json:
        payload = []
        for c in conversations:
            payload.append({
                "id": c.id,
                "title": c.title,
                "status": c.status.value if hasattr(c.status, "value") else str(c.status),
                "message_count": c.message_count,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                "model": c.metadata.model,
                "provider": c.metadata.provider,
            })
        print(json.dumps(payload, indent=2, default=str))
        return 0

    header = f"  {'ID':<38} {'Title':<42} {'Msgs':<5} {'Model':<20} Updated"
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in conversations:
        title = (c.title or "(untitled)")[:40]
        model = (c.metadata.model or "-")[:19]
        updated = c.updated_at.isoformat() if c.updated_at else "-"
        print(f"  {c.id:<38} {title:<42} {c.message_count:<5} {model:<20} {updated}")
    print()
    return 0


async def cmd_session_delete(session_id: str) -> int:
    """Delete a conversation and its messages from the database."""
    engine, factory = await _engine_and_session()
    try:
        async with session_context(factory) as session:
            repo = ConversationRepository(session)
            conversation = await repo.get(session_id)
            if conversation is None:
                print(f"  Session '{session_id}' not found.")
                return 1
            # Delete messages first.
            msg_repo = MessageRepository(session)
            messages = await msg_repo.list_by_conversation(session_id, limit=100000)
            for msg in messages:
                await session.execute(
                    select(MessageModel).where(MessageModel.id == msg.id)
                )
                await session.delete(await session.get(MessageModel, msg.id))
            await repo.delete(session_id)
    finally:
        await engine.dispose()
    print(f"  Deleted session '{session_id}'.")
    return 0


async def cmd_session_export(session_id: str | None) -> int:
    """Export session data as JSON (to stdout)."""
    engine, factory = await _engine_and_session()
    try:
        async with session_context(factory) as session:
            conv_repo = ConversationRepository(session)
            msg_repo = MessageRepository(session)

            if session_id is not None:
                conversations = [c for c in [await conv_repo.get(session_id)] if c is not None]
            else:
                conversations = await conv_repo.list_all()
                conversations.sort(key=lambda c: c.updated_at or datetime.min, reverse=True)

            if not conversations:
                print("  No sessions found to export.", file=sys.stderr)
                return 1

            payload: list[dict[str, Any]] = []
            for conv in conversations:
                messages = await msg_repo.list_by_conversation(conv.id, limit=100000)
                payload.append({
                    "conversation": conv.model_dump(mode="json"),
                    "messages": [m.model_dump(mode="json") for m in messages],
                })
    finally:
        await engine.dispose()

    print(json.dumps(payload, indent=2, default=str))
    return 0


async def cmd_session_import(path: str) -> int:
    """Import session data from a JSON file."""
    import shutil
    from uuid import uuid4

    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        print(f"  File not found: {path}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"  Invalid JSON: {exc}")
        return 1

    # Accept either a single session dict or a list of sessions.
    entries = payload if isinstance(payload, list) else [payload]

    engine, factory = await _engine_and_session()
    created = 0
    try:
        async with session_context(factory) as session:
            from app.domain.conversation import Conversation
            from app.domain.message import Message
            from app.database.repositories.message_repository import MessageRepository

            conv_repo = ConversationRepository(session)
            msg_repo = MessageRepository(session)

            for entry in entries:
                conv_data = entry.get("conversation") or {}
                conv = Conversation.model_validate(conv_data)
                # Reassign id to avoid PK conflicts on re-import.
                conv = conv.model_copy(update={"id": str(uuid4())})
                await conv_repo.add(conv)

                for msg_data in entry.get("messages", []):
                    msg: Message = Message.model_validate(msg_data)
                    msg = msg.model_copy(update={
                        "id": str(uuid4()),
                        "conversation_id": conv.id,
                    })
                    await msg_repo.add(msg)
                created += 1
    finally:
        await engine.dispose()

    print(f"  Imported {created} session(s) from {path}.")
    return 0


async def cmd_stats(days: int | None) -> int:
    """Show token usage statistics across all sessions."""
    engine, factory = await _engine_and_session()
    try:
        async with session_context(factory) as session:
            stmt = select(UsageModel)
            if days:
                cutoff = datetime.now(UTC) - timedelta(days=days)
                # Usage rows carry no timestamp; approximate via message join.
                stmt = (
                    select(UsageModel)
                    .join(MessageModel, UsageModel.message_id == MessageModel.id)
                    .where(MessageModel.created_at >= cutoff)
                )
            result = await session.execute(stmt)
            usage_rows = result.scalars().all()
    finally:
        await engine.dispose()

    total_prompt = sum(u.prompt_tokens for u in usage_rows)
    total_completion = sum(u.completion_tokens for u in usage_rows)
    total_tokens = sum(u.total_tokens for u in usage_rows)
    total_cost = sum(u.cost_usd or 0.0 for u in usage_rows)

    print()
    print("  Token usage stats")
    print("  " + "-" * 40)
    print(f"  Sessions billed messages : {len(usage_rows)}")
    print(f"  Prompt tokens            : {total_prompt:,}")
    print(f"  Completion tokens        : {total_completion:,}")
    print(f"  Total tokens             : {total_tokens:,}")
    if total_cost:
        print(f"  Est. cost                : ${total_cost:.4f}")
    print()
    return 0
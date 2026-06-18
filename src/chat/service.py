"""Chat orchestration.

Per turn:
  1. Load conversation (project + persona).
  2. Persist the user message immediately (Postgres) + push to Redis history.
  3. Best-effort, time-boxed memory recall (pgvector ANN; returns short strings).
  4. Build [system persona] + [recalled context] + [recent history] + [new turn].
  5. Stream the reply (model chosen by persona — chat vs. coder), segmenting it
     into sentences as it goes so the frontend can drive the real-time reading
     highlighter and TTS from the same boundaries.
  6. Persist the reply; store the exchange in memory in the BACKGROUND.

v2 changes:
 • HISTORY now comes from Redis first (cache_get_history); Postgres is the
   fallback/rebuild path. This removes a DB round trip from the hot path and
   is the other half of the fix for "sloppy after bulk session usage".
 • Model routing: stream_chat(..., persona=conv.persona) so developer /
   architect / coder turns run on the coder model (qwen3-coder by default).
 • SENTENCE SEGMENTS: stream_reply now yields dicts {"type": "token", "t": ...}
   and {"type": "sentence", "text": ..., "index": N} as sentence boundaries are
   crossed. The router turns these into SSE frames; the frontend uses the
   sentence events to align the avatar's speech with on-screen highlighting
   (Real-Time Reading Progress Highlighter).
"""
import asyncio
import re
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.prompts import system_prompt
from src.core import redis_client
from src.core.database import SessionLocal
from src.core.llm import llm
from src.memory import service as memory
from src.memory.models import Conversation, Message

HISTORY_WINDOW = 8
RECALL_TIMEOUT = 0.5

_SENTENCE_RE = re.compile(r"[^.!?:\n]*[.!?:\n]+")


async def _recent_history_db(session: AsyncSession, conversation_id: str) -> list[dict]:
    rows = list(
        (
            await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(HISTORY_WINDOW)
            )
        ).scalars().all()
    )
    rows.reverse()
    return [{"role": m.role, "content": m.content} for m in rows]


async def _recent_history(session: AsyncSession, conversation_id: str) -> list[dict]:
    cached = await redis_client.cache_get_history(conversation_id)
    if cached is not None:
        return cached[-HISTORY_WINDOW:]
    history = await _recent_history_db(session, conversation_id)
    await redis_client.cache_set_history(conversation_id, history)
    return history


async def build_messages(
    session: AsyncSession, conv: Conversation, user_text: str
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt(conv.persona)}]

    try:
        chunks = await asyncio.wait_for(
            memory.recall(session, conv.project_id, user_text, k=3), RECALL_TIMEOUT
        )
    except Exception:
        chunks = []
    if chunks:
        recalled = "\n".join(f"- {c}" for c in chunks)
        messages.append(
            {
                "role": "system",
                "content": "Relevant context from earlier work on this project:\n"
                + recalled,
            }
        )

    for m in await _recent_history(session, conv.id):
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_text})
    return messages


async def _remember_bg(project_id: str, text: str) -> None:
    """Detached: store the exchange in memory without delaying the response."""
    try:
        async with SessionLocal() as s:
            await memory.remember(s, project_id, text, source="message")
    except Exception:
        pass


async def stream_reply(
    session: AsyncSession, conversation_id: str, user_text: str, remember: bool
) -> AsyncGenerator[dict, None]:
    conv = await session.get(Conversation, conversation_id)
    if conv is None:
        raise ValueError("Conversation not found")

    user_msg = Message(conversation_id=conv.id, role="user", content=user_text)
    session.add(user_msg)
    await session.commit()
    await redis_client.cache_append_messages(
        conv.id, {"role": "user", "content": user_text}
    )

    messages = await build_messages(session, conv, user_text)

    parts: list[str] = []
    buffer = ""
    sentence_index = 0

    async for piece in llm.stream_chat(messages, persona=conv.persona):
        parts.append(piece)
        buffer += piece
        yield {"type": "token", "t": piece}

        while True:
            m = _SENTENCE_RE.match(buffer)
            if not m:
                break
            sentence = m.group(0)
            if not sentence.strip():
                buffer = buffer[m.end():]
                continue
            yield {"type": "sentence", "index": sentence_index, "text": sentence}
            sentence_index += 1
            buffer = buffer[m.end():]

    if buffer.strip():
        yield {"type": "sentence", "index": sentence_index, "text": buffer}

    reply = "".join(parts).strip()
    assistant_msg = Message(conversation_id=conv.id, role="assistant", content=reply)
    session.add(assistant_msg)
    await session.commit()
    await redis_client.cache_append_messages(
        conv.id, {"role": "assistant", "content": reply}
    )

    if remember and reply:
        asyncio.create_task(_remember_bg(conv.project_id, f"Q: {user_text}\nA: {reply}"))

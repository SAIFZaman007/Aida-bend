"""Chat orchestration.

Per turn:
  1. Load the conversation (project + persona).
  2. Read PRIOR history (Redis first, Postgres fallback).
  3. Persist the user message (Postgres) and push it to the Redis cache.
  4. Best-effort, time-boxed, relevance-filtered memory recall.
  5. Build [system persona] + [recalled notes] + [prior history] + [new turn].
  6. Stream the reply, segmenting it into sentences (Bengali + English aware) so
     the frontend can speak and highlight from the SAME boundaries.
  7. Persist the reply — even a partial one if the client disconnects or the
     provider fails mid-stream — and store the exchange in memory in the
     BACKGROUND.

v2.1 fixes (all reproduced with tests/test_history.py):
 • The new user message used to be sent to the model TWICE per turn (it was
   saved + cached before history was read, then appended again).
 • After a Redis restart or the 6 h TTL, the whole earlier conversation was
   dropped (a cold cache was "rebuilt" from the single just-appended message).
 • A dropped connection left a user turn with no reply, producing consecutive
   user turns that providers reject or misread; partial replies are now saved.
 • Sentence segmentation lives in core/text.py (understands "।", decimals,
   times, URLs, code fences) and emits `code` flags so code is never read aloud.
 • Recalled memory is filtered by relevance and presented as unverified notes.
"""
import asyncio
import logging
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.prompts import system_prompt
from src.core import redis_client
from src.core.config import settings
from src.core.database import SessionLocal
from src.core.errors import AidaError
from src.core.llm import llm
from src.core.text import SentenceSegmenter, detect_language
from src.memory import service as memory
from src.memory.models import Conversation, Message

log = logging.getLogger("aida.chat")

HISTORY_WINDOW = 20  # messages sent as context (10 exchanges) — long live conversations stay coherent

# Keep strong references so fire-and-forget tasks are not garbage-collected.
_bg_tasks: set[asyncio.Task] = set()

_BN_NOTE = (
    "The user's latest message is written in Bengali script: reply in natural "
    "Bengali script."
)


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
    """PRIOR turns only. Must be called BEFORE the new user message is saved."""
    cached = await redis_client.cache_get_history(conversation_id)
    if cached is not None:
        return cached[-HISTORY_WINDOW:]
    history = await _recent_history_db(session, conversation_id)
    await redis_client.cache_set_history(conversation_id, history)
    return history


async def build_messages(
    session: AsyncSession, conv: Conversation, user_text: str, history: list[dict]
) -> list[dict]:
    persona_prompt = system_prompt(conv.persona)
    if detect_language(user_text) == "bn":
        persona_prompt += " " + _BN_NOTE
    messages: list[dict] = [{"role": "system", "content": persona_prompt}]

    try:
        chunks = await asyncio.wait_for(
            memory.recall(session, conv.project_id, user_text),
            settings.memory_recall_timeout,
        )
    except Exception:
        chunks = []
    if chunks:
        recalled = "\n".join(f"- {c}" for c in chunks)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Notes from earlier work on this project. They may be outdated "
                    "or wrong: do not present them as verified fact, use them only "
                    "if clearly relevant, and prefer the user's latest message if "
                    "they conflict:\n" + recalled
                ),
            }
        )

    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


async def _remember_bg(project_id: str, text: str) -> None:
    """Detached: store the exchange in memory without delaying the response."""
    try:
        async with SessionLocal() as s:
            await memory.remember(s, project_id, text, source="message")
    except Exception:
        log.debug("background memory write failed", exc_info=True)


async def _persist_assistant(conversation_id: str, reply: str) -> None:
    """Save the assistant turn on a FRESH session (the request session may be
    closing when a client disconnects) and mirror it into the Redis cache."""
    async with SessionLocal() as s:
        s.add(Message(conversation_id=conversation_id, role="assistant", content=reply))
        await s.commit()
    await redis_client.cache_append_messages(
        conversation_id, {"role": "assistant", "content": reply}
    )


async def stream_reply(
    session: AsyncSession, conversation_id: str, user_text: str, remember: bool
) -> AsyncGenerator[dict, None]:
    conv = await session.get(Conversation, conversation_id)
    if conv is None:
        raise AidaError("not_found", conversation_id)
    conv_id, project_id, persona = conv.id, conv.project_id, conv.persona

    # 1) PRIOR history first — never include the message we are about to send.
    history = await _recent_history(session, conv_id)

    # 2) Persist the user turn.
    session.add(Message(conversation_id=conv_id, role="user", content=user_text))
    await session.commit()
    await redis_client.cache_append_messages(
        conv_id, {"role": "user", "content": user_text}
    )

    messages = await build_messages(session, conv, user_text, history)

    parts: list[str] = []
    segmenter = SentenceSegmenter()
    sentence_index = 0
    completed = False

    def sentence_events(segments) -> list[dict]:
        nonlocal sentence_index
        out = []
        for seg in segments:
            out.append(
                {"type": "sentence", "index": sentence_index, "text": seg.text, "code": seg.code}
            )
            sentence_index += 1
        return out

    try:
        async for piece in llm.stream_chat(messages, persona=persona):
            parts.append(piece)
            yield {"type": "token", "t": piece}
            for evt in sentence_events(segmenter.feed(piece)):
                yield evt

        for evt in sentence_events(segmenter.flush()):
            yield evt
        completed = True
    finally:
        # Runs on success, provider error AND client disconnect (cancellation).
        reply = "".join(parts).strip()
        if reply:
            try:
                await asyncio.shield(_persist_assistant(conv_id, reply))
            except Exception:
                log.exception("failed to persist assistant reply")

    if remember and completed and reply:
        task = asyncio.create_task(_remember_bg(project_id, f"Q: {user_text}\nA: {reply}"))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

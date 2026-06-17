"""Redis 8 client — chat history cache + pub/sub backbone for SSE.

Two responsibilities:

1. HISTORY CACHE
   `build_messages()` in chat/service.py used to hit Postgres for the last
   N messages on every single turn. We now keep a rolling list of the most
   recent messages per conversation in Redis (capped, TTL'd), so steady-state
   chat avoids a DB round trip on the hot path. Postgres remains the source
   of truth — the cache is refreshed on every write and lazily rebuilt on a
   miss.

2. SSE FAN-OUT (pub/sub)
   Streaming chat now publishes each token to a per-conversation Redis
   channel in addition to yielding it directly to the requesting connection.
   This lets multiple tabs/clients (or the desktop + a future mobile client)
   subscribe to the same in-flight response, and lets the highlighter's
   timing-mark stream be delivered over the same transport without a second
   HTTP round trip.

Both features are best-effort: if Redis is unreachable, the app falls back
to direct DB reads and single-consumer streaming (logged, never fatal).
"""
import json
import logging
from collections.abc import AsyncIterator

from redis import asyncio as redis_asyncio

from src.core.config import settings

log = logging.getLogger("aida.redis")

_redis: "redis_asyncio.Redis | None" = None


def get_redis() -> "redis_asyncio.Redis":
    global _redis
    if _redis is None:
        _redis = redis_asyncio.from_url(
            settings.redis_url, encoding="utf-8", decode_responses=True
        )
    return _redis


async def aclose() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


# ----------------------------------------------------------------- History cache
_HIST_KEY = "aida:hist:{conversation_id}"
_HIST_MAX = 40  # keep more than HISTORY_WINDOW so trimming server-side is cheap


async def cache_get_history(conversation_id: str) -> list[dict] | None:
    try:
        r = get_redis()
        raw = await r.lrange(_HIST_KEY.format(conversation_id=conversation_id), 0, -1)
        if not raw:
            return None
        return [json.loads(x) for x in raw]
    except Exception:
        log.debug("redis history get failed; falling back to DB", exc_info=True)
        return None


async def cache_append_messages(conversation_id: str, *messages: dict) -> None:
    if not messages:
        return
    try:
        r = get_redis()
        key = _HIST_KEY.format(conversation_id=conversation_id)
        pipe = r.pipeline()
        for m in messages:
            pipe.rpush(key, json.dumps(m))
        pipe.ltrim(key, -_HIST_MAX, -1)
        pipe.expire(key, settings.redis_history_ttl)
        await pipe.execute()
    except Exception:
        log.debug("redis history append failed (non-fatal)", exc_info=True)


async def cache_set_history(conversation_id: str, messages: list[dict]) -> None:
    try:
        r = get_redis()
        key = _HIST_KEY.format(conversation_id=conversation_id)
        pipe = r.pipeline()
        pipe.delete(key)
        for m in messages[-_HIST_MAX:]:
            pipe.rpush(key, json.dumps(m))
        pipe.expire(key, settings.redis_history_ttl)
        await pipe.execute()
    except Exception:
        log.debug("redis history set failed (non-fatal)", exc_info=True)


async def cache_invalidate(conversation_id: str) -> None:
    try:
        r = get_redis()
        await r.delete(_HIST_KEY.format(conversation_id=conversation_id))
    except Exception:
        pass


# ----------------------------------------------------------------- SSE pub/sub
def stream_channel(conversation_id: str) -> str:
    return f"{settings.redis_stream_channel_prefix}{conversation_id}"


async def publish_event(conversation_id: str, payload: dict) -> None:
    try:
        r = get_redis()
        await r.publish(stream_channel(conversation_id), json.dumps(payload))
    except Exception:
        log.debug("redis publish failed (non-fatal)", exc_info=True)


async def subscribe(conversation_id: str) -> AsyncIterator[dict]:
    """Yield published events for a conversation until [DONE] or cancellation."""
    r = get_redis()
    pubsub = r.pubsub()
    await pubsub.subscribe(stream_channel(conversation_id))
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                data = json.loads(message["data"])
            except Exception:
                continue
            yield data
            if data.get("done"):
                break
    finally:
        await pubsub.unsubscribe(stream_channel(conversation_id))
        await pubsub.aclose()

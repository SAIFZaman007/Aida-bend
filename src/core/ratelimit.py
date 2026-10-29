"""Abuse protection for a public, login-free API.

A public endpoint that holds a paid API key is a spending risk, so there are
two layers (both env-tunable, both disable-able):

  1. PER-CLIENT rate limit — fixed window per client IP and route group.
  2. GLOBAL DAILY CAP — a ceiling on total calls per day per route group, which
     bounds worst-case spend even if many IPs are used.

Counters live in Redis (shared across workers/restarts). If Redis is down they
fall back to per-process memory, so protection degrades instead of vanishing.

Client IP comes from `request.client.host`; behind a reverse proxy run uvicorn
with --proxy-headers (the Dockerfile does) so that is the real client address.
"""
import logging
import time
from datetime import datetime, timezone

from fastapi import HTTPException, Request

from src.core import redis_client
from src.core.config import settings
from src.core.errors import http_detail, request_lang

log = logging.getLogger("aida.ratelimit")


class _MemoryCounters:
    def __init__(self) -> None:
        self._d: dict[str, tuple[int, float]] = {}

    def hit(self, key: str, window: int) -> tuple[int, int]:
        now = time.monotonic()
        count, expires = self._d.get(key, (0, 0.0))
        if now >= expires:
            count, expires = 0, now + window
        count += 1
        self._d[key] = (count, expires)
        if len(self._d) > 10_000:  # bound memory
            self._d = {k: v for k, v in self._d.items() if v[1] > now}
        return count, max(1, int(expires - now))


_memory = _MemoryCounters()


async def hit(key: str, window: int) -> tuple[int, int]:
    """Increment `key`; return (count_in_window, seconds_until_reset)."""
    try:
        r = redis_client.get_redis()
        count = await r.incr(key)
        ttl = await r.ttl(key)
        if count == 1 or ttl < 0:  # (re)arm expiry; guards a crash between calls
            await r.expire(key, window)
            ttl = window
        return int(count), int(ttl)
    except Exception:
        log.debug("redis unavailable for rate limiting; using memory", exc_info=True)
        return _memory.hit(key, window)


def _reject(code: str, request: Request, retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=http_detail(code, request_lang(request)),
        headers={"Retry-After": str(retry_after)},
    )


async def check(group: str, ip: str, per_min_attr: str, daily_attr: str | None = None) -> tuple[str, int] | None:
    """Count one call. Returns (error_code, retry_after_seconds) when the client
    is over its per-minute limit or the global daily cap, else None. Shared by
    the HTTP dependency below and by the WebSocket speech endpoint (which counts
    one call per utterance)."""
    if not settings.rate_limit_enabled:
        return None
    per_min = int(getattr(settings, per_min_attr))
    if per_min > 0:
        count, ttl = await hit(f"aida:rl:{group}:{ip}", 60)
        if count > per_min:
            return "rate_limited", ttl
    cap = int(getattr(settings, daily_attr)) if daily_attr else 0
    if cap > 0:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        count, ttl = await hit(f"aida:cap:{group}:{day}", 90_000)
        if count > cap:
            log.warning("daily cap reached for %s (%d)", group, cap)
            return "daily_cap", ttl
    return None


def limit(group: str, per_min_attr: str, daily_attr: str | None = None):
    """FastAPI dependency factory. `*_attr` are Settings field names, read at
    call time so changing env/config needs no code change."""

    async def dependency(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        verdict = await check(group, ip, per_min_attr, daily_attr)
        if verdict:
            raise _reject(verdict[0], request, verdict[1])

    return dependency


limit_chat = limit("chat", "rate_chat_per_min", "daily_chat_cap")
limit_stt = limit("stt", "rate_stt_per_min", "daily_stt_cap")
limit_tts = limit("tts", "rate_tts_per_min", "daily_tts_cap")
limit_crud = limit("crud", "rate_crud_per_min")

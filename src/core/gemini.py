"""Shared Google Gemini client + exception mapping.

One lazily-created `genai.Client` (connection pooled) serves chat, embeddings,
speech-to-text and text-to-speech. Provider/SDK exceptions are translated to
`AidaError` codes so callers never leak raw error text to users.
"""
import asyncio
import logging

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.core.config import settings
from src.core.errors import AidaError

log = logging.getLogger("aida.gemini")

_client: "genai.Client | None" = None

# Per-read network timeout (ms). Our own first-token / idle timeouts in llm.py
# are the real streaming guards; this just stops a dead socket hanging forever.
_HTTP_TIMEOUT_MS = 60_000


def get_client() -> "genai.Client":
    global _client
    if not settings.gemini_api_key:
        raise AidaError("not_configured", "GEMINI_API_KEY is empty")
    if _client is None:
        _client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=_HTTP_TIMEOUT_MS),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aio.aclose()
        except Exception:
            pass
        _client = None


def thinking_config(level: str) -> "types.ThinkingConfig | None":
    level = (level or "").strip().lower()
    if not level:
        return None
    return types.ThinkingConfig(thinking_level=level)  # validated by the SDK enum


def map_exception(exc: BaseException) -> AidaError:
    """Translate any provider/network exception into a user-safe AidaError."""
    if isinstance(exc, AidaError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return AidaError("timeout", repr(exc))
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code == 429:
            return AidaError("rate_limited", repr(exc))
        if code in (401, 403):
            # Bad/expired key or no access: an operator problem, not the user's.
            log.error("Gemini rejected credentials (HTTP %s): %s", code, exc)
            return AidaError("not_configured", repr(exc))
        if isinstance(code, int) and code >= 500:
            return AidaError("unavailable", repr(exc))
        if code == 404:
            log.error("Gemini model not found — check GEMINI_*_MODEL: %s", exc)
            return AidaError("unavailable", repr(exc))
        return AidaError("internal", repr(exc))
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return AidaError("unavailable", repr(exc))
    # Optional Anthropic adapter: map its exceptions by class name so this
    # module does not import the anthropic SDK.
    name = type(exc).__name__
    if name == "RateLimitError":
        return AidaError("rate_limited", repr(exc))
    if name == "APITimeoutError":
        return AidaError("timeout", repr(exc))
    if name in ("APIConnectionError", "InternalServerError", "OverloadedError"):
        return AidaError("unavailable", repr(exc))
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return AidaError("not_configured", repr(exc))
    return AidaError("internal", repr(exc))


async def call_with_retry(make_call, *, timeout: float, attempts: int = 2):
    """Run `make_call()` (a coroutine factory) with a timeout and one retry on
    transient errors. For NON-streaming calls (STT, TTS) only."""
    last: AidaError | None = None
    for i in range(attempts):
        try:
            return await asyncio.wait_for(make_call(), timeout)
        except Exception as exc:
            last = map_exception(exc)
            if not last.retryable or i == attempts - 1:
                raise last from exc
            await asyncio.sleep(0.5 * (2**i))
    raise last or AidaError("internal")

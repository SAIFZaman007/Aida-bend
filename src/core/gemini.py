"""Shared Google Gemini client + exception mapping.

One lazily-created `genai.Client` (connection pooled) serves chat, embeddings,
speech-to-text and text-to-speech. Provider/SDK exceptions are translated to
`AidaError` codes so callers never leak raw error text to users.
"""
import asyncio
import logging
import math
import re
import time

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


_tts_client: "genai.Client | None" = None


def tts_http_options() -> "types.HttpOptions":
    """HTTP options of the voice client (see `get_tts_client`)."""
    return types.HttpOptions(
        timeout=_HTTP_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(
            attempts=1,
            http_status_codes=[500, 502, 503, 504],
            initial_delay=0.3,
            max_delay=1.0,
        ),
    )


def get_tts_client() -> "genai.Client":
    """Client for the voice (Interactions) API.

    Same key and pool settings, but the SDK's own retry policy is narrowed. By default
    the Interactions client retries EVERY 429 and 5xx up to three times with backoff
    (0.5 s, 1 s, 2 s ... measured), silently, inside one call. When Google answers
    "quota exceeded" that turns a 0.3 s refusal into several seconds of waiting AND
    multiplies the requests that count against the quota. Here only transient server
    errors (500/502/503/504) and connection errors get ONE quick retry; a 429 is
    returned at once so the caller can switch to the browser voice immediately.
    (`attempts=0` is not usable: the SDK silently raises it to 1.)
    """
    global _tts_client
    if not settings.gemini_api_key:
        raise AidaError("not_configured", "GEMINI_API_KEY is empty")
    if _tts_client is None:
        _tts_client = genai.Client(api_key=settings.gemini_api_key, http_options=tts_http_options())
    return _tts_client


async def aclose() -> None:
    global _client, _tts_client
    for c in (_client, _tts_client):
        if c is not None:
            try:
                await c.aio.aclose()
            except Exception:
                pass
    _client = _tts_client = None


# Models that rejected our thinking_level at runtime. Allowed levels differ per
# model (e.g. gemini-3.8-flash has no "minimal"), so instead of failing every
# request we drop thinking_config for that model once and keep working.
_NO_THINKING: set[str] = set()


# Circuit breaker. When a model stalls, is overloaded (503) or out of quota (429),
# every following request would wait the full timeout again. Instead the model is
# remembered as unhealthy for `llm_cooldown_seconds` and a healthy backup is
# tried first. Per process, in memory: a restart simply forgets it.
_UNHEALTHY: dict[str, float] = {}


def mark_unhealthy(model: str) -> None:
    _UNHEALTHY[model] = time.monotonic() + settings.llm_cooldown_seconds


def mark_healthy(model: str) -> None:
    _UNHEALTHY.pop(model, None)


def is_unhealthy(model: str) -> bool:
    until = _UNHEALTHY.get(model)
    if until is None:
        return False
    if time.monotonic() >= until:
        _UNHEALTHY.pop(model, None)
        return False
    return True


def order_models(models: list[str]) -> list[str]:
    """Healthy models first, otherwise the configured order (stable)."""
    return sorted(models, key=is_unhealthy)


def thinking_config(level: str) -> "types.ThinkingConfig | None":
    level = (level or "").strip().lower()
    if not level:
        return None
    try:
        return types.ThinkingConfig(thinking_level=level)  # validated by the SDK enum
    except Exception:
        log.error("Ignoring invalid thinking level %r (use low|medium|high)", level)
        return None


def thinking_config_for(model: str, level: str) -> "types.ThinkingConfig | None":
    """thinking_config unless `model` already rejected it."""
    return None if model in _NO_THINKING else thinking_config(level)


def is_thinking_rejected(exc: BaseException) -> bool:
    """True if the API refused the thinking level for this model (HTTP 400)."""
    return (
        isinstance(exc, genai_errors.APIError)
        and getattr(exc, "code", None) == 400
        and "thinking" in str(exc).lower()
    )


def mark_no_thinking(model: str) -> bool:
    """Remember that `model` rejects thinking_config. True if newly recorded."""
    if model in _NO_THINKING:
        return False
    _NO_THINKING.add(model)
    log.warning(
        "Model %r rejected the thinking level; continuing WITHOUT thinking_config. "
        "Fix GEMINI_THINKING_LEVEL / GEMINI_STT_THINKING_LEVEL (valid for "
        "gemini-3.8-flash: low, medium, high).", model)
    return True


_RETRY_IN = re.compile(r"retry(?:\s+in|Delay\W+)\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.I)
_PER_DAY = re.compile(r"per\s*day|perday|daily", re.I)


def quota_retry_after(exc: BaseException) -> int:
    """Seconds Google asks us to wait after a 429 (0 = it did not say).

    Looks at the `Retry-After` header, then at the text ("Please retry in 12.5s",
    RetryInfo `retryDelay: "12s"`). A per-DAY quota says nothing useful, so it is
    reported as a long wait (the caller clamps it)."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    try:
        hv = headers.get("retry-after") if headers is not None else None
        if hv and str(hv).strip().isdigit():
            return int(str(hv).strip())
    except Exception:
        pass
    text = f"{exc!r} {getattr(exc, 'message', '')} {getattr(exc, 'body', '')}"
    m = _RETRY_IN.search(text)
    if m:
        return max(1, math.ceil(float(m.group(1))))
    if _PER_DAY.search(text):
        return 900
    return 0


def map_exception(exc: BaseException) -> AidaError:
    """Translate any provider/network exception into a user-safe AidaError."""
    if isinstance(exc, AidaError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return AidaError("timeout", repr(exc))
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code == 429:
            return AidaError("rate_limited", repr(exc), retry_after=quota_retry_after(exc))
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
    # The Interactions client (voice) raises its own error classes with `status_code`.
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        if sc == 429:
            return AidaError("rate_limited", repr(exc), retry_after=quota_retry_after(exc))
        if sc in (401, 403):
            log.error("Gemini rejected credentials (HTTP %s): %s", sc, exc)
            return AidaError("not_configured", repr(exc))
        if sc >= 500 or sc == 404:
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


async def call_with_retry(make_call, *, timeout: float, attempts: int = 2,
                          fail_fast: tuple[str, ...] = ()):
    """Run `make_call()` (a coroutine factory) with a timeout and one retry on
    transient errors. For NON-streaming calls (STT, TTS) only. Error codes listed in
    `fail_fast` (e.g. "rate_limited": asking again at once cannot help) are raised
    without a retry."""
    last: AidaError | None = None
    for i in range(attempts):
        try:
            return await asyncio.wait_for(make_call(), timeout)
        except Exception as exc:
            last = map_exception(exc)
            if not last.retryable or i == attempts - 1 or last.code in fail_fast:
                raise last from exc
            await asyncio.sleep(0.5 * (2**i))
    raise last or AidaError("internal")

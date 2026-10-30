"""Voice quota handling (v2.5.1): a Google 429 must cost ONE fast round trip, never seconds.

Measured with the real SDK against a fake HTTP server: the Interactions client retries every
429 three times by default (4 requests, ~3 s). `gemini.tts_http_options()` narrows that; these
tests pin the behaviour so an SDK upgrade cannot silently bring the delay back."""
import asyncio
import time
import types

import httpx
import pytest
from google import genai

from src.core import gemini
from src.core.config import settings
from src.core.errors import AidaError
from src.voice import tts

from tests.test_live_voice import api_error, collect_stream, event, pcm, tts_stream_client

QUOTA_BODY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                        "message": "You exceeded your current quota. Please retry in 12.5s."}}


# ------------------------------------------------ the SDK really stops retrying a 429
async def _calls_for(status: int, options) -> int:
    seen = []

    def handler(request):
        seen.append(request)
        body = QUOTA_BODY if status == 429 else {"error": {"code": status, "message": "x", "status": "UNAVAILABLE"}}
        return httpx.Response(status, json=body)

    opts = options.model_copy(update={"async_client_args": {"transport": httpx.MockTransport(handler)}})
    client = genai.Client(api_key="k", http_options=opts)
    with pytest.raises(Exception):
        await client.aio.interactions.create(
            model="m", input=[{"type": "user_input", "content": [{"type": "text", "text": "hi"}]}],
            response_format={"type": "audio"}, stream=True)
    return len(seen)


async def test_quota_error_is_asked_once_not_retried_by_the_sdk():
    t0 = time.monotonic()
    assert await _calls_for(429, gemini.tts_http_options()) == 1
    assert time.monotonic() - t0 < 1.0


async def test_transient_server_error_still_gets_one_quick_retry():
    assert await _calls_for(503, gemini.tts_http_options()) == 2


async def test_default_sdk_policy_would_have_retried_the_quota_error():
    """Documents WHY the options exist; if the SDK default ever changes this tells us."""
    from google.genai import types as gt
    assert await _calls_for(429, gt.HttpOptions(timeout=60_000)) > 1


# ------------------------------------------------ parsing what Google says
@pytest.mark.parametrize("message,expected", [
    ("You exceeded your current quota. Please retry in 12.5s.", 13),
    ('{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}', 30),
    ("Quota exceeded for metric ...PerDay... limit", 900),
    ("something else entirely", 0),
])
def test_quota_retry_after_parsing(message, expected):
    assert gemini.quota_retry_after(Exception(message)) == expected


def test_rate_limited_error_carries_retry_after():
    err = gemini.map_exception(api_error(429))
    assert err.code == "rate_limited" and err.retry_after == 0
    sdk_style = type("RateLimitError", (Exception,), {})("retry in 7s")
    sdk_style.status_code = 429
    assert gemini.map_exception(sdk_style).retry_after == 7


# ------------------------------------------------ the quota gate
async def test_quota_error_falls_to_backup_at_once_then_gates_both_models(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "lite-tts")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([api_error(429)], calls))
    with pytest.raises(AidaError) as exc:
        await collect_stream()
    assert exc.value.code == "rate_limited"
    assert exc.value.retry_after >= settings.tts_quota_cooldown
    assert sorted(c["model"] for c in calls) == sorted([settings.gemini_tts_model, "lite-tts"])

    # Both models are now paused: the next clip is refused WITHOUT calling Google.
    before = len(calls)
    t0 = time.monotonic()
    with pytest.raises(AidaError) as again:
        await collect_stream()
    assert again.value.code == "rate_limited" and again.value.retry_after >= 1
    assert len(calls) == before and time.monotonic() - t0 < 0.05


async def test_gate_leaves_the_healthy_model_in_use(monkeypatch):
    calls = []
    good = pcm(240)
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "lite-tts")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([api_error(429), [event(good)]], calls))
    out, _ = await collect_stream()
    assert out == [good]
    assert tts.quota_remaining(settings.gemini_tts_model) > 0 and tts.quota_remaining("lite-tts") == 0
    await collect_stream()   # main is skipped from now on
    assert [c["model"] for c in calls][2:] == ["lite-tts"]


async def test_gate_expires_and_the_model_is_tried_again(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([api_error(429), [event(pcm(240))]], calls))
    with pytest.raises(AidaError):
        await collect_stream()
    tts._QUOTA_UNTIL[settings.gemini_tts_model] = time.monotonic() - 1     # cool-down is over
    out, _ = await collect_stream()
    assert out and len(calls) == 2
    assert tts.quota_remaining(settings.gemini_tts_model) == 0


async def test_gate_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "tts_quota_cooldown", 0)
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([api_error(429)], calls))
    for _ in range(2):
        with pytest.raises(AidaError):
            await collect_stream()
    assert len(calls) == 2


def test_cooldown_is_clamped(monkeypatch):
    err = AidaError("rate_limited", "x", retry_after=2)
    tts.note_quota("m", err)
    assert err.retry_after == settings.tts_quota_cooldown            # floor
    big = AidaError("rate_limited", "x", retry_after=99999)
    tts.note_quota("m2", big)
    assert big.retry_after == settings.tts_quota_cooldown_max        # ceiling


def test_quota_logged_once_per_pause(monkeypatch, caplog):
    caplog.set_level("WARNING", logger="aida.tts")
    for _ in range(5):
        tts.note_quota("m", AidaError("rate_limited", "quota text from Google"))
    msgs = [r.getMessage() for r in caplog.records if "quota exceeded" in r.getMessage()]
    assert len(msgs) == 1 and "quota text from Google" in msgs[0]


# ------------------------------------------------ the HTTP answer
async def test_speak_stream_answers_429_with_retry_after(monkeypatch):
    import app as app_module

    async def boom(text, gender, info):
        raise AidaError("rate_limited", "paused", retry_after=17)
        yield b""  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(tts, "synthesize_stream", boom)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://t") as c:
        r = await c.post("/api/voice/speak_stream", json={"text": "Hello there.", "gender": "male"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "17"
    assert r.json()["detail"]["code"] == "rate_limited"

"""Google Cloud TTS (Chirp 3 HD) as the first-choice voice, Gemini behind it."""
import asyncio
import base64
import io
import math
import struct
import wave

import httpx
import pytest

from src.core.config import Settings, settings
from src.core.errors import AidaError
from src.voice import gcloud_tts, tts

G = gcloud_tts.GCLOUD


def wav(seconds=1.0, rate=24000, tone=True) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(i / 20)) if tone else 0) for i in range(n)))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def on(monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "google_cloud")
    monkeypatch.setattr(settings, "google_tts_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_tts_model", "main-tts")
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "lite-tts")
    monkeypatch.setattr(settings, "google_tts_voice_female_bn", "")
    monkeypatch.setattr(settings, "google_tts_voice_male_bn", "")


def serve(monkeypatch, handler):
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    monkeypatch.setattr(gcloud_tts, "_client", client)
    return seen


def ok(request):
    return httpx.Response(200, json={"audioContent": base64.b64encode(wav(1.0)).decode()})


# ------------------------------------------------------------------ settings
def test_default_is_gemini_and_a_key_is_needed(monkeypatch):
    assert Settings(_env_file=None).tts_provider == "gemini"
    assert not Settings(_env_file=None).tts_google_cloud
    monkeypatch.setattr(settings, "google_tts_api_key", "")
    assert not gcloud_tts.enabled()                      # selected but no key -> Gemini only
    assert tts.tts_models() == ["main-tts", "lite-tts"]


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "azure")
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_cloud_voice_goes_first_then_the_gemini_models():
    assert tts.tts_models() == [G, "main-tts", "lite-tts"]


# --------------------------------------------------------------- voice names
def test_voice_names_follow_language_and_gender(monkeypatch):
    assert gcloud_tts.voice_for("female", "bn") == ("bn-IN", "bn-IN-Chirp3-HD-Kore")
    assert gcloud_tts.voice_for("male", "en") == ("en-US", "en-US-Chirp3-HD-Charon")
    monkeypatch.setattr(settings, "google_tts_voice_female_bn", "Aoede")
    assert gcloud_tts.voice_for("female", "bn")[1] == "bn-IN-Chirp3-HD-Aoede"
    monkeypatch.setattr(settings, "google_tts_voice_male_bn", "bn-IN-Chirp3-HD-Puck")
    assert gcloud_tts.voice_for("male", "bn") == ("bn-IN", "bn-IN-Chirp3-HD-Puck")


# ------------------------------------------------------------------- request
async def test_request_shape_and_key_in_header_not_url(monkeypatch):
    seen = serve(monkeypatch, ok)
    pcm, rate = await gcloud_tts.fetch("আমি ভালো আছি।", "female", "bn")
    req = seen[0]
    assert req.headers["x-goog-api-key"] == "test-key" and "test-key" not in str(req.url)
    import json
    body = json.loads(req.content)
    assert body["voice"] == {"languageCode": "bn-IN", "name": "bn-IN-Chirp3-HD-Kore"}
    assert body["audioConfig"] == {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}
    assert body["input"] == {"text": "আমি ভালো আছি।"} and rate == 24000 and len(pcm) == 48000


async def test_unary_returns_a_valid_wav(monkeypatch):
    serve(monkeypatch, ok)
    out, duration = await tts.synthesize("Hello there, how are you today?", "male")
    assert out[:4] == b"RIFF" and 0.9 < duration < 1.1


async def test_stream_yields_pcm_and_reports_the_model(monkeypatch):
    serve(monkeypatch, ok)
    info = {}
    chunks = [c async for c in tts.synthesize_stream("Hello there, how are you today?", "female", info)]
    assert b"".join(chunks) and all(len(c) % 2 == 0 for c in chunks)
    assert info["model"] == G and info["sample_rate"] == 24000 and not info.get("hedged")


# ------------------------------------------------------------------ failures
async def test_quota_closes_the_gate_and_gemini_takes_over(monkeypatch):
    serve(monkeypatch, lambda r: httpx.Response(
        429, headers={"retry-after": "30"}, json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}}))
    got = []

    async def fake(model, text, voice, info):
        got.append(model)
        info["sample_rate"] = 24000
        yield b"\x01\x00" * 100

    real = tts._stream_model

    async def route(model, text, voice, info):
        async for c in (real if model == G else fake)(model, text, voice, info):
            yield c

    monkeypatch.setattr(tts, "_stream_model", route)
    out = [c async for c in tts.synthesize_stream("Hello there, how are you today?", "female", {})]
    assert out and got and got[0] == "main-tts"          # Gemini answered
    assert tts.quota_remaining(G) > 25                  # Cloud TTS left alone meanwhile
    assert G not in tts.tts_models()


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_config_errors_fall_back_instead_of_blocking(monkeypatch, status):
    serve(monkeypatch, lambda r: httpx.Response(status, json={"error": {"status": "X", "message": "bad"}}))
    with pytest.raises(AidaError) as e:
        await gcloud_tts.fetch("Hello there friend.", "female", "en")
    assert e.value.code == "unavailable" and e.value.retryable   # the chain moves on to Gemini


async def test_server_error_and_network_error_are_unavailable(monkeypatch):
    serve(monkeypatch, lambda r: httpx.Response(503, json={}))
    with pytest.raises(AidaError) as e:
        await gcloud_tts.fetch("Hello there friend.", "female", "en")
    assert e.value.code == "unavailable"

    def boom(request):
        raise httpx.ConnectError("down")
    serve(monkeypatch, boom)
    with pytest.raises(AidaError) as e2:
        await gcloud_tts.fetch("Hello there friend.", "female", "en")
    assert e2.value.code == "unavailable"


async def test_empty_audio_is_an_error(monkeypatch):
    serve(monkeypatch, lambda r: httpx.Response(200, json={"audioContent": base64.b64encode(wav(0.0)).decode()}))
    with pytest.raises(AidaError):
        await gcloud_tts.fetch("Hello there friend.", "female", "en")


async def test_per_day_quota_waits_long(monkeypatch):
    serve(monkeypatch, lambda r: httpx.Response(
        429, json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded per day"}}))
    with pytest.raises(AidaError) as e:
        await gcloud_tts.fetch("Hello there friend.", "female", "en")
    assert e.value.code == "rate_limited" and e.value.retry_after == 900


async def test_gemini_hedging_is_off_while_cloud_tts_leads(monkeypatch):
    monkeypatch.setattr(settings, "tts_hedge_after", 0.01)
    started = []

    async def fake(model, text, voice, info):
        started.append(model)
        if model == G:
            await asyncio.sleep(0.1)       # slow, but must NOT wake the Gemini models
        info["sample_rate"] = 24000
        yield b"\x01\x00" * 10

    monkeypatch.setattr(tts, "_stream_model", fake)
    out = [c async for c in tts.synthesize_stream("Hello there, how are you today?", "female", {})]
    assert out and started == [G]

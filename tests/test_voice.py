import array
import base64
import io
import math
import types
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from src.core import gemini
from src.core.config import settings
from src.core.errors import AidaError
from src.voice import stt, tts
from src.voice.router import build_marks


def make_wav(seconds=1.0, amp=0.3, rate=24000, tone=True) -> bytes:
    n = int(seconds * rate)
    samples = array.array("h", (int(amp * 32767 * math.sin(2 * math.pi * 220 * i / rate)) if tone else 0 for i in range(n)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------- audio checks
def test_wav_info_and_plausibility():
    dur, rms = tts.wav_info(make_wav(2.0))
    assert abs(dur - 2.0) < 0.01 and rms > 0.05
    assert tts.is_plausible("a" * 30, dur, rms)
    assert not tts.is_plausible("a" * 30, 0.1, rms)          # clipped to nothing
    assert not tts.is_plausible("hi", 30.0, rms)             # rambled for 30 s
    d2, r2 = tts.wav_info(make_wav(2.0, tone=False))
    assert not tts.is_plausible("a" * 30, d2, r2)            # silent


def test_ensure_wav_wraps_raw_pcm():
    pcm = array.array("h", [0, 1000, -1000, 0] * 100).tobytes()
    wav = tts.ensure_wav(pcm, 24000)
    assert wav[:4] == b"RIFF" and tts.ensure_wav(wav, 24000) is wav


def test_build_marks_cover_audio_and_text():
    marks = build_marks("আমি ভালো আছি", 3.0)
    assert [m["word"] for m in marks] == ["আমি", "ভালো", "আছি"]
    assert marks[0]["start"] == 0 and abs(marks[-1]["end"] - 3.0) < 1e-3
    assert all(marks[i]["end"] <= marks[i + 1]["start"] + 1e-6 for i in range(2))


# ---------------- TTS synthesis (mocked Gemini)
def tts_client(wav_bytes, calls):
    async def create(**kw):
        calls.append(kw)
        audio = types.SimpleNamespace(data=base64.b64encode(wav_bytes).decode()) if wav_bytes is not None else None
        return types.SimpleNamespace(output_audio=audio)
    return types.SimpleNamespace(aio=types.SimpleNamespace(interactions=types.SimpleNamespace(create=create)))


async def test_synthesize_uses_language_specific_voice(monkeypatch):
    calls = []
    monkeypatch.setattr(gemini, "get_client", lambda: tts_client(make_wav(2.0), calls))
    monkeypatch.setattr(settings, "tts_voice_female", "EnVoice")
    monkeypatch.setattr(settings, "tts_voice_female_bn", "BnVoice")
    await tts.synthesize("আমি আজ ভালো আছি, ধন্যবাদ।", "female")
    await tts.synthesize("I am fine today, thank you.", "female")
    voices = [c["generation_config"]["speech_config"][0]["voice"] for c in calls]
    assert voices == ["BnVoice", "EnVoice"]
    assert calls[0]["model"] == settings.gemini_tts_model
    assert calls[0]["response_format"] == {"type": "audio"}


async def test_synthesize_retries_then_rejects_silent_audio(monkeypatch):
    calls = []
    monkeypatch.setattr(gemini, "get_client", lambda: tts_client(make_wav(2.0, tone=False), calls))
    with pytest.raises(AidaError) as ei:
        await tts.synthesize("This should be audible text.", "male")
    assert ei.value.code == "unavailable" and len(calls) == 2


async def test_synthesize_missing_audio_is_an_error(monkeypatch):
    monkeypatch.setattr(gemini, "get_client", lambda: tts_client(None, []))
    with pytest.raises(AidaError):
        await tts.synthesize("Hello there friend.", "female")


# ---------------- STT
def test_clean_transcript():
    assert stt.clean_transcript("<NO_SPEECH>") == ""
    assert stt.clean_transcript("  ") == ""
    assert stt.clean_transcript('Transcript: "আমি ভালো আছি"') == "আমি ভালো আছি"
    assert stt.clean_transcript("```\nhello\n```") == "hello"
    assert stt.clean_transcript("Docker কী?") == "Docker কী?"


def test_safe_mime():
    assert stt.safe_mime("audio/webm;codecs=opus") == "audio/webm"
    assert stt.safe_mime("application/x-evil") == "audio/webm"
    assert stt.safe_mime(None) == "audio/webm"


async def test_transcribe_sends_audio_and_strict_prompt(monkeypatch):
    seen = {}
    async def generate_content(*, model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return types.SimpleNamespace(text="আমি ভালো আছি")
    monkeypatch.setattr(gemini, "get_client", lambda: types.SimpleNamespace(
        aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))))
    assert await stt.transcribe(b"\x00" * 3000, "audio/webm") == "আমি ভালো আছি"
    assert "<NO_SPEECH>" in seen["config"].system_instruction
    assert seen["model"] == settings.gemini_stt_model


async def test_stt_default_thinking_level_is_valid_for_gemini_3_8_flash():
    # Production incident: "minimal" is rejected by gemini-3.8-flash (low|medium|high).
    assert settings.gemini_stt_thinking_level in {"low", "medium", "high"}


async def test_transcribe_self_heals_when_thinking_level_rejected(monkeypatch):
    from google.genai import errors as genai_errors
    gemini._NO_THINKING.clear()
    seen = []
    async def generate_content(*, model, contents, config):
        seen.append(config.thinking_config)
        if config.thinking_config is not None:
            raise genai_errors.APIError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                "message": "Thinking level MINIMAL is not supported for this model."}})
        return types.SimpleNamespace(text="hello there")
    monkeypatch.setattr(gemini, "get_client", lambda: types.SimpleNamespace(
        aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))))
    assert await stt.transcribe(b"\x00" * 3000, "audio/webm") == "hello there"
    assert seen[0] is not None and seen[1] is None
    gemini._NO_THINKING.clear()


# ---------------- HTTP endpoints
@pytest.fixture
def client(fake_redis):
    import app as app_module
    return AsyncClient(transport=ASGITransport(app=app_module.app), base_url="http://t")


async def test_transcribe_endpoint_contract(client, monkeypatch):
    async def fake(data, mime): return "আমি ভালো আছি"
    monkeypatch.setattr(stt, "transcribe", fake)
    async with client as c:
        r = await c.post("/api/voice/transcribe", files={"file": ("s.webm", b"x" * 5000, "audio/webm")})
    assert r.status_code == 200 and r.json() == {"text": "আমি ভালো আছি", "language": "bn"}


async def test_transcribe_tiny_audio_returns_empty(client):
    async with client as c:
        r = await c.post("/api/voice/transcribe", files={"file": ("s.webm", b"x" * 100, "audio/webm")})
    assert r.json()["text"] == ""


async def test_transcribe_provider_failure_is_200_with_code_not_500(client, monkeypatch):
    async def boom(data, mime): raise AidaError("unavailable", "secret-internal-detail")
    monkeypatch.setattr(stt, "transcribe", boom)
    async with client as c:
        r = await c.post("/api/voice/transcribe", files={"file": ("s.webm", b"x" * 5000, "audio/webm")},
                         headers={"X-Lang": "bn"})
    body = r.json()
    assert r.status_code == 200 and body["text"] == "" and body["error"] == "unavailable"
    assert "secret-internal-detail" not in r.text and "অনুগ্রহ" in body["message"]


async def test_transcribe_rejects_oversized_audio(client, monkeypatch):
    monkeypatch.setattr(settings, "max_audio_bytes", 2000)
    async with client as c:
        r = await c.post("/api/voice/transcribe", files={"file": ("s.webm", b"x" * 5000, "audio/webm")})
    assert r.status_code == 413 and r.json()["detail"]["code"] == "too_large"


async def test_speak_marks_contract_unchanged(client, monkeypatch):
    wav = make_wav(2.0)
    async def fake(text, gender): return wav, 2.0
    monkeypatch.setattr(tts, "synthesize", fake)
    async with client as c:
        r = await c.post("/api/voice/speak_marks", json={"text": "আমি ভালো আছি।", "gender": "female"})
    j = r.json()
    assert r.status_code == 200 and set(j) == {"audio", "duration", "text", "marks"}
    assert base64.b64decode(j["audio"]) == wav and j["marks"][0]["char_start"] == 0


async def test_speak_returns_wav(client, monkeypatch):
    wav = make_wav(1.0)
    async def fake(text, gender): return wav, 1.0
    monkeypatch.setattr(tts, "synthesize", fake)
    async with client as c:
        r = await c.post("/api/voice/speak", json={"text": "Hello there.", "gender": "male"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == wav


async def test_speak_validation_and_failure_codes(client, monkeypatch):
    async def boom(text, gender): raise AidaError("rate_limited", "quota exceeded for key sk-123")
    monkeypatch.setattr(tts, "synthesize", boom)
    async with client as c:
        assert (await c.post("/api/voice/speak", json={"text": "  "})).status_code == 400
        assert (await c.post("/api/voice/speak", json={"text": "x", "gender": "robot"})).status_code == 422
        big = await c.post("/api/voice/speak", json={"text": "x" * (settings.max_tts_chars + 1)})
        assert big.status_code == 413
        r = await c.post("/api/voice/speak", json={"text": "Hello"})
    assert r.status_code == 503 and "sk-123" not in r.text
    assert r.json()["detail"]["code"] == "rate_limited"

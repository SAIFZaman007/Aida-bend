"""Real-time voice: live transcription WebSocket, streaming TTS, dedicated STT model.
No network: Google's SDK objects are replaced by small fakes."""
import asyncio
import base64
import types

import pytest
from google.genai import errors as genai_errors
from starlette.testclient import TestClient

from src.core import gemini
from src.core.config import settings
from src.core.errors import AidaError
from src.voice import live_stt, stt, tts



def api_error(code):
    return genai_errors.APIError(code, {"error": {"code": code, "message": "x", "status": "S"}})


# ------------------------------------------------------------------ dedicated STT model
def test_dedicated_model_detection():
    assert stt.is_dedicated("gemini-3.5-transcribe")
    assert stt.is_dedicated("gemini-3.5-transcribe-live")
    assert not stt.is_dedicated("gemini-3.8-flash")


async def test_dedicated_model_request_has_no_prompt_or_thinking(monkeypatch):
    monkeypatch.setattr(settings, "gemini_stt_model", "gemini-3.5-transcribe")
    monkeypatch.setattr(settings, "stt_languages", "")
    monkeypatch.setattr(settings, "gemini_stt_mode", "SMART")
    seen = {}

    async def generate_content(*, model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return types.SimpleNamespace(text="আমি ভালো আছি")

    monkeypatch.setattr(gemini, "get_client", lambda: types.SimpleNamespace(
        aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))))
    assert await stt.transcribe(b"\x00" * 3000, "audio/wav") == "আমি ভালো আছি"
    cfg = seen["config"]
    assert seen["model"] == "gemini-3.5-transcribe"
    assert cfg.system_instruction is None and cfg.thinking_config is None
    assert cfg.audio_transcription_config.language_codes == []        # automatic detection
    assert str(cfg.audio_transcription_config.mode).endswith("SMART")
    assert len(seen["contents"]) == 1                                  # audio only, no prompt text


async def test_stt_language_hints_and_verbatim_mode(monkeypatch):
    monkeypatch.setattr(settings, "stt_languages", "bn-BD, en-US")
    monkeypatch.setattr(settings, "gemini_stt_mode", "verbatim")
    cfg = stt.audio_transcription_config()
    assert cfg.language_codes == ["bn-BD", "en-US"]
    assert str(cfg.mode).endswith("VERBATIM")


async def test_stt_falls_back_from_dedicated_to_general_model(monkeypatch):
    monkeypatch.setattr(settings, "gemini_stt_model", "gemini-3.5-transcribe")
    monkeypatch.setattr(settings, "gemini_fallback_model", "gemini-3.7-flash")
    monkeypatch.setattr(settings, "gemini_stt_fallback_model", "")
    seen = []

    async def generate_content(*, model, contents, config):
        seen.append((model, config.system_instruction is not None))
        if model == "gemini-3.5-transcribe":
            raise api_error(404)  # e.g. the key has no access to it
        return types.SimpleNamespace(text="hello")

    monkeypatch.setattr(gemini, "get_client", lambda: types.SimpleNamespace(
        aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))))
    assert await stt.transcribe(b"\x00" * 3000, "audio/wav") == "hello"
    # the backup is a general model, so it got the strict prompt
    assert seen == [("gemini-3.5-transcribe", False), ("gemini-3.7-flash", True)]


# ------------------------------------------------------------------ merge_final
def test_merge_final_joins_without_duplicating():
    m = live_stt.merge_final
    assert m("", "hello") == "hello"
    assert m("hello", "world") == "hello world"
    assert m("hello", "hello world") == "hello world"   # cumulative resend replaces
    assert m("hello world", "world") == "hello world"   # exact tail repeat dropped
    assert m("আমি", "ভালো আছি") == "আমি ভালো আছি"
    assert m("hi", "  ") == "hi"


# ------------------------------------------------------------------ live WebSocket
class FakeLive:
    """Stands in for client.aio.live.connect(...): records what the server sends
    to Google and plays back scripted transcription events."""

    def __init__(self, script=None, fail_on_connect=None):
        self.sent = []
        self.script = script or {}
        self.fail = fail_on_connect
        self.q: asyncio.Queue | None = None

    async def __aenter__(self):
        if self.fail:
            raise self.fail
        self.q = asyncio.Queue()
        return self

    async def __aexit__(self, *a):
        return False

    @staticmethod
    def msg(interim=None, final=None, turn_complete=False):
        sc = types.SimpleNamespace(
            interim_input_transcription=types.SimpleNamespace(text=interim) if interim else None,
            input_transcription=types.SimpleNamespace(text=final) if final else None,
            turn_complete=turn_complete,
        )
        return types.SimpleNamespace(server_content=sc)

    async def send_realtime_input(self, **kw):
        self.sent.append(kw)
        key = next(iter(kw))
        for m in self.script.get(key, []):
            self.q.put_nowait(m)
            self.script[key] = []  # each scripted reply plays once

    async def receive(self):
        while True:
            m = await self.q.get()
            yield m
            if m.server_content.turn_complete:
                return


@pytest.fixture
def ws_client(monkeypatch, fake_redis):
    import app as app_module
    monkeypatch.setattr(settings, "stt_live_final_wait", 0.15)
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    live_stt._connections.clear()

    def install(fake):
        monkeypatch.setattr(live_stt, "open_session", lambda model, config: fake)
        fake.model_used = None
        return fake

    return TestClient(app_module.app), install


def test_live_transcription_roundtrip(ws_client):
    c, install = ws_client
    fake = install(FakeLive(script={
        "audio": [FakeLive.msg(interim="hello wor")],
        "activity_end": [FakeLive.msg(final="hello world", turn_complete=True)],
    }))
    with c.websocket_connect("/api/voice/stream") as ws:
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(b"\x01" * 640)                 # before `start`: must be ignored
        ws.send_json({"type": "start"})
        ws.send_bytes(b"\x01" * 3200)
        interim = ws.receive_json()
        assert interim["type"] == "interim" and interim["text"] == "hello wor"
        ws.send_json({"type": "end"})
        final = ws.receive_json()
        assert final["type"] == "final" and final["text"] == "hello world"
        assert final["language"] == "en" and final["id"] == interim["id"]
    kinds = [next(iter(k)) for k in fake.sent]
    assert kinds == ["activity_start", "audio", "activity_end"]   # pre-start audio never sent
    blob = fake.sent[1]["audio"]
    assert blob.mime_type == "audio/pcm;rate=16000" and len(blob.data) == 3200


def test_live_bengali_final_is_detected(ws_client):
    c, install = ws_client
    install(FakeLive(script={"activity_end": [FakeLive.msg(final="আমি ভালো আছি", turn_complete=True)]}))
    with c.websocket_connect("/api/voice/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.send_bytes(b"\x01" * 3200)
        ws.send_json({"type": "end"})
        final = ws.receive_json()
        assert final["text"] == "আমি ভালো আছি" and final["language"] == "bn"


def test_live_final_arrives_even_without_turn_complete(ws_client):
    c, install = ws_client
    install(FakeLive(script={"activity_end": [FakeLive.msg(final="no turn marker")]}))
    with c.websocket_connect("/api/voice/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.send_bytes(b"\x01" * 3200)
        ws.send_json({"type": "end"})
        assert ws.receive_json()["text"] == "no turn marker"   # settles after STT_LIVE_FINAL_WAIT


def test_live_final_is_released_quickly_once_google_sent_it(ws_client, monkeypatch):
    """After a FINAL segment arrives we wait only the short settle time, not the
    full STT_LIVE_FINAL_WAIT — that wait is dead air the person feels."""
    import time
    c, install = ws_client
    monkeypatch.setattr(settings, "stt_live_final_wait", 5.0)
    monkeypatch.setattr(settings, "stt_live_final_settle", 0.05)
    install(FakeLive(script={"activity_end": [FakeLive.msg(final="quick")]}))
    with c.websocket_connect("/api/voice/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.send_bytes(b"\x01" * 3200)
        t0 = time.monotonic()
        ws.send_json({"type": "end"})
        assert ws.receive_json()["text"] == "quick"
        assert time.monotonic() - t0 < 1.5


def test_live_two_utterances_in_one_session(ws_client):
    c, install = ws_client
    fake = install(FakeLive(script={"activity_end": [FakeLive.msg(final="one", turn_complete=True)]}))
    with c.websocket_connect("/api/voice/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"}); ws.send_bytes(b"\x01" * 640); ws.send_json({"type": "end"})
        assert ws.receive_json()["text"] == "one"
        fake.script["activity_end"] = [FakeLive.msg(final="two", turn_complete=True)]
        ws.send_json({"type": "start"}); ws.send_bytes(b"\x01" * 640); ws.send_json({"type": "end"})
        second = ws.receive_json()
        assert second["text"] == "two" and second["id"] == 2


def test_live_provider_failure_sends_coded_error(ws_client):
    c, install = ws_client
    install(FakeLive(fail_on_connect=api_error(503)))
    with c.websocket_connect("/api/voice/stream") as ws:
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "unavailable"
        assert "503" not in err["message"] and "api_error" not in err["message"].lower()


def test_live_disabled_closes_so_client_falls_back(ws_client, monkeypatch):
    c, install = ws_client
    monkeypatch.setattr(settings, "stt_live_enabled", False)
    install(FakeLive())
    with pytest.raises(Exception):
        with c.websocket_connect("/api/voice/stream"):
            pass


def test_live_foreign_origin_is_refused(ws_client, monkeypatch):
    c, install = ws_client
    monkeypatch.setattr(settings, "cors_origins", "https://aida.example.com")
    install(FakeLive())
    with pytest.raises(Exception):
        with c.websocket_connect("/api/voice/stream", headers={"origin": "https://evil.example"}):
            pass
    install(FakeLive())
    with c.websocket_connect("/api/voice/stream", headers={"origin": "https://aida.example.com"}) as ws:
        assert ws.receive_json()["type"] == "ready"


def test_live_rate_limit_rejects_utterance_but_keeps_socket(ws_client, monkeypatch):
    c, install = ws_client
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_stt_per_min", 1)
    monkeypatch.setattr(settings, "daily_stt_cap", 0)
    fake = install(FakeLive(script={"activity_end": [FakeLive.msg(final="ok", turn_complete=True)]}))
    with c.websocket_connect("/api/voice/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"}); ws.send_bytes(b"\x01" * 640); ws.send_json({"type": "end"})
        assert ws.receive_json()["text"] == "ok"
        ws.send_json({"type": "start"})                       # second one in the same minute
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "rate_limited" and err["recoverable"]
        assert err["id"] == 2                                  # tells the client WHICH utterance
        ws.send_bytes(b"\x01" * 640); ws.send_json({"type": "end"})
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}          # still alive, audio dropped
    assert [next(iter(k)) for k in fake.sent].count("audio") == 1


# ------------------------------------------------------------------ streaming TTS
def pcm(n_samples=240):
    import array
    return array.array("h", [1000, -1000] * (n_samples // 2)).tobytes()


def event(data: bytes, rate=24000):
    return types.SimpleNamespace(
        event_type="step.delta",
        delta=types.SimpleNamespace(type="audio", data=base64.b64encode(data).decode(), sample_rate=rate))


def tts_stream_client(plans, calls):
    """plans: per call, a list of events (or an Exception to raise on create / mid-stream)."""
    async def create(**kw):
        plan = plans[min(len(calls), len(plans) - 1)]
        calls.append(kw)
        if isinstance(plan, Exception):
            raise plan

        async def gen():
            for item in plan:
                if isinstance(item, Exception):
                    raise item
                yield item
        return gen()
    return types.SimpleNamespace(aio=types.SimpleNamespace(interactions=types.SimpleNamespace(create=create)))


async def collect_stream(text="Hello there, friend.", gender="female"):
    info = {}
    out = [c async for c in tts.synthesize_stream(text, gender, info)]
    return out, info


async def test_stream_yields_pcm_chunks_in_order(monkeypatch):
    calls = []
    a, b = pcm(240), pcm(480)
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([[event(a), event(b)]], calls))
    out, info = await collect_stream()
    assert out == [a, b] and info["sample_rate"] == 24000
    assert calls[0]["stream"] is True and calls[0]["model"] == settings.gemini_tts_model


async def test_stream_never_splits_a_16bit_sample(monkeypatch):
    calls = []
    whole = pcm(240)
    # chunk boundaries fall in the middle of samples
    parts = [whole[:101], whole[101:300], whole[300:]]
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([[event(p) for p in parts]], calls))
    out, _ = await collect_stream()
    assert all(len(c) % 2 == 0 for c in out) and b"".join(out) == whole


async def test_stream_fails_over_before_first_byte(monkeypatch):
    calls = []
    good = pcm(240)
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "lite-tts")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([api_error(503), [event(good)]], calls))
    out, _ = await collect_stream()
    assert out == [good]
    assert [c["model"] for c in calls] == [settings.gemini_tts_model, "lite-tts"]
    assert gemini.is_unhealthy(settings.gemini_tts_model)


async def test_stream_skips_model_that_just_failed_on_next_clip(monkeypatch):
    calls = []
    good = pcm(240)
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "lite-tts")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([api_error(429), [event(good)]], calls))
    await collect_stream()
    await collect_stream()
    assert [c["model"] for c in calls] == [settings.gemini_tts_model, "lite-tts", "lite-tts"]


async def test_stream_failure_mid_clip_ends_quietly_without_second_voice(monkeypatch):
    calls = []
    first = pcm(240)
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "lite-tts")
    monkeypatch.setattr(gemini, "get_client",
                        lambda: tts_stream_client([[event(first), api_error(503)], [event(pcm(480))]], calls))
    out, _ = await collect_stream()
    assert out == [first] and len(calls) == 1          # no splice of a different voice


async def test_stream_with_no_audio_is_an_error(monkeypatch):
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([[]], []))
    with pytest.raises(AidaError) as ei:
        await collect_stream()
    assert ei.value.code == "unavailable"


async def test_stream_first_audio_timeout_triggers_failover(monkeypatch):
    calls = []
    good = pcm(240)
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "lite-tts")
    monkeypatch.setattr(settings, "tts_first_audio_timeout", 0.1)

    async def create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            async def hang():
                await asyncio.sleep(5)
                yield event(good)
            return hang()

        async def ok():
            yield event(good)
        return ok()

    monkeypatch.setattr(gemini, "get_client", lambda: types.SimpleNamespace(
        aio=types.SimpleNamespace(interactions=types.SimpleNamespace(create=create))))
    out, _ = await collect_stream()
    assert out == [good] and [c["model"] for c in calls] == [settings.gemini_tts_model, "lite-tts"]


# ------------------------------------------------------------------ /voice/speak_stream endpoint
@pytest.fixture
def http(fake_redis, monkeypatch):
    import app as app_module
    from httpx import ASGITransport, AsyncClient
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    return lambda: AsyncClient(transport=ASGITransport(app=app_module.app), base_url="http://t")


async def test_speak_stream_endpoint_streams_pcm_with_headers(http, monkeypatch):
    a, b = pcm(240), pcm(480)
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([[event(a), event(b)]], []))
    async with http() as c:
        r = await c.post("/api/voice/speak_stream", json={"text": "Hello there, friend.", "gender": "male"})
    assert r.status_code == 200 and r.content == a + b
    assert r.headers["x-sample-rate"] == "24000" and r.headers["x-audio-format"] == "pcm_s16le"
    assert r.headers["cache-control"] == "no-store"


async def test_speak_stream_endpoint_is_a_clean_503_before_any_audio(http, monkeypatch):
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", "")
    monkeypatch.setattr(gemini, "get_client", lambda: tts_stream_client([api_error(503)], []))
    async with http() as c:
        r = await c.post("/api/voice/speak_stream", json={"text": "Hello there, friend."})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "unavailable"


async def test_speak_stream_endpoint_validates_input(http):
    async with http() as c:
        assert (await c.post("/api/voice/speak_stream", json={"text": "   "})).status_code == 400
        too_long = "a" * (settings.max_tts_chars + 1)
        assert (await c.post("/api/voice/speak_stream", json={"text": too_long})).status_code == 413
        assert (await c.post("/api/voice/speak_stream", json={"text": "hi", "gender": "x"})).status_code == 422


async def test_health_reports_live_and_streaming_voice(http):
    async with http() as c:
        v = (await c.get("/api/health")).json()["voice"]
    assert v["stream"] is True and v["stt_live"] and v["tts_backup"]

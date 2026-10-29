import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from src.chat import router as chat_router
from src.core.config import settings
from src.core.database import get_session
from src.core.errors import AidaError


def parse_sse(text: str):
    frames = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            frames.append(("comment", block))
            continue
        ev = data = None
        for line in block.split("\n"):
            if line.startswith("event:"): ev = line[6:].strip()
            elif line.startswith("data:"): data = json.loads(line[5:].strip())
        frames.append((ev, data))
    return frames


@pytest.fixture
def client(fake_redis):
    import app as app_module
    async def no_session():
        yield object()
    app_module.app.dependency_overrides[get_session] = no_session
    yield AsyncClient(transport=ASGITransport(app=app_module.app), base_url="http://t")
    app_module.app.dependency_overrides.clear()


def patch_stream(monkeypatch, events=(), error=None, delay=0.0):
    async def fake(session, conv_id, content, remember, persona=None):
        if delay:
            await asyncio.sleep(delay)
        for e in events:
            yield e
        if error:
            raise error
    monkeypatch.setattr(chat_router, "stream_reply", fake)


BODY = {"conversation_id": "c1", "content": "hi"}


async def test_happy_path_events(client, monkeypatch):
    patch_stream(monkeypatch, [
        {"type": "token", "t": "আমি "},
        {"type": "sentence", "index": 0, "text": "আমি ভালো।", "code": False},
        {"type": "sentence", "index": 1, "text": "x = 1.5\n", "code": True},
    ])
    async with client as c:
        r = await c.post("/api/chat", json=BODY)
    assert r.headers["content-type"].startswith("text/event-stream") and r.headers["x-request-id"]
    frames = parse_sse(r.text)
    assert [f[0] for f in frames] == ["token", "sentence", "sentence", "done"]
    assert frames[1][1] == {"index": 0, "text": "আমি ভালো।", "code": False}
    assert frames[2][1]["code"] is True
    assert "আমি ভালো" in r.text      # ensure_ascii=False: readable on the wire


async def test_aida_error_becomes_localized_frame(client, monkeypatch):
    patch_stream(monkeypatch, error=AidaError("rate_limited", "429 RESOURCE_EXHAUSTED project 12345"))
    async with client as c:
        r = await c.post("/api/chat", json={**BODY, "content": "আপনি কেমন আছেন?"})
    err = [f for f in parse_sse(r.text) if f[0] == "error"][0][1]
    assert err["code"] == "rate_limited" and "অনুগ্রহ" in err["error"] and err["request_id"]
    assert "12345" not in r.text and parse_sse(r.text)[-1][0] == "done"


async def test_unexpected_exception_never_leaks_internals(client, monkeypatch):
    patch_stream(monkeypatch, error=RuntimeError("postgresql://user:p4ssw0rd@db/secret"))
    async with client as c:
        r = await c.post("/api/chat", json=BODY)
    assert "p4ssw0rd" not in r.text
    err = [f for f in parse_sse(r.text) if f[0] == "error"][0][1]
    assert err["code"] == "internal"


async def test_english_message_gets_english_error(client, monkeypatch):
    patch_stream(monkeypatch, error=AidaError("timeout"))
    async with client as c:
        r = await c.post("/api/chat", json=BODY)
    err = [f for f in parse_sse(r.text) if f[0] == "error"][0][1]
    assert "taking too long" in err["error"]


async def test_heartbeat_while_waiting_for_first_token(client, monkeypatch):
    monkeypatch.setattr(settings, "sse_heartbeat_seconds", 0.05)
    patch_stream(monkeypatch, [{"type": "token", "t": "ok"}], delay=0.3)
    async with client as c:
        r = await c.post("/api/chat", json=BODY)
    frames = parse_sse(r.text)
    assert sum(1 for f in frames if f[0] == "comment") >= 2
    assert [f[0] for f in frames if f[0] != "comment"] == ["token", "done"]


async def test_input_validation(client):
    async with client as c:
        assert (await c.post("/api/chat", json={**BODY, "content": ""})).status_code == 422
        assert (await c.post("/api/chat", json={**BODY, "content": "x" * (settings.max_chat_chars + 1)})).status_code == 422


async def test_chat_rate_limit_returns_coded_429(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_chat_per_min", 1)
    patch_stream(monkeypatch, [])
    async with client as c:
        assert (await c.post("/api/chat", json=BODY)).status_code == 200
        r = await c.post("/api/chat", json=BODY)
    assert r.status_code == 429 and r.json()["detail"]["code"] == "rate_limited"


async def test_unknown_persona_rejected(client):
    async with client as c:
        r = await c.post("/api/conversations", json={"project_id": "p", "persona": "hacker"})
    assert r.status_code == 422


async def test_health_reports_status_without_secrets(client):
    async with client as c:
        r = await c.get("/api/health")
    j = r.json()
    assert j["status"] == "ok" and j["languages"] == ["bn", "en"] and j["ai_configured"] is True
    assert "test-key" not in r.text and j["model"] == settings.chat_model

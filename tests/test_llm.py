import asyncio
import types

import pytest
from google.genai import errors as genai_errors

from src.core import gemini, llm as llm_mod
from src.core.config import settings
from src.core.errors import AidaError
from src.core.llm import LLM, normalize_turns

_REAL_SLEEP = asyncio.sleep  # captured before the autouse fixture patches it


class Chunk:
    def __init__(self, text="", block=False):
        self.text = text
        self.prompt_feedback = types.SimpleNamespace(block_reason="SAFETY") if block else None
        self.candidates = []


def fake_client(plans):
    """plans: list (one per call) of callables returning an async iterator of chunks,
    or an exception to raise when the stream is created."""
    calls = {"n": 0, "configs": []}

    async def generate_content_stream(*, model, contents, config):
        plan = plans[min(calls["n"], len(plans) - 1)]
        calls["n"] += 1
        calls["configs"].append((model, contents, config))
        if isinstance(plan, Exception):
            raise plan
        return plan()

    client = types.SimpleNamespace(
        aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=generate_content_stream))
    )
    return client, calls


def stream_of(*items, then_raise=None, delay=0.0):
    async def gen():
        for it in items:
            if delay:
                await _REAL_SLEEP(delay)
            yield it
        if then_raise:
            raise then_raise
    return lambda: gen()


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_retries", 2)
    monkeypatch.setattr(settings, "llm_first_token_timeout", 0.3)
    monkeypatch.setattr(settings, "llm_idle_timeout", 0.3)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep(_REAL_SLEEP))


def _fast_sleep(real):
    async def s(delay, *a, **k):
        await real(min(delay, 0.01))
    return s


def use(monkeypatch, client):
    monkeypatch.setattr(gemini, "get_client", lambda: client)


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


async def collect(llm=None, messages=MSGS):
    return [p async for p in (llm or LLM()).stream_chat(messages)]


def api_error(code):
    return genai_errors.APIError(code, {"error": {"code": code, "message": "x", "status": "S"}})


# ---- turn normalisation
def test_normalize_turns_merges_and_drops_leading_assistant():
    system, turns = normalize_turns([
        {"role": "system", "content": "A"}, {"role": "system", "content": "B"},
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "q1"}, {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a"}, {"role": "assistant", "content": ""},
    ])
    assert system == "A\n\nB"
    assert turns == [{"role": "user", "content": "q1\n\nq2"}, {"role": "assistant", "content": "a"}]


# ---- streaming
async def test_streams_text_and_passes_config(monkeypatch):
    client, calls = fake_client([stream_of(Chunk("Hel"), Chunk("lo"))])
    use(monkeypatch, client)
    assert await collect() == ["Hel", "lo"]
    model, contents, cfg = calls["configs"][0]
    assert model == settings.gemini_chat_model
    assert contents[0].role == "user"
    assert cfg.system_instruction == "sys"
    assert cfg.temperature is None  # Google: keep default for Gemini 3


async def test_retries_transient_error_before_first_token(monkeypatch):
    client, calls = fake_client([api_error(429), api_error(503), stream_of(Chunk("ok"))])
    use(monkeypatch, client)
    assert await collect() == ["ok"]
    assert calls["n"] == 3


async def test_no_retry_after_tokens_were_produced(monkeypatch):
    client, calls = fake_client([stream_of(Chunk("part"), then_raise=api_error(503)), stream_of(Chunk("DUP"))])
    use(monkeypatch, client)
    got = []
    with pytest.raises(AidaError) as ei:
        async for p in LLM().stream_chat(MSGS):
            got.append(p)
    assert got == ["part"] and calls["n"] == 1 and ei.value.code == "unavailable"


async def test_retries_exhausted_raises_friendly_code(monkeypatch):
    client, calls = fake_client([api_error(503)])
    use(monkeypatch, client)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "unavailable" and calls["n"] == 3  # 1 try + 2 retries


async def test_auth_error_is_not_retried_and_maps_to_not_configured(monkeypatch):
    client, calls = fake_client([api_error(403)])
    use(monkeypatch, client)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "not_configured" and calls["n"] == 1


async def test_first_token_timeout(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_retries", 0)
    client, _ = fake_client([stream_of(Chunk("late"), delay=1.0)])
    use(monkeypatch, client)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "timeout"


async def test_blocked_content_is_final_not_retried(monkeypatch):
    client, calls = fake_client([stream_of(Chunk("", block=True))])
    use(monkeypatch, client)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "blocked" and calls["n"] == 1


async def test_empty_response_is_retried_then_succeeds(monkeypatch):
    client, calls = fake_client([stream_of(Chunk("")), stream_of(Chunk("fine"))])
    use(monkeypatch, client)
    assert await collect() == ["fine"] and calls["n"] == 2


async def test_missing_key_gives_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(gemini, "_client", None)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "not_configured"


async def test_coder_persona_uses_coder_model(monkeypatch):
    monkeypatch.setattr(settings, "gemini_coder_model", "coder-x")
    client, calls = fake_client([stream_of(Chunk("x"))])
    use(monkeypatch, client)
    [p async for p in LLM().stream_chat(MSGS, persona="developer")]
    assert calls["configs"][0][0] == "coder-x"


# ---- embeddings
def embed_client(values, raises=None):
    async def embed_content(*, model, contents, config):
        if raises:
            raise raises
        return types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=values)])
    return types.SimpleNamespace(aio=types.SimpleNamespace(models=types.SimpleNamespace(embed_content=embed_content)))


async def test_embed_returns_unit_vector_of_configured_dim(monkeypatch):
    use(monkeypatch, embed_client([3.0, 4.0] + [0.0] * (settings.embedding_dim - 2)))
    v = await LLM().embed("hello")
    assert len(v) == settings.embedding_dim
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9


async def test_embed_rejects_wrong_dimension(monkeypatch):
    use(monkeypatch, embed_client([0.1] * 10))
    assert await LLM().embed("hello") is None


async def test_embed_never_raises(monkeypatch):
    use(monkeypatch, embed_client([], raises=api_error(500)))
    assert await LLM().embed("hello") is None
    assert await LLM().embed("   ") is None


async def test_embed_times_out_gracefully(monkeypatch):
    monkeypatch.setattr(settings, "embed_timeout", 0.05)
    async def slow(*, model, contents, config):
        await _REAL_SLEEP(1)
    monkeypatch.setattr(gemini, "get_client", lambda: types.SimpleNamespace(
        aio=types.SimpleNamespace(models=types.SimpleNamespace(embed_content=slow))))
    assert await LLM().embed("x") is None

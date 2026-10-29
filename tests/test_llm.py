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


def fake_client(plans, once_plans=None):
    """plans: list (one per call) of callables returning an async iterator of chunks,
    or an exception to raise when the stream is created.
    once_plans: same idea for the non-streaming generate_content (value = response
    object, an Exception to raise, or a float = seconds to hang)."""
    calls = {"n": 0, "configs": [], "once": 0, "models": []}

    async def generate_content_stream(*, model, contents, config):
        plan = plans[min(calls["n"], len(plans) - 1)]
        calls["n"] += 1
        calls["configs"].append((model, contents, config))
        calls["models"].append(("stream", model))
        if isinstance(plan, Exception):
            raise plan
        return plan()

    async def generate_content(*, model, contents, config):
        plan = (once_plans or [Exception("no once plan")])[min(calls["once"], len(once_plans or [0]) - 1)]
        calls["once"] += 1
        calls["configs"].append((model, contents, config))
        calls["models"].append(("once", model))
        if isinstance(plan, Exception):
            raise plan
        if isinstance(plan, float):
            await _REAL_SLEEP(plan)
        return plan

    client = types.SimpleNamespace(
        aio=types.SimpleNamespace(models=types.SimpleNamespace(
            generate_content_stream=generate_content_stream, generate_content=generate_content))
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
    monkeypatch.setattr(settings, "llm_once_timeout", 0.3)
    monkeypatch.setattr(settings, "gemini_fallback_model", "")
    gemini._NO_THINKING.clear()
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


def once_response(text):
    return types.SimpleNamespace(text=text, prompt_feedback=None, candidates=[])


async def test_stalled_stream_falls_back_to_non_streaming(monkeypatch):
    # The production incident: stream never yields a first token.
    client, calls = fake_client([stream_of(Chunk("late"), delay=1.0)], [once_response("hello")])
    use(monkeypatch, client)
    assert await collect() == ["hello"]
    assert calls["models"] == [("stream", settings.gemini_chat_model), ("once", settings.gemini_chat_model)]


async def test_stall_everywhere_then_fallback_model(monkeypatch):
    monkeypatch.setattr(settings, "gemini_fallback_model", "backup-m")
    client, calls = fake_client(
        [stream_of(Chunk("late"), delay=1.0), stream_of(Chunk("from backup"))], [3.0])
    use(monkeypatch, client)
    assert await collect() == ["from backup"]
    # A stall goes to the backup model first; the same model without streaming is last.
    assert calls["models"] == [("stream", settings.gemini_chat_model), ("stream", "backup-m")]


async def test_timeout_when_every_route_stalls(monkeypatch):
    client, _ = fake_client([stream_of(Chunk("late"), delay=1.0)], [3.0])
    use(monkeypatch, client)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "timeout"


async def test_thinking_level_rejected_is_self_healed(monkeypatch):
    bad = genai_errors.APIError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
        "message": "Thinking level MINIMAL is not supported for this model."}})
    client, calls = fake_client([bad, stream_of(Chunk("ok"))])
    use(monkeypatch, client)
    assert await collect() == ["ok"]
    assert calls["configs"][0][2].thinking_config is not None
    assert calls["configs"][1][2].thinking_config is None          # retried without it
    assert settings.gemini_chat_model in gemini._NO_THINKING
    # ...and later calls skip the bad setting up front.
    client2, calls2 = fake_client([stream_of(Chunk("x"))])
    use(monkeypatch, client2)
    await collect()
    assert calls2["configs"][0][2].thinking_config is None


async def test_unrelated_400_is_not_swallowed(monkeypatch):
    client, calls = fake_client([api_error(400)])
    use(monkeypatch, client)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "internal" and calls["n"] == 1


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


# ---- overload / quota failover (production incident: 503 "high demand" then 429)
async def test_quota_error_fails_over_to_backup_immediately(monkeypatch):
    monkeypatch.setattr(settings, "gemini_fallback_model", "backup-m")
    client, calls = fake_client([api_error(429), stream_of(Chunk("from backup"))])
    use(monkeypatch, client)
    assert await collect() == ["from backup"]
    assert calls["models"] == [("stream", settings.gemini_chat_model), ("stream", "backup-m")]


async def test_overload_503_gets_one_retry_then_backup(monkeypatch):
    monkeypatch.setattr(settings, "gemini_fallback_model", "backup-m")
    client, calls = fake_client([api_error(503), api_error(503), stream_of(Chunk("ok"))])
    use(monkeypatch, client)
    assert await collect() == ["ok"]
    assert [m for _, m in calls["models"]] == [settings.gemini_chat_model] * 2 + ["backup-m"]


async def test_without_backup_503_still_retries_same_model(monkeypatch):
    client, calls = fake_client([api_error(503), api_error(503), stream_of(Chunk("ok"))])
    use(monkeypatch, client)
    assert await collect() == ["ok"] and calls["n"] == 3


async def test_both_models_down_raises_last_error(monkeypatch):
    monkeypatch.setattr(settings, "gemini_fallback_model", "backup-m")
    client, _ = fake_client([api_error(503)])
    use(monkeypatch, client)
    with pytest.raises(AidaError) as ei:
        await collect()
    assert ei.value.code == "unavailable"


async def test_stalled_primary_then_backup_then_non_streaming_last_resort(monkeypatch):
    monkeypatch.setattr(settings, "gemini_fallback_model", "backup-m")
    client, calls = fake_client(
        [stream_of(Chunk("late"), delay=1.0), api_error(503), api_error(503)],
        [once_response("last resort")])
    use(monkeypatch, client)
    assert await collect() == ["last resort"]
    assert calls["models"][0] == ("stream", settings.gemini_chat_model)
    assert calls["models"][-1] == ("once", settings.gemini_chat_model)


async def test_failed_model_is_skipped_for_the_next_request(monkeypatch):
    # The live-demo fix: after the primary failed, the next question must not
    # wait for it again; the healthy backup answers first.
    monkeypatch.setattr(settings, "gemini_fallback_model", "backup-m")
    client, calls = fake_client([api_error(429), stream_of(Chunk("one")), stream_of(Chunk("two"))])
    use(monkeypatch, client)
    assert await collect() == ["one"]
    assert gemini.is_unhealthy(settings.gemini_chat_model)
    assert await collect() == ["two"]
    assert [m for _, m in calls["models"]] == [settings.gemini_chat_model, "backup-m", "backup-m"]


async def test_model_recovers_after_cooldown(monkeypatch):
    monkeypatch.setattr(settings, "llm_cooldown_seconds", 0.0)
    gemini.mark_unhealthy("m1")
    assert not gemini.is_unhealthy("m1")
    monkeypatch.setattr(settings, "llm_cooldown_seconds", 60.0)
    gemini.mark_unhealthy("m1")
    assert gemini.is_unhealthy("m1")
    gemini.mark_healthy("m1")
    assert gemini.order_models(["m1", "m2"]) == ["m1", "m2"]
    gemini.mark_unhealthy("m1")
    assert gemini.order_models(["m1", "m2"]) == ["m2", "m1"]

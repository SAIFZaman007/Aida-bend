"""The backup voice model is started when the main one is SLOW, not only when it failed."""
import asyncio

import pytest

from src.core import gemini
from src.core.config import settings
from src.core.errors import AidaError
from src.voice import tts

MAIN, BACKUP = "main-tts", "lite-tts"


@pytest.fixture(autouse=True)
def two_models(monkeypatch):
    monkeypatch.setattr(settings, "gemini_tts_model", MAIN)
    monkeypatch.setattr(settings, "gemini_tts_fallback_model", BACKUP)
    monkeypatch.setattr(settings, "tts_hedge_after", 0.05)


def install(monkeypatch, plans):
    """plans: model -> (delay_before_first_chunk, [chunks], error_or_None)."""
    started, cancelled = [], []

    async def fake(model, text, voice, info):
        started.append(model)
        delay, chunks, error = plans[model]
        try:
            await asyncio.sleep(delay)
            info["sample_rate"] = 24000 if model == MAIN else 22050
            for c in chunks:
                yield c
                await asyncio.sleep(0)
            if error:
                raise error
        except asyncio.CancelledError:
            cancelled.append(model)
            raise

    monkeypatch.setattr(tts, "_stream_model", fake)
    return started, cancelled


async def run(text="Hello there, friend.", gender="female"):
    info = {}
    out = [c async for c in tts.synthesize_stream(text, gender, info)]
    return out, info


async def test_fast_main_model_never_wakes_the_backup(monkeypatch):
    started, _ = install(monkeypatch, {MAIN: (0.0, [b"aa", b"bb"], None), BACKUP: (0.0, [b"zz"], None)})
    out, _ = await run()
    assert out == [b"aa", b"bb"] and started == [MAIN]


async def test_slow_main_model_loses_to_the_backup_and_is_cancelled(monkeypatch):
    started, cancelled = install(monkeypatch, {MAIN: (2.0, [b"aa"], None), BACKUP: (0.01, [b"zz", b"yy"], None)})
    t0 = asyncio.get_event_loop().time()
    out, info = await run()
    assert asyncio.get_event_loop().time() - t0 < 1.0          # did not wait for the slow model
    assert out == [b"zz", b"yy"]                                # one voice only: the winner's
    assert started == [MAIN, BACKUP] and cancelled == [MAIN]
    assert info["sample_rate"] == 22050                         # the winner's rate
    assert gemini.is_unhealthy(MAIN) and not gemini.is_unhealthy(BACKUP)


async def test_main_model_that_answers_during_the_grace_period_still_wins(monkeypatch):
    monkeypatch.setattr(settings, "tts_hedge_after", 0.3)
    started, _ = install(monkeypatch, {MAIN: (0.05, [b"aa"], None), BACKUP: (0.0, [b"zz"], None)})
    out, _ = await run()
    assert out == [b"aa"] and started == [MAIN]


async def test_main_model_failing_starts_the_backup_at_once(monkeypatch):
    monkeypatch.setattr(settings, "tts_hedge_after", 5.0)       # would be far too slow to wait for
    started, _ = install(monkeypatch, {MAIN: (0.0, [], AidaError("unavailable", "503")),
                                       BACKUP: (0.0, [b"zz"], None)})
    t0 = asyncio.get_event_loop().time()
    out, _ = await run()
    assert asyncio.get_event_loop().time() - t0 < 1.0
    assert out == [b"zz"] and started == [MAIN, BACKUP] and gemini.is_unhealthy(MAIN)


async def test_both_failing_raises_the_error(monkeypatch):
    install(monkeypatch, {MAIN: (0.0, [], AidaError("unavailable", "a")),
                          BACKUP: (0.0, [], AidaError("rate_limited", "b"))})
    with pytest.raises(AidaError) as ei:
        await run()
    assert ei.value.code in {"unavailable", "rate_limited"}


async def test_a_non_retryable_error_does_not_start_the_backup(monkeypatch):
    started, _ = install(monkeypatch, {MAIN: (0.0, [], AidaError("blocked", "policy")),
                                       BACKUP: (0.0, [b"zz"], None)})
    with pytest.raises(AidaError) as ei:
        await run()
    assert ei.value.code == "blocked" and started == [MAIN]


async def test_winner_failing_mid_clip_ends_quietly_without_a_second_voice(monkeypatch):
    started, _ = install(monkeypatch, {MAIN: (0.0, [b"aa"], AidaError("unavailable", "cut")),
                                       BACKUP: (0.0, [b"zz"], None)})
    out, _ = await run()
    assert out == [b"aa"] and started == [MAIN]


async def test_no_audio_from_either_model_is_an_error(monkeypatch):
    install(monkeypatch, {MAIN: (0.0, [], None), BACKUP: (0.0, [], None)})
    with pytest.raises(AidaError):
        await run()


async def test_a_client_that_leaves_cancels_everything(monkeypatch):
    started, cancelled = install(monkeypatch, {MAIN: (0.0, [b"a" * 2] * 50, None), BACKUP: (5.0, [b"zz"], None)})
    gen = tts.synthesize_stream("Hello there, friend.", "female", {})
    await gen.__anext__()
    await gen.aclose()
    await asyncio.sleep(0.05)
    leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert not leftover


async def test_hedge_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "tts_hedge_after", 0)
    started, _ = install(monkeypatch, {MAIN: (0.3, [b"aa"], None), BACKUP: (0.0, [b"zz"], None)})
    out, _ = await run()
    assert out == [b"aa"] and started == [MAIN]                 # waited for the main model only

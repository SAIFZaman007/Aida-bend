"""Text-to-speech via Gemini TTS (auto-detects Bangla / English).

Returns a 24 kHz mono 16-bit WAV, the same container the old Piper path
returned, so the frontend's decodeAudioData + lip-sync pipeline is unchanged.

Generative TTS can occasionally return silence or clipped audio. Each result is
checked (not silent, duration plausible for the text length); one retry is made
and, if it still fails, the endpoint answers 503 so the client falls back to the
browser voice rather than playing garbage.
"""
import array
import asyncio
import base64
import io
import logging
import wave

from src.core import gemini
from src.core.config import settings
from src.core.errors import AidaError
from src.core.text import dominant_language

log = logging.getLogger("aida.tts")

_sem: asyncio.Semaphore | None = None

# Speech runs ~3–25 chars/second across Bengali/English; bounds are deliberately
# wide — they catch "clipped to nothing" and "rambled for a minute", not style.
_MIN_SEC_PER_CHAR = 0.02
_MAX_SEC_PER_CHAR = 0.5
_MAX_SLACK_SEC = 2.5


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(max(1, settings.tts_concurrency))
    return _sem


def ensure_wav(raw: bytes, sample_rate: int) -> bytes:
    """Gemini returns WAV for unary calls; wrap raw PCM defensively if not."""
    if raw[:4] == b"RIFF":
        return raw
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return buf.getvalue()


def wav_info(wav_bytes: bytes) -> tuple[float, float]:
    """Return (duration_seconds, rms_amplitude_0..1) of a 16-bit PCM WAV."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or settings.tts_sample_rate
        width = wf.getsampwidth()
        pcm = wf.readframes(frames)
    duration = frames / float(rate)
    if width != 2 or not pcm:
        return duration, 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    step = max(1, len(samples) // 20000)  # subsample: plenty for an RMS estimate
    picked = samples[::step]
    if not picked:
        return duration, 0.0
    rms = (sum(s * s for s in picked) / len(picked)) ** 0.5 / 32768.0
    return duration, rms


def is_plausible(text: str, duration: float, rms: float) -> bool:
    chars = max(1, len(text.strip()))
    if rms < 0.002:  # effectively silent
        return False
    if duration < chars * _MIN_SEC_PER_CHAR:
        return False
    if duration > chars * _MAX_SEC_PER_CHAR + _MAX_SLACK_SEC:
        return False
    return True


def _build_input(text: str) -> list[dict]:
    block: dict = {"type": "text", "text": text}
    if settings.tts_style.strip():
        block["annotations"] = [
            {"type": "speech_metadata", "style": settings.tts_style.strip()}
        ]
    return [{"type": "user_input", "content": [block]}]


def tts_models() -> list[str]:
    """Main voice model, then the backup; one that just failed goes last."""
    models = [settings.gemini_tts_model]
    if settings.tts_fallback_model:
        models.append(settings.tts_fallback_model)
    return gemini.order_models(models)


def _interaction_args(model: str, text: str, voice: str) -> dict:
    return dict(
        model=model,
        input=_build_input(text),
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": voice}]},
    )


async def _synthesize_once(text: str, voice: str, model: str, *, attempts: int = 2) -> bytes:
    client = gemini.get_client()

    async def call():
        return await client.aio.interactions.create(**_interaction_args(model, text, voice))

    async with _semaphore():
        interaction = await gemini.call_with_retry(call, timeout=20.0, attempts=attempts)
    audio = getattr(interaction, "output_audio", None)
    data = getattr(audio, "data", None) if audio is not None else None
    if not data:
        raise AidaError("unavailable", "TTS returned no audio")
    raw = base64.b64decode(data) if isinstance(data, (str, bytes)) else bytes(data)
    return ensure_wav(raw, settings.tts_sample_rate)


async def synthesize(text: str, gender: str = "female") -> tuple[bytes, float]:
    """Return (wav_bytes, duration_seconds) for `text` in the matching voice.
    Fails over to the backup voice model when the main one is overloaded."""
    voice = settings.tts_voice(gender, dominant_language(text))
    models = tts_models()
    last: AidaError | None = None
    for i, model in enumerate(models):
        more = i + 1 < len(models)
        try:
            for attempt in (1, 2):
                wav = await _synthesize_once(text, voice, model, attempts=1 if more else 2)
                duration, rms = wav_info(wav)
                if is_plausible(text, duration, rms):
                    gemini.mark_healthy(model)
                    return wav, duration
                last = AidaError(
                    "unavailable", f"implausible audio (dur={duration:.2f}s rms={rms:.4f})"
                )
                log.warning("TTS result rejected on attempt %d: %s", attempt, last.detail)
        except AidaError as err:
            last = err
            if not err.retryable:
                raise
        gemini.mark_unhealthy(model)
        if more:
            log.warning("tts failing over: %s -> %s", model, models[i + 1])
    raise last or AidaError("unavailable", "no voice model succeeded")


# ----------------------------------------------------------------- streaming
async def _stream_model(model: str, text: str, voice: str, info: dict):
    """Yield raw PCM (16-bit LE mono) from one model as Google generates it.

    Interactions API with stream=True: each `step.delta` event whose delta type
    is "audio" carries a base64 chunk of headerless 24 kHz PCM (Google docs:
    ai.google.dev/gemini-api/docs/speech-generation). First audio can arrive
    within a fraction of the time a whole-clip request needs.
    """
    client = gemini.get_client()
    async with _semaphore():
        try:
            stream = await asyncio.wait_for(
                client.aio.interactions.create(
                    **_interaction_args(model, text, voice), stream=True),
                settings.tts_first_audio_timeout,
            )
        except Exception as exc:
            raise gemini.map_exception(exc) from exc
        it = stream.__aiter__()
        first, carry, produced = True, b"", False
        try:
            while True:
                wait = settings.tts_first_audio_timeout if first else settings.tts_idle_timeout
                try:
                    event = await asyncio.wait_for(it.__anext__(), wait)
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    raise gemini.map_exception(exc) from exc
                if getattr(event, "event_type", None) != "step.delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) != "audio" or not getattr(delta, "data", None):
                    continue
                raw = base64.b64decode(delta.data)
                if first and raw[:4] == b"RIFF":  # defensive: a WAV header slipped in
                    raw = raw[44:]
                rate = getattr(delta, "sample_rate", None)
                if rate:
                    info["sample_rate"] = int(rate)
                buf = carry + raw
                usable = len(buf) - (len(buf) % 2)  # never split a 16-bit sample
                carry = buf[usable:]
                if usable:
                    first = False
                    produced = True
                    yield buf[:usable]
        finally:
            aclose = getattr(stream, "aclose", None) or getattr(stream, "close", None)
            if aclose:
                try:
                    res = aclose()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass
        if not produced:
            raise AidaError("unavailable", "TTS stream carried no audio")


async def _synthesize_sequential(models: list[str], text: str, voice: str, info: dict):
    """One model at a time; the next only after the previous failed BEFORE any audio."""
    last: AidaError | None = None
    for i, model in enumerate(models):
        started = False
        try:
            async for chunk in _stream_model(model, text, voice, info):
                started = True
                yield chunk
            gemini.mark_healthy(model)
            return
        except AidaError as err:
            if started:
                log.warning("tts stream cut mid-clip (%s): %s", err.code, err.detail[:120])
                return
            last = err
            if not err.retryable:
                raise
            gemini.mark_unhealthy(model)
            if i + 1 < len(models):
                log.warning("tts failing over: %s -> %s (%s)", model, models[i + 1], err.code)
    raise last or AidaError("unavailable", "no voice model succeeded")


async def _synthesize_hedged(models: list[str], text: str, voice: str, info: dict):
    """Start the backup voice model when the main one is merely SLOW, not only when it
    has failed, and use whichever produces audio first.

    Waiting for a model to time out before trying the next one stacks the delays (8 s
    + 8 s ...); this caps the wait for first audio at roughly `tts_hedge_after` plus the
    backup's own time. The loser is cancelled, and the listener only ever hears ONE
    voice model for a clip: nothing is yielded until a winner has produced a chunk."""
    first, backup = models[0], models[1]
    queue: asyncio.Queue = asyncio.Queue()
    tasks: dict[str, asyncio.Task] = {}
    infos: dict[str, dict] = {}

    async def pump(model: str) -> None:
        sub: dict = infos.setdefault(model, dict(info))
        try:
            async for chunk in _stream_model(model, text, voice, sub):
                await queue.put((model, "chunk", chunk))
            await queue.put((model, "end", None))
        except asyncio.CancelledError:
            raise
        except AidaError as err:
            await queue.put((model, "err", err))
        except Exception as exc:  # defensive: map anything else
            await queue.put((model, "err", gemini.map_exception(exc)))

    def launch(model: str) -> None:
        tasks[model] = asyncio.create_task(pump(model))

    async def stop_all(keep: str | None = None) -> None:
        for m, t in tasks.items():
            if m != keep and not t.done():
                t.cancel()
        for m, t in tasks.items():
            if m != keep:
                try:
                    await t
                except BaseException:
                    pass

    winner: str | None = None
    last: AidaError | None = None
    errors: set[str] = set()
    try:
        launch(first)
        hedge_at = settings.tts_hedge_after
        while winner is None:
            if backup not in tasks:
                try:
                    model, kind, payload = await asyncio.wait_for(queue.get(), hedge_at)
                except asyncio.TimeoutError:
                    log.info("tts %s slow (> %.1fs): also asking %s", first, hedge_at, backup)
                    launch(backup)
                    continue
            else:
                model, kind, payload = await queue.get()
            if kind == "chunk":
                winner = model
                await stop_all(keep=model)
                if model != first and first not in errors:
                    gemini.mark_unhealthy(first)  # too slow: the backup goes first for a while
                info.update({k: v for k, v in infos[model].items() if v})
                gemini.mark_healthy(model)
                yield payload
            elif kind == "err":
                errors.add(model)
                last = payload
                if not payload.retryable:
                    raise payload
                gemini.mark_unhealthy(model)
                if backup not in tasks:
                    log.warning("tts failing over: %s -> %s (%s)", first, backup, payload.code)
                    launch(backup)
                elif errors >= {first, backup}:
                    raise last
            else:  # "end" without audio: treat as a failed model
                errors.add(model)
                last = AidaError("unavailable", "TTS stream carried no audio")
                gemini.mark_unhealthy(model)
                if backup not in tasks:
                    launch(backup)
                elif errors >= {first, backup}:
                    raise last
        # A winner is streaming: pass its chunks through.
        while True:
            model, kind, payload = await queue.get()
            if model != winner:
                continue
            if kind == "chunk":
                yield payload
            elif kind == "end":
                return
            else:
                log.warning("tts stream cut mid-clip (%s): %s", payload.code, payload.detail[:120])
                return
    finally:
        await stop_all()
        if winner is not None and winner in tasks:
            try:
                await tasks[winner]
            except BaseException:
                pass


async def synthesize_stream(text: str, gender: str, info: dict):
    """Async generator of PCM chunks. Falls over to the backup voice model only
    BEFORE the first byte (so the listener never hears two voices); after that a
    failure just ends the clip. `info["sample_rate"]` is filled in as known.

    With a backup model configured the two are HEDGED (see _synthesize_hedged): the
    backup starts as soon as the main model fails or is slower than
    `tts_hedge_after` seconds, instead of after its full timeout."""
    voice = settings.tts_voice(gender, dominant_language(text))
    info.setdefault("sample_rate", settings.tts_sample_rate)
    models = tts_models()
    if len(models) > 1 and settings.tts_hedge_after > 0:
        gen = _synthesize_hedged(models, text, voice, info)
    else:
        gen = _synthesize_sequential(models, text, voice, info)
    try:
        async for chunk in gen:
            yield chunk
    finally:
        await gen.aclose()

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


async def _synthesize_once(text: str, voice: str) -> bytes:
    client = gemini.get_client()

    async def call():
        return await client.aio.interactions.create(
            model=settings.gemini_tts_model,
            input=_build_input(text),
            response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": voice}]},
        )

    async with _semaphore():
        interaction = await gemini.call_with_retry(call, timeout=30.0)
    audio = getattr(interaction, "output_audio", None)
    data = getattr(audio, "data", None) if audio is not None else None
    if not data:
        raise AidaError("unavailable", "TTS returned no audio")
    raw = base64.b64decode(data) if isinstance(data, (str, bytes)) else bytes(data)
    return ensure_wav(raw, settings.tts_sample_rate)


async def synthesize(text: str, gender: str = "female") -> tuple[bytes, float]:
    """Return (wav_bytes, duration_seconds) for `text` in the matching voice."""
    voice = settings.tts_voice(gender, dominant_language(text))
    last: AidaError | None = None
    for attempt in (1, 2):
        wav = await _synthesize_once(text, voice)
        duration, rms = wav_info(wav)
        if is_plausible(text, duration, rms):
            return wav, duration
        last = AidaError(
            "unavailable", f"implausible audio (dur={duration:.2f}s rms={rms:.4f})"
        )
        log.warning("TTS result rejected on attempt %d: %s", attempt, last.detail)
    raise last  # type: ignore[misc]

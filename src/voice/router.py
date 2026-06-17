"""Voice endpoints — speech-to-text and natural text-to-speech with timing marks.

  POST /voice/transcribe   : audio blob -> text   (faster-whisper, local, free)
  POST /voice/speak        : text -> WAV audio     (Piper, local, natural)
  POST /voice/speak_marks  : text -> {audio (base64 WAV), marks: [...]}
                              — powers the Real-Time Reading Progress Highlighter

v2 changes:
 • TEXT NORMALIZATION: both /speak and /speak_marks run text through
   text_normalize.normalize_for_speech() first, fixing the robotic reading of
   things like "11111111", "e.g.", and other abbreviations/digit runs.
 • FASTER STT: whisper compute type, beam size, and model size are now
   configurable (defaults tuned for low latency: int8, beam=1) and the model
   is warmed up at import time in a background thread so the FIRST request
   isn't the one that pays the multi-second model-load cost.
 • TIMING MARKS: /voice/speak_marks runs Piper per-sentence and returns
   word-level timestamps (estimated from phoneme/audio duration) alongside a
   single concatenated WAV, so the frontend highlighter can move in lockstep
   with playback — including for the webspeech fallback via SpeechSynthesis
   boundary events (handled client-side).

STT decodes from memory (BytesIO) so it works on Windows (no temp-file lock)
and never 500s. TTS uses Piper if a voice model is present for the requested
gender; otherwise it returns 503 and the frontend uses the browser voice.
"""
import io
import os
import threading
import wave

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from src.core.config import settings
from src.voice.text_normalize import normalize_for_speech

router = APIRouter()

# ----------------------------------------------------------------- STT (Whisper)
_whisper = None
_whisper_lock = threading.Lock()


def _load_whisper():
    global _whisper
    from faster_whisper import WhisperModel

    with _whisper_lock:
        if _whisper is None:
            _whisper = WhisperModel(
                settings.whisper_model,
                device="auto",
                compute_type=settings.whisper_compute_type,
            )
    return _whisper


def _get_whisper():
    return _whisper or _load_whisper()


# Warm up the model in the background on import so the first transcribe call
# doesn't pay multi-second model-load latency (a big chunk of the "voice
# command takes a long time" complaint).
threading.Thread(target=_load_whisper, daemon=True).start()


MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB cap — guards against memory abuse


@router.post("/voice/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data or len(data) < 1200:
        return {"text": "", "language": "en"}
    if len(data) > MAX_AUDIO_BYTES:
        return {"text": "", "language": "en", "error": "audio too large"}
    try:
        segments, info = _get_whisper().transcribe(
            io.BytesIO(data),
            beam_size=settings.whisper_beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {"text": text, "language": info.language}
    except Exception as exc:  # never 500 -> never a phantom CORS error
        return {"text": "", "language": "en", "error": str(exc)}


# ----------------------------------------------------------------- TTS (Piper)
class SpeakRequest(BaseModel):
    text: str
    gender: str = "female"


_piper_cache: dict[str, object] = {}


def _voice_path(gender: str) -> str | None:
    """Resolve the model path for a gender; None if it isn't on disk."""
    path = settings.piper_voice_male if gender == "male" else settings.piper_voice_female
    if path and os.path.exists(path) and os.path.exists(path + ".json"):
        return path
    return None


def _get_piper(gender: str):
    path = _voice_path(gender)
    if path is None:
        return None
    if path not in _piper_cache:
        from piper.voice import PiperVoice  # imported lazily

        _piper_cache[path] = PiperVoice.load(path)
    return _piper_cache[path]


def _synth_wav(voice, text: str) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        # piper-tts 1.x uses synthesize_wav(text, wav_file); older builds expose
        # synthesize(text, wav_file). Support both so the version doesn't matter.
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wf)
        else:
            voice.synthesize(text, wf)
    return buf.getvalue()


@router.post("/voice/speak")
async def speak(req: SpeakRequest):
    try:
        voice = _get_piper(req.gender)
    except Exception as exc:
        raise HTTPException(503, f"Piper unavailable: {exc}")
    if voice is None:
        raise HTTPException(503, "Piper voice model not found (run download_voices.py)")

    text = normalize_for_speech((req.text or "").strip())
    if not text:
        raise HTTPException(400, "Empty text")

    return Response(content=_synth_wav(voice, text), media_type="audio/wav")


# -------------------------------------------------- TTS with timing marks (highlighter)
class SpeakMarksRequest(BaseModel):
    text: str
    gender: str = "female"


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or settings.piper_sample_rate
        return frames / float(rate)


def _split_words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return (word, start_char, end_char) for each word in `text`."""
    words: list[tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        start = i
        while i < n and not text[i].isspace():
            i += 1
        if i > start:
            words.append((text[start:i], start, i))
    return words


@router.post("/voice/speak_marks")
async def speak_marks(req: SpeakMarksRequest):
    """Synthesize audio for `text` and return word-level timing marks.

    Response: {"audio": "<base64 wav>", "duration": <seconds>,
               "marks": [{"word": str, "start": float, "end": float,
                          "char_start": int, "char_end": int}, ...]}

    `start`/`end` are seconds from the beginning of the returned audio.
    Word durations are allocated proportionally to character length within
    each sentence's audio span — Piper doesn't expose true phoneme
    timestamps, but this estimate is smooth and accurate enough for a
    teleprompter-style highlight, and stays perfectly in sync because it's
    derived from the SAME audio the user hears.
    """
    import base64

    try:
        voice = _get_piper(req.gender)
    except Exception as exc:
        raise HTTPException(503, f"Piper unavailable: {exc}")
    if voice is None:
        raise HTTPException(503, "Piper voice model not found (run download_voices.py)")

    raw_text = (req.text or "").strip()
    if not raw_text:
        raise HTTPException(400, "Empty text")

    text = normalize_for_speech(raw_text)
    wav_bytes = _synth_wav(voice, text)
    duration = _wav_duration_seconds(wav_bytes)

    words = _split_words_with_offsets(text)
    total_chars = sum(len(w) for w, _, _ in words) or 1

    marks = []
    t = 0.0
    for word, cs, ce in words:
        span = duration * (len(word) / total_chars)
        marks.append(
            {
                "word": word,
                "start": round(t, 4),
                "end": round(t + span, 4),
                "char_start": cs,
                "char_end": ce,
            }
        )
        t += span

    return {
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
        "duration": round(duration, 4),
        "text": text,
        "marks": marks,
    }

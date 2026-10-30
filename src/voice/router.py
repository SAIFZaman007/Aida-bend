"""Voice endpoints — speech-to-text and natural text-to-speech with timing marks.

  WS   /voice/stream       : live mic PCM -> interim + final text, WHILE speaking
                              (Gemini Live transcription; see voice/live_stt.py)
  POST /voice/transcribe   : audio blob -> text   (per-clip fallback for the above)
  POST /voice/speak_stream : text -> raw PCM streamed as it is generated (low latency)
  POST /voice/speak        : text -> WAV audio     (Gemini TTS, whole clip)
  POST /voice/speak_marks  : text -> {audio (base64 WAV), marks: [...]}
                              — whole-clip variant with word timings

v2.1: the local faster-whisper / Piper engines were replaced by Gemini so the
assistant can listen and speak Bengali as well as English. REQUEST AND RESPONSE
SHAPES ARE UNCHANGED, so the frontend lip-sync and highlighter keep working.

Failure semantics (preserved from v2):
  • /transcribe never returns a 5xx for provider problems — it answers 200 with
    empty text + an `error` code so the hands-free loop just keeps listening and
    no phantom CORS error appears.
  • /speak and /speak_marks answer 503 with a coded body when synthesis is
    unavailable, which makes the frontend fall back to the browser voice.
"""
import base64
import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.errors import AidaError, http_detail, request_lang
from src.core.ratelimit import limit_stt, limit_tts
from src.core.text import detect_language, dominant_language
from src.voice import live_stt, stt, tts
from src.voice.text_normalize import normalize_for_speech

log = logging.getLogger("aida.voice")
router = APIRouter()

MIN_AUDIO_BYTES = 1200  # shorter than this is a click/blip, not speech


# ----------------------------------------------------------------- STT
@router.post("/voice/transcribe", dependencies=[Depends(limit_stt)])
async def transcribe(request: Request, file: UploadFile = File(...)) -> dict:
    lang = request_lang(request)
    data = await file.read(settings.max_audio_bytes + 1)
    if len(data) > settings.max_audio_bytes:
        raise HTTPException(413, detail=http_detail("too_large", lang))
    if len(data) < MIN_AUDIO_BYTES:
        return {"text": "", "language": "unknown"}
    try:
        text = await stt.transcribe(data, stt.safe_mime(file.content_type))
    except AidaError as err:
        log.warning("transcribe failed code=%s detail=%s", err.code, err.detail[:200])
        return {"text": "", "language": "unknown", "error": err.code,
                "message": err.message(lang)}
    except Exception:
        log.exception("transcribe crashed")
        return {"text": "", "language": "unknown", "error": "internal",
                "message": AidaError("internal").message(lang)}
    return {"text": text, "language": detect_language(text)}


# ----------------------------------------------------------------- live STT
def _origin_allowed(origin: str | None) -> bool:
    """Browsers do not apply CORS to WebSockets, so enforce the same allow-list.
    A missing Origin (non-browser client) is allowed, exactly like plain HTTP."""
    if not origin:
        return True
    allowed = settings.cors_list
    return "*" in allowed or origin in allowed


@router.websocket("/voice/stream")
async def stream_transcribe(ws: WebSocket) -> None:
    if not settings.stt_live_enabled or not _origin_allowed(ws.headers.get("origin")):
        await ws.close(code=1008)  # policy violation: the client falls back to /transcribe
        return
    await ws.accept()
    ip = ws.client.host if ws.client else "unknown"
    await live_stt.serve(ws, ip)


# ----------------------------------------------------------------- TTS
class SpeakRequest(BaseModel):
    text: str = Field(max_length=20_000)
    gender: Literal["female", "male"] = "female"


class SpeakMarksRequest(SpeakRequest):
    pass


async def _synth(req: SpeakRequest, request: Request) -> tuple[bytes, float, str]:
    lang = request_lang(request)
    raw_text = (req.text or "").strip()
    if not raw_text:
        raise HTTPException(400, detail={"code": "bad_request", "message": "Empty text"})
    if len(raw_text) > settings.max_tts_chars:
        raise HTTPException(413, detail=http_detail("too_large", lang))
    text = normalize_for_speech(raw_text, dominant_language(raw_text))
    if not text:
        raise HTTPException(400, detail={"code": "bad_request", "message": "Empty text"})
    try:
        wav, duration = await tts.synthesize(text, req.gender)
    except AidaError as err:
        if err.code == "rate_limited":
            raise HTTPException(
                429,
                detail=http_detail(err.code, lang),
                headers={"Retry-After": str(max(1, err.retry_after))},
            )
        log.warning("speak failed code=%s detail=%s", err.code, err.detail[:200])
        raise HTTPException(503, detail=http_detail(err.code, lang))
    except Exception:
        log.exception("speak crashed")
        raise HTTPException(503, detail=http_detail("unavailable", lang))
    return wav, duration, text


@router.post("/voice/speak", dependencies=[Depends(limit_tts)])
async def speak(req: SpeakRequest, request: Request):
    wav, _, _ = await _synth(req, request)
    return Response(content=wav, media_type="audio/wav")


# -------------------------------------------------- TTS with timing marks (highlighter)
def _split_words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return (word, start_char, end_char) for each word in `text`."""
    words: list[tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        start = i
        while i < n and not text[i].isspace():
            i += 1
        if i > start:
            words.append((text[start:i], start, i))
    return words


def build_marks(text: str, duration: float) -> list[dict]:
    """Word timings estimated proportionally to word length over the audio.

    The TTS API exposes no word timestamps; this estimate is derived from the
    SAME audio the user hears, so it cannot drift. Offsets index into `text`
    (the spoken text); the frontend re-maps timing onto the displayed text.
    """
    words = _split_words_with_offsets(text)
    total_chars = sum(len(w) for w, _, _ in words) or 1
    marks, t = [], 0.0
    for word, cs, ce in words:
        span = duration * (len(word) / total_chars)
        marks.append({"word": word, "start": round(t, 4), "end": round(t + span, 4),
                      "char_start": cs, "char_end": ce})
        t += span
    return marks


@router.post("/voice/speak_marks", dependencies=[Depends(limit_tts)])
async def speak_marks(req: SpeakMarksRequest, request: Request):
    """Response: {"audio": "<base64 wav>", "duration": <seconds>, "text": str,
                  "marks": [{"word","start","end","char_start","char_end"}...]}"""
    wav, duration, text = await _synth(req, request)
    return {
        "audio": base64.b64encode(wav).decode("ascii"),
        "duration": round(duration, 4),
        "text": text,
        "marks": build_marks(text, duration),
    }


# ------------------------------------------------------ TTS streamed as it is generated
@router.post("/voice/speak_stream", dependencies=[Depends(limit_tts)])
async def speak_stream(req: SpeakRequest, request: Request):
    """Raw 16-bit little-endian mono PCM (`X-Sample-Rate`, normally 24000) sent
    while Google is still generating, so playback can begin within a moment.

    The first chunk is awaited BEFORE answering, so a failure (overload, quota)
    is still a clean 503 and the browser can switch to a fallback voice at once.
    """
    lang = request_lang(request)
    raw_text = (req.text or "").strip()
    if not raw_text:
        raise HTTPException(400, detail={"code": "bad_request", "message": "Empty text"})
    if len(raw_text) > settings.max_tts_chars:
        raise HTTPException(413, detail=http_detail("too_large", lang))
    text = normalize_for_speech(raw_text, dominant_language(raw_text))
    if not text:
        raise HTTPException(400, detail={"code": "bad_request", "message": "Empty text"})

    info: dict = {}
    agen = tts.synthesize_stream(text, req.gender, info)
    try:
        first = await agen.__anext__()
    except AidaError as err:
        if err.code == "rate_limited":
            # Quota: say so plainly (429 + Retry-After) so the browser stops asking and
            # speaks with the device voice at once. Already logged once by the quota gate.
            raise HTTPException(
                429,
                detail=http_detail(err.code, lang),
                headers={"Retry-After": str(max(1, err.retry_after))},
            )
        log.warning("speak_stream failed code=%s detail=%s", err.code, err.detail[:200])
        raise HTTPException(503, detail=http_detail(err.code, lang))
    except StopAsyncIteration:
        raise HTTPException(503, detail=http_detail("unavailable", lang))
    except Exception:
        log.exception("speak_stream crashed")
        raise HTTPException(503, detail=http_detail("unavailable", lang))

    async def body():
        try:
            yield first
            async for chunk in agen:
                yield chunk
        finally:
            await agen.aclose()  # client left: stop paying for audio nobody hears

    return StreamingResponse(
        body(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(info.get("sample_rate", settings.tts_sample_rate)),
            "X-Audio-Format": "pcm_s16le",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",  # tell nginx-style proxies not to buffer the stream
        },
    )

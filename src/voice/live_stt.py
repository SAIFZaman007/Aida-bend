"""Real-time speech-to-text over WebSocket (Gemini Live transcription).

    browser ──raw PCM 16 kHz──▶  /api/voice/stream  ──▶  Gemini Live
            ◀──interim / final text──                ◀──  (gemini-3.5-transcribe-live)

Why this exists: the per-clip path uploads the whole sentence AFTER the person
stops talking and only then waits for a transcript (1–5 s of dead air). Here the
audio streams while they speak, Google returns low-latency interim hypotheses
(`interim_input_transcription`, shown as live captions) and a final text
(`input_transcription`) a moment after they stop. The API key never leaves the
server. Language is auto-detected, including Bengali/English code-switching.

Turn control is MANUAL (documented "push-to-talk" mode): the browser's own
voice-activity detector sends `start` / `end`, so one long-lived session serves
many utterances and silence is never uploaded or billed.

Wire protocol (JSON text frames unless noted)
  client → server   binary  : raw 16-bit little-endian mono PCM at 16 kHz
                    {"type":"start"} | {"type":"end"} | {"type":"ping"}
  server → client   {"type":"ready"}                      session usable
                    {"type":"interim","id":n,"text":s}    live caption
                    {"type":"final","id":n,"text":s,"language":"bn|en|unknown"}
                    {"type":"error","code":c,"message":m}  then the socket closes, unless
                      "recoverable":true (+ "id"): only that utterance was refused
                    {"type":"recycle"}                    reconnect (10-min cap)
                    {"type":"pong"}

Google's session cap is 10 minutes, so the server recycles proactively between
utterances; the client reconnects and queues audio meanwhile. If anything here
fails, the client falls back to POST /voice/transcribe (voice/stt.py).
"""
import asyncio
import json
import logging
import time

from fastapi import WebSocket
from google.genai import types

from src.core import gemini, ratelimit
from src.core.config import settings
from src.core.errors import AidaError
from src.core.text import detect_language
from src.voice import stt

log = logging.getLogger("aida.voice.live")

PCM_MIME = "audio/pcm;rate=16000"
BYTES_PER_SECOND = 16000 * 2
MAX_CHUNK_BYTES = 64 * 1024
IDLE_CLOSE_SECONDS = 120.0
FINALIZE_MAX_SECONDS = 4.0
MAX_CONNECTIONS_PER_IP = 4

_connections: dict[str, int] = {}


def merge_final(acc: str, new: str) -> str:
    """Join final segments of ONE utterance without duplicating text.

    Segments are normally consecutive pieces; if the server resends a cumulative
    text (new starts with what we have) it replaces, and an exact repeat is dropped.
    """
    new = (new or "").strip()
    if not new:
        return acc
    if not acc:
        return new
    if new.startswith(acc):
        return new
    if acc.endswith(new):
        return acc
    return f"{acc} {new}"


class _Utterance:
    """Everything known about the utterance currently being transcribed."""

    def __init__(self, uid: int) -> None:
        self.id = uid
        self.acc = ""            # finalized segments, merged
        self.interim = ""        # latest speculative text (after `acc`)
        self.audio_bytes = 0
        self.ended_at: float | None = None
        self.last_event = time.monotonic()
        self.turn_complete = False
        self.final_at: float | None = None   # when a FINAL segment last arrived
        self.rejected = False    # over the rate limit: ignore its audio

    def caption(self) -> str:
        return merge_final(self.acc, self.interim) if self.interim else self.acc


class _Conn:
    def __init__(self, ws: WebSocket, session, ip: str) -> None:
        self.ws = ws
        self.session = session
        self.ip = ip
        self.next_id = 0
        self.cur: _Utterance | None = None   # collecting audio, or awaiting its final
        self.active = False                  # between start and end
        self.finalizer: asyncio.Task | None = None
        self.closing = False

    async def send(self, obj: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            self.closing = True  # client went away


# --------------------------------------------------------------------- client → Gemini
async def _begin(conn: _Conn) -> None:
    if conn.finalizer and not conn.finalizer.done():
        await _flush_now(conn)  # previous utterance still settling: close it first
    conn.next_id += 1
    utt = _Utterance(conn.next_id)
    conn.cur = utt
    conn.active = True
    verdict = await ratelimit.check("stt", conn.ip, "rate_stt_per_min", "daily_stt_cap")
    if verdict:
        utt.rejected = True
        code = verdict[0]
        await conn.send({"type": "error", "code": code, "recoverable": True, "id": utt.id,
                         "message": AidaError(code).message("en")})
        return
    await conn.session.send_realtime_input(activity_start=types.ActivityStart())


async def _end(conn: _Conn) -> None:
    utt = conn.cur
    if not conn.active or utt is None:
        return
    conn.active = False
    utt.ended_at = time.monotonic()
    if utt.rejected:
        conn.cur = None
        return
    await conn.session.send_realtime_input(activity_end=types.ActivityEnd())
    conn.finalizer = asyncio.create_task(_finalize(conn, utt))


async def _finalize(conn: _Conn, utt: _Utterance, force: bool = False) -> None:
    """Emit the final text once Google has gone quiet (or says the turn is done)."""
    started = time.monotonic()
    while not force:
        now = time.monotonic()
        quiet = now - max(utt.ended_at or now, utt.last_event)
        settled = (utt.final_at is not None and utt.ended_at is not None
                   and now - utt.final_at >= settings.stt_live_final_settle)
        if utt.turn_complete or settled or quiet >= settings.stt_live_final_wait:
            break
        if now - started >= FINALIZE_MAX_SECONDS:
            break
        await asyncio.sleep(0.05)
    raw = utt.caption()
    text = stt.clean_transcript(raw)
    if conn.cur is utt:
        conn.cur = None
    msg = {"type": "final", "id": utt.id, "text": text,
           "language": detect_language(text) if text else "unknown"}
    if not text and raw.strip() and stt.wrong_script(raw):
        msg["reason"] = "script"  # noise heard as another language: the client must not retry it
    await conn.send(msg)


async def _flush_now(conn: _Conn) -> None:
    task, utt = conn.finalizer, conn.cur
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if utt is not None:
        await _finalize(conn, utt, force=True)


async def _from_client(conn: _Conn) -> None:
    ws = conn.ws
    while not conn.closing:
        try:
            msg = await asyncio.wait_for(ws.receive(), IDLE_CLOSE_SECONDS)
        except asyncio.TimeoutError:
            return  # nobody there: free the Google session
        if msg["type"] == "websocket.disconnect":
            return
        data = msg.get("bytes")
        if data is not None:
            utt = conn.cur
            if not conn.active or utt is None or utt.rejected or len(data) > MAX_CHUNK_BYTES:
                continue  # audio outside an utterance is never sent to Google
            utt.audio_bytes += len(data)
            if utt.audio_bytes > settings.stt_live_max_utterance_seconds * BYTES_PER_SECOND:
                await _end(conn)  # runaway utterance: cut it here, the browser will start a new one
                continue
            await conn.session.send_realtime_input(
                audio=types.Blob(data=data, mime_type=PCM_MIME))
            continue
        text = msg.get("text")
        if not text:
            continue
        try:
            kind = json.loads(text).get("type")
        except (ValueError, AttributeError):
            continue
        if kind == "start" and not conn.active:
            await _begin(conn)
        elif kind == "end":
            await _end(conn)
        elif kind == "ping":
            await conn.send({"type": "pong"})


# --------------------------------------------------------------------- Gemini → client
async def _from_gemini(conn: _Conn) -> None:
    while not conn.closing:
        async for response in conn.session.receive():
            sc = getattr(response, "server_content", None)
            if not sc:
                continue
            utt = conn.cur
            if utt is None or utt.rejected:
                continue
            now = time.monotonic()
            interim = getattr(sc, "interim_input_transcription", None)
            final = getattr(sc, "input_transcription", None)
            if final is not None and getattr(final, "text", None):
                utt.acc = merge_final(utt.acc, final.text)
                utt.interim = ""
                utt.last_event = now
                utt.final_at = now
                if conn.active and not stt.wrong_script(utt.caption()):
                    await conn.send({"type": "interim", "id": utt.id, "text": utt.caption()})
            elif interim is not None and getattr(interim, "text", None):
                utt.interim = interim.text.strip()
                utt.last_event = now
                if not stt.wrong_script(utt.caption()):  # no live captions in a language we do not speak
                    await conn.send({"type": "interim", "id": utt.id, "text": utt.caption()})
            if getattr(sc, "turn_complete", False):
                utt.turn_complete = True


async def _recycle(conn: _Conn) -> None:
    """Close before Google's 10-minute session cap, but never mid-utterance."""
    await asyncio.sleep(max(30, settings.stt_live_session_seconds))
    while conn.active or (conn.finalizer and not conn.finalizer.done()):
        await asyncio.sleep(0.2)
    await conn.send({"type": "recycle"})


# --------------------------------------------------------------------- entry point
def open_session(model: str, config):
    """Context manager for one Live session (a seam so tests need no network)."""
    return gemini.get_client().aio.live.connect(model=model, config=config)


async def serve(ws: WebSocket, ip: str) -> None:
    """Run one browser connection until it ends. Never raises."""
    if _connections.get(ip, 0) >= MAX_CONNECTIONS_PER_IP:
        await ws.send_text(json.dumps({"type": "error", "code": "rate_limited",
                                       "message": AidaError("rate_limited").message("en")}))
        await ws.close(code=1013)
        return
    _connections[ip] = _connections.get(ip, 0) + 1
    try:
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=stt.audio_transcription_config(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)),
        )
        async with open_session(settings.gemini_stt_live_model, config) as session:
            conn = _Conn(ws, session, ip)
            await conn.send({"type": "ready"})
            tasks = [asyncio.create_task(c(conn)) for c in (_from_client, _from_gemini, _recycle)]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                conn.closing = True
                if conn.finalizer and not conn.finalizer.done():
                    conn.finalizer.cancel()
                for t in tasks:
                    t.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                    raise r
    except Exception as exc:  # the browser then falls back to POST /voice/transcribe
        err = exc if isinstance(exc, AidaError) else gemini.map_exception(exc)
        log.warning("live stt ended code=%s detail=%s", err.code, str(err.detail)[:200])
        try:
            await ws.send_text(json.dumps({"type": "error", "code": err.code,
                                           "message": err.message("en")}))
        except Exception:
            pass
    finally:
        _connections[ip] = max(0, _connections.get(ip, 1) - 1)
        try:
            await ws.close()
        except Exception:
            pass

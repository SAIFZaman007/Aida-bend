"""Find out WHY Gemini calls are slow/failing — in about a minute, with real timings.

    python scripts/diag_gemini.py

Run it inside the deployed backend container (Coolify -> backend -> Terminal) so it
uses the exact same network, key and env vars as production. It talks to the SDK
directly (not through the app), so each result isolates one variable:

  1. which models your key can actually use           (typos / wrong IDs)
  2. chat, non-streaming, with your thinking level      (is the model answering?)
  3. chat, STREAMING, with your thinking level          (does the stream stall?)
  4. chat, streaming, WITHOUT a thinking config         (is the thinking setting the cause?)
  5. speech-to-text call path (a generated test tone)   (STT config accepted?)
  6. text-to-speech                                     (TTS path works?)
  7. embeddings
  8. the fallback model, if GEMINI_FALLBACK_MODEL is set

It prints PASS/FAIL with seconds, then a plain-English verdict. No secrets are printed.
Exit code 0 only if every step passed.
"""
import asyncio
import io
import math
import os
import struct
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.genai import types  # noqa: E402

from src.core import gemini  # noqa: E402
from src.core.config import settings  # noqa: E402

TIMEOUT = 25.0
PROMPT = "Reply with exactly one word: hello"
results: dict[str, bool] = {}


def line(ok: bool, name: str, secs: float | None, note: str = "") -> None:
    t = f"{secs:5.1f}s" if secs is not None else "   -  "
    print(f"[{'PASS' if ok else 'FAIL'}] {t}  {name}  {note}".rstrip(), flush=True)


async def timed(key: str, name: str, factory, timeout: float = TIMEOUT):
    t0 = time.perf_counter()
    try:
        out = await asyncio.wait_for(factory(), timeout)
        results[key] = True
        line(True, name, time.perf_counter() - t0, f"-> {str(out)[:60]!r}")
        return out
    except asyncio.TimeoutError:
        results[key] = False
        line(False, name, time.perf_counter() - t0, f"NO RESPONSE within {timeout:.0f}s")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        results[key] = False
        line(False, name, time.perf_counter() - t0, f"{type(exc).__name__}: {str(exc)[:230]}")
    return None


def cfg(model: str, level: str | None) -> types.GenerateContentConfig:
    tc = None if level is None else gemini.thinking_config(level)
    return types.GenerateContentConfig(
        max_output_tokens=64,
        thinking_config=tc,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def tone_wav(seconds: float = 1.2, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(int(seconds * rate))))
    return buf.getvalue()


async def list_models(client) -> list[str]:
    names: list[str] = []
    pager = await client.aio.models.list()
    async for m in pager:
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions or not actions:
            names.append((m.name or "").removeprefix("models/"))
    return names


async def main() -> int:
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set in this environment.")
        return 2
    client = gemini.get_client()
    chat, level = settings.chat_model, settings.gemini_thinking_level or None
    print(f"chat model={chat}  thinking={level}  stt model={settings.gemini_stt_model} "
          f"(thinking={settings.gemini_stt_thinking_level})  tts={settings.gemini_tts_model}  "
          f"fallback={settings.fallback_model or '-'}")
    print(f"key: set ({len(settings.gemini_api_key)} chars)\n")

    # 1. models
    names = await timed("models", "list models your key can use",
                        lambda: list_models(client), 30) or []
    if names:
        flash = sorted(n for n in names if "flash" in n and "tts" not in n and "live" not in n)
        print(f"       generateContent-capable 'flash' models: {', '.join(flash[:25]) or '(none)'}")
        for label, m in (("chat", chat), ("stt", settings.gemini_stt_model)):
            if m not in names:
                print(f"       !! configured {label} model {m!r} is NOT in that list")

    # 2-4 chat
    await timed("once", f"chat non-streaming  [{chat}, thinking={level}]",
                lambda: _once(client, chat, level))
    await timed("stream", f"chat streaming (first token)  [{chat}, thinking={level}]",
                lambda: _first_token(client, chat, level))
    if level is not None:
        await timed("stream_nothink", f"chat streaming (first token)  [{chat}, NO thinking config]",
                    lambda: _first_token(client, chat, None))

    # 5 STT path
    stt_level = settings.gemini_stt_thinking_level or None
    audio = tone_wav()

    async def stt_call():
        r = await client.aio.models.generate_content(
            model=settings.gemini_stt_model,
            contents=[types.Part.from_bytes(data=audio, mime_type="audio/wav"), "Transcribe this audio."],
            config=cfg(settings.gemini_stt_model, stt_level))
        return r.text
    await timed("stt", f"speech-to-text call  [{settings.gemini_stt_model}, thinking={stt_level}]", stt_call)

    # 6 TTS
    async def tts_call():
        from src.voice import tts
        wav, dur = await tts.synthesize("Hello, this is a test.", "female")
        return f"{dur:.1f}s audio, {len(wav) // 1024} KB"
    await timed("tts", f"text-to-speech  [{settings.gemini_tts_model}]", tts_call, 40)

    # 7 embeddings
    async def emb():
        from src.core.llm import llm
        v = await llm.embed("hello world")
        if not v:
            raise RuntimeError("embedding returned nothing (see logs)")
        return f"{len(v)} dims"
    await timed("embed", f"embeddings  [{settings.gemini_embed_model}]", emb, 20)

    # 8 fallback
    if settings.fallback_model:
        await timed("fallback", f"fallback model streaming  [{settings.fallback_model}]",
                    lambda: _first_token(client, settings.fallback_model, None))

    verdict()
    return 0 if all(results.values()) else 1


async def _once(client, model, level):
    r = await client.aio.models.generate_content(model=model, contents=PROMPT, config=cfg(model, level))
    return r.text


async def _first_token(client, model, level):
    stream = await client.aio.models.generate_content_stream(
        model=model, contents=PROMPT, config=cfg(model, level))
    async for ch in stream:
        if ch.text:
            return ch.text
    raise RuntimeError("stream ended without text")


def verdict() -> None:
    r = results
    print("\n=== What this means ===")
    if r.get("once") and r.get("stream"):
        print("Chat works (streaming and non-streaming). If the app still stalls, send me the "
              "app log lines from the failing request.")
    elif r.get("once") and not r.get("stream"):
        print("STREAMING is the problem: the model answers when asked without streaming but the "
              "stream never delivers. The app now falls back to a non-streaming call by itself "
              "after LLM_FIRST_TOKEN_TIMEOUT seconds (default 15).")
    elif not r.get("once") and r.get("stream_nothink"):
        print("The THINKING setting is the problem: it works without a thinking config. Set "
              "GEMINI_THINKING_LEVEL to a level your model accepts (gemini-3.8-flash: low, medium, "
              "high) or leave it empty.")
    elif not r.get("once") and not r.get("stream"):
        print("The chat model is not answering at all (see the error text above: wrong ID, no "
              "access, quota, or the service is overloaded). Set GEMINI_FALLBACK_MODEL to a "
              "'flash' model from the list above that works, then re-run this script.")
    if r.get("stt") is False:
        print("Speech-to-text FAILS here, so voice input cannot work until this step passes.")
    if r.get("tts") is False:
        print("Text-to-speech FAILS: replies will use the browser voice (no Bengali guarantee).")
    if r.get("embed") is False:
        print("Embeddings FAIL: memory recall is skipped (chat still works).")
    if all(r.values()):
        print("Everything passed.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

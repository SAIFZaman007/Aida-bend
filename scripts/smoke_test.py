"""Live end-to-end check of every AI role with YOUR key and model IDs.

    python scripts/smoke_test.py

Verifies, for Bengali AND English: chat reply language, embeddings (dimension +
distances, to help tune MEMORY_MAX_DISTANCE), text-to-speech, and a TTS -> STT
round trip (the synthesized audio is transcribed back and compared with the
original text — a self-contained check that Bengali audio is both spoken and
understood). Exit code 0 only if everything passes. It makes ~10 small API calls.
"""
import asyncio
import difflib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.prompts import system_prompt  # noqa: E402
from src.core import gemini  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.core.llm import llm  # noqa: E402
from src.core.text import script_counts  # noqa: E402
from src.voice import stt, tts  # noqa: E402

BN = "আজকের আবহাওয়া খুব সুন্দর, আমি একটু হাঁটতে বের হব।"
EN = "The weather is lovely today, so I will go for a short walk."
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = "") -> None:
    results.append((name, ok, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {note}")


def bn_ratio(t: str) -> float:
    bn, lat = script_counts(t)
    return bn / (bn + lat) if bn + lat else 0.0


def cosine_distance(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def chat(prompt: str) -> str:
    msgs = [{"role": "system", "content": system_prompt("executive")},
            {"role": "user", "content": prompt}]
    return await llm.complete(msgs)


async def main() -> int:
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set"); return 2

    # ---- chat language behaviour
    r = await chat("আপনি কেমন আছেন? এক বাক্যে বলুন Docker কী।")
    check("chat: Bengali in -> Bengali out", bn_ratio(r) >= 0.6, f"bn_ratio={bn_ratio(r):.2f}")
    check("chat: keeps 'Docker' in Latin letters", "Docker" in r)
    r = await chat("In one sentence, what is Docker?")
    check("chat: English in -> English out", bn_ratio(r) <= 0.05, f"bn_ratio={bn_ratio(r):.2f}")
    r = await chat("ami bhalo achi, tumi kemon acho? ek line e bolo")
    check("chat: Banglish in -> Bengali script out", bn_ratio(r) >= 0.6, f"bn_ratio={bn_ratio(r):.2f}")

    # ---- embeddings
    v = [await llm.embed(t, kind="document") for t in (BN, EN, "Kubernetes pod scheduling policy")]
    check("embed: returns EMBEDDING_DIM vectors", all(x and len(x) == settings.embedding_dim for x in v))
    if all(v):
        same, other = cosine_distance(v[0], v[1]), cosine_distance(v[0], v[2])
        check("embed: cross-lingual same-meaning closer than unrelated", same < other,
              f"distance bn<->en={same:.3f} (same meaning)  bn<->unrelated={other:.3f}"
              f"  | MEMORY_MAX_DISTANCE={settings.memory_max_distance} should sit between them")

    # ---- TTS + round-trip STT, both languages
    for label, text in (("Bengali", BN), ("English", EN)):
        try:
            wav, dur = await tts.synthesize(text, "female")
        except Exception as exc:
            check(f"tts: {label}", False, repr(exc)[:160]); continue
        check(f"tts: {label} audio plausible", True, f"{dur:.1f}s, {len(wav)//1024} KB")
        try:
            heard = await stt.transcribe(wav, "audio/wav")
        except Exception as exc:
            check(f"stt round-trip: {label}", False, repr(exc)[:160]); continue
        sim = difflib.SequenceMatcher(None, text.replace(" ", ""), heard.replace(" ", "")).ratio()
        check(f"stt round-trip: {label} (heard ≈ spoken)", sim >= 0.8,
              f"similarity={sim:.2f}  heard={heard!r}")

    await gemini.aclose()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Measure how long the voice takes to START, per language and sentence length.

    python scripts/bench_latency.py [runs-per-sample]     (default 3)

Calls the same code path as POST /voice/speak_stream (src.voice.tts.synthesize_stream) with
YOUR key and settings, and prints the time to the first audio byte, the total time and the
audio length for a few typical sentences in English and Bengali. Run it before and after
changing GEMINI_TTS_MODEL, TTS_HEDGE_AFTER or the key's plan, to see what actually helps.
Each run is a real request (it counts against your quota).
"""
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings  # noqa: E402
from src.voice import tts  # noqa: E402

SAMPLES = [
    ("en short", "Sure, here is the plan."),
    ("en medium", "A load balancer spreads incoming requests across several servers so that no single one is overloaded."),
    ("bn short", "ঠিক আছে, পরিকল্পনাটি এখানে।"),
    ("bn medium", "লোড ব্যালেন্সার আসা অনুরোধগুলো কয়েকটি সার্ভারের মধ্যে ভাগ করে দেয়, যাতে কোনো একটির উপর চাপ না পড়ে।"),
]


async def one(text: str) -> tuple[float | None, float, float, str]:
    info: dict = {}
    t0 = time.monotonic()
    first = None
    pcm = 0
    try:
        async for chunk in tts.synthesize_stream(text, "female", info):
            if first is None:
                first = time.monotonic() - t0
            pcm += len(chunk)
    except Exception as exc:  # report, keep going
        return None, time.monotonic() - t0, 0.0, f"{type(exc).__name__}: {exc}"[:80]
    seconds = pcm / 2 / int(info.get("sample_rate", settings.tts_sample_rate))
    return first, time.monotonic() - t0, seconds, str(info.get("model", "")) + (" (hedged)" if info.get("hedged") else "")


async def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"models: {tts.tts_models()}  hedge_after={settings.tts_hedge_after}s\n")
    print(f"{'sample':<10} {'first audio':>12} {'total':>8} {'audio':>7}  notes")
    for name, text in SAMPLES:
        firsts = []
        for _ in range(runs):
            f, total, secs, note = await one(text)
            if f is not None:
                firsts.append(f)
            print(f"{name:<10} {('%.2fs' % f) if f is not None else 'FAILED':>12} {total:>7.2f}s {secs:>6.1f}s  {note}")
        if len(firsts) > 1:
            print(f"{'':<10} median {statistics.median(firsts):.2f}s  worst {max(firsts):.2f}s\n")


if __name__ == "__main__":
    asyncio.run(main())

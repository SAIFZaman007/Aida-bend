"""List the Google Cloud Text-to-Speech voices for a language, to confirm the Chirp 3 HD
voice names before putting them in GOOGLE_TTS_VOICE_*.

    GOOGLE_TTS_API_KEY=... python scripts/list_gcloud_voices.py bn-IN
    python scripts/list_gcloud_voices.py en-US chirp      # optional name filter

Uses the key from .env / the environment (src.core.config). Prints name, gender, rate.
"""
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings  # noqa: E402


async def main() -> None:
    lang = sys.argv[1] if len(sys.argv) > 1 else "bn-IN"
    needle = (sys.argv[2] if len(sys.argv) > 2 else "chirp3").lower()
    if not settings.google_tts_api_key:
        sys.exit("GOOGLE_TTS_API_KEY is empty (set it in .env or the environment)")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://texttospeech.googleapis.com/v1/voices",
            params={"languageCode": lang},
            headers={"X-Goog-Api-Key": settings.google_tts_api_key},
        )
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text[:400]}")
    voices = [v for v in r.json().get("voices", []) if needle in v.get("name", "").lower()]
    for v in sorted(voices, key=lambda v: v["name"]):
        print(f'{v["name"]:36} {v.get("ssmlGender", ""):8} {v.get("naturalSampleRateHertz", "")} Hz')
    print(f"\n{len(voices)} voice(s) for {lang}" + (f" matching {needle!r}" if needle else ""))


if __name__ == "__main__":
    asyncio.run(main())

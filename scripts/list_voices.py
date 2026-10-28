"""List the Gemini voices available to your key (incl. the Extended Voice Library)
so you can pick Bangla voices for TTS_VOICE_FEMALE_BN / TTS_VOICE_MALE_BN.

    python scripts/list_voices.py [filter-text]

The voice object's fields are printed as returned by the API; the shape is
provider-defined, so this prints raw attributes instead of assuming a schema.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import gemini  # noqa: E402


async def main() -> None:
    needle = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    client = gemini.get_client()
    pager = await client.aio.voices.list()
    shown = 0
    async for v in pager if hasattr(pager, "__aiter__") else _iter(pager):
        blob = json.dumps(
            v.model_dump() if hasattr(v, "model_dump") else getattr(v, "__dict__", str(v)),
            ensure_ascii=False, default=str,
        )
        if needle in blob.lower():
            print(blob)
            shown += 1
    print(f"\n{shown} voice(s) shown" + (f" matching {needle!r}" if needle else ""))
    await gemini.aclose()


async def _iter(pager):
    for item in pager:
        yield item


if __name__ == "__main__":
    asyncio.run(main())

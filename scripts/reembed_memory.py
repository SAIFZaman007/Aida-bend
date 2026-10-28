"""Re-embed ALL stored memory chunks with the configured embedding model.

REQUIRED once after switching embedding models (e.g. Ollama `nomic-embed-text`
-> Gemini): vectors from different models live in different spaces, so old
vectors would make similarity search return nonsense even though the column
dimension (768) still matches.

    python scripts/reembed_memory.py [--dry-run] [--batch 50]

Safe to re-run; it only rewrites the `embedding` column. Chunks whose embedding
fails are left untouched and reported (re-run later to retry them).
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func, select, update  # noqa: E402

from src.core.config import settings  # noqa: E402
from src.core.database import SessionLocal, aclose, init_db  # noqa: E402
from src.core.llm import llm  # noqa: E402
from src.memory.models import MemoryChunk  # noqa: E402


async def main(batch: int, dry: bool) -> int:
    await init_db()
    async with SessionLocal() as s:
        total = (await s.execute(select(func.count()).select_from(MemoryChunk))).scalar_one()
    print(f"{total} chunk(s); model={settings.gemini_embed_model} dim={settings.embedding_dim}")
    if dry or total == 0:
        return 0
    done = failed = 0
    last_id = ""
    while True:
        async with SessionLocal() as s:
            rows = (await s.execute(
                select(MemoryChunk.id, MemoryChunk.content)
                .where(MemoryChunk.id > last_id)
                .order_by(MemoryChunk.id).limit(batch)
            )).all()
        if not rows:
            break
        for cid, content in rows:
            last_id = cid
            vec = await llm.embed(content, kind="document")
            if vec is None:
                failed += 1
                continue
            async with SessionLocal() as s:
                await s.execute(update(MemoryChunk).where(MemoryChunk.id == cid)
                                .values(embedding=vec))
                await s.commit()
            done += 1
        print(f"  re-embedded {done}/{total} (failed {failed})")
    await llm.aclose()
    await aclose()
    print(f"done: {done} updated, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.batch, a.dry_run)))

"""Long-term memory — pgvector-backed, SQL-side similarity search.

v2 change: recall() now runs a single SQL query using pgvector's cosine
distance operator (`<=>`) against the HNSW index, instead of pulling every
chunk's embedding into Python and scoring with numpy. This is the main fix
for "system gets slower after bulk session usage" — the old approach grew
linearly (and re-parsed JSON) with the number of stored chunks; this is now
an index-accelerated ANN lookup regardless of project size.

remember() is unchanged in spirit: embed the text, store the chunk. The
embedding is now passed straight through as a Python list of floats — pgvector
handles the storage format.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.llm import llm
from src.memory.models import MemoryChunk

_MAX_CHARS = 300  # cap injected context length per chunk


async def remember(
    session: AsyncSession, project_id: str, content: str, source: str = "note"
) -> MemoryChunk | None:
    content = content.strip()
    if not content:
        return None
    vector = await llm.embed(content)
    if vector is None:
        return None
    chunk = MemoryChunk(
        project_id=project_id, content=content, source=source, embedding=vector
    )
    session.add(chunk)
    await session.commit()
    await session.refresh(chunk)
    return chunk


async def recall(
    session: AsyncSession, project_id: str, query: str, k: int = 3
) -> list[str]:
    query_vec = await llm.embed(query)
    if query_vec is None:
        return []

    # Cosine distance (smaller = more similar); pgvector's `<=>` uses the
    # HNSW index created in core/database.py.
    stmt = (
        select(MemoryChunk.content)
        .where(MemoryChunk.project_id == project_id)
        .order_by(MemoryChunk.embedding.cosine_distance(query_vec))
        .limit(k)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [c[:_MAX_CHARS] for c in rows]

"""Long-term memory — pgvector-backed, SQL-side similarity search.

recall() runs a single SQL query using pgvector's cosine distance operator
(`<=>`) against the HNSW index.

v2.1 anti-hallucination changes:
 • RELEVANCE CUT-OFF. recall() used to return the top-k chunks no matter how
   unrelated they were, so every turn injected "context" that could mislead
   the model. Chunks farther than settings.memory_max_distance are now dropped
   (zero chunks is a valid, safe answer).
 • Query vs document embeddings are requested separately (`kind=`), so models
   or prefixes that distinguish them can be used.
"""
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.llm import llm
from src.memory.models import MemoryChunk

log = logging.getLogger("aida.memory")

_MAX_CHARS = 300  # cap injected context length per chunk


async def remember(
    session: AsyncSession, project_id: str, content: str, source: str = "note"
) -> MemoryChunk | None:
    content = content.strip()
    if not content:
        return None
    vector = await llm.embed(content, kind="document")
    if vector is None:
        return None
    chunk = MemoryChunk(
        project_id=project_id, content=content, source=source, embedding=vector
    )
    session.add(chunk)
    await session.commit()
    await session.refresh(chunk)
    return chunk


def recall_statement(project_id: str, query_vec: list[float], k: int, max_distance: float):
    """The recall query (separate so it can be compiled/inspected in tests)."""
    dist = MemoryChunk.embedding.cosine_distance(query_vec)
    return (
        select(MemoryChunk.content, dist.label("distance"))
        .where(MemoryChunk.project_id == project_id)
        .where(dist <= max_distance)
        .order_by(dist)
        .limit(k)
    )


async def recall(
    session: AsyncSession, project_id: str, query: str, k: int | None = None
) -> list[str]:
    query_vec = await llm.embed(query, kind="query")
    if query_vec is None:
        return []
    stmt = recall_statement(
        project_id, query_vec, k or settings.memory_recall_k, settings.memory_max_distance
    )
    rows = (await session.execute(stmt)).all()
    if rows:
        log.debug("recall: %d chunk(s), best distance %.3f", len(rows), rows[0][1])
    return [content[:_MAX_CHARS] for content, _ in rows]

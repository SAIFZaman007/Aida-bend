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

v2.5 — a KNOWLEDGE BASE PER MODE (no schema change, no migration):
 • An exchange is stored with source "message:<persona>" (see message_source()).
   A specialist mode (Coder, Security, …) recalls ONLY its own exchanges plus the
   shared notes of the project (any source that does not start with "message");
   it no longer sees what was said in the other modes.
 • Non-stop is the generalist: it recalls everything in the project.
 • Exchanges stored before v2.5 (plain source "message", mode unknown) are
   therefore visible to Non-stop only.
"""
import logging

from sqlalchemy import not_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.prompts import UNRESTRICTED_PERSONA
from src.core.config import settings
from src.core.llm import llm
from src.memory.models import MemoryChunk

log = logging.getLogger("aida.memory")

_MAX_CHARS = 300  # cap injected context length per chunk

MESSAGE_PREFIX = "message"


def message_source(persona: str | None) -> str:
    """The `source` tag for an exchange stored while `persona` was active."""
    return f"{MESSAGE_PREFIX}:{persona}" if persona else MESSAGE_PREFIX


def sees_everything(persona: str | None) -> bool:
    """Only the unrestricted mode (and an unknown caller) reads every partition."""
    return not persona or persona == UNRESTRICTED_PERSONA


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


def recall_statement(
    project_id: str,
    query_vec: list[float],
    k: int,
    max_distance: float,
    persona: str | None = None,
):
    """The recall query (separate so it can be compiled/inspected in tests).

    With a specialist `persona` the rows are limited to that mode's own exchanges
    and to shared notes; with Non-stop (or no persona) nothing is filtered."""
    dist = MemoryChunk.embedding.cosine_distance(query_vec)
    stmt = (
        select(MemoryChunk.content, dist.label("distance"))
        .where(MemoryChunk.project_id == project_id)
        .where(dist <= max_distance)
    )
    if not sees_everything(persona):
        stmt = stmt.where(
            or_(
                MemoryChunk.source == message_source(persona),
                not_(MemoryChunk.source.startswith(MESSAGE_PREFIX)),
            )
        )
    return stmt.order_by(dist).limit(k)


async def recall(
    session: AsyncSession,
    project_id: str,
    query: str,
    k: int | None = None,
    persona: str | None = None,
) -> list[str]:
    query_vec = await llm.embed(query, kind="query")
    if query_vec is None:
        return []
    stmt = recall_statement(
        project_id,
        query_vec,
        k or settings.memory_recall_k,
        settings.memory_max_distance,
        persona,
    )
    rows = (await session.execute(stmt)).all()
    if rows:
        log.debug("recall: %d chunk(s), best distance %.3f", len(rows), rows[0][1])
    return [content[:_MAX_CHARS] for content, _ in rows]

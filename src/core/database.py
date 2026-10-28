"""Database wiring — Postgres 18 + pgvector.

v2: replaced SQLite (aiosqlite) with Postgres + the pgvector extension.
SQLite was the root cause of the "DATABASE Error after bulk session usage"
(SQLite serializes writes at the file level — under concurrent streaming
chat + background memory writes it throws "database is locked" once enough
sessions pile up). Postgres gives us real concurrent connections, a proper
connection pool, and native vector similarity search via pgvector — which
also removes the old Python-side numpy cosine-similarity loop in
memory/service.py (now done in SQL with an index).

`init_db()`:
  1. Ensures the `vector` extension exists (prints a clear install guide if missing).
  2. Creates all tables.
  3. Creates an HNSW index on memory_chunks.embedding for fast ANN search.

This module is intentionally the ONLY place that knows about pgvector setup,
so the rest of the app just uses normal SQLAlchemy.

WINDOWS SETUP (if you get "extension vector is not available"):
  See WINDOWS_SETUP.md in the backend folder — it walks through installing
  pgvector from source or via the pre-built Windows zip.
"""
import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings

log = logging.getLogger("aida.db")


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create the pgvector extension, tables, and the HNSW index.

    Safe to run repeatedly (CREATE ... IF NOT EXISTS everywhere).

    Raises RuntimeError with a human-readable install guide if pgvector is
    not installed on the Postgres server — this is the most common setup
    failure on Windows where bare Postgres installs don't include pgvector.
    """
    from src.memory import models  # noqa: F401  (register models on Base)

    async with engine.begin() as conn:
        # --- pgvector extension ---
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as exc:
            _raise_pgvector_guide(exc)

        # --- tables ---
        await conn.run_sync(Base.metadata.create_all)

        # --- HNSW index for fast approximate nearest-neighbour search ---
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_memory_chunks_embedding_hnsw "
                "ON memory_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )

    log.info("Database schema ready (pgvector + HNSW index OK)")


def _raise_pgvector_guide(original_exc: Exception) -> None:
    """Raise a RuntimeError explaining how to get pgvector onto the server.

    The message is environment-neutral (Docker/Coolify, managed, Windows) and
    always includes the database's own error, because `CREATE EXTENSION` can
    also fail for other reasons (e.g. missing privilege), not only "not installed".
    """
    guide = f"""
============================================================
 Could not enable the pgvector extension (CREATE EXTENSION vector)
============================================================
Database said: {original_exc}

If it says the extension is "not available": the Postgres SERVER has no
pgvector installed. Plain Postgres images and the standard Windows installer
do not include it. Use one that does:

  * Docker / Coolify : run Postgres from the image  pgvector/pgvector:pg18
                       (compose service or the database resource's image field).
  * Local Windows    : build pgvector against your Postgres (needs Visual Studio
                       C++ Build Tools) - steps: https://github.com/pgvector/pgvector#windows
  * Hosted Postgres  : pick a provider/plan that lists pgvector, then enable it.

If it says "permission denied": run once as a superuser in this database:
    CREATE EXTENSION IF NOT EXISTS vector;

Then restart the app.
============================================================
"""
    raise RuntimeError(guide) from original_exc


async def aclose() -> None:
    await engine.dispose()
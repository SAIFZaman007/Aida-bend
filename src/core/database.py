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
    """Raise a RuntimeError with clear Windows install instructions."""
    guide = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 pgvector extension not found on your Postgres server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The official Postgres Windows installer does NOT include pgvector.
You need to install it separately. Two options:

OPTION A — Pre-built Windows ZIP (fastest, no compiler needed)
  1. Go to: https://github.com/pgvector/pgvector/releases
  2. Download the ZIP matching your Postgres version (e.g. pgvector-pg18-win64.zip)
  3. Extract it — you'll find:
       vector.dll   → copy to  C:\\Program Files\\PostgreSQL\\18\\lib\\
       vector.control → copy to  C:\\Program Files\\PostgreSQL\\18\\share\\extension\\
       vector--*.sql  → copy to  C:\\Program Files\\PostgreSQL\\18\\share\\extension\\
  4. Restart Postgres: Services → postgresql-x64-18 → Restart
  5. Run:  python app.py

OPTION B — Build from source with Visual Studio
  1. Install Visual Studio Build Tools (C++ workload)
  2. Open "x64 Native Tools Command Prompt for VS"
  3. Run:
       set "PGROOT=C:\\Program Files\\PostgreSQL\\18"
       git clone https://github.com/pgvector/pgvector.git
       cd pgvector
       nmake /F Makefile.win
       nmake /F Makefile.win install
  4. Restart Postgres service.
  5. Run:  python app.py

OPTION C — Use Docker (recommended for production, skips all of this)
  docker compose up -d
  (The pgvector/pgvector:pg18 image has everything pre-installed.)

See WINDOWS_SETUP.md for a full step-by-step with screenshots.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    raise RuntimeError(guide) from original_exc


async def aclose() -> None:
    await engine.dispose()
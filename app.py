"""Application entrypoint.

Run locally with:
    python app.py
or:
    uvicorn app:app --host 127.0.0.1 --port 8000

Pre-flight checks run at startup:
  • Postgres + pgvector connectivity (fatal — app cannot work without it)
  • Redis connectivity (non-fatal — degrades to DB-only, logs a warning)
  • Ollama model availability (non-fatal — warns, chat will fail at runtime)
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.chat.router import router as chat_router
from src.core import redis_client
from src.core.config import settings
from src.core.database import aclose as db_aclose
from src.core.database import init_db
from src.core.llm import llm
from src.voice.router import router as voice_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("aida")


async def _check_redis() -> bool:
    """Ping Redis. Returns True if reachable, False + logs warning otherwise."""
    try:
        r = redis_client.get_redis()
        await r.ping()
        return True
    except Exception as exc:
        log.warning(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            " Redis not reachable — running WITHOUT cache\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            " Chat history will be read from Postgres on every turn\n"
            " (a little slower, but fully functional).\n\n"
            " To enable Redis on Windows:\n"
            "   Option A — WSL2 (recommended):\n"
            "     wsl --install                    # one-time setup\n"
            "     wsl -e sh -c 'sudo apt install redis-server -y && redis-server --daemonize yes'\n\n"
            "   Option B — Memurai (native Windows Redis port):\n"
            "     https://www.memurai.com/get-memurai  (free for dev)\n\n"
            "   Option C — Docker:\n"
            "     docker run -d -p 6379:6379 redis:8-alpine\n\n"
            " Error: %s\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            exc,
        )
        return False


async def _check_ollama() -> None:
    """Log a warning if the configured Ollama models aren't available."""
    if settings.is_anthropic:
        return
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            tags = {m["name"] for m in r.json().get("models", [])}
            missing = []
            for model in [settings.chat_model, settings.coder_model, settings.embed_model]:
                # Ollama tag matching: "llama3.2" matches "llama3.2:latest"
                if not any(t == model or t.startswith(model + ":") for t in tags):
                    missing.append(model)
            if missing:
                log.warning(
                    "Ollama models not found (run `ollama pull <model>` to fix): %s",
                    missing,
                )
    except Exception as exc:
        log.warning("Could not reach Ollama at %s: %s", settings.ollama_base_url, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Database (fatal if Postgres/pgvector is missing — raises RuntimeError
    #    with install instructions for Windows users).
    await init_db()

    # 2. Redis (non-fatal — warns and continues without cache).
    redis_ok = await _check_redis()

    # 3. Ollama model check (non-fatal warning only).
    await _check_ollama()

    log.info(
        "AIDA backend ready  provider=%s  chat=%s  coder=%s  db=postgres+pgvector  cache=%s",
        settings.llm_provider,
        settings.chat_model,
        settings.coder_model,
        "redis" if redis_ok else "DISABLED (Postgres fallback)",
    )
    yield

    # Cleanly close pooled clients on shutdown.
    await llm.aclose()
    await redis_client.aclose()
    await db_aclose()


app = FastAPI(title="A.I.D.A — Personal AI Assistant", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(voice_router, prefix="/api")


@app.get("/api/health")
async def health() -> dict:
    redis_alive = False
    try:
        r = redis_client.get_redis()
        await r.ping()
        redis_alive = True
    except Exception:
        pass

    return {
        "status": "ok",
        "name": "AIDA",
        "version": "2.0.0",
        "provider": settings.llm_provider,
        "model": settings.chat_model,
        "coder_model": settings.coder_model,
        "database": "postgres+pgvector",
        "cache": "redis" if redis_alive else "unavailable (DB fallback active)",
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.app_host, port=settings.app_port, reload=True)
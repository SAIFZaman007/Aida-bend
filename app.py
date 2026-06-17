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
    try:
        r = redis_client.get_redis()
        await r.ping()
        return True
    except Exception as exc:
        log.warning("Redis not reachable — running without cache. Error: %s", exc)
        return False


async def _check_ollama() -> None:
    if settings.is_anthropic:
        return
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            tags = {m["name"] for m in r.json().get("models", [])}
            missing = [
                m for m in [settings.chat_model, settings.coder_model, settings.embed_model]
                if not any(t == m or t.startswith(m + ":") for t in tags)
            ]
            if missing:
                log.warning("Ollama models not yet pulled: %s", missing)
    except Exception as exc:
        log.warning("Could not reach Ollama at %s: %s", settings.ollama_base_url, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    redis_ok = await _check_redis()
    await _check_ollama()
    log.info(
        "AIDA backend ready  provider=%s  chat=%s  coder=%s  db=postgres+pgvector  cache=%s",
        settings.llm_provider,
        settings.chat_model,
        settings.coder_model,
        "redis" if redis_ok else "DISABLED (Postgres fallback)",
    )
    yield
    await llm.aclose()
    await redis_client.aclose()
    await db_aclose()


app = FastAPI(title="A.I.D.A — Personal AI Assistant", version="2.0.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600, 
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
"""
AIDA Backend — Application Entrypoint v2.2.0
"""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.chat.router import router as chat_router
from src.core import gemini, redis_client
from src.core.config import settings
from src.core.database import aclose as db_aclose
from src.core.database import init_db
from src.core.errors import http_detail, request_lang
from src.core.llm import llm
from src.voice.router import router as voice_router

VERSION = "2.5.1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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


async def _check_models() -> None:
    """Warn early (never fail startup) if the key or model IDs are wrong."""
    if not settings.gemini_api_key:
        log.warning(
            "GEMINI_API_KEY is empty — chat, memory and voice will answer "
            "'not configured' until it is set."
        )
        return
    client = gemini.get_client()
    wanted = {settings.gemini_embed_model}
    if not settings.is_anthropic:
        wanted.add(settings.gemini_chat_model)
        wanted.update(m for m in (settings.fallback_model, settings.stt_fallback_model,
                                  settings.gemini_stt_model, settings.gemini_tts_model,
                                  settings.tts_fallback_model) if m)
        if not settings.fallback_model:
            log.warning(
                "GEMINI_FALLBACK_MODEL is empty: when %s is overloaded or out of quota "
                "there is no backup and chat/voice will fail. Set a second model.",
                settings.gemini_chat_model,
            )
    for model in sorted(wanted):
        try:
            await asyncio.wait_for(client.aio.models.get(model=model), 8)
        except Exception as exc:
            err = gemini.map_exception(exc)
            log.warning("Gemini model check failed for %r (%s): %s", model, err.code,
                        err.detail[:200])


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    redis_ok = await _check_redis()
    await _check_models()
    log.info(
        "AIDA backend ready v%s  provider=%s  chat=%s  fallback=%s  thinking=%s/%s  "
        "coder=%s  stt=%s  stt_live=%s  tts=%s  tts_backup=%s  embed=%s  db=postgres+pgvector  cache=%s",
        VERSION,
        settings.llm_provider,
        settings.chat_model,
        settings.fallback_model or "-",
        settings.gemini_thinking_level or "default",
        settings.gemini_stt_thinking_level or "default",
        settings.coder_model,
        settings.gemini_stt_model,
        settings.gemini_stt_live_model if settings.stt_live_enabled else "off",
        settings.gemini_tts_model,
        settings.tts_fallback_model or "-",
        settings.gemini_embed_model,
        "redis" if redis_ok else "DISABLED (Postgres fallback)",
    )
    yield
    await llm.aclose()
    await redis_client.aclose()
    await db_aclose()


app = FastAPI(
    title="A.I.D.A — Personal AI Assistant",
    version=VERSION,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Retry-After", "X-Sample-Rate", "X-Audio-Format"],
    max_age=600,
)


# ── Last-resort error handler ─────────────────────────────────────────
# Unhandled errors would otherwise leak a stack trace AND, because they escape
# the CORS middleware, surface in the browser as a misleading CORS failure.
@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = uuid.uuid4().hex[:12]
    log.exception("unhandled error rid=%s %s %s", request_id, request.method, request.url.path)
    headers = {"X-Request-ID": request_id}
    origin = request.headers.get("origin")
    if origin and origin in settings.cors_list:
        headers.update({
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        })
    detail = http_detail("internal", request_lang(request)) | {"request_id": request_id}
    return JSONResponse(status_code=500, content={"detail": detail}, headers=headers)


app.include_router(chat_router, prefix="/api")
app.include_router(voice_router, prefix="/api")


# ── Root route ────────────────────────────────────────────────────────
@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "aida-backend"})


# ── Health endpoint ───────────────────────────────────────────────────
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
        "version": VERSION,
        "provider": settings.llm_provider,
        "model": settings.chat_model,
        "coder_model": settings.coder_model,
        "voice": {
            "stt": settings.gemini_stt_model,
            "stt_live": settings.gemini_stt_live_model if settings.stt_live_enabled else None,
            "tts": settings.gemini_tts_model,
            "tts_backup": settings.tts_fallback_model or None,
            "stream": True,
        },
        "languages": ["bn", "en"],
        # Reports whether a key is configured — never the key itself.
        "ai_configured": bool(settings.gemini_api_key),
        "database": "postgres+pgvector",
        "cache": "redis" if redis_alive else "unavailable (DB fallback active)",
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )

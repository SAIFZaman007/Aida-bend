# ── AIDA Backend — Production Dockerfile ─────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is only needed for the container health check.
# (ffmpeg / libgomp were only needed by local Whisper/Piper — removed with them.)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application source
COPY . .

# Remove any local .env that may have been accidentally committed.
RUN rm -f .env .env.local

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 aida && chown -R aida:aida /app
USER aida

EXPOSE 8000

# ── Health check (30s start grace period) ─────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health | grep -q '"status":"ok"' || exit 1

# --proxy-headers: behind Coolify/Traefik the real client IP comes from
# X-Forwarded-For, which the per-IP rate limiter depends on. Keep the container
# reachable ONLY through your proxy, otherwise clients could spoof that header.
# Secrets (GEMINI_API_KEY, DATABASE_URL, ...) come from the platform environment.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]

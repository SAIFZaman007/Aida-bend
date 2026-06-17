# ── AIDA Backend — Production Dockerfile ─────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source (voices/*.onnx are committed to repo — no download needed)
COPY . .

# Remove any local .env that may have been accidentally committed.
RUN rm -f .env .env.local

EXPOSE 8000

# ── Health check (30s start grace period) ─────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health | grep -q '"status":"ok"' || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
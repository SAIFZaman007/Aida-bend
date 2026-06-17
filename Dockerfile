# ── AIDA Backend — Production Dockerfile ─────────────────────────────
 
FROM python:3.11-slim
 
WORKDIR /app
 
# System dependencies:w
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Installs Python dependencies first (separate layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copis the rest of the application source.
# voices / directory (committed to repo) is included automatically.
COPY . .
 
EXPOSE 8000
 
# ── Health check ──────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1
 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
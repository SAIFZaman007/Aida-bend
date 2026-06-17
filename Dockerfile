FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg for faster-whisper audio, libgomp for ONNX runtime (Piper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Download Piper voice models (graceful — doesn't fail build if network is slow)
RUN python download_voices.py || echo "Voice download skipped — run manually if needed"

EXPOSE 8000

# Use 0.0.0.0 so Coolify/Docker can reach the port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

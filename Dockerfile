FROM python:3.11-slim
WORKDIR /app
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
# Download Piper voice models (graceful fail)
RUN python download_voices.py || echo "Voice download skipped"
 
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# linux/amd64 is required by the judging VM. The platform is set by the buildx
# --platform flag at build time (not hardcoded here), e.g.:
#   docker buildx build --platform linux/amd64 -t <registry>/<image>:latest --push .
FROM python:3.12-slim

# ffmpeg (+ffprobe) is needed for keyframe extraction.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py styles.py ./

# FIREWORKS_API_KEY is provided at runtime via `-e FIREWORKS_API_KEY=...` — never baked in.
# Reads /input/tasks.json, writes /output/results.json, exits 0.
ENTRYPOINT ["python", "main.py"]

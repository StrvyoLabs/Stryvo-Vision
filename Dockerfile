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

# The judging VM injects NO env vars, so credentials must travel inside the image.
# They are passed at build time as --build-arg (kept out of the repo/Dockerfile source)
# and baked into ENV so a plain `docker run` (no -e) works. A runtime --env-file/-e still
# overrides these, so local dev keeps using .env.
#   docker buildx build --platform linux/amd64 \
#     --build-arg FIREWORKS_API_KEY=fw_... --build-arg GROQ_API_KEY=gsk_... \
#     -t <registry>/<image>:latest --push .
# NOTE: a public image exposes these keys — use spend-capped keys and rotate after judging.
ARG FIREWORKS_API_KEY=""
ARG GROQ_API_KEY=""
ENV FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
    GROQ_API_KEY=$GROQ_API_KEY

# Reads /input/tasks.json, writes /output/results.json, exits 0.
ENTRYPOINT ["python", "main.py"]

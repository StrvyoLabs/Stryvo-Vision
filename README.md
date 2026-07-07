# Stryvo Vision — Video Captioning Agent

A containerized agent that watches short video clips and writes captions in four
styles. It reads `/input/tasks.json`, processes each video through Fireworks AI, writes
`/output/results.json`, and exits `0`.

All model calls go through **Fireworks AI** (OpenAI-compatible API) using
`FIREWORKS_API_KEY`, supplied **at runtime** via env var — never baked into the image.

## Pipeline

```
tasks.json → download video → ffmpeg keyframe sampling → downscale (768px, JPEG)
           → base64 (budget-enforced) → ONE vision request (neutral description)
           → one styling call per requested style (concurrent) → results.json
```

One neutral scene description feeds all four style passes (saves time + cost). The four
style rewrites run concurrently; tasks run concurrently too.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Full pipeline: download, sample, analyze, style, atomic write. |
| `styles.py` | The four style system prompts + few-shot examples — tune here. |
| `Dockerfile` | `python:3.12-slim` + `ffmpeg`, `linux/amd64`. |
| `requirements.txt` | `openai`, `httpx`. |
| `input/tasks.json` | Sample input for the three example clips. |
| `run_local.ps1` / `run_local.sh` | Local end-to-end test harness. |
| `demo/` | Presentation UI (`server.py` + `index.html`) that runs the published image live. |

## Models (swap without hunting)

Set near the top of `main.py`, overridable via env vars:

| Role | Env var | Default (verified LIVE against this account, Jul 2026) |
|------|---------|------------------------------------|
| Vision | `VISION_MODEL` | `accounts/fireworks/models/kimi-k2p6` |
| Styling | `TEXT_MODEL` | `accounts/fireworks/models/glm-5p2` |
| Reasoning | `REASONING_EFFORT` | `none` — required; these are reasoning models |

> These IDs were chosen by querying the account's live `/v1/models` list and probing
> which models accept image input. On this account **`kimi-k2p6` is the only
> image-capable model** — it does the frame analysis. `glm-5p2` (also `deepseek-v4-pro`)
> handles styling. All are reasoning models, so `REASONING_EFFORT=none` is passed to keep
> chain-of-thought OUT of the captions. If you swap in a non-reasoning model, set
> `REASONING_EFFORT=` (empty) to omit the param. Re-run `python probe.py`-style checks if
> your account's catalog differs — Fireworks model IDs change often.

## Build & push (must be linux/amd64, publicly pullable)

```bash
docker buildx build --platform linux/amd64 \
  --tag <registry>/<image>:latest \
  --push .
```

The judging VM is `linux/amd64`; an image without that manifest scores zero. Confirm the
image is **publicly pullable** before submitting.

## Where to set FIREWORKS_API_KEY

Runtime only — never in the image:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=fw_xxx \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  <registry>/<image>:latest
```

## Local test (before pushing)

1. Edit `input/tasks.json` — replace the `https://REPLACE_ME/...` URLs with real,
   publicly-reachable clip URLs (autumn boulevard / orange kitten / office worker).
   The third task requests only 2 styles on purpose — it verifies that **only requested
   styles are emitted**.
2. Build the local image:
   ```bash
   docker build -t stryvo-vision:local .
   ```
3. Set your key and run the harness:
   ```powershell
   $env:FIREWORKS_API_KEY = "fw_xxx"     # PowerShell
   .\run_local.ps1
   ```
   ```bash
   export FIREWORKS_API_KEY=fw_xxx        # bash
   ./run_local.sh
   ```
4. Inspect `output/results.json`.

## Demo UI (for presentations)

A small web UI that lets you paste a video link, pick styles, and see the description +
captions — by running the **published Docker Hub image** live (not a reimplementation).

```powershell
# Docker Desktop running + .env has your key
./demo/start_demo.ps1        # or: python demo/server.py
```

Opens `http://localhost:8000`. Paste a `.mp4` URL (or click an example), choose styles,
hit **Describe & Caption**. It plays the clip, shows the `formal` caption as the video
description, and shows a card per selected style. Each request takes ~20–40s (the backend
does a full `docker run` of the image per clip). The image is **unmodified** — the demo
just always requests the `formal` style and presents it as the description line.

> Backend: `demo/server.py` (Python stdlib only, no pip installs). It writes a one-task
> `tasks.json`, runs `saieesh09/stryvo-vision:latest` with your `.env`, and returns the
> `results.json`. Set `DEMO_PORT` / `DEMO_IMAGE` env vars to override.

## Hosted demo on Streamlit Community Cloud

Streamlit Cloud **can't run Docker**, so the hosted demo reuses the pipeline code
(`main.py` / `styles.py`) directly — same models, prompts, and `reasoning_effort`. The
**Docker image is not modified**; this is a parallel demo path. Because it runs the
pipeline live, it shows the real neutral scene description plus captions.

Files: `streamlit/streamlit_app.py` (entrypoint), `streamlit/requirements.txt`
(subfolder — takes precedence, keeps the Docker `requirements.txt` untouched), and
`packages.txt` at the **repo root** (`ffmpeg` — Streamlit requires `packages.txt` at
root; Docker ignores it).

**Run locally:**
```powershell
./run_streamlit.ps1        # loads .env, installs deps, launches; needs ffmpeg on PATH
```

**Deploy to Community Cloud:**
1. Push this repo to GitHub (see the git steps below).
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo/branch.
3. Set **Main file path** to `streamlit/streamlit_app.py`.
4. **Advanced settings → Secrets**, paste:
   ```toml
   FIREWORKS_API_KEY = "fw_your_key_here"
   ```
5. Deploy. First build installs `ffmpeg` (from `packages.txt`) + Python deps (~1–2 min).

> ⚠️ **Cost/security:** a public Streamlit app runs on **your** `FIREWORKS_API_KEY` —
> every visitor's caption request is billed to you. Unlike the judged Docker image (where
> the caller brings their own key), don't share the URL widely, or add a password
> (`st.text_input(type="password")` gate) / take it down after the demo.

## Output contract

```json
[
  {"task_id": "v1",
   "captions": {"formal": "...", "sarcastic": "...",
                "humorous_tech": "...", "humorous_non_tech": "..."}}
]
```

Only the styles each task requests are emitted, but **every requested style is always
present** — each generation is wrapped in `try/except` with a graceful fallback caption,
and `results.json` is written via temp-file + atomic rename so it is never malformed.

## Tunables (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `NUM_FRAMES` | `12` | Target keyframes per clip. |
| `FRAME_LONGEST_SIDE` | `768` | Downscale longest side (px). |
| `JPEG_QSCALE` | `5` | ffmpeg mjpeg `-q:v` (2=best…31); ~5 ≈ JPEG q80. |
| `SCENE_DETECT` | `0` | `1` = scene-change sampling (`select gt(scene,0.3)`); default even sampling. |
| `MAX_BASE64_MB` | `9.0` | Safety cap under the 10MB Fireworks limit; actual bytes measured in code. |
| `MAX_CONCURRENT_TASKS` | `3` | Lower if your Fireworks tier rate-limits. |
| `REQUEST_TIMEOUT` | `28` | Per Fireworks call (< 30s). |
| `MAX_RETRIES` | `3` | Downloads + Fireworks calls, exponential backoff. |
| `INPUT_PATH` / `OUTPUT_PATH` | `/input/tasks.json` / `/output/results.json` | |

## Frame-budget math (the real limiter is 10MB base64, not the 30-image cap)

- 768px-longest-side JPEG (q≈80) of typical footage ≈ 40–120 KB.
- Base64 inflates ×4/3 → ~55–160 KB/frame.
- 12 frames × 160 KB (worst case) ≈ **1.9 MB** — well under 10 MB.
- `frames_to_budgeted_b64()` sums the **actual** base64 bytes; if it ever exceeds the
  9 MB cap it drops frames evenly (keeping a temporally spread subset) and re-checks.

## Robustness

- Download retries with exponential backoff; Fireworks calls retried on transient errors.
- If vision analysis fails entirely, a neutral stub description still drives best-effort
  captions per style — keys are never dropped.
- All logs go to **stderr**; `results.json` on stdout-adjacent path stays clean JSON.
- Exit `0` on success; `1` only on a truly unrecoverable error (still attempts to write
  valid output first).

## Assumptions / tier notes

- Your key has serverless access to both models above with enough concurrency for
  parallel style calls + parallel tasks. On a constrained tier, lower
  `MAX_CONCURRENT_TASKS` (and consider the 7B vision model).
- Judge clips are ≤ 2 min and downloadable from within the container.
- No answers are cached/hardcoded for the example clips — the pipeline generalizes to the
  hidden eval set.

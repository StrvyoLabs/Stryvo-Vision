#!/usr/bin/env python3
"""
main.py — Video-captioning agent.

Flow per task:
  download video -> ffmpeg keyframe sampling + downscale -> base64 (budget-enforced)
  -> ONE vision request for a neutral scene description (Fireworks -> Groq fallback)
  -> one styling call per requested style (concurrent, Fireworks -> Groq fallback)
  -> collect captions.

Reads   /input/tasks.json
Writes  /output/results.json   (atomic: temp file + rename)
Exits   0 on success.

Design guarantees:
  * Every requested style key is always present (try/except + fallbacks).
  * Output is always valid JSON (build dict -> json.dumps -> temp -> atomic rename).
  * All logging goes to stderr so results.json stays clean.
  * Retries with backoff on downloads and transient Fireworks errors.

All tunables are env-overridable constants near the top so model IDs / frame
settings can be swapped without hunting through the code.
"""

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from styles import (
    SUPPORTED_STYLES,
    STYLE_FALLBACKS,
    build_style_messages,
)

# ---------------------------------------------------------------------------
# Configuration (override any of these via environment variables)
# ---------------------------------------------------------------------------
# Credentials. The judging VM injects NO env vars — it runs the container with your own
# credentials baked in (via Docker build-args -> ENV). At runtime, a mounted --env-file
# still overrides these, so local dev uses .env and the judge uses the baked values.
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Primary provider: Fireworks (verified live against this account, July 2026) ---
# kimi-k2p6 is the only image-capable model on this account; glm-5p2 is the fast styler.
VISION_MODEL = os.getenv("VISION_MODEL", "accounts/fireworks/models/kimi-k2p6")
TEXT_MODEL = os.getenv("TEXT_MODEL", "accounts/fireworks/models/glm-5p2")
# Fireworks models are reasoning models; "none" keeps chain-of-thought OUT of captions.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")

# --- Fallback provider: Groq (used automatically if Fireworks fails/auth/timeouts) ---
# Llama-4 Scout is multimodal (vision); llama-3.3-70b is a fast, clean styler. Neither is
# a reasoning model, so no reasoning_effort param is sent to Groq.
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
# Groq's Llama-4 vision models accept at most 5 images per request (Fireworks allows 30).
GROQ_MAX_IMAGES = int(os.getenv("GROQ_MAX_IMAGES", "5"))


class Provider:
    """One OpenAI-compatible backend with its own client, models, and per-provider caps."""

    def __init__(self, name, base_url, api_key, vision_model, text_model, extra_body, max_images):
        self.name = name
        self.vision_model = vision_model
        self.text_model = text_model
        self.extra_body = extra_body
        self.max_images = max_images  # cap on images per vision request for this provider
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "missing",
            timeout=REQUEST_TIMEOUT, max_retries=0,
        )


def build_providers() -> List["Provider"]:
    """Providers in priority order; only those with a key are included."""
    provs = []
    if FIREWORKS_API_KEY:
        provs.append(Provider(
            "fireworks", FIREWORKS_BASE_URL, FIREWORKS_API_KEY,
            VISION_MODEL, TEXT_MODEL,
            {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else None,
            MAX_IMAGES_PER_REQUEST,
        ))
    if GROQ_API_KEY:
        provs.append(Provider(
            "groq", GROQ_BASE_URL, GROQ_API_KEY,
            GROQ_VISION_MODEL, GROQ_TEXT_MODEL, None,
            GROQ_MAX_IMAGES,
        ))
    return provs

# Frame sampling / downscale settings.
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "12"))          # target keyframes per clip
FRAME_LONGEST_SIDE = int(os.getenv("FRAME_LONGEST_SIDE", "1024"))  # px, longest side
JPEG_QSCALE = int(os.getenv("JPEG_QSCALE", "5"))          # ffmpeg mjpeg -q:v (2=best..31); ~5 ≈ q80
SCENE_DETECT = os.getenv("SCENE_DETECT", "0") == "1"      # optional scene-change sampling

# Fireworks request limits (hard caps from the provider).
MAX_IMAGES_PER_REQUEST = 30
# 10MB hard limit on total base64; use 9MB as a safety cap and measure actual bytes.
MAX_BASE64_BYTES = int(float(os.getenv("MAX_BASE64_MB", "9.0")) * 1024 * 1024)

# Concurrency / timeouts.
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "28"))   # per Fireworks call, < 30s
DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")


def log(*args) -> None:
    """Log to stderr only — stdout/results.json must stay clean."""
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Video download (retry with exponential backoff)
# ---------------------------------------------------------------------------
async def download_video(url: str, dest_path: str) -> bool:
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1 << 16):
                            f.write(chunk)
            size = os.path.getsize(dest_path)
            if size <= 0:
                raise ValueError("downloaded file is empty")
            log(f"[download] ok ({size} bytes) attempt {attempt}: {url}")
            return True
        except Exception as e:  # noqa: BLE001 - broad by design; we retry then fall back
            log(f"[download] attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
    return False


# ---------------------------------------------------------------------------
# Frame extraction (blocking ffmpeg/ffprobe; call via asyncio.to_thread)
# ---------------------------------------------------------------------------
def _probe_duration(video_path: str) -> Optional[float]:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        val = out.stdout.strip()
        return float(val) if val else None
    except Exception as e:  # noqa: BLE001
        log(f"[ffprobe] duration probe failed: {e}")
        return None


def _scale_filter() -> str:
    # Resize so the LONGEST side == FRAME_LONGEST_SIDE, preserve aspect, keep dims even.
    s = FRAME_LONGEST_SIDE
    return (
        f"scale=w='if(gt(iw,ih),{s},-2)':h='if(gt(iw,ih),-2,{s})'"
    )


def extract_frames(video_path: str, work_dir: str) -> List[str]:
    """Extract up to NUM_FRAMES downscaled JPEG frames. Returns file paths (sorted)."""
    scale = _scale_filter()

    if SCENE_DETECT:
        vf = f"select='gt(scene,0.3)',{scale}"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
            "-vf", vf, "-vsync", "vfr", "-q:v", str(JPEG_QSCALE),
            "-frames:v", str(NUM_FRAMES),
            os.path.join(work_dir, "frame_%04d.jpg"),
        ]
        _run_ffmpeg(cmd)
        frames = _list_frames(work_dir)
        if frames:
            return frames[:NUM_FRAMES]
        log("[ffmpeg] scene-detect produced no frames; falling back to even sampling")

    # Even sampling: derive an fps rate from duration so we get ~NUM_FRAMES frames.
    duration = _probe_duration(video_path)
    if duration and duration > 0:
        rate = max(NUM_FRAMES / duration, 0.1)
        vf = f"fps={rate:.6f},{scale}"
    else:
        # Unknown duration: grab 1 fps and cap the count.
        vf = f"fps=1,{scale}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
        "-vf", vf, "-q:v", str(JPEG_QSCALE),
        "-frames:v", str(NUM_FRAMES),
        os.path.join(work_dir, "frame_%04d.jpg"),
    ]
    _run_ffmpeg(cmd)
    return _list_frames(work_dir)[:NUM_FRAMES]


def _run_ffmpeg(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except Exception as e:  # noqa: BLE001
        log(f"[ffmpeg] error running {' '.join(cmd[:6])}...: {e}")


def _list_frames(work_dir: str) -> List[str]:
    return sorted(
        os.path.join(work_dir, f)
        for f in os.listdir(work_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )


def frames_to_budgeted_b64(frame_paths: List[str]) -> List[str]:
    """Base64-encode frames and enforce the <10MB total budget + 30-image cap.

    Measures ACTUAL base64 byte length rather than trusting a magic number; if over
    budget it drops frames evenly (keeps a temporally spread-out subset).
    """
    encoded = []
    for p in frame_paths:
        try:
            with open(p, "rb") as f:
                encoded.append(base64.b64encode(f.read()).decode("ascii"))
        except Exception as e:  # noqa: BLE001
            log(f"[encode] skip frame {p}: {e}")

    # Cap image count first.
    if len(encoded) > MAX_IMAGES_PER_REQUEST:
        encoded = _evenly_subsample(encoded, MAX_IMAGES_PER_REQUEST)

    # Enforce byte budget: drop frames evenly until under the cap.
    def total_bytes(lst):
        return sum(len(x) for x in lst)

    while encoded and total_bytes(encoded) > MAX_BASE64_BYTES:
        keep = max(1, len(encoded) - 1)
        log(
            f"[budget] {total_bytes(encoded)} b64 bytes > {MAX_BASE64_BYTES}; "
            f"reducing {len(encoded)} -> {keep} frames"
        )
        encoded = _evenly_subsample(encoded, keep)
        if keep == 1:
            break

    log(f"[budget] sending {len(encoded)} frames, {total_bytes(encoded)} base64 bytes")
    return encoded


def _evenly_subsample(items: List, k: int) -> List:
    if k >= len(items) or k <= 0:
        return items
    n = len(items)
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


# ---------------------------------------------------------------------------
# Model calls (multi-provider: primary -> fallback)
# ---------------------------------------------------------------------------
VISION_INSTRUCTION = (
    "You are a meticulous visual analyst. The images are keyframes sampled in order from "
    "ONE short video clip. Write a single dense, neutral, factual paragraph (4-6 sentences) "
    "that a caption writer can fully rely on. Be specific and concrete. Explicitly cover:\n"
    "1) The MAIN SUBJECT(S): who/what, how many, appearance (clothing, color, species, type).\n"
    "2) The KEY ACTION or motion happening across the frames (what actually changes/moves).\n"
    "3) The SETTING/location and time of day or weather if visible.\n"
    "4) Notable details: dominant colors, lighting, mood, and anything distinctive.\n"
    "5) Any TEXT, signage, logos/brands, screens, or technology: quote text ONLY if it is "
    "clearly legible, exactly as written; if text is blurry or partial, say 'some signage' "
    "without guessing the words or brand.\n"
    "State only what is actually visible. Do not add humor, opinion, or invented details; "
    "if something is uncertain, describe what is plausibly shown without overstating. "
    "Lead with the single most important, most caption-worthy fact. English only."
)


async def get_scene_description(providers: List[Provider], b64_frames: List[str], task_id: str) -> str:
    # Try each provider; each gets frames capped to its own per-request image limit
    # (Groq's Llama-4 allows 5, Fireworks 30), evenly subsampled so coverage stays broad.
    for prov in providers:
        frames = _evenly_subsample(b64_frames, prov.max_images)
        content = [{"type": "text", "text": VISION_INSTRUCTION}]  # static first, images last
        for b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages = [{"role": "user", "content": content}]
        resp = await _chat_with_retry(
            prov.client, model=prov.vision_model, messages=messages,
            max_tokens=400, temperature=0.2, extra_body=prov.extra_body,
            label=f"vision:{task_id}@{prov.name}({len(frames)} imgs)",
        )
        if resp:
            text = _clean_text(resp.choices[0].message.content or "")
            if text:
                if prov is not providers[0]:
                    log(f"[fallback] vision:{task_id} served by {prov.name}")
                return text
    raise RuntimeError(f"vision analysis returned no text for {task_id}")


async def generate_style_caption(
    providers: List[Provider], style: str, description: str, task_id: str
) -> str:
    try:
        messages = build_style_messages(style, description)
        text = await _call_with_fallback(
            providers, "text_model", messages,
            max_tokens=90, temperature=0.7, label=f"style:{style}:{task_id}",
        )
        if text:
            return text
        raise RuntimeError("empty styling response from all providers")
    except Exception as e:  # noqa: BLE001 - never drop a style key
        log(f"[style] {style} for {task_id} failed, using fallback: {e}")
        return STYLE_FALLBACKS.get(style, "A short video clip.")


async def _call_with_fallback(providers, model_attr, messages, *, max_tokens, temperature, label):
    """Try each provider in order (with per-provider retries); return first clean text."""
    for prov in providers:
        model = getattr(prov, model_attr)
        resp = await _chat_with_retry(
            prov.client, model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            extra_body=prov.extra_body, label=f"{label}@{prov.name}",
        )
        if resp:
            text = _clean_text(resp.choices[0].message.content or "")
            if text:
                if prov is not providers[0]:
                    log(f"[fallback] {label} served by {prov.name}")
                return text
    return None


async def _chat_with_retry(client, *, model, messages, max_tokens, temperature, extra_body, label):
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                timeout=REQUEST_TIMEOUT,
                extra_body=extra_body,
            )
        except Exception as e:  # noqa: BLE001
            log(f"[api] {label} attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
    return None


def _clean_text(text: str) -> str:
    """Strip any leaked reasoning wrappers and surrounding quotes/whitespace."""
    if not text:
        return ""
    # Remove <think>...</think> blocks if a model still emits them.
    while "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>") + len("</think>")
        text = (text[:start] + text[end:]).strip()
    # If an unmatched reasoning tag remains, keep only what follows it.
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip().strip('"').strip()


# ---------------------------------------------------------------------------
# Per-task orchestration
# ---------------------------------------------------------------------------
async def process_task(providers: List[Provider], task: dict) -> dict:
    task_id = task.get("task_id", "unknown")
    video_url = task.get("video_url", "")
    styles = task.get("styles") or SUPPORTED_STYLES
    # De-dup while preserving order; only requested styles are emitted.
    styles = list(dict.fromkeys(styles))

    log(f"[task] start {task_id} styles={styles}")
    description = None
    tmp_dir = tempfile.mkdtemp(prefix=f"vid_{task_id}_")
    try:
        video_path = os.path.join(tmp_dir, "video.mp4")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        if await download_video(video_url, video_path):
            frames = await asyncio.to_thread(extract_frames, video_path, frames_dir)
            log(f"[task] {task_id}: extracted {len(frames)} frames")
            if frames:
                b64 = await asyncio.to_thread(frames_to_budgeted_b64, frames)
                if b64:
                    try:
                        description = await get_scene_description(providers, b64, task_id)
                        log(f"[task] {task_id}: description ok ({len(description)} chars)")
                    except Exception as e:  # noqa: BLE001
                        log(f"[task] {task_id}: vision failed: {e}")
        else:
            log(f"[task] {task_id}: video download failed after retries")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # If vision failed entirely, still produce best-effort captions from a neutral stub.
    if not description:
        description = (
            "A short real-world video clip showing a scene with some activity and "
            "movement. The exact contents could not be analyzed in detail."
        )
        log(f"[task] {task_id}: using stub description for best-effort captions")

    # Run the requested style rewrites concurrently.
    results = await asyncio.gather(
        *[generate_style_caption(providers, s, description, task_id) for s in styles]
    )
    captions = {style: cap for style, cap in zip(styles, results)}

    # Safety net: guarantee every requested key exists.
    for s in styles:
        if not captions.get(s):
            captions[s] = STYLE_FALLBACKS.get(s, "A short video clip.")

    log(f"[task] done {task_id}")
    return {"task_id": task_id, "captions": captions}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_tasks() -> List[dict]:
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("tasks.json must be a JSON array")
        return data
    except Exception as e:  # noqa: BLE001
        log(f"[input] failed to read {INPUT_PATH}: {e}")
        return []


def write_results_atomic(results: List[dict]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    # ensure_ascii=True -> pure-ASCII output (an em dash is written as its \\uXXXX escape).
    # Valid JSON that parses back to the identical string regardless of the reader's
    # assumed file encoding -- eliminates all mojibake/decoding risk for the judge.
    payload = json.dumps(results, ensure_ascii=True, indent=2)
    dir_name = os.path.dirname(OUTPUT_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, OUTPUT_PATH)  # atomic rename
        log(f"[output] wrote {len(results)} results -> {OUTPUT_PATH}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def run() -> int:
    start = time.time()
    providers = build_providers()
    if not providers:
        log("[fatal] no API keys available (FIREWORKS_API_KEY / GROQ_API_KEY both empty)")
        # Still emit valid (fallback) output so results.json is never missing.
    else:
        chain = " -> ".join(f"{p.name}({p.vision_model.split('/')[-1]}/{p.text_model.split('/')[-1]})"
                            for p in providers)
        log(f"[main] provider chain: {chain}")

    tasks = load_tasks()
    log(f"[main] loaded {len(tasks)} task(s)")

    if not tasks:
        write_results_atomic([])
        return 0

    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def guarded(t):
        async with sem:
            try:
                return await process_task(providers, t)
            except Exception as e:  # noqa: BLE001 - never let one task kill the batch
                tid = t.get("task_id", "unknown")
                styles = list(dict.fromkeys(t.get("styles") or SUPPORTED_STYLES))
                log(f"[task] {tid}: unexpected failure {e}; emitting fallbacks")
                return {
                    "task_id": tid,
                    "captions": {s: STYLE_FALLBACKS.get(s, "A short video clip.") for s in styles},
                }

    results = await asyncio.gather(*[guarded(t) for t in tasks])
    write_results_atomic(list(results))
    log(f"[main] completed in {time.time() - start:.1f}s")
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except Exception as e:  # noqa: BLE001 - last-resort: still try to write something valid
        log(f"[fatal] {e}")
        try:
            write_results_atomic([])
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())

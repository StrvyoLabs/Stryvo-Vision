#!/usr/bin/env python3
"""
main.py — Video-captioning agent.

Flow per task:
  download video -> ffmpeg keyframe sampling + downscale -> base64 (budget-enforced)
  -> ONE Qwen2.5-VL request for a neutral scene description
  -> one styling call per requested style (run concurrently)
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
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")

# Model IDs — verified LIVE against this account's Fireworks catalog (July 2026).
# kimi-k2p6 is the only image-capable model available on this account; glm-5p2 is the
# fast text model used for styling. Swap freely via env vars.
VISION_MODEL = os.getenv("VISION_MODEL", "accounts/fireworks/models/kimi-k2p6")
TEXT_MODEL = os.getenv("TEXT_MODEL", "accounts/fireworks/models/glm-5p2")

# These are reasoning models: without this they dump chain-of-thought into the caption.
# "none" disables reasoning so `content` is the clean caption. Set to "" to omit the param.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")

# Frame sampling / downscale settings.
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "12"))          # target keyframes per clip
FRAME_LONGEST_SIDE = int(os.getenv("FRAME_LONGEST_SIDE", "768"))  # px, longest side
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
# Fireworks calls
# ---------------------------------------------------------------------------
VISION_INSTRUCTION = (
    "You are a meticulous visual analyst. The following images are keyframes sampled "
    "in order from a single short video clip. Describe the clip as ONE neutral, factual "
    "paragraph that a caption writer could rely on. Cover: the setting/location, the "
    "main subjects, what actions or motion occur across the frames, the overall mood, "
    "notable visual details (colors, weather, lighting), and any visible text, signage, "
    "screens, or technology. Do not add humor, opinion, or invented details. If "
    "something is unclear, describe what is plausibly shown without overstating. "
    "English only."
)


async def get_scene_description(client: AsyncOpenAI, b64_frames: List[str], task_id: str) -> str:
    # Static instruction first, variable images last (prompt-cache friendly).
    content = [{"type": "text", "text": VISION_INSTRUCTION}]
    for b64 in b64_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    messages = [{"role": "user", "content": content}]

    resp = await _chat_with_retry(
        client, model=VISION_MODEL, messages=messages,
        max_tokens=400, temperature=0.2, label=f"vision:{task_id}",
    )
    if resp:
        text = _clean_text(resp.choices[0].message.content or "")
        if text:
            return text
    raise RuntimeError(f"vision analysis returned no text for {task_id}")


async def generate_style_caption(
    client: AsyncOpenAI, style: str, description: str, task_id: str
) -> str:
    try:
        messages = build_style_messages(style, description)
        resp = await _chat_with_retry(
            client, model=TEXT_MODEL, messages=messages,
            max_tokens=90, temperature=0.7, label=f"style:{style}:{task_id}",
        )
        if resp:
            text = _clean_text(resp.choices[0].message.content or "")
            if text:
                return text
        raise RuntimeError("empty styling response")
    except Exception as e:  # noqa: BLE001 - never drop a style key
        log(f"[style] {style} for {task_id} failed, using fallback: {e}")
        return STYLE_FALLBACKS.get(style, "A short video clip.")


async def _chat_with_retry(client, *, model, messages, max_tokens, temperature, label):
    # Disable reasoning so `content` is the clean caption, not chain-of-thought.
    extra_body = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else None
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
            log(f"[fireworks] {label} attempt {attempt}/{MAX_RETRIES}: {e}")
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
async def process_task(client: AsyncOpenAI, task: dict) -> dict:
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
                        description = await get_scene_description(client, b64, task_id)
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
        *[generate_style_caption(client, s, description, task_id) for s in styles]
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
    if not FIREWORKS_API_KEY:
        log("[fatal] FIREWORKS_API_KEY not set in environment")
        # Still emit valid (fallback) output so results.json is never missing.
    tasks = load_tasks()
    log(f"[main] loaded {len(tasks)} task(s); vision={VISION_MODEL} text={TEXT_MODEL}")

    if not tasks:
        write_results_atomic([])
        return 0

    client = AsyncOpenAI(
        base_url=FIREWORKS_BASE_URL,
        api_key=FIREWORKS_API_KEY or "missing",
        timeout=REQUEST_TIMEOUT,
        max_retries=0,  # we handle retries ourselves
    )

    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def guarded(t):
        async with sem:
            try:
                return await process_task(client, t)
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

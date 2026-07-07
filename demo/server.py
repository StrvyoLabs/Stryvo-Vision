#!/usr/bin/env python3
"""
Stryvo Vision — local demo backend.

Serves the demo UI and, on each request, runs the PUBLISHED Docker Hub image
(saieesh09/stryvo-vision:latest) exactly as the judge would: writes a one-task
tasks.json, runs the container, reads results.json, returns the captions.

The image is UNMODIFIED — it outputs only captions. The demo always requests the
"formal" style and shows it as the video description (per the chosen design).

Stdlib only — no pip installs. Run:  python demo/server.py   then open http://localhost:8000
Requires: Docker running, and a ../.env with FIREWORKS_API_KEY.
"""

import json
import os
import shutil
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
ENV_FILE = os.path.join(PROJECT, ".env")
WORK_ROOT = os.path.join(HERE, "_work")

IMAGE = os.getenv("DEMO_IMAGE", "saieesh09/stryvo-vision:latest")
PORT = int(os.getenv("DEMO_PORT", "8000"))
RUN_TIMEOUT = int(os.getenv("DEMO_RUN_TIMEOUT", "300"))

SUPPORTED = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run_container(video_url: str, styles: list) -> dict:
    """Run the published image for a single clip and return its result dict."""
    # Always include "formal" so we have a description to show.
    run_styles = list(dict.fromkeys(["formal"] + [s for s in styles if s in SUPPORTED]))

    job = os.path.join(WORK_ROOT, uuid.uuid4().hex)
    in_dir = os.path.join(job, "input")
    out_dir = os.path.join(job, "output")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    try:
        tasks = [{"task_id": "demo", "video_url": video_url, "styles": run_styles}]
        with open(os.path.join(in_dir, "tasks.json"), "w", encoding="utf-8") as f:
            json.dump(tasks, f)

        cmd = [
            "docker", "run", "--rm",
            "--env-file", ENV_FILE,
            "-v", f"{in_dir}:/input:ro",
            "-v", f"{out_dir}:/output",
            IMAGE,
        ]
        log(f"[demo] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT)
        log(proc.stderr[-2000:] if proc.stderr else "(no container stderr)")

        results_path = os.path.join(out_dir, "results.json")
        if not os.path.exists(results_path):
            raise RuntimeError(
                f"container produced no results.json (exit {proc.returncode}). "
                f"stderr tail: {proc.stderr[-400:]}"
            )
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        if not results:
            raise RuntimeError("results.json was empty")
        return results[0]
    finally:
        shutil.rmtree(job, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # quieter default logging
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "index.html not found", "text/plain")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True, "image": IMAGE}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            video_url = (payload.get("video_url") or "").strip()
            styles = payload.get("styles") or SUPPORTED
            if not video_url:
                self._send(400, json.dumps({"error": "video_url is required"}))
                return
            result = run_container(video_url, styles)
            self._send(200, json.dumps({
                "task_id": result.get("task_id", "demo"),
                "captions": result.get("captions", {}),
            }))
        except subprocess.TimeoutExpired:
            self._send(504, json.dumps({"error": f"container timed out after {RUN_TIMEOUT}s"}))
        except Exception as e:  # noqa: BLE001
            log(f"[demo] error: {e}")
            self._send(500, json.dumps({"error": str(e)}))


def main():
    if not os.path.exists(ENV_FILE):
        log(f"WARNING: no .env at {ENV_FILE} — the container will have no API key.")
    os.makedirs(WORK_ROOT, exist_ok=True)
    log(f"Stryvo Vision demo on http://localhost:{PORT}  (image: {IMAGE})")
    log("Open that URL in your browser. Ctrl+C to stop.")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

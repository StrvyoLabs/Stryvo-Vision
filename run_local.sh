#!/usr/bin/env bash
# Local end-to-end test (Linux/macOS/Git-Bash).
#   1) Put real, publicly-reachable video URLs in ./input/tasks.json  (done)
#   2) Put your key in ./.env  (FIREWORKS_API_KEY=fw_...)
#   3) ./run_local.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${HERE}/.env"

[ -f "${ENV_FILE}" ] || { echo "No .env found. Create one with FIREWORKS_API_KEY=fw_..."; exit 1; }
if grep -q "fw_paste_your_key_here" "${ENV_FILE}"; then
  echo "Edit .env and replace fw_paste_your_key_here with your real Fireworks key."; exit 1
fi

docker run --rm \
  --env-file "${ENV_FILE}" \
  -v "${HERE}/input:/input:ro" \
  -v "${HERE}/output:/output" \
  stryvo-vision:local

echo "--- output/results.json ---"
cat "${HERE}/output/results.json"

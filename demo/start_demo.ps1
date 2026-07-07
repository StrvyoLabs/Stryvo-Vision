# Launches the Stryvo Vision demo UI and opens it in your browser.
# Requires: Docker Desktop running, and ..\.env with FIREWORKS_API_KEY.
$demo = $PSScriptRoot
$envFile = Join-Path (Split-Path $demo -Parent) ".env"

if (-not (Test-Path $envFile)) { Write-Error "No .env found at $envFile"; exit 1 }
if (Select-String -Path $envFile -Pattern "fw_paste_your_key_here" -Quiet) {
    Write-Error "Put your real Fireworks key in .env first."; exit 1
}

# Make sure the published image is present (pulls if not).
docker image inspect saieesh09/stryvo-vision:latest *> $null
if (-not $?) { Write-Host "Pulling image..."; docker pull saieesh09/stryvo-vision:latest }

Start-Process "http://localhost:8000"
Write-Host "Demo running at http://localhost:8000  (Ctrl+C to stop)"
python "$demo\server.py"

# Local end-to-end test (Windows PowerShell).
# 1) Put real, publicly-reachable video URLs in .\input\tasks.json  (done)
# 2) Put your key in .\.env  (FIREWORKS_API_KEY=fw_...)
# 3) Run this script.

$here = $PSScriptRoot
$envFile = Join-Path $here ".env"

if (-not (Test-Path $envFile)) {
    Write-Error "No .env file found. Create one with: FIREWORKS_API_KEY=fw_..."
    exit 1
}
if (Select-String -Path $envFile -Pattern "fw_paste_your_key_here" -Quiet) {
    Write-Error "Edit .env and replace fw_paste_your_key_here with your real Fireworks key."
    exit 1
}

docker run --rm `
    --env-file $envFile `
    -v "${here}\input:/input:ro" `
    -v "${here}\output:/output" `
    stryvo-vision:local

Write-Host "`n--- output/results.json ---"
Get-Content "${here}\output\results.json"

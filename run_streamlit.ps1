# Runs the Streamlit demo locally. Loads FIREWORKS_API_KEY from .env, installs deps,
# and launches the app. Requires ffmpeg on PATH locally (Streamlit Cloud gets it from
# the root packages.txt instead).
$root = $PSScriptRoot
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    $key = ((Get-Content $envFile | Where-Object { $_ -match '^FIREWORKS_API_KEY=' }) -replace '^FIREWORKS_API_KEY=', '').Trim()
    if ($key) { $env:FIREWORKS_API_KEY = $key }
}
if (-not $env:FIREWORKS_API_KEY) { Write-Warning "FIREWORKS_API_KEY not set — the app will show a key error." }

python -m pip install -q -r "$root\streamlit\requirements.txt"
python -m streamlit run "$root\streamlit\streamlit_app.py"

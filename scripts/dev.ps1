# Start FastAPI backend + Next.js frontend for local development.
# Usage (from repo root): pwsh scripts/dev.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

Write-Host "Starting NeuraX API on http://127.0.0.1:8000 ..."
$api = Start-Process -FilePath $python -ArgumentList @(
  "-m", "uvicorn", "backend.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"
) -PassThru -NoNewWindow

$frontend = Join-Path $Root "frontend"
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
  Write-Host "Installing frontend dependencies..."
  Push-Location $frontend
  npm install
  Pop-Location
}

Write-Host "Starting Next.js on http://127.0.0.1:3000 ..."
Push-Location $frontend
try {
  npm run dev
} finally {
  Pop-Location
  if ($api -and -not $api.HasExited) {
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
  }
}

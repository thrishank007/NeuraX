# PowerShell script to start the NeuraX FastAPI backend
# Usage: .\scripts\start-backend.ps1

Write-Host "Starting NeuraX Backend API..." -ForegroundColor Green

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Error: Python not found. Please install Python 3.8+ and add it to PATH." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Check if FastAPI and uvicorn are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$fastapiInstalled = python -c "import fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing FastAPI dependencies..." -ForegroundColor Yellow
    pip install fastapi uvicorn python-multipart
}

# Start the FastAPI server
Write-Host "Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

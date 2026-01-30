# PowerShell setup script for NeuraX
# Usage: .\scripts\setup.ps1

Write-Host "NeuraX Setup Script" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check Node.js
Write-Host "Checking Node.js installation..." -ForegroundColor Yellow
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Setup Python virtual environment
Write-Host ""
Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
Write-Host "✓ Python dependencies installed" -ForegroundColor Green

# Setup frontend
Write-Host ""
Write-Host "Setting up frontend..." -ForegroundColor Yellow
Set-Location neurax-frontend

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
    Write-Host "✓ Node.js dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✓ Node.js dependencies already installed" -ForegroundColor Green
}

# Create .env.local if it doesn't exist
if (-not (Test-Path ".env.local")) {
    Copy-Item ".env.local.example" ".env.local"
    Write-Host "✓ Created .env.local from example" -ForegroundColor Green
    Write-Host "  Please edit .env.local with your configuration" -ForegroundColor Yellow
}

Set-Location ..

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the backend:" -ForegroundColor Cyan
Write-Host "  .\scripts\start-backend.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To start the frontend (in a new terminal):" -ForegroundColor Cyan
Write-Host "  .\scripts\start-frontend.ps1" -ForegroundColor White
Write-Host ""

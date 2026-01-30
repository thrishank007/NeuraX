# PowerShell script to start the NeuraX Next.js frontend
# Usage: .\scripts\start-frontend.ps1

Write-Host "Starting NeuraX Frontend..." -ForegroundColor Green

# Check if Node.js is available
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) {
    Write-Host "Error: Node.js not found. Please install Node.js 18+ and add it to PATH." -ForegroundColor Red
    exit 1
}

# Navigate to frontend directory
Set-Location neurax-frontend

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Check if .env.local exists
if (-not (Test-Path ".env.local")) {
    Write-Host "Creating .env.local from example..." -ForegroundColor Yellow
    Copy-Item ".env.local.example" ".env.local"
    Write-Host "Please edit .env.local with your configuration" -ForegroundColor Yellow
}

# Start the development server
Write-Host "Starting Next.js development server on http://localhost:3000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

npm run dev

# Setup and Run script for Windows (PowerShell)

# 1. Check Python Version
Write-Host "Checking Python version..." -ForegroundColor Cyan
$pythonVersion = py -3.12 --version 2>&1
if ($pythonVersion -match "Python 3.12") {
    Write-Host "Python 3.12 found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Error "Python 3.12.x was not found. Please install Python 3.12 (64-bit) from python.org."
    exit 1
}

# 2. Create required directories
Write-Host "Creating log, screenshot, and recording directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force logs, screenshots, recordings | Out-Null

# 3. Create virtual environment
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment .venv..." -ForegroundColor Cyan
    py -3.12 -m venv .venv
} else {
    Write-Host "Existing .venv found, skipping creation." -ForegroundColor Yellow
}

# 4. Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
.venv\Scripts\pip.exe install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt

# 5. Start application
Write-Host "Starting the application..." -ForegroundColor Green
.venv\Scripts\python.exe main.py

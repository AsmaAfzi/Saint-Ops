# Create isolated Python environments for SAINT-OPS (Option A).
# Run from repo root:  .\scripts\setup_env.ps1
#   -All        recreate both venvs (default)
#   -MlOnly     only .venv-ml
#   -BackendOnly only .venv-backend

param(
    [switch]$MlOnly,
    [switch]$BackendOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function New-SaintVenv {
    param(
        [string]$Name,
        [string]$RequirementsPath,
        [string[]]$ExtraPackages = @()
    )

    $venvPath = Join-Path $Root $Name
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan

    if (Test-Path $venvPath) {
        Write-Host "Removing existing $Name..."
        Remove-Item -Recurse -Force $venvPath
    }

    Write-Host "Creating $Name..."
    python -m venv $venvPath

    $pip = Join-Path $venvPath "Scripts\pip.exe"
    $python = Join-Path $venvPath "Scripts\python.exe"

    & $python -m pip install --upgrade pip wheel
    & $pip install -r $RequirementsPath --no-cache-dir
    foreach ($pkg in $ExtraPackages) {
        & $pip install $pkg --no-cache-dir
    }

    Write-Host "Verifying $Name..."
    if ($Name -eq ".venv-ml") {
        & $python -c "from tensorflow.keras.models import Model; import tensorflow as tf; print('tensorflow', tf.__version__)"
    } else {
        & $python -c "import fastapi, uvicorn; print('fastapi', fastapi.__version__)"
    }
    & $python -m pip check
    Write-Host "$Name ready: $venvPath\Scripts\Activate.ps1" -ForegroundColor Green
}

$doMl = -not $BackendOnly
$doBackend = -not $MlOnly

if ($doMl) {
    New-SaintVenv -Name ".venv-ml" -RequirementsPath "ml\requirements.txt"
}

if ($doBackend) {
    New-SaintVenv -Name ".venv-backend" -RequirementsPath "backend\requirements.txt" -ExtraPackages @("pytest", "httpx")
}

Write-Host "`nUsage:" -ForegroundColor Yellow
Write-Host "  ML:      .\.venv-ml\Scripts\Activate.ps1"
Write-Host "  Backend: .\.venv-backend\Scripts\Activate.ps1"

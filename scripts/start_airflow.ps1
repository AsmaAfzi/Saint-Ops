# Clean Airflow startup for SAINT-OPS demo (run from repo root).
# Use when you see "Ooops!" or ERR_EMPTY_RESPONSE on http://localhost:8081

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\lib\docker.ps1"

Ensure-DockerEngine
Stop-K8sIfRunning

Write-Host "Stopping existing stack (all profiles)..." -ForegroundColor Cyan
Invoke-Docker compose --profile monitoring --profile airflow down --remove-orphans

Write-Host "Resetting Airflow database (fixes Fernet / corrupted DB)..." -ForegroundColor Yellow
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker volume rm miniproj_airflow-db-data 2>$null | Out-Null
$ErrorActionPreference = $prevEap

Write-Host "Starting core services + Airflow (first start may take 3-5 min)..." -ForegroundColor Cyan
Invoke-Docker compose --profile monitoring --profile airflow up --detach

Write-Host ""
Write-Host "Wait until webserver is healthy:" -ForegroundColor Cyan
Write-Host "  docker compose --profile airflow ps airflow-webserver"
Write-Host ""
Write-Host "Then open: http://localhost:8081  (auto-signs in as admin for demo)"
Write-Host "Do NOT use port 8080 - local Apache may be bound there."

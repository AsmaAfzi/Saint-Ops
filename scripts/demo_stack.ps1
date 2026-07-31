# SAINT-OPS full demo stack - the ONLY supported way to start Docker Compose.
# Usage:  .\scripts\demo_stack.ps1
#         .\scripts\demo_stack.ps1 -SkipBuild

param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\lib\docker.ps1"

Ensure-DockerEngine
Stop-K8sIfRunning

Write-Host "Stopping any existing stack..." -ForegroundColor Cyan
Invoke-Docker compose --profile monitoring --profile airflow down --remove-orphans

$upArgs = @("compose", "--profile", "monitoring", "--profile", "airflow", "up", "--detach")
if (-not $SkipBuild) { $upArgs += "--build" }

Write-Host "Starting SAINT-OPS core + monitoring + Airflow..." -ForegroundColor Cyan
Invoke-Docker @upArgs

Write-Host ""
Write-Host "Demo URLs:" -ForegroundColor Green
Write-Host "  Dashboard     http://localhost:3200"
Write-Host "  API / Swagger http://localhost:8000/docs"
Write-Host "  MLflow        http://localhost:5000"
Write-Host "  Grafana       http://localhost:3301  (anonymous admin)"
Write-Host "  Prometheus    http://localhost:9090"
Write-Host "  Alertmanager  http://localhost:9093"
Write-Host "  Airflow       http://localhost:8081  (auto-signs in for demo)"
Write-Host ""
Write-Host "Airflow may take 2-5 min to become healthy on first start."
Write-Host ""
Write-Host "Full word-for-word demo script: docs\DEMO_SCRIPT.md" -ForegroundColor Cyan

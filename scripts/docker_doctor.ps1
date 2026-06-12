# Diagnose and recover Docker Desktop when you see 500 Internal Server Error.
# Usage:  .\scripts\docker_doctor.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\lib\docker.ps1"

Write-Host "=== SAINT-OPS Docker Doctor ===" -ForegroundColor Cyan

if (Test-DockerEngine) {
    Write-Host "Docker engine: OK" -ForegroundColor Green
    docker ps --format "table {{.Names}}\t{{.Status}}" 2>$null | Select-Object -First 20
    exit 0
}

Write-Host "Docker engine: NOT RESPONDING (500 errors)" -ForegroundColor Red
Ensure-DockerEngine -MaxRestarts 2

Write-Host ""
Write-Host "Recovery complete. Start the demo with:" -ForegroundColor Green
Write-Host "  .\scripts\demo_stack.ps1"

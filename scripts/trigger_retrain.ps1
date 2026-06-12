# Trigger SAINT-OPS retrain via backend API (Step 4 demo).
# Usage:  .\scripts\trigger_retrain.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\lib\docker.ps1"

Ensure-DockerEngine | Out-Null

Write-Host "Triggering retrain..." -ForegroundColor Cyan
$result = Invoke-RestMethod -Uri "http://localhost:8000/retrain/trigger/demo"

Write-Host "OK - retrain queued" -ForegroundColor Green
$result | ConvertTo-Json -Depth 4

Write-Host ""
Write-Host "Status:" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8000/retrain/status" | ConvertTo-Json -Depth 4

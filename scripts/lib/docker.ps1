# Shared Docker helpers for SAINT-OPS demo scripts (PowerShell).
# Dot-source:  . "$PSScriptRoot\lib\docker.ps1"

function Test-DockerEngine {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    docker info *> $null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap
    return $ok
}

function Restart-DockerDesktop {
    Write-Host "Docker engine not responding - restarting Docker Desktop..." -ForegroundColor Yellow
    Write-Host "(This fixes the recurring '500 Internal Server Error' from dockerDesktopLinuxEngine.)" -ForegroundColor DarkGray

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    Get-Process -Name "Docker Desktop", "com.docker.backend", "com.docker.build" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3

    wsl --shutdown 2>$null | Out-Null
    Start-Sleep -Seconds 5

    $dockerExe = Join-Path ${env:ProgramFiles} "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $dockerExe)) {
        $ErrorActionPreference = $prevEap
        throw "Docker Desktop not found at $dockerExe"
    }
    Start-Process $dockerExe | Out-Null

    $deadline = (Get-Date).AddMinutes(4)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerEngine) {
            Write-Host "Docker engine is ready." -ForegroundColor Green
            $ErrorActionPreference = $prevEap
            return $true
        }
        Write-Host "  Waiting for Docker engine..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 8
    }

    $ErrorActionPreference = $prevEap
    return $false
}

function Ensure-DockerEngine {
    param([int]$MaxRestarts = 1)

    if (Test-DockerEngine) { return }

    for ($attempt = 0; $attempt -le $MaxRestarts; $attempt++) {
        if ($attempt -gt 0) {
            Write-Host "Retrying Docker recovery (attempt $($attempt + 1))..." -ForegroundColor Yellow
        }
        if (Restart-DockerDesktop) { return }
    }

    throw @"
Docker Desktop is still not responding.

Manual recovery (do once, then re-run the script):
  1. Quit Docker Desktop from the system tray
  2. Open PowerShell as Administrator and run:  wsl --shutdown
  3. Start Docker Desktop and wait until the whale icon shows Running
  4. Docker Desktop -> Settings -> Resources: set Memory to at least 8 GB
  5. If Kubernetes is enabled, scale down K8s while using Compose:
       kubectl scale deployment --all --replicas=0 -n saint-dev

Do NOT use plain 'docker compose up' - use .\scripts\demo_stack.ps1 instead.
"@
}

function Invoke-Docker {
    Ensure-DockerEngine | Out-Null
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker @args
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap

    if ($exit -ne 0 -and -not (Test-DockerEngine)) {
        Write-Host "Docker command failed and engine is down - attempting one recovery..." -ForegroundColor Yellow
        if (Restart-DockerDesktop) {
            $ErrorActionPreference = "Continue"
            & docker @args
            $exit = $LASTEXITCODE
            $ErrorActionPreference = $prevEap
        }
    }

    if ($exit -ne 0) { exit $exit }
}

function Stop-K8sIfRunning {
    $kubectl = Get-Command kubectl -ErrorAction SilentlyContinue
    if (-not $kubectl) { return }

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $deploys = kubectl get deploy -n saint-dev -o name 2>$null
    $ErrorActionPreference = $prevEap
    if (-not $deploys) { return }

    Write-Host "Scaling down Kubernetes saint-dev deployments (reduces Docker Desktop memory pressure)..." -ForegroundColor Yellow
    $ErrorActionPreference = "Continue"
    kubectl scale deployment --all --replicas=0 -n saint-dev 2>$null | Out-Null
    $ErrorActionPreference = $prevEap
}

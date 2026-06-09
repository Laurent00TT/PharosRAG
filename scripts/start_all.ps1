<#
.SYNOPSIS
    Knowledge Base 一键启动 — 检查外部依赖,然后在独立窗口里拉起本项目服务。

.DESCRIPTION
    必需依赖(本脚本只 check,不替你启动):
      - Docker Desktop  → qdrant-kb container at :6333

    可选依赖(脚本不检查,仅在你跑 /ask 实验时需要手动启动):
      - LM Studio  → Qwen3.6-27B at :1235  (KBAgent 用本地 LLM 跑 HyDE/MQ/Agent loop)

    Production 路径(MCP / Claude 当 agent / kb CLI search)完全不需要本地 LLM。

    本项目服务(脚本会启,各开一个 PowerShell 窗口):
      - tool_server     :8000  (HTTP API + MCP /mcp/)
      - worker                  (ingestion 队列消费)

    模型服务(mineru / embedding / rerank / sparse)跑在云端 GPU,经 SSH 隧道
    够到本地栈 —— 本脚本不启动它们,见 the project README。

.EXAMPLE
    pwsh scripts/start_all.ps1
    pwsh scripts/start_all.ps1 -SkipWorker     # 只起 tool_server,不起 worker
    pwsh scripts/start_all.ps1 -CheckOnly      # 只检查依赖,不启任何服务
#>

[CmdletBinding()]
param(
    [switch]$SkipWorker,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "找不到 .venv: $PythonExe — 先在项目根目录跑 'uv sync' 或 'python -m venv .venv && pip install -e .'"
    exit 1
}

function Test-Endpoint {
    param([string]$Url, [string]$Name, [string]$HintIfDown)
    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        Write-Host ("  [ OK ] {0,-22} @ {1}" -f $Name, $Url) -ForegroundColor Green
        return $true
    } catch {
        Write-Host ("  [FAIL] {0,-22} @ {1}" -f $Name, $Url) -ForegroundColor Yellow
        Write-Host ("         => $HintIfDown") -ForegroundColor DarkYellow
        return $false
    }
}

function Test-QdrantContainer {
    try {
        $running = docker ps --filter "name=qdrant-kb" --filter "status=running" --format "{{.Names}}" 2>$null
        if ($running -eq "qdrant-kb") {
            Write-Host "  [ OK ] qdrant-kb container   running" -ForegroundColor Green
            return $true
        }
        # Container exists but stopped — try start
        $exists = docker ps -a --filter "name=qdrant-kb" --format "{{.Names}}" 2>$null
        if ($exists -eq "qdrant-kb") {
            Write-Host "  [WARN] qdrant-kb container   exists but stopped — starting..." -ForegroundColor Yellow
            docker start qdrant-kb | Out-Null
            Start-Sleep -Seconds 2
            return Test-QdrantContainer  # recurse once to re-check
        }
        Write-Host "  [FAIL] qdrant-kb container   not found" -ForegroundColor Yellow
        Write-Host "         => Run:  docker run -d --name qdrant-kb --restart unless-stopped \\" -ForegroundColor DarkYellow
        Write-Host "                  -p 6333:6333 -p 6334:6334 \\" -ForegroundColor DarkYellow
        Write-Host "                  -v `"$ProjectRoot/qdrant_data:/qdrant/storage`" qdrant/qdrant:v1.18.0" -ForegroundColor DarkYellow
        return $false
    } catch {
        Write-Host "  [FAIL] qdrant-kb container   docker not reachable: $_" -ForegroundColor Yellow
        return $false
    }
}

Write-Host ""
Write-Host "=== 外部依赖检查 ===" -ForegroundColor Cyan
$qdrant = Test-QdrantContainer

# Production MCP path only needs Qdrant (Claude is the agent).
# Local Qwen3.6 @ :1235 (LM Studio) is an OPTIONAL research-only dependency:
#   - Required ONLY if you run /ask  (KBAgent uses local LLM for HyDE/MQ/Agent loop)
#   - NOT required for MCP path, /search_docs, /ingestion (description channel disabled)
# Bridge (:3456) is fully decommissioned — see commit history.

$missing = -not $qdrant
if ($missing) {
    Write-Host ""
    Write-Host "⚠  Qdrant 不在线 → 整个 stack 起不来" -ForegroundColor Yellow
    if (-not $CheckOnly) {
        $ans = Read-Host "继续?(y/N)"
        if ($ans -notmatch '^[Yy]') { exit 1 }
    }
}

if ($CheckOnly) { if ($missing) { exit 1 } else { exit 0 } }

# ── 启动本项目服务,各开一个独立窗口 ─────────────────────────────────────
function Start-Service {
    param([string]$Title, [string]$ScriptPath, [string[]]$ExtraArgs)
    $argsList = @($ScriptPath) + $ExtraArgs
    $cmd = "Set-Location '$ProjectRoot'; & '$PythonExe' " + ($argsList -join ' ')
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "`$Host.UI.RawUI.WindowTitle = '$Title'; $cmd"
    )
    Write-Host ("  [ ↑ ] {0,-22} (new window)" -f $Title) -ForegroundColor Green
}

Write-Host ""
Write-Host "=== 启动本项目服务 ===" -ForegroundColor Cyan

# 1. tool_server  (FastAPI HTTP API). Model servers (mineru / embedding /
#    rerank / sparse) run on the cloud GPU and reach this stack via the SSH
#    tunnel — they are NOT launched here.
Start-Service -Title "tool_server :8000" `
    -ScriptPath "scripts/serve.py" `
    -ExtraArgs @("--port", "8000")

# 2. worker(可选 — pure ingestion query 不需要;但日常默认开)
if (-not $SkipWorker) {
    Start-Service -Title "worker (ingestion)" `
        -ScriptPath "scripts/worker.py" `
        -ExtraArgs @("--queue", "ingestion", "--poll-interval", "2")
}

Write-Host ""
Write-Host "✓ 已派发启动命令。每个服务在独立 PowerShell 窗口里跑,Ctrl+C 关单个服务。" -ForegroundColor Green
Write-Host "  健康检查:  curl http://localhost:8000/health" -ForegroundColor Gray
Write-Host "  停止全部:  关闭对应窗口 或 Stop-Process -Name python -Force(粗暴)" -ForegroundColor Gray
